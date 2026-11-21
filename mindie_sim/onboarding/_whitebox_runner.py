"""True level-specific L1/L2/L3 WhiteboxSpineRunner bound to the real workload Engine.

[Role] Implement ``EvaluationRunner`` for onboarding pipelines by running
  distinct single-operator (L1), bounded subgraph/layer (L2), and full body
  (L3) evaluations through ``default_spine()`` + ``Execution`` on ONE real
  workload ``runtime.Engine``, binding ``EngineRunReceipt`` directly to THAT
  real workload ``Engine`` and its real boundary ``Event`` handles (zero
  synthetic second Engine), binding calibrated ``PrimitiveTable`` ``sim_rate_p50``
  supplies, supporting newly scaffolded SKUs via ``profile_override``, honoring
  all planned dimensions (``context_len``, ``tokens``, ``batch_size``,
  ``query_tokens``, ``batch_x_kv_heads``, ``parallel_topology`` with ``dcp->cp``),
  computing zero-truncation streaming ``busy_union``, and emitting canonical
  workload ``(program, event_dag)`` digests for Lean non-inversion joining.
[Invariants]
  * ZERO dummy/second Engine: ``bind_engine_interval`` signs the actual
    workload ``execution.engine`` UID and boundary events.
  * L1 enforces ``target_op_id`` / ``target_family`` even for single-op graphs;
    L2 executes a strict subgraph strictly smaller than L3; L3 executes full body.
  * Streaming ``_BusyUnionObserver`` captures every active interval per physical
    row in bounded memory, producing set-inclusion-sound resource lower bounds.
  * Rejects ``service_regimes`` crutches and enforces ``table.sku == sku``.
[Timeouts & Locks] Deterministic local Engine replay; zero locks or device
  contexts.
[Side-Effect Policy] Zero filesystem writes or caller-mapping mutations;
  temporary scaffold profile registrations always unwind in ``finally``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import concurrent.futures
from dataclasses import asdict, is_dataclass, replace
import hashlib
import multiprocessing
import os
import re
import threading
from typing import Any

from mindie_sim.observations.multilevel_gate import (
    EngineInterval,
    GateCase,
    bind_engine_interval,
)
from mindie_sim.primitives.schema import PrimitiveTable
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import local_kv_heads as compute_local_kv_heads

from ._lean_and_receipts import program_dag_sha256
from ._sku_calibration import temporary_scaffold_profile
from ._whitebox_binding import _opt_pos_int, _verify_measured_binding
from ._whitebox_engine import (
    _apply_calibrated_primitive_rates,
    _build_executed_witness_program_dag,
    _run_graphs_on_real_engine,
    check_no_service_regimes,
    union_interval_span_ms,
)
from .contracts import (
    DEFAULT_CLOCK_OPERATOR,
    NS_PER_MS,
    VALID_CLOCK_OPERATORS,
    canonical_json_bytes,
)

_MAX_DIAG_OP_IDS = 16
_MAX_REPLAY_WORKERS = 16
_MIN_PARALLEL_CASES = 2
_ALLOWED_MODEL_OVERRIDES = frozenset({"layers", "index_topk", "index_dtype"})
_TOPO_TOKEN_RE = re.compile(r"(tp|pp|dp|ep|pcp|dcp)(\d+)")
_INTERVAL_HAS_CLOCK_OP = hasattr(EngineInterval, "clock_operator")


def _parse_parallel_topology(topo: str) -> dict[str, int]:
    return {
        axis: int(val_str)
        for axis, val_str in _TOPO_TOKEN_RE.findall(str(topo or "").lower())
    }



def _apply_model_overrides(
    raw_dict: Mapping[str, Any], overrides: Mapping[str, Any], *, case_id: str
) -> dict[str, Any]:
    """Apply supported declarative ``model_overrides`` onto a composed case mapping."""
    unknown = sorted(str(k) for k in overrides if str(k) not in _ALLOWED_MODEL_OVERRIDES)
    if unknown:
        raise ModelingError(f"whitebox_unsupported_model_override:{case_id}:{','.join(unknown)}")
    model_d = dict(raw_dict["model"])
    if "layers" in overrides:
        n_layers = _opt_pos_int(overrides["layers"])
        if n_layers is None:
            raise ModelingError(f"whitebox_invalid_model_override:{case_id}:layers={overrides['layers']!r}")
        model_d["layers"] = n_layers
        model_d["body_layers"] = n_layers
        if isinstance(model_d.get("layer_plans"), list):
            model_d["layer_plans"] = list(model_d["layer_plans"])[:n_layers]
    if "index_topk" in overrides:
        tk = _opt_pos_int(overrides["index_topk"])
        if tk is None:
            raise ModelingError(f"whitebox_invalid_model_override:{case_id}:index_topk={overrides['index_topk']!r}")
        mla_d = dict(model_d.get("mla") or {})
        mla_d["index_topk"] = tk
        model_d["mla"] = mla_d
        if isinstance(model_d.get("layer_plans"), list):
            updated_lps = []
            for lp in model_d["layer_plans"]:
                lp_d = dict(lp) if isinstance(lp, Mapping) else lp
                if isinstance(lp_d, dict) and lp_d.get("index_topk") is not None:
                    lp_d["index_topk"] = tk
                updated_lps.append(lp_d)
            model_d["layer_plans"] = updated_lps
    if "index_dtype" in overrides:
        idtype = str(overrides["index_dtype"] or "").strip()
        if not idtype:
            raise ModelingError(f"whitebox_invalid_model_override:{case_id}:index_dtype")
        quant_d = dict(model_d.get("quant") or {})
        quant_d["index_dtype"] = idtype
        model_d["quant"] = quant_d
    return {**dict(raw_dict), "model": model_d}


def _select_level_ops(
    all_ops: tuple[Any, ...],
    *,
    level: str,
    payload: Mapping[str, Any],
    case_id: str,
) -> tuple[Any, ...]:
    if level == "L3":
        return all_ops
    if level == "L1":
        binding = payload.get("measured_binding") if isinstance(payload.get("measured_binding"), Mapping) else {}
        target_op_id = str(payload.get("target_op_id") or binding.get("target_op_id") or "").strip()
        target_family = str(payload.get("target_family") or binding.get("family") or "").strip()
        if target_op_id:
            matches = [op for op in all_ops if op.op_id == target_op_id]
        elif target_family:
            matches = [op for op in all_ops if op.family == target_family]
        elif len(all_ops) == 1:
            matches = [all_ops[0]]
        else:
            raise ModelingError(f"whitebox_l1_requires_single_op_target:{case_id}")
        if not matches:
            raise ModelingError(
                f"whitebox_l1_target_not_found:{case_id}:{target_op_id or target_family}"
            )
        chosen = matches[0]
        if target_family and chosen.family != target_family:
            raise ModelingError(
                f"whitebox_l1_family_mismatch:{case_id}:expected={target_family}:got={chosen.family}"
            )
        return (replace(chosen, deps=()),)

    if level == "L2":
        target_ids = payload.get("target_op_ids")
        target_layer = payload.get("target_layer")
        if isinstance(target_ids, Sequence) and not isinstance(target_ids, (str, bytes)) and target_ids:
            wanted = {str(x) for x in target_ids}
            chosen = [op for op in all_ops if op.op_id in wanted]
            missing = sorted(wanted - {op.op_id for op in chosen})
            if missing:
                raise ModelingError(
                    f"whitebox_l2_target_op_ids_missing:{case_id}:{','.join(missing)}"
                )
        elif target_layer is not None and not isinstance(target_layer, bool):
            layer_num = int(target_layer)
            chosen = [op for op in all_ops if int(op.layer) == layer_num]
        else:
            raise ModelingError(f"whitebox_l2_requires_subgraph_target:{case_id}")
        if not chosen:
            raise ModelingError(f"whitebox_l2_subgraph_empty:{case_id}")
        if len(chosen) >= len(all_ops) and len(all_ops) > 1:
            raise ModelingError(f"whitebox_l2_equals_full_l3_body_forbidden:{case_id}")
        chosen_ids = {op.op_id for op in chosen}
        return tuple(
            replace(op, deps=tuple(d for d in op.deps if d in chosen_ids))
            for op in chosen
        )
    raise ModelingError(f"whitebox_level_unsupported:{level}")


def _resolve_step_dimensions(
    payload: Mapping[str, Any],
    *,
    local_kv_heads: int,
    case_id: str,
) -> tuple[int, int, int, int]:
    ctx_raw = payload.get("context_len") if payload.get("context_len") is not None else payload.get("context")
    context = int(ctx_raw) if ctx_raw is not None and not isinstance(ctx_raw, bool) else 0

    b_raw = payload.get("batch_size") if payload.get("batch_size") is not None else payload.get("sequences")
    batch = int(b_raw) if b_raw is not None and not isinstance(b_raw, bool) else 1

    axis_name = str(payload.get("axis") or "").strip()
    bxkv_raw = payload.get("batch_x_kv_heads")
    if bxkv_raw is None and axis_name == "batch_x_kv_heads":
        bxkv_raw = payload.get("value")
    bxkv = int(bxkv_raw) if bxkv_raw is not None and not isinstance(bxkv_raw, bool) else None
    kv_per_rank = max(1, int(local_kv_heads))
    if bxkv is not None:
        if bxkv < 1 or bxkv % kv_per_rank != 0:
            raise ModelingError(
                f"whitebox_batch_x_kv_heads_indivisible:{case_id}:target={bxkv}:local_kv_heads={kv_per_rank}"
            )
        derived_seqs = bxkv // kv_per_rank
        if batch > 1 and batch * kv_per_rank != bxkv:
            raise ModelingError(
                f"whitebox_batch_x_kv_heads_conflict:{case_id}:batch={batch}:local_kv_heads={kv_per_rank}:target={bxkv}"
            )
        sequences = max(1, derived_seqs)
    else:
        sequences = max(1, batch)

    q_raw = payload.get("query_tokens")
    t_raw = payload.get("tokens")
    q_val = int(q_raw) if q_raw is not None and not isinstance(q_raw, bool) else None
    t_val = int(t_raw) if t_raw is not None and not isinstance(t_raw, bool) else None
    if axis_name == "tokens" and t_val is not None and t_val > 0:
        tokens = max(1, t_val // sequences) if sequences > 1 and t_val >= sequences else t_val
    elif q_val is not None and q_val > 0:
        if sequences == 1 and q_val == 1 and t_val is not None and t_val > 1 and axis_name in {"", "tokens"}:
            tokens = t_val
        else:
            tokens = q_val
    elif t_val is not None and t_val > 0:
        tokens = max(1, t_val // sequences) if sequences > 1 and t_val >= sequences else t_val
    else:
        tokens = 1
    actual_bxkv = int(sequences) * kv_per_rank
    return tokens, context, sequences, actual_bxkv



_WORKER_RUNNER_CACHE: dict[str, Any] = {}


def eligible_replay_cases(
    cases: Sequence[GateCase], *, target_split: str = "all"
) -> list[GateCase]:
    """Filter ``cases`` using the exact replay-eligibility criteria of ``multilevel_gate.evaluate``."""
    out: list[GateCase] = []
    for case in cases:
        if target_split != "all" and case.split != target_split:
            continue
        if case.is_counterfactual and not case.has_matched_capture:
            continue
        is_rung = case.boundary.boundary_kind == "paired_rung_delta"
        arm_a, arm_b = case.rung_arm_a_measured_ns, case.rung_arm_b_measured_ns
        if is_rung and (arm_a is None or arm_b is None or arm_a <= 0 or arm_b <= 0):
            continue
        if not is_rung and (case.measured_ns is None or case.measured_ns <= 0):
            continue
        out.append(case)
    return out


def _plain_for_fingerprint(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain_for_fingerprint(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain_for_fingerprint(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _plain_for_fingerprint(asdict(value))
    return repr(value)


def _case_replay_work_key(case: GateCase) -> int:
    payload = dict(case.payload)
    ctx = max(1, int(payload.get("context_len") or payload.get("context") or 1))
    seq = max(1, int(payload.get("batch_size") or payload.get("sequences") or payload.get("batch_x_kv_heads") or 1))
    return ctx * seq


def _init_replay_worker(
    clock_operator: str,
    primitive_envelope: dict[str, Any] | None,
    profile_override: dict[str, Any] | None,
) -> None:
    from mindie_sim.primitives.schema import parse_primitive_table

    prim_table = (
        parse_primitive_table(primitive_envelope)
        if primitive_envelope is not None
        else None
    )
    _WORKER_RUNNER_CACHE["runner"] = WhiteboxSpineRunner(
        clock_operator=clock_operator,
        primitive_table=prim_table,
        profile_override=profile_override,
        max_workers=1,
    )


def _replay_whitebox_case_worker(
    args: tuple[GateCase, str],
) -> EngineInterval:
    case, expected_level = args
    runner: WhiteboxSpineRunner = _WORKER_RUNNER_CACHE["runner"]
    return runner._run_level(case, expected_level=expected_level)


class WhiteboxSpineRunner:
    """Sealed ``EvaluationRunner`` executing true level-specific L1/L2/L3 Engine replays."""

    def __init__(
        self,
        *,
        clock_operator: str = DEFAULT_CLOCK_OPERATOR,
        primitive_table: PrimitiveTable | None = None,
        profile_override: Mapping[str, Any] | None = None,
        max_workers: int | None = None,
    ) -> None:
        if clock_operator not in VALID_CLOCK_OPERATORS:
            raise ModelingError(f"whitebox_runner_clock_operator_invalid:{clock_operator}")
        self.clock_operator = clock_operator
        self.primitive_table = primitive_table
        self.profile_override = dict(profile_override) if profile_override is not None else None
        self.max_workers = max_workers
        self._prefetched: dict[tuple[str, str, str], EngineInterval] = {}

    def _case_cache_key(self, case: GateCase, expected_level: str) -> tuple[str, str, str]:
        payload_plain = _plain_for_fingerprint(dict(case.payload))
        fp_obj = {
            "case_id": case.case_id,
            "level": expected_level,
            "split": case.split,
            "sku": case.cohort.sku,
            "phase": case.cohort.phase,
            "topology": case.cohort.parallel_topology,
            "raw_sha256": case.boundary.raw_sha256,
            "start_event": case.boundary.start_event,
            "end_event": case.boundary.end_event,
            "boundary_kind": case.boundary.boundary_kind,
            "clock_operator": self.clock_operator,
            "primitives_sha": self.primitive_table.envelope_sha256 if self.primitive_table is not None else "",
            "profile_override": _plain_for_fingerprint(self.profile_override or {}),
            "payload": payload_plain,
        }
        fp = hashlib.sha256(canonical_json_bytes(fp_obj)).hexdigest()
        return (case.case_id, expected_level, fp)

    def _resolve_workers(self, case_count: int) -> int:
        if case_count < _MIN_PARALLEL_CASES:
            return 1
        if self.max_workers is not None:
            return max(1, min(case_count, int(self.max_workers)))
        env_raw = os.environ.get("MINDIE_SIM_REPLAY_WORKERS", "").strip()
        if env_raw:
            try:
                return max(1, min(case_count, int(env_raw)))
            except ValueError:
                pass
        return max(1, min(case_count, os.cpu_count() or 1, _MAX_REPLAY_WORKERS))

    def prefetch_cases(self, cases: Sequence[GateCase]) -> int:
        """Execute independent GateCase replays concurrently via bounded ProcessPoolExecutor."""
        self._prefetched.clear()
        seen_ids: set[tuple[str, str]] = set()
        for case in cases:
            pair = (case.case_id, str(case.level).upper())
            if pair in seen_ids:
                raise ModelingError(f"whitebox_duplicate_prefetch_case:{pair[0]}:{pair[1]}")
            seen_ids.add(pair)
        workers = self._resolve_workers(len(cases))
        if (
            workers <= 1
            or "fork" not in multiprocessing.get_all_start_methods()
            or threading.active_count() > 1
        ):
            return 0
        prim_env = (
            self.primitive_table.to_envelope_dict()
            if self.primitive_table is not None
            else None
        )
        indexed = [
            (
                idx,
                _case_replay_work_key(case),
                self._case_cache_key(case, str(case.level).upper()),
                (case, str(case.level).upper()),
            )
            for idx, case in enumerate(cases)
        ]
        try:
            ctx = multiprocessing.get_context("fork")
            pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_init_replay_worker,
                initargs=(self.clock_operator, prim_env, self.profile_override),
            )
        except TimeoutError:
            raise
        except (PermissionError, OSError):
            return 0
        staged: dict[tuple[str, str, str], EngineInterval] = {}
        with pool:
            futures_by_idx: dict[int, tuple[tuple[str, str, str], concurrent.futures.Future[EngineInterval]]] = {}
            for idx, _work, cache_key, task in sorted(indexed, key=lambda item: item[1], reverse=True):
                futures_by_idx[idx] = (cache_key, pool.submit(_replay_whitebox_case_worker, task))
            for idx in range(len(indexed)):
                cache_key, fut = futures_by_idx[idx]
                staged[cache_key] = fut.result()
        self._prefetched = staged
        return len(indexed)

    def _materialize_case(self, case: GateCase, payload: Mapping[str, Any]) -> Any:
        from mindie_sim.api import compose_case
        from mindie_sim.compose import compose_case as compose_case_dict
        from mindie_sim.prediction.structure import parse_case
        from mindie_sim.schema.case import Case

        raw_case = payload.get("sim_case")
        if raw_case is None:
            raise ModelingError(f"whitebox_case_missing_sim_case:{case.case_id}")
        topo_req = _parse_parallel_topology(
            str(payload.get("parallel_topology") or case.cohort.parallel_topology or "")
        )
        with temporary_scaffold_profile(case.cohort.sku, self.profile_override):
            if isinstance(raw_case, Case):
                sim_case: Any = raw_case
            elif isinstance(raw_case, Mapping):
                check_no_service_regimes(raw_case, path=f"{case.case_id}.sim_case")
                if "compose_model" in raw_case:
                    opts = dict(raw_case.get("compose_options") or {})
                    for axis, val in topo_req.items():
                        opts.setdefault(axis, val)
                    overrides = raw_case.get("model_overrides")
                    if isinstance(overrides, Mapping) and overrides:
                        raw_dict = compose_case_dict(
                            model=str(raw_case["compose_model"]),
                            hardware=str(raw_case.get("compose_hardware", case.cohort.sku)),
                            **opts,
                        )
                        sim_case = parse_case(
                            _apply_model_overrides(raw_dict, overrides, case_id=case.case_id)
                        )
                    else:
                        sim_case = compose_case(
                            raw_case["compose_model"],
                            raw_case.get("compose_hardware", case.cohort.sku),
                            options=opts or None,
                        )
                else:
                    sim_case = parse_case(dict(raw_case))
            else:
                raise ModelingError(f"whitebox_case_invalid_type:{case.case_id}")
        mat_sku = str(getattr(sim_case.machine, "sku", "") or "").strip()
        if mat_sku != str(case.cohort.sku).strip():
            raise ModelingError(
                f"whitebox_materialized_sku_mismatch:{case.case_id}:expected={case.cohort.sku}:materialized={mat_sku}"
            )
        par = sim_case.runtime.parallel
        for axis, req_val in topo_req.items():
            actual_val = int(getattr(par, axis, 1))
            if actual_val != req_val:
                raise ModelingError(
                    f"whitebox_parallel_topology_mismatch:{case.case_id}:{axis}:expected={req_val}:got={actual_val}"
                )
        if self.primitive_table is not None:
            sim_case = _apply_calibrated_primitive_rates(sim_case, self.primitive_table)
        return sim_case

    def _run_level(self, case: GateCase, *, expected_level: str) -> EngineInterval:
        from mindie_sim.prediction.results import StepRequest
        from mindie_sim.prediction.spine import default_spine
        from mindie_sim.stack_registry import LoweringRequest

        payload = dict(case.payload)
        sim_case = self._materialize_case(case, payload)
        local_kv = int(
            compute_local_kv_heads(
                int(getattr(sim_case.model, "num_kv_heads", 1) or 1),
                max(1, int(getattr(sim_case.runtime.parallel, "tp", 1) or 1)),
            )
        )
        tokens, context, sequences, actual_bxkv = _resolve_step_dimensions(
            payload, local_kv_heads=local_kv, case_id=case.case_id
        )
        step_req = StepRequest(
            phase=str(payload.get("phase") or case.cohort.phase or "decode"),  # type: ignore[arg-type]
            tokens=tokens,
            context=context,
            sequences=sequences,
            scope="body",
        )
        spine = default_spine()
        prepared = spine.lower(sim_case, LoweringRequest(step=step_req, tier="mainline"))
        graphs: dict[int, tuple[Any, ...]] = {}
        for entry in prepared.graph.executable:
            chosen = _select_level_ops(
                entry.ops, level=expected_level, payload=payload, case_id=case.case_id
            )
            _verify_measured_binding(
                chosen,
                prepared.case,
                payload,
                local_kv_heads=local_kv,
                expected_level=expected_level,
                case_id=case.case_id,
            )
            graphs[entry.replica] = chosen
        external_inputs = {"analysis_input": 0.0} if any(
            "analysis_input" in op.deps for ops in graphs.values() for op in ops
        ) else None
        engine, start_ev, end_ev, elapsed_ms, busy_ms, observer = _run_graphs_on_real_engine(
            prepared.case,
            graphs,
            start_event_name=case.boundary.start_event,
            end_event_name=case.boundary.end_event,
            external_inputs=external_inputs,
            family_registry=spine.families,
            sequences=sequences,
            context=context,
        )
        elapsed_ns = int(round(elapsed_ms * NS_PER_MS))
        busy_ns = int(round(busy_ms * NS_PER_MS))
        if busy_ns == elapsed_ns + 1:
            busy_ns = elapsed_ns
        case_clock = str(payload.get("clock_operator") or self.clock_operator)
        if case_clock not in VALID_CLOCK_OPERATORS:
            raise ModelingError(f"whitebox_case_clock_operator_invalid:{case.case_id}:{case_clock}")
        if elapsed_ns <= 0:
            raise ModelingError(f"whitebox_prediction_nonpositive:{case.case_id}")
        if case_clock == "busy_union" and busy_ns <= 0:
            raise ModelingError(f"whitebox_busy_union_empty:{case.case_id}")
        target_ns = busy_ns if case_clock == "busy_union" else elapsed_ns
        lean_prog, lean_dag = _build_executed_witness_program_dag(
            graphs, observer, target_ns=target_ns
        )
        prog_sha = program_dag_sha256(lean_prog, lean_dag)
        exec_families = sorted({str(op.family) for ops in graphs.values() for op in ops})
        exec_op_ids = [str(op.op_id) for ops in graphs.values() for op in ops][:_MAX_DIAG_OP_IDS]
        lowered_dims: dict[str, Any] = {
            "materialized_sku": str(sim_case.machine.sku),
            "tokens": step_req.tokens * step_req.sequences,
            "query_tokens": step_req.tokens,
            "context": step_req.context,
            "sequences": step_req.sequences,
            "local_kv_heads": local_kv,
            "batch_x_kv_heads": actual_bxkv,
            "topk": payload.get("topk"),
            "parallel_topology": str(payload.get("parallel_topology") or case.cohort.parallel_topology or "tp1"),
        }
        diag: list[tuple[str, Any]] = [
            ("clock_operator", case_clock),
            ("busy_union_ns", busy_ns),
            ("elapsed_span_ns", elapsed_ns),
            ("replay_level", expected_level),
            ("op_count", sum(len(v) for v in graphs.values())),
            ("engine_uid", str(engine.uid)),
            ("program_dag_sha256", prog_sha),
            ("lean_program", lean_prog),
            ("lean_dag", lean_dag),
            ("executed_op_families", exec_families),
            ("executed_op_ids", exec_op_ids),
            ("lowered_dimensions", lowered_dims),
        ]
        if self.primitive_table is not None:
            diag.append(("primitives_envelope_sha256", self.primitive_table.envelope_sha256))
        bind_kw: dict[str, Any] = {
            "source_level": expected_level,
            "diagnostics": tuple(diag),
        }
        if _INTERVAL_HAS_CLOCK_OP:
            bind_kw["clock_operator"] = case_clock
        return bind_engine_interval(case, engine, start_ev, end_ev, **bind_kw)

    def run_l1(self, case: GateCase) -> EngineInterval:
        cached = self._prefetched.pop(self._case_cache_key(case, "L1"), None)
        return cached if cached is not None else self._run_level(case, expected_level="L1")

    def run_l2(self, case: GateCase) -> EngineInterval:
        cached = self._prefetched.pop(self._case_cache_key(case, "L2"), None)
        return cached if cached is not None else self._run_level(case, expected_level="L2")

    def run_l3(self, case: GateCase) -> EngineInterval:
        cached = self._prefetched.pop(self._case_cache_key(case, "L3"), None)
        return cached if cached is not None else self._run_level(case, expected_level="L3")


__all__ = [
    "WhiteboxSpineRunner",
    "check_no_service_regimes",
    "eligible_replay_cases",
    "union_interval_span_ms",
]
