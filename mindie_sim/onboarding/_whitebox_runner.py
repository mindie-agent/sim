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
from dataclasses import replace
import re
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
from .contracts import (
    DEFAULT_CLOCK_OPERATOR,
    NS_PER_MS,
    VALID_CLOCK_OPERATORS,
)

_MIN_WORKLOAD_TRACE_LIMIT = 65536
_TRACE_PER_OP_FACTOR = 256
_TRACE_PER_SEQ_FACTOR = 128
_MAX_WORKLOAD_TRACE_LIMIT = 524288
_DEFAULT_AIV_PER_AIC_RATIO = 2
_PERIODIC_BOUNDARY_REPS = 2
_OBSERVER_COMPACT_THRESHOLD = 4096
_MAX_DIAG_OP_IDS = 16
_TOPO_TOKEN_RE = re.compile(r"(tp|pp|dp|ep|pcp|dcp)(\d+)")
_INTERVAL_HAS_CLOCK_OP = hasattr(EngineInterval, "clock_operator")


def check_no_service_regimes(payload: Any, *, path: str = "case") -> None:
    """Fail closed if any nested mapping declares non-empty ``service_regimes``."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_s = str(key)
            if key_s == "service_regimes" and value:
                raise ModelingError(f"whitebox_service_regimes_forbidden:{path}")
            check_no_service_regimes(value, path=f"{path}.{key_s}")
    elif isinstance(payload, (list, tuple)):
        for idx, item in enumerate(payload):
            check_no_service_regimes(item, path=f"{path}[{idx}]")


def union_interval_span_ms(intervals: Sequence[tuple[float, float]]) -> float:
    """Compute exact union duration of closed ``[start_ms, end_ms]`` intervals."""
    merged = _coalesce_intervals(intervals)
    return sum(hi - lo for lo, hi in merged)


def _coalesce_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    valid = sorted((float(s), float(e)) for s, e in intervals if float(e) > float(s))
    if not valid:
        return []
    merged: list[tuple[float, float]] = []
    cur_s, cur_e = valid[0]
    for s, e in valid[1:]:
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


class _BusyUnionObserver:
    """Zero-truncation streaming ``Engine`` observer computing exact ``busy_union`` & row spans."""

    def __init__(self) -> None:
        self._intervals: list[tuple[float, float]] = []
        self._by_row: dict[str, list[tuple[float, float]]] = {}
        self._by_op: dict[tuple[int, str], list[tuple[float, float]]] = {}
        self._task_op: dict[int, tuple[int, str]] = {}
        self._periodic_extra_ms = 0.0
        self._periodic_by_row: dict[str, float] = {}
        self._slot_gen: dict[tuple[str, int], int] = {}

    def register_pipe(self, name: str) -> int:
        return 0

    def on_submit(
        self,
        task_id: int = 0,
        _stage_id: str = "",
        _aliases: Sequence[str] = (),
        _predecessors: Sequence[int] = (),
        _signals: Sequence[str] = (),
        occurrence: Any = None,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        if occurrence is not None:
            op_id = str(getattr(occurrence, "op_id", "") or "").strip()
            if op_id:
                replica = int(getattr(occurrence, "replica", 0) or 0)
                self._task_op[int(task_id)] = (replica, op_id)

    def on_admit(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def on_allocation(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def allocate_slot_generation(self, pool: str, owner: int, *_args: Any, **_kwargs: Any) -> tuple[int, int]:
        key = (str(pool), int(owner))
        gen = self._slot_gen.get(key, 0) + 1
        self._slot_gen[key] = gen
        return 0, gen

    def release_slot_generation(self, pool: str, owner: int, *_args: Any, **_kwargs: Any) -> tuple[int, int]:
        key = (str(pool), int(owner))
        return 0, self._slot_gen.get(key, 0)

    def on_service(
        self,
        task_id: int,
        physical_row: str,
        _owner: int,
        t0_ms: float,
        t1_ms: float,
        delivered: float,
        *,
        scope_id: int | None = None,
    ) -> None:
        del scope_id
        if delivered <= 0.0 or t1_ms <= t0_ms:
            return
        span = (float(t0_ms), float(t1_ms))
        self._intervals.append(span)
        row_key = str(physical_row or "device")
        bucket = self._by_row.setdefault(row_key, [])
        bucket.append(span)
        if len(bucket) >= _OBSERVER_COMPACT_THRESHOLD:
            self._by_row[row_key] = _coalesce_intervals(bucket)
        op_key = self._task_op.get(int(task_id))
        if op_key is not None:
            op_bucket = self._by_op.setdefault(op_key, [])
            op_bucket.append(span)
            if len(op_bucket) >= _OBSERVER_COMPACT_THRESHOLD:
                self._by_op[op_key] = _coalesce_intervals(op_bucket)
        if len(self._intervals) >= _OBSERVER_COMPACT_THRESHOLD:
            self._intervals = _coalesce_intervals(self._intervals)

    def extract_cycle_pattern(
        self, start_ms: float, end_ms: float, *_args: Any, **_kwargs: Any
    ) -> tuple[tuple[tuple[str, float, float], ...], tuple[Any, ...]]:
        s0, s1 = float(start_ms), float(end_ms)
        if s1 <= s0:
            return (), ()
        merged = _coalesce_intervals(self._intervals)
        self._intervals = merged
        clipped: list[tuple[str, float, float]] = []
        for lo, hi in merged:
            if hi <= s0 or lo >= s1:
                continue
            c_lo = max(lo, s0) - s0
            c_hi = min(hi, s1) - s0
            if c_hi > c_lo:
                clipped.append(("active", c_lo, c_hi))
        return tuple(clipped), ()

    def on_period(
        self,
        anchor_ms: float,
        period_ms: float,
        repeats: int,
        service_pattern: Sequence[tuple[str, float, float]],
        handoff_pattern: Sequence[Any] = (),
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        del handoff_pattern
        if repeats <= 0 or period_ms <= 0.0 or not service_pattern:
            return
        rel = _coalesce_intervals([(float(lo), float(hi)) for _pn, lo, hi in service_pattern])
        if not rel:
            return
        a_ms, p_ms = float(anchor_ms), float(period_ms)
        for lo, hi in rel:
            self._intervals.append((a_ms + lo, a_ms + hi))
        if repeats >= _PERIODIC_BOUNDARY_REPS:
            last_base = a_ms + float(repeats - 1) * p_ms
            for lo, hi in rel:
                self._intervals.append((last_base + lo, last_base + hi))
        interior = repeats - _PERIODIC_BOUNDARY_REPS
        if interior > 0:
            extra = float(interior) * sum(hi - lo for lo, hi in rel)
            self._periodic_extra_ms += extra
            self._periodic_by_row["periodic"] = self._periodic_by_row.get("periodic", 0.0) + extra

    def total_busy_ms(self) -> float:
        return union_interval_span_ms(self._intervals) + self._periodic_extra_ms

    def row_busy_ns(self, *, cap_ns: int) -> tuple[int, ...]:
        """Return sorted positive per-physical-row coalesced busy spans bounded by ``cap_ns``."""
        spans: list[int] = []
        for row_name in sorted(self._by_row):
            ms = union_interval_span_ms(self._by_row[row_name]) + self._periodic_by_row.get(row_name, 0.0)
            ns = min(cap_ns, int(round(ms * NS_PER_MS)))
            if ns > 0:
                spans.append(ns)
        if not spans and cap_ns > 0:
            spans.append(cap_ns)
        return tuple(spans)


    def op_busy_ns(self, op_key: tuple[int, str], *, cap_ns: int) -> int:
        """Return coalesced busy duration in ns observed for one executed op."""
        intervals = self._by_op.get(op_key)
        if not intervals:
            return 0
        return min(cap_ns, int(round(union_interval_span_ms(intervals) * NS_PER_MS)))


def _parse_parallel_topology(topo: str) -> dict[str, int]:
    return {
        axis: int(val_str)
        for axis, val_str in _TOPO_TOKEN_RE.findall(str(topo or "").lower())
    }


def _build_executed_witness_program_dag(
    graphs: Mapping[int, tuple[Any, ...]],
    observer: _BusyUnionObserver,
    *,
    target_ns: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build exact Lean ``(program, dag)`` from executed graph ops, deps, and row work."""
    row_ns = observer.row_busy_ns(cap_ns=target_ns)
    lean_prog = {"num_resources": len(row_ns), "global_work": list(row_ns), "site_modes": []}
    node_idx: dict[tuple[int, str], int] = {}
    node_quanta: list[int] = []
    ordered_nodes: list[tuple[int, Any]] = []
    for replica in sorted(graphs):
        for op in graphs[replica]:
            key = (int(replica), str(op.op_id))
            if key in node_idx:
                continue
            node_idx[key] = len(ordered_nodes)
            ordered_nodes.append((int(replica), op))
            q_ns = observer.op_busy_ns(key, cap_ns=target_ns)
            node_quanta.append(q_ns if q_ns > 0 else (1 if target_ns > 0 else 0))
    edges: list[dict[str, int]] = []
    for dst_idx, (replica, op) in enumerate(ordered_nodes):
        for dep in getattr(op, "deps", ()) or ():
            src_idx = node_idx.get((replica, str(dep)))
            if src_idx is not None and src_idx != dst_idx:
                edges.append({"src": src_idx, "dst": dst_idx, "wire_ns": 0})
    if not node_quanta:
        node_quanta = [max(row_ns)]
    lean_dag = {"node_quantum_ns": node_quanta, "edges": edges}
    return lean_prog, lean_dag


def _apply_calibrated_primitive_rates(case: Any, table: PrimitiveTable) -> Any:
    """Apply calibrated ``sim_rate_p50`` rates from ``table`` onto every die in ``case``."""
    if table.sku != case.machine.sku:
        raise ModelingError(
            f"primitive_table_sku_mismatch:table={table.sku}:case={case.machine.sku}"
        )
    p01 = table.primitives.get("P01_datacopy")
    p04 = table.primitives.get("P04_mte3_store")
    p06 = table.primitives.get("P06_mmad")
    p07 = table.primitives.get("P07_vec_alu")
    p09 = table.primitives.get("P09_vec_reduce_cast")

    updated_dies = []
    for die in case.machine.dies:
        rates = dict(die.rates)
        peaks = dict(die.compute_peaks)
        aiv = max(1, int(dict(die.resource_units).get("vector", die.compute_units * _DEFAULT_AIV_PER_AIC_RATIO)))
        aic = max(1, int(die.compute_units))
        if p01 is not None:
            hbm_r = p01.metrics.get("hbm_read_bytes_per_s")
            hbm_w = p01.metrics.get("hbm_write_bytes_per_s")
            mte2_aiv = p01.metrics.get("mte2_aiv_hit_bytes_per_s_per_core")
            mte2_aic = p01.metrics.get("mte2_aic_hit_bytes_per_s_per_core")
            if hbm_r is not None and hbm_r.p50:
                rates["hbm"] = float(hbm_r.p50)
            if hbm_r is not None and hbm_w is not None and hbm_r.p50 and hbm_w.p50:
                rates["hbm"] = float(min(hbm_r.p50, hbm_w.p50))
            if mte2_aiv is not None and mte2_aiv.p50:
                rates["mte2_aiv"] = float(mte2_aiv.p50) * aiv
            if mte2_aic is not None and mte2_aic.p50:
                rates["mte2_aic"] = float(mte2_aic.p50) * aic
        if p04 is not None:
            mte3_b = (
                p04.metrics.get("mte3_store_bytes_per_s_per_aiv")
                or p04.metrics.get("mte3_bytes_per_s_per_aiv")
            )
            if mte3_b is not None and mte3_b.p50:
                rates["mte3"] = float(mte3_b.p50) * aiv
        if p06 is not None:
            for mode_name, mode_metrics in p06.modes.items():
                band = (
                    mode_metrics.get("cube_only_rate")
                    or mode_metrics.get("cube_ops_per_s_per_aic")
                )
                if band is not None and band.p50:
                    peaks[mode_name] = float(band.p50) * aic
        vec_candidates: list[float] = []
        if p07 is not None:
            v_alu = p07.metrics.get("vec_alu_ops_per_s_per_aiv")
            if v_alu is not None and v_alu.p50:
                vec_candidates.append(float(v_alu.p50) * aiv)
        if p09 is not None:
            v_red = (
                p09.metrics.get("vec_reduce_ops_per_s_per_aiv")
                or p09.metrics.get("reduce_ops_per_s_per_aiv")
            )
            if v_red is not None and v_red.p50:
                vec_candidates.append(float(v_red.p50) * aiv)
        if vec_candidates:
            rates["vector"] = float(min(vec_candidates))
        updated_dies.append(
            replace(
                die,
                rates=tuple(sorted(rates.items())),
                compute_peaks=tuple(sorted(peaks.items())),
            )
        )
    return replace(case, machine=replace(case.machine, dies=tuple(updated_dies)))


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
        target_op_id = str(payload.get("target_op_id") or "").strip()
        target_family = str(payload.get("target_family") or "").strip()
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
    q_raw = payload.get("query_tokens")
    t_raw = payload.get("tokens")
    q_val = int(q_raw) if q_raw is not None and not isinstance(q_raw, bool) else None
    t_val = int(t_raw) if t_raw is not None and not isinstance(t_raw, bool) else None
    if t_val is not None and t_val > 0 and (q_val is None or q_val == 1 or str(payload.get("axis") or "") == "tokens"):
        tokens = t_val
    elif q_val is not None and q_val > 0:
        tokens = q_val
    else:
        tokens = 1

    ctx_raw = payload.get("context_len") if payload.get("context_len") is not None else payload.get("context")
    context = int(ctx_raw) if ctx_raw is not None and not isinstance(ctx_raw, bool) else 0

    b_raw = payload.get("batch_size") if payload.get("batch_size") is not None else payload.get("sequences")
    batch = int(b_raw) if b_raw is not None and not isinstance(b_raw, bool) else 1

    bxkv_raw = payload.get("batch_x_kv_heads")
    if bxkv_raw is None and str(payload.get("axis") or "") == "batch_x_kv_heads":
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
    actual_bxkv = int(sequences) * kv_per_rank
    return tokens, context, sequences, actual_bxkv


def _run_graphs_on_real_engine(
    sim_case: Any,
    graphs: Mapping[int, tuple[Any, ...]],
    *,
    start_event_name: str,
    end_event_name: str,
    external_inputs: Mapping[str, float] | None,
    family_registry: Any = None,
    sequences: int = 1,
) -> tuple[Any, Any, Any, float, float, _BusyUnionObserver]:
    from mindie_sim.prediction.cohort_traffic import bind_cohort_traffic
    from mindie_sim.prediction.cost import CostLedger
    from mindie_sim.prediction.dispatch import cohorts
    from mindie_sim.prediction.evaluation import EvaluationContext
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.graph_analysis import _name
    from mindie_sim.prediction.implementation_selection import graph_selections
    from mindie_sim.prediction.program import RecipeBook
    from mindie_sim.schema.operators import IterationWork
    from mindie_sim.schema.workload import IterationPlan

    book = RecipeBook.from_mapping(sim_case.runtime.kernels, family_registry=family_registry)
    ctx = EvaluationContext.create(sim_case, book)
    expanded = {r: tuple(book.expand(items)) for r, items in graphs.items()}
    expanded = bind_cohort_traffic(sim_case, expanded)
    external = dict(external_inputs or {})
    groups = cohorts(sim_case.runtime.parallel, sim_case.machine)
    op_count = sum(len(group) for group in expanded.values())
    trace_limit = min(
        _MAX_WORKLOAD_TRACE_LIMIT,
        max(_MIN_WORKLOAD_TRACE_LIMIT, op_count * _TRACE_PER_OP_FACTOR, int(sequences) * _TRACE_PER_SEQ_FACTOR),
    )
    execution = Execution(
        sim_case, ctx.cost, recipes=book, bindings=ctx.bindings,
        ledger=CostLedger(), include_submission=False,
        trace_limit=trace_limit, evaluation=ctx, accel_eager_attach=op_count >= 1,
    )
    execution._compact_step = not execution.legacy_diagnostics
    execution.engine._compact_step = execution._compact_step
    engine = execution.engine
    observer = _BusyUnionObserver()
    engine.attach_wave_recorder(observer)
    engine.now = 0.0
    start_ev = engine.submit(start_event_name, release=0.0)
    inputs: dict[int, dict[str, Any]] = {}
    for r, items in expanded.items():
        r_inputs: dict[str, Any] = {}
        for key, val in external.items():
            if any(key in op.deps for op in items):
                r_inputs[key] = (
                    start_ev
                    if float(val) <= 0.0
                    else engine.submit(_name(r, "input:" + key), release=float(val), after=(start_ev,))
                )
        inputs[r] = r_inputs
    works = {
        r: IterationWork(
            items, IterationPlan(()), sim_case.runtime.parallel.pp, 0, 0, None,
            "graph", "declared_graph", replay_scope=None,
            implementation_selections=graph_selections(items),
        )
        for r, items in expanded.items()
    }
    steps: dict[int, Any] = {}
    for group in groups:
        selected = {r: works[r] for r in group if r in works}
        if len(selected) > 1:
            steps.update(execution.add_group(selected, 0.0, external_inputs=inputs))
        elif selected:
            r, work = next(iter(selected.items()))
            steps[r] = execution.add(work, r, 0.0, external_inputs=inputs[r])
    end_ev = engine.submit(end_event_name, after=tuple(step.completion for step in steps.values()))
    engine.advance(until=(start_ev, end_ev), first=False)
    elapsed_ms = float(end_ev.time - start_ev.time) if (end_ev.time is not None and start_ev.time is not None) else 0.0
    return engine, start_ev, end_ev, elapsed_ms, float(observer.total_busy_ms()), observer


class WhiteboxSpineRunner:
    """Sealed ``EvaluationRunner`` executing true level-specific L1/L2/L3 Engine replays."""

    def __init__(
        self,
        *,
        clock_operator: str = DEFAULT_CLOCK_OPERATOR,
        primitive_table: PrimitiveTable | None = None,
        profile_override: Mapping[str, Any] | None = None,
    ) -> None:
        if clock_operator not in VALID_CLOCK_OPERATORS:
            raise ModelingError(f"whitebox_runner_clock_operator_invalid:{clock_operator}")
        self.clock_operator = clock_operator
        self.primitive_table = primitive_table
        self.profile_override = dict(profile_override) if profile_override is not None else None

    def _materialize_case(self, case: GateCase, payload: Mapping[str, Any]) -> Any:
        from mindie_sim.api import compose_case
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
            graphs[entry.replica] = _select_level_ops(
                entry.ops, level=expected_level, payload=payload, case_id=case.case_id
            )
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
        )
        elapsed_ns = int(round(elapsed_ms * NS_PER_MS))
        busy_ns = int(round(busy_ms * NS_PER_MS))
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
            "tokens": step_req.tokens,
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
        return self._run_level(case, expected_level="L1")

    def run_l2(self, case: GateCase) -> EngineInterval:
        return self._run_level(case, expected_level="L2")

    def run_l3(self, case: GateCase) -> EngineInterval:
        return self._run_level(case, expected_level="L3")


__all__ = [
    "WhiteboxSpineRunner",
    "check_no_service_regimes",
    "union_interval_span_ms",
]
