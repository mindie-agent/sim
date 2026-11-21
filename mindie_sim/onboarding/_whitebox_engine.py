"""Real workload Engine execution, zero-truncation busy-union observer, and calibrated rate binding.

[Role] Execute lowered L1/L2/L3 operator graphs on ONE real workload
  ``runtime.Engine`` via ``Execution``, attach a zero-truncation streaming
  ``_BusyUnionObserver`` to measure exact coalesced ``busy_union`` spans and
  per-row/per-op work without trace truncation, apply calibrated
  ``PrimitiveTable`` ``sim_rate_p50`` supplies onto every die, construct
  canonical executed Lean ``(program, event_dag)`` witnesses, and forbid
  ``service_regimes`` crutches.
[Invariants]
  * ZERO synthetic second Engine: returns the actual workload ``execution.engine``
    and real boundary ``Event`` handles for HMAC receipt binding.
  * Streaming ``_BusyUnionObserver`` compacts intervals in bounded memory so
    ``busy_union`` and per-row bounds never drop events on long contexts.
  * Rejects ``PrimitiveTable`` SKU mismatches (``table.sku != case.machine.sku``)
    and any nested ``service_regimes`` payload keys.
[Timeouts & Locks] Deterministic in-memory Engine replay; zero I/O, locks, or
  device contexts.
[Side-Effect Policy] Zero filesystem writes or caller-mapping mutations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from mindie_sim.primitives.schema import PrimitiveTable
from mindie_sim.schema.common import ModelingError

from .contracts import NS_PER_MS

_MIN_WORKLOAD_TRACE_LIMIT = 65536
_TRACE_PER_OP_FACTOR = 256
_TRACE_PER_SEQ_FACTOR = 128
_TRACE_KV_PAGE_SIZE = 128
_TRACE_PER_KV_PAGE_FACTOR = 6
_MAX_WORKLOAD_TRACE_LIMIT = 1_048_576
_DEFAULT_AIV_PER_AIC_RATIO = 2
_PERIODIC_BOUNDARY_REPS = 2
_OBSERVER_COMPACT_THRESHOLD = 4096


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


def _run_graphs_on_real_engine(
    sim_case: Any,
    graphs: Mapping[int, tuple[Any, ...]],
    *,
    start_event_name: str,
    end_event_name: str,
    external_inputs: Mapping[str, float] | None,
    family_registry: Any = None,
    sequences: int = 1,
    context: int = 0,
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
    kv_pages = max(1, int(sequences)) * max(1, (max(0, int(context)) + _TRACE_KV_PAGE_SIZE) // _TRACE_KV_PAGE_SIZE + 1)
    trace_limit = min(
        _MAX_WORKLOAD_TRACE_LIMIT,
        max(
            _MIN_WORKLOAD_TRACE_LIMIT,
            op_count * _TRACE_PER_OP_FACTOR,
            int(sequences) * _TRACE_PER_SEQ_FACTOR,
            kv_pages * _TRACE_PER_KV_PAGE_FACTOR,
        ),
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


__all__ = [
    "_BusyUnionObserver",
    "_apply_calibrated_primitive_rates",
    "_build_executed_witness_program_dag",
    "_run_graphs_on_real_engine",
    "check_no_service_regimes",
    "union_interval_span_ms",
]
