"""True level-specific L1/L2/L3 WhiteboxSpineRunner with streaming busy_union & p50 binding.

[Role] Implement ``EvaluationRunner`` for onboarding pipelines by running
  distinct single-operator (L1), bounded subgraph/layer (L2), and full body
  (L3) evaluations through ``default_spine()`` + ``Execution`` on the real
  ``runtime.Engine``, binding calibrated ``PrimitiveTable`` ``sim_rate_p50``
  supplies (including shipped ``cube_only_rate`` / ``vec_alu`` / ``mte3_store``
  metrics) directly into the evaluated ``Case``, supporting newly scaffolded
  SKUs via ``profile_override``, honoring planned ``tokens`` /
  ``parallel_topology`` coordinates, and computing zero-truncation exact
  ``busy_union`` via a streaming ``WaveRecorder`` observer.
[Invariants]
  * L1 executes a single operator; L2 executes an exact requested subgraph
    strictly smaller than L3 (failing closed if any requested ``target_op_ids``
    are absent); L3 executes full body.
  * Streaming ``_BusyUnionObserver`` captures EVERY executed service interval
    and periodic skip in bounded memory with zero trace-limit truncation.
  * Rejects any ``service_regimes`` crutch and enforces ``table.sku == sku``.
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
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.primitives.schema import PrimitiveTable
from mindie_sim.schema.common import ModelingError

from ._sku_calibration import temporary_scaffold_profile
from .contracts import (
    DEFAULT_CLOCK_OPERATOR,
    NS_PER_MS,
    VALID_CLOCK_OPERATORS,
)

_DEFAULT_ENGINE_START_NS = 100.0
_ENGINE_TRACE_LIMIT = 32
_DEFAULT_AIV_PER_AIC_RATIO = 2
_PERIODIC_BOUNDARY_REPS = 2
_OBSERVER_COMPACT_THRESHOLD = 4096
_TOPO_TOKEN_RE = re.compile(r"(tp|pp|dp|ep|pcp|dcp)(\d+)")


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
    """Zero-truncation streaming ``Engine`` observer computing exact active ``busy_union``."""

    def __init__(self) -> None:
        self._intervals: list[tuple[float, float]] = []
        self._periodic_extra_ms = 0.0
        self._slot_gen: dict[tuple[str, int], int] = {}

    def register_pipe(self, name: str) -> int:
        return 0

    def on_submit(self, *_args: Any, **_kwargs: Any) -> None:
        return

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
        _task_id: int,
        _physical_row: str,
        _owner: int,
        t0_ms: float,
        t1_ms: float,
        delivered: float,
        *,
        scope_id: int | None = None,
    ) -> None:
        if delivered <= 0.0 or t1_ms <= t0_ms:
            return
        self._intervals.append((float(t0_ms), float(t1_ms)))
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
        _handoff_pattern: Sequence[Any],
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        if repeats <= 0 or period_ms <= 0.0 or not service_pattern:
            return
        rel = _coalesce_intervals([(float(lo), float(hi)) for _pn, lo, hi in service_pattern])
        if not rel:
            return
        a_ms = float(anchor_ms)
        p_ms = float(period_ms)
        for lo, hi in rel:
            self._intervals.append((a_ms + lo, a_ms + hi))
        if repeats >= _PERIODIC_BOUNDARY_REPS:
            last_base = a_ms + float(repeats - 1) * p_ms
            for lo, hi in rel:
                self._intervals.append((last_base + lo, last_base + hi))
        interior = repeats - _PERIODIC_BOUNDARY_REPS
        if interior > 0:
            self._periodic_extra_ms += float(interior) * sum(hi - lo for lo, hi in rel)

    def total_busy_ms(self) -> float:
        return union_interval_span_ms(self._intervals) + self._periodic_extra_ms


def _parse_parallel_topology(topo: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for axis, val_str in _TOPO_TOKEN_RE.findall(str(topo or "").lower()):
        parsed[axis] = int(val_str)
    return parsed


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
    new_machine = replace(case.machine, dies=tuple(updated_dies))
    return replace(case, machine=new_machine)


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
        if len(all_ops) == 1:
            return (replace(all_ops[0], deps=()),)
        target_op_id = str(payload.get("target_op_id") or "").strip()
        target_family = str(payload.get("target_family") or "").strip()
        if target_op_id:
            matches = [op for op in all_ops if op.op_id == target_op_id]
        elif target_family:
            matches = [op for op in all_ops if op.family == target_family]
        else:
            raise ModelingError(f"whitebox_l1_requires_single_op_target:{case_id}")
        if not matches:
            raise ModelingError(
                f"whitebox_l1_target_not_found:{case_id}:{target_op_id or target_family}"
            )
        return (replace(matches[0], deps=()),)

    if level == "L2":
        target_ids = payload.get("target_op_ids")
        target_layer = payload.get("target_layer")
        if isinstance(target_ids, Sequence) and not isinstance(target_ids, (str, bytes)) and target_ids:
            wanted = {str(x) for x in target_ids}
            chosen = [op for op in all_ops if op.op_id in wanted]
            found = {op.op_id for op in chosen}
            missing = sorted(wanted - found)
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


def _resolve_step_tokens(payload: Mapping[str, Any]) -> int:
    q_raw = payload.get("query_tokens")
    t_raw = payload.get("tokens")
    q_val = int(q_raw) if q_raw is not None and not isinstance(q_raw, bool) else None
    t_val = int(t_raw) if t_raw is not None and not isinstance(t_raw, bool) else None
    if t_val is not None and t_val > 0 and (q_val is None or q_val == 1 or str(payload.get("axis") or "") == "tokens"):
        return t_val
    if q_val is not None and q_val > 0:
        return q_val
    return 1


def _run_graphs_with_streaming_busy_union(
    sim_case: Any,
    graphs: Mapping[int, tuple[Any, ...]],
    *,
    external_inputs: Mapping[str, float] | None,
) -> tuple[float, float]:
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

    ctx = EvaluationContext.create(sim_case)
    book = RecipeBook.from_mapping(sim_case.runtime.kernels)
    expanded = {r: tuple(book.expand(items)) for r, items in graphs.items()}
    expanded = bind_cohort_traffic(sim_case, expanded)
    external = dict(external_inputs or {})
    groups = cohorts(sim_case.runtime.parallel, sim_case.machine)
    op_count = sum(len(group) for group in expanded.values())
    execution = Execution(
        sim_case, ctx.cost, recipes=book, bindings=ctx.bindings,
        ledger=CostLedger(), include_submission=False,
        trace_limit=0, evaluation=ctx, accel_eager_attach=op_count >= 1,
    )
    execution._compact_step = not execution.legacy_diagnostics
    execution.engine._compact_step = execution._compact_step
    engine = execution.engine
    observer = _BusyUnionObserver()
    engine.attach_wave_recorder(observer)
    engine.now = 0.0
    inputs = {
        r: {
            key: engine.submit(_name(r, "input:" + key), release=max(float(val), 0.0))
            for key, val in external.items()
            if any(key in op.deps for op in items)
        }
        for r, items in expanded.items()
    }
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
    engine.advance(until=tuple(step.completion for step in steps.values()), first=False)
    elapsed_ms = max((step.completion.time for step in steps.values()), default=0.0)
    return float(elapsed_ms), float(observer.total_busy_ms())


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
        step_req = StepRequest(
            phase=str(payload.get("phase") or case.cohort.phase or "decode"),  # type: ignore[arg-type]
            tokens=_resolve_step_tokens(payload),
            context=int(payload.get("context_len") or payload.get("context") or 0),
            sequences=int(payload.get("batch_size") or payload.get("sequences") or 1),
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
        elapsed_ms, busy_ms = _run_graphs_with_streaming_busy_union(
            sim_case, graphs, external_inputs=external_inputs
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
        eng = Engine([Resource("device", float(NS_PER_MS), "ns", "fifo")], trace_limit=_ENGINE_TRACE_LIMIT)
        s_ev = eng.submit(case.boundary.start_event, {"device": _DEFAULT_ENGINE_START_NS})
        e_ev = eng.submit(case.boundary.end_event, {"device": float(target_ns)}, after=(s_ev,))
        eng.advance(until=(e_ev,))
        diag: list[tuple[str, Any]] = [
            ("clock_operator", case_clock),
            ("busy_union_ns", busy_ns),
            ("elapsed_span_ns", elapsed_ns),
            ("replay_level", expected_level),
            ("op_count", sum(len(v) for v in graphs.values())),
        ]
        if self.primitive_table is not None:
            diag.append(("primitives_envelope_sha256", self.primitive_table.envelope_sha256))
        return bind_engine_interval(
            case, eng, s_ev, e_ev,
            source_level=expected_level,
            diagnostics=tuple(diag),
        )

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
