"""True level-specific L1/L2/L3 WhiteboxSpineRunner with PrimitiveTable p50 binding.

[Role] Implement ``EvaluationRunner`` for onboarding pipelines by running
  distinct single-operator (L1), bounded subgraph/layer (L2), and full body
  (L3) evaluations through ``default_spine()`` + ``graph_costs`` on the real
  ``runtime.Engine``, binding calibrated ``PrimitiveTable`` ``sim_rate_p50``
  supplies directly into the evaluated ``Case``, and computing exact active
  ``busy_union`` (from Engine executed ``events``) alongside ``elapsed_span``.
[Invariants]
  * L1 (``device_kernel``) MUST execute a single operator (never whole-model
    body); L2 (``fx_region``) MUST execute a bounded subgraph strictly smaller
    than L3 (never whole-model body); L3 (``model_step``) executes full body.
  * Active ``busy_union`` sweeps ONLY executed ``report["events"]`` intervals
    (excluding queue waits) and fails closed if empty (never substituting
    ``elapsed_span``).
  * Rejects any ``service_regimes`` crutch and enforces ``table.sku == sku``.
[Timeouts & Locks] Deterministic local Engine replay; zero locks or device
  contexts.
[Side-Effect Policy] Zero filesystem writes or caller-mapping mutations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from mindie_sim.observations.multilevel_gate import (
    EngineInterval,
    GateCase,
    bind_engine_interval,
)
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.primitives.schema import PrimitiveTable
from mindie_sim.schema.common import ModelingError

from .contracts import (
    DEFAULT_CLOCK_OPERATOR,
    NS_PER_MS,
    VALID_CLOCK_OPERATORS,
)

_DEFAULT_ENGINE_START_NS = 100.0
_ENGINE_TRACE_LIMIT = 32
_GRAPH_EVENT_LIMIT = 65536
_DEFAULT_AIV_PER_AIC_RATIO = 2


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
    valid = sorted((float(s), float(e)) for s, e in intervals if float(e) > float(s))
    if not valid:
        return 0.0
    total = 0.0
    cur_s, cur_e = valid[0]
    for s, e in valid[1:]:
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def _apply_calibrated_primitive_rates(case: Any, table: PrimitiveTable) -> Any:
    """Apply calibrated ``sim_rate_p50`` rates from ``table`` onto every die in ``case``."""
    if table.sku != case.machine.sku:
        raise ModelingError(
            f"primitive_table_sku_mismatch:table={table.sku}:case={case.machine.sku}"
        )
    p01 = table.primitives.get("P01_datacopy")
    p04 = table.primitives.get("P04_mte3_store")
    p06 = table.primitives.get("P06_mmad")
    p09 = table.primitives.get("P09_vec_reduce_cast")

    updated_dies = []
    for die in case.machine.dies:
        rates = dict(die.rates)
        peaks = dict(die.compute_peaks)
        aiv = max(1, int(dict(die.resource_units).get("vector", die.compute_units * _DEFAULT_AIV_PER_AIC_RATIO)))
        aic = max(1, int(die.compute_units))
        if p01 is not None and p01.status == "calibrated":
            hbm_r = p01.metrics.get("hbm_read_bytes_per_s")
            hbm_w = p01.metrics.get("hbm_write_bytes_per_s")
            mte2_aiv = p01.metrics.get("mte2_aiv_hit_bytes_per_s_per_core")
            if hbm_r is not None and hbm_r.p50:
                rates["hbm"] = float(hbm_r.p50)
            if hbm_r is not None and hbm_w is not None and hbm_r.p50 and hbm_w.p50:
                rates["hbm"] = float(min(hbm_r.p50, hbm_w.p50))
            if mte2_aiv is not None and mte2_aiv.p50:
                rates["mte2_aiv"] = float(mte2_aiv.p50) * aiv
        if p04 is not None and p04.status == "calibrated":
            mte3_b = p04.metrics.get("mte3_bytes_per_s_per_aiv")
            if mte3_b is not None and mte3_b.p50:
                rates["mte3"] = float(mte3_b.p50) * aiv
        if p06 is not None and p06.status == "calibrated":
            for mode_name, mode_metrics in p06.modes.items():
                band = mode_metrics.get("cube_ops_per_s_per_aic")
                if band is not None and band.p50:
                    peaks[mode_name] = float(band.p50) * aic
        if p09 is not None and p09.status == "calibrated":
            vec_b = p09.metrics.get("reduce_ops_per_s_per_aiv")
            if vec_b is not None and vec_b.p50:
                rates["vector"] = float(vec_b.p50) * aiv
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
            raise ModelingError(
                f"whitebox_l1_requires_single_op_target:{case_id}"
            )
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
        elif target_layer is not None and not isinstance(target_layer, bool):
            layer_num = int(target_layer)
            chosen = [op for op in all_ops if int(op.layer) == layer_num]
        else:
            raise ModelingError(
                f"whitebox_l2_requires_subgraph_target:{case_id}"
            )
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


class WhiteboxSpineRunner:
    """Sealed ``EvaluationRunner`` executing true level-specific L1/L2/L3 Engine replays."""

    def __init__(
        self,
        *,
        clock_operator: str = DEFAULT_CLOCK_OPERATOR,
        primitive_table: PrimitiveTable | None = None,
    ) -> None:
        if clock_operator not in VALID_CLOCK_OPERATORS:
            raise ModelingError(f"whitebox_runner_clock_operator_invalid:{clock_operator}")
        self.clock_operator = clock_operator
        self.primitive_table = primitive_table

    def _materialize_case(self, case: GateCase, payload: Mapping[str, Any]) -> Any:
        from mindie_sim.api import compose_case
        from mindie_sim.prediction.structure import parse_case
        from mindie_sim.schema.case import Case

        raw_case = payload.get("sim_case")
        if raw_case is None:
            raise ModelingError(f"whitebox_case_missing_sim_case:{case.case_id}")
        if isinstance(raw_case, Case):
            sim_case: Any = raw_case
        elif isinstance(raw_case, Mapping):
            check_no_service_regimes(raw_case, path=f"{case.case_id}.sim_case")
            if "compose_model" in raw_case:
                sim_case = compose_case(
                    raw_case["compose_model"],
                    raw_case.get("compose_hardware", case.cohort.sku),
                    options=raw_case.get("compose_options"),
                )
            else:
                sim_case = parse_case(dict(raw_case))
        else:
            raise ModelingError(f"whitebox_case_invalid_type:{case.case_id}")
        if self.primitive_table is not None:
            sim_case = _apply_calibrated_primitive_rates(sim_case, self.primitive_table)
        return sim_case

    def _run_level(self, case: GateCase, *, expected_level: str) -> EngineInterval:
        from mindie_sim.prediction.graph_analysis import graph_costs
        from mindie_sim.prediction.results import StepRequest
        from mindie_sim.prediction.spine import default_spine
        from mindie_sim.stack_registry import LoweringRequest

        payload = dict(case.payload)
        sim_case = self._materialize_case(case, payload)
        step_req = StepRequest(
            phase=str(payload.get("phase") or case.cohort.phase or "decode"),  # type: ignore[arg-type]
            tokens=int(payload.get("query_tokens") or payload.get("tokens") or 1),
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
        report = graph_costs(
            sim_case,
            graphs,
            external_inputs=external_inputs,
            isolated=False,
            trace_limit=_GRAPH_EVENT_LIMIT,
        )
        elapsed_ms = float(report.get("duration_ms", 0.0))
        events = report.get("events") or ()
        active_intervals = [
            (float(ev["start_ms"]), float(ev["end_ms"]))
            for ev in events
            if (ev.get("serviced") or ev.get("demand"))
            and ev.get("start_ms") is not None
            and ev.get("end_ms") is not None
            and float(ev["end_ms"]) > float(ev["start_ms"])
        ]
        busy_ms = union_interval_span_ms(active_intervals)
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
