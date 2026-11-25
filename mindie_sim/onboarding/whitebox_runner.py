"""White-box single-Engine L1/L2/L3 evaluation runner and sealed gate scorer.

[Role] Provide context-local scaffold profile injection
  (``temporary_scaffold_profile``), single-Engine ``WhiteboxSpineRunner`` with
  zero-truncation streaming ``_BusyUnionObserver`` and calibrated
  ``PrimitiveTable`` rate binding, lowered-op parity verification for attention,
  indexer, GEMM / ``grouped_gemm``, and DCP ``attention_merge``, and
  ``score_sealed_gate_cases`` / ``validate_disjoint_lineage`` over real
  ``multilevel_gate.evaluate`` with unified ``max(10%, 2.0 us)`` tolerance.
[Invariants]
  * ZERO synthetic second Engine: ``bind_engine_interval`` signs the real
    workload ``execution.engine`` UID and boundary events.
  * Rejects ``service_regimes`` crutches, caller-supplied ``predicted_*`` keys,
    ``PrimitiveTable`` SKU mismatches, and ``train`` vs ``heldout`` lineage
    overlap.
  * Enforces physical operand parity (``recipe``, paged layout, ``sequences``,
    ``q_len``, ``block_table``, GEMM/``grouped_gemm`` ``(m, n, k, matrix_groups,
    dtype, weight_dtype, quant_mode)``, indexer ``topk``/dtypes, DCP merge
    ``dtype``/``partial_count``).
[Timeouts & Locks] Deterministic in-memory Engine replay; ``ContextVar`` +
  one-time hook lock for scaffold profile isolation; zero device contexts.
[Side-Effect Policy] Zero filesystem writes or global ``ascend.json`` mutations.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib
import math
from pathlib import Path
import re
import threading
from typing import Any

import mindie_sim.backends as _backends_pkg
from mindie_sim.backends import ascend
import mindie_sim.compose.inputs as _compose_inputs_mod
from mindie_sim.observations.multilevel import (
    EngineInterval, FrozenFitManifest, GateCase, Manifest, bind_engine_interval,
    evaluate as evaluate_multilevel, parse_gate_case, validate_lineage,
)
from mindie_sim.primitives.schema import PrimitiveTable
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.parallel import local_kv_heads as compute_local_kv_heads
from mindie_sim.schema.workload import RequestView

from .contracts import (
    DEFAULT_CLOCK_OPERATOR, DEFAULT_HELDOUT_APE_THRESHOLD_PCT, DEFAULT_PHYSICAL_ATOL_NS,
    DISPLAY_ROUND_DIGITS, FORBIDDEN_CALLER_PREDICTION_KEYS, NS_PER_MS, PERCENT_SCALE,
    SHORT_DIGEST_LEN, STANDARD_LEVEL_BOUNDARY_KINDS, VALID_CLOCK_OPERATORS,
    assert_sanitized_payload, exceeds_physical_tolerance, load_frozen_fit_manifest,
    parse_lineage_edges, program_dag_sha256, sanitize_public_alias, sanitize_public_capture_uri,
    sanitize_public_run_id, validate_parallel_topology, validate_strict_atol_ns,
    validate_strict_threshold_pct, verify_capture_file_identity,
)

_ascend_mod = importlib.import_module("mindie_sim.backends.ascend")
_BASE_ASCEND_PROFILES = _ascend_mod.profiles
_SCOPED_SCAFFOLD_PROFILES: ContextVar[tuple[tuple[str, Mapping[str, Any]], ...]] = ContextVar(
    "mindie_sim_onboarding_scoped_scaffold_profiles", default=()
)
_PROFILE_HOOK_LOCK = threading.Lock()
_PROFILE_HOOK_INSTALLED = False

_MIN_WORKLOAD_TRACE_LIMIT = 65536
_TRACE_PER_OP_FACTOR = 256
_TRACE_PER_SEQ_FACTOR = 128
_TRACE_KV_PAGE_SIZE = 128
_TRACE_PER_KV_PAGE_FACTOR = 6
_MAX_WORKLOAD_TRACE_LIMIT = 1_048_576
_PERIODIC_BOUNDARY_REPS = 2
_OBSERVER_COMPACT_THRESHOLD = 4096
_MAX_DIAG_OP_IDS = 64
_MIN_GEMM_RANK = 3
_BLOCK_TABLE_PAIR_LEN = 2
_MIN_DCP_PARTIALS = 2
_P90_QUANTILE = 0.90
_TOPO_TOKEN_RE = re.compile(r"(tp|pp|dp|ep|pcp|dcp)(\d+)")
_GEMM_FAMILIES = frozenset({"gemm", "grouped_gemm", "gate_up_projection", "down_projection"})
_ACTUAL_SEQ_SHAPE_KEYS = frozenset({"q_len", "query_tokens", "seq_len", "kv_len", "context_len"})
_MLA_RECIPES = frozenset({
    "ascend_paged_mla_decode_arch22", "ascend_paged_mla_decode_arch35",
    "latent_attention0", "latent_attention_batch", "paired_mla_attention_analytic",
})
_DEFAULT_MODEL = "deepseek-v2-lite"


def _scoped_ascend_profiles() -> dict[str, Any]:
    base = _BASE_ASCEND_PROFILES()
    scoped = _SCOPED_SCAFFOLD_PROFILES.get()
    if not scoped:
        return base
    merged = dict(base)
    for k, v in scoped:
        merged[k] = v
    return merged


def _ensure_scoped_profiles_hook() -> None:
    global _PROFILE_HOOK_INSTALLED
    if _PROFILE_HOOK_INSTALLED:
        return
    with _PROFILE_HOOK_LOCK:
        if _PROFILE_HOOK_INSTALLED:
            return
        _ascend_mod.profiles = _scoped_ascend_profiles
        _backends_pkg.ascend_profiles = _scoped_ascend_profiles
        _compose_inputs_mod.ascend_profiles = _scoped_ascend_profiles
        _PROFILE_HOOK_INSTALLED = True


def resolve_sku_profile(sku: str, profile_override: Mapping[str, Any] | None = None) -> Mapping[str, Any] | None:
    if profile_override is not None:
        return profile_override
    for k, v in reversed(_SCOPED_SCAFFOLD_PROFILES.get()):
        if k == sku:
            return v
    found = _BASE_ASCEND_PROFILES().get(sku)
    return found if isinstance(found, Mapping) else None


@contextmanager
def temporary_scaffold_profile(sku: str, profile_override: Mapping[str, Any] | None) -> Iterator[None]:
    if profile_override is None:
        yield
        return
    _ensure_scoped_profiles_hook()
    token = _SCOPED_SCAFFOLD_PROFILES.set((*_SCOPED_SCAFFOLD_PROFILES.get(), (sku, deepcopy(dict(profile_override)))))
    try:
        yield
    finally:
        _SCOPED_SCAFFOLD_PROFILES.reset(token)


def materialize_sku_hardware(sku: str, *, profile_override: Mapping[str, Any] | None = None) -> Hardware:
    if profile_override is not None:
        return Hardware.from_mapping({
            "sku": sku, "architecture": profile_override["architecture"],
            "dies_per_package": profile_override["dies_per_package"],
            "ranks_per_node": int(profile_override.get("ranks_per_node", 1)),
            "num_devices": 1, "device": deepcopy(profile_override["device"]),
            "transfer_channels": deepcopy(profile_override.get("transfer_channels", {})),
        })
    return Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1}))


def check_no_service_regimes(payload: Any, *, path: str = "case") -> None:
    if isinstance(payload, Mapping):
        for k, v in payload.items():
            ks = str(k)
            if ks == "service_regimes" and v:
                raise ModelingError(f"whitebox_service_regimes_forbidden:{path}")
            check_no_service_regimes(v, path=f"{path}.{ks}")
    elif isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            check_no_service_regimes(item, path=f"{path}[{i}]")


def _coalesce_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    valid = sorted((float(s), float(e)) for s, e in intervals if float(e) > float(s))
    if not valid:
        return []
    merged: list[tuple[float, float]] = []
    cur_s, cur_e = valid[0]
    for s, e in valid[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def union_interval_span_ms(intervals: Sequence[tuple[float, float]]) -> float:
    return sum(hi - lo for lo, hi in _coalesce_intervals(intervals))


class _BusyUnionObserver:
    """Zero-truncation streaming ``EngineObserver`` for exact ``busy_union`` and per-row work."""

    def __init__(self) -> None:
        self._intervals: list[tuple[float, float]] = []
        self._by_row: dict[str, list[tuple[float, float]]] = {}
        self._by_op: dict[tuple[int, str], list[tuple[float, float]]] = {}
        self._task_op: dict[int, tuple[int, str]] = {}
        self._periodic_extra_ms = 0.0
        self._periodic_by_row: dict[str, float] = {}
        self._periodic_by_op: dict[tuple[int, str], float] = {}
        self._slot_gen: dict[tuple[str, int, str], int] = {}

    def register_pipe(self, name: str) -> int:
        return 0 * len(name)

    def on_submit(
        self, task_id: int = 0, name: str = "", *_a: Any,
        aliases: tuple[str, ...] = (), predecessors: tuple[int, ...] = (),
        occurrence: Any = None, op_occ: Any = None, parent_task_id: int | None = None,
        **_kw: Any,
    ) -> None:
        del name, aliases, predecessors
        src = op_occ if op_occ is not None else occurrence
        if src is not None and (oid := str(getattr(src, "op_id", "") or "").strip()):
            self._task_op[int(task_id)] = (int(getattr(src, "replica", 0) or 0), oid)
        elif parent_task_id is not None and int(parent_task_id) in self._task_op:
            self._task_op[int(task_id)] = self._task_op[int(parent_task_id)]

    def on_admit(self, *_a: Any, **_kw: Any) -> None:
        return

    def on_allocation(self, *_a: Any, **_kw: Any) -> None:
        return

    def allocate_slot_generation(self, pool: str, unit: int, name: str = "") -> tuple[int, int]:
        key = (str(pool), int(unit), str(name))
        self._slot_gen[key] = self._slot_gen.get(key, 0) + 1
        return 0, self._slot_gen[key]

    def release_slot_generation(self, pool: str, unit: int, name: str = "") -> tuple[int, int]:
        return 0, self._slot_gen.get((str(pool), int(unit), str(name)), 0)

    def _append_compact(self, bucket: list[tuple[float, float]], span: tuple[float, float]) -> list[tuple[float, float]]:
        bucket.append(span)
        return _coalesce_intervals(bucket) if len(bucket) >= _OBSERVER_COMPACT_THRESHOLD else bucket

    def on_service(self, task_id: int, physical_row: str, owner: int, t0_ms: float, t1_ms: float, delivered: float, *, scope_id: int | None = None) -> None:
        del owner, scope_id
        if delivered <= 0.0 or t1_ms <= t0_ms:
            return
        span = (float(t0_ms), float(t1_ms))
        self._intervals = self._append_compact(self._intervals, span)
        rk = str(physical_row or "device")
        self._by_row[rk] = self._append_compact(self._by_row.setdefault(rk, []), span)
        if (ok := self._task_op.get(int(task_id))) is not None:
            self._by_op[ok] = self._append_compact(self._by_op.setdefault(ok, []), span)

    def extract_cycle_pattern(
        self, start_ms: float, end_ms: float, *_a: Any, **_kw: Any,
    ) -> tuple[tuple[tuple[str, float, float], ...], tuple[tuple[tuple[int, str], float, float], ...]]:
        s0, s1 = float(start_ms), float(end_ms)
        if s1 <= s0:
            return (), ()
        row_pat: list[tuple[str, float, float]] = []
        for rk in sorted(self._by_row):
            ivs = _coalesce_intervals(self._by_row[rk])
            self._by_row[rk] = ivs
            for lo, hi in ivs:
                if hi > s0 and lo < s1 and (c_hi := min(hi, s1)) > (c_lo := max(lo, s0)):
                    row_pat.append((rk, c_lo - s0, c_hi - s0))
        op_pat: list[tuple[tuple[int, str], float, float]] = []
        for op_key in sorted(self._by_op):
            ivs = _coalesce_intervals(self._by_op[op_key])
            self._by_op[op_key] = ivs
            for lo, hi in ivs:
                if hi > s0 and lo < s1 and (c_hi := min(hi, s1)) > (c_lo := max(lo, s0)):
                    op_pat.append((op_key, c_lo - s0, c_hi - s0))
        return tuple(row_pat), tuple(op_pat)

    @staticmethod
    def _integrate_periodic(
        bucket: list[tuple[float, float]], raw_spans: Sequence[tuple[float, float]],
        a_ms: float, p_ms: float, repeats: int,
    ) -> float:
        rel = _coalesce_intervals(raw_spans)
        if not rel:
            return 0.0
        bucket.extend((a_ms + lo, a_ms + hi) for lo, hi in rel)
        if repeats >= _PERIODIC_BOUNDARY_REPS:
            last_base = a_ms + float(repeats - 1) * p_ms
            bucket.extend((last_base + lo, last_base + hi) for lo, hi in rel)
        interior = repeats - _PERIODIC_BOUNDARY_REPS
        return float(interior) * sum(hi - lo for lo, hi in rel) if interior > 0 else 0.0

    def on_period(
        self, anchor_ms: float, period_ms: float, repeats: int,
        service_pattern: Sequence[tuple[str, float, float]] = (),
        *, template_id: str = "", state_digest: str = "",
        handoff_pattern: Sequence[Any] = (), **_kw: Any,
    ) -> None:
        del template_id, state_digest
        if repeats <= 0 or period_ms <= 0.0 or (not service_pattern and not handoff_pattern):
            return
        a_ms, p_ms = float(anchor_ms), float(period_ms)
        by_row_raw: dict[str, list[tuple[float, float]]] = {}
        all_raw: list[tuple[float, float]] = []
        for entry in service_pattern:
            rk, lo, hi = str(entry[0] or "device"), float(entry[1]), float(entry[2])
            if hi > lo:
                by_row_raw.setdefault(rk, []).append((lo, hi))
                all_raw.append((lo, hi))
        if all_raw:
            self._periodic_extra_ms += self._integrate_periodic(self._intervals, all_raw, a_ms, p_ms, repeats)
        for rk, spans in by_row_raw.items():
            extra = self._integrate_periodic(self._by_row.setdefault(rk, []), spans, a_ms, p_ms, repeats)
            if extra > 0.0:
                self._periodic_by_row[rk] = self._periodic_by_row.get(rk, 0.0) + extra
        by_op_raw: dict[tuple[int, str], list[tuple[float, float]]] = {}
        for entry in handoff_pattern:
            if isinstance(entry, tuple) and len(entry) == 3 and isinstance(entry[0], tuple) and len(entry[0]) == 2:
                op_key = (int(entry[0][0]), str(entry[0][1]))
                lo, hi = float(entry[1]), float(entry[2])
                if hi > lo:
                    by_op_raw.setdefault(op_key, []).append((lo, hi))
        for op_key, spans in by_op_raw.items():
            extra = self._integrate_periodic(self._by_op.setdefault(op_key, []), spans, a_ms, p_ms, repeats)
            if extra > 0.0:
                self._periodic_by_op[op_key] = self._periodic_by_op.get(op_key, 0.0) + extra

    def total_busy_ms(self) -> float:
        return union_interval_span_ms(self._intervals) + self._periodic_extra_ms

    def row_busy_ns(self, *, cap_ns: int) -> tuple[int, ...]:
        row_keys = sorted(set(self._by_row) | set(self._periodic_by_row))
        spans = [
            ns for k in row_keys
            if (ns := min(cap_ns, int(math.floor((union_interval_span_ms(self._by_row.get(k, ())) + self._periodic_by_row.get(k, 0.0)) * NS_PER_MS)))) > 0
        ]
        return tuple(spans) if spans else ((cap_ns,) if cap_ns > 0 else ())

    def op_busy_ns(self, op_key: tuple[int, str], *, cap_ns: int) -> int:
        total_ms = union_interval_span_ms(self._by_op.get(op_key, ())) + self._periodic_by_op.get(op_key, 0.0)
        return min(cap_ns, int(math.floor(total_ms * NS_PER_MS))) if total_ms > 0.0 else 0


def _build_executed_witness_program_dag(graphs: Mapping[int, tuple[Any, ...]], observer: _BusyUnionObserver, *, target_ns: int) -> tuple[dict[str, Any], dict[str, Any]]:
    row_ns = observer.row_busy_ns(cap_ns=target_ns)
    node_idx: dict[tuple[int, str], int] = {}
    quanta: list[int] = []
    ordered: list[tuple[int, Any]] = []
    for rep in sorted(graphs):
        for op in graphs[rep]:
            key = (int(rep), str(op.op_id))
            if key in node_idx:
                continue
            node_idx[key] = len(ordered)
            ordered.append((int(rep), op))
            quanta.append(observer.op_busy_ns(key, cap_ns=target_ns))
    edges = [
        {"src": node_idx[(rep, str(d))], "dst": dst, "wire_ns": 0}
        for dst, (rep, op) in enumerate(ordered)
        for d in (getattr(op, "deps", ()) or ())
        if (rep, str(d)) in node_idx and node_idx[(rep, str(d))] != dst
    ]
    return {"num_resources": len(row_ns), "global_work": list(row_ns), "site_modes": []}, {"node_quantum_ns": quanta or [max(row_ns) if row_ns else 0], "edges": edges}


def _run_graphs_on_real_engine(
    sim_case: Any, graphs: Mapping[int, tuple[Any, ...]], *, start_event_name: str, end_event_name: str,
    external_inputs: Mapping[str, float] | None, family_registry: Any = None, sequences: int = 1, context: int = 0,
    primitive_table: PrimitiveTable | None = None,
) -> tuple[Any, Any, Any, float, float, _BusyUnionObserver]:
    from mindie_sim.prediction.execution.cohort_traffic import bind_cohort_traffic
    from mindie_sim.prediction.efficiency.cost import CostLedger
    from mindie_sim.prediction.execution.dispatch import cohorts
    from mindie_sim.prediction.evaluation import EvaluationContext
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.graph_analysis import _name
    from mindie_sim.prediction.implementation_selection import graph_selections
    from mindie_sim.prediction.program import RecipeBook
    from mindie_sim.schema.operators import IterationWork
    from mindie_sim.schema.workload import IterationPlan

    if primitive_table is not None and primitive_table.sku != sim_case.machine.sku:
        raise ModelingError(f"primitive_table_sku_mismatch:table={primitive_table.sku}:case={sim_case.machine.sku}")
    book = RecipeBook.from_mapping(sim_case.runtime.kernels, family_registry=family_registry)
    ctx = EvaluationContext.create(sim_case, book, primitive_table=primitive_table)
    expanded = bind_cohort_traffic(sim_case, {r: tuple(book.expand(items)) for r, items in graphs.items()})
    external = dict(external_inputs or {})
    op_count = sum(len(g) for g in expanded.values())
    kv_pages = max(1, int(sequences)) * max(1, (max(0, int(context)) + _TRACE_KV_PAGE_SIZE) // _TRACE_KV_PAGE_SIZE + 1)
    trace_limit = min(_MAX_WORKLOAD_TRACE_LIMIT, max(_MIN_WORKLOAD_TRACE_LIMIT, op_count * _TRACE_PER_OP_FACTOR, int(sequences) * _TRACE_PER_SEQ_FACTOR, kv_pages * _TRACE_PER_KV_PAGE_FACTOR))
    execution = Execution(sim_case, ctx.cost, recipes=book, bindings=ctx.bindings, ledger=CostLedger(), include_submission=False, trace_limit=trace_limit, evaluation=ctx, accel_eager_attach=op_count >= 1)
    execution._compact_step = not execution.legacy_diagnostics
    execution.engine._compact_step = execution._compact_step
    engine, observer = execution.engine, _BusyUnionObserver()
    engine.attach_wave_recorder(observer)
    engine.now = 0.0
    start_ev = engine.submit(start_event_name, release=0.0)
    inputs = {
        r: {k: (start_ev if float(v) <= 0.0 else engine.submit(_name(r, "input:" + k), release=float(v), after=(start_ev,))) for k, v in external.items() if any(k in op.deps for op in items)}
        for r, items in expanded.items()
    }
    works = {
        r: IterationWork(items, IterationPlan(()), sim_case.runtime.parallel.pp, 0, 0, None, "graph", "declared_graph", replay_scope=None, implementation_selections=graph_selections(items))
        for r, items in expanded.items()
    }
    steps: dict[int, Any] = {}
    for group in cohorts(sim_case.runtime.parallel, sim_case.machine):
        selected = {r: works[r] for r in group if r in works}
        if len(selected) > 1:
            steps.update(execution.add_group(selected, 0.0, external_inputs=inputs))
        elif selected:
            r, work = next(iter(selected.items()))
            steps[r] = execution.add(work, r, 0.0, external_inputs=inputs[r])
    end_ev = engine.submit(end_event_name, after=tuple(s.completion for s in steps.values()))
    engine.advance(until=(start_ev, end_ev), first=False)
    elapsed_ms = float(end_ev.time - start_ev.time) if (end_ev.time is not None and start_ev.time is not None) else 0.0
    return engine, start_ev, end_ev, elapsed_ms, float(observer.total_busy_ms()), observer


def _opt_pos_int(*values: Any) -> int | None:
    for val in values:
        if val is not None and not isinstance(val, bool):
            try:
                iv = int(val)
                if iv > 0:
                    return iv
            except (TypeError, ValueError):
                continue
    return None


def _norm_node_op_id(raw: str, cid: str) -> str:
    clean = str(raw or "").strip()
    if ":" in clean:
        pfx, op_part = clean.split(":", 1)
        if pfx != cid or not op_part.strip():
            raise ModelingError(f"whitebox_node_foreign_prefix:{cid}:{clean}")
        return op_part.strip()
    return clean


def _assert_bind_eq(cid: str, field: str, measured: Any, lowered: Any) -> None:
    if measured is not None and lowered is not None and measured != lowered:
        raise ModelingError(f"whitebox_measured_binding_mismatch:{cid}:{field}:measured={measured}:lowered={lowered}")


def _verify_attention_binding(op: Any, sim_case: Any, nb: Mapping[str, Any], actual: Mapping[str, Any], *, local_kv: int, cid: str, step_tokens: int, step_seqs: int) -> None:
    _assert_bind_eq(cid, "recipe", str(nb["recipe"]) if nb.get("recipe") else None, str(op.recipe))
    if nb.get("paged") is False and (op.recipe in _MLA_RECIPES or "paged" in op.op_id):
        raise ModelingError(f"whitebox_measured_binding_mismatch:{cid}:layout:measured_unpaged_vs_lowered_paged")
    meta = dict(getattr(op, "metadata", ()) or ())
    nabi = meta.get("native_attention_abi") if isinstance(meta.get("native_attention_abi"), Mapping) else {}
    pgeo = meta.get("paged_mla_geometry") if isinstance(meta.get("paged_mla_geometry"), Mapping) else {}
    meas_seqs = _opt_pos_int(actual.get("sequences"), nb.get("sequences"))
    bt = actual.get("block_table")
    bt_ok = isinstance(bt, Sequence) and not isinstance(bt, (str, bytes)) and len(bt) >= _BLOCK_TABLE_PAIR_LEN
    bt_seqs = int(bt[0]) if bt_ok and type(bt[0]) is int else None
    bt_pages = int(bt[1]) if bt_ok and type(bt[1]) is int else None
    if bt_seqs is not None and meas_seqs is not None and bt_seqs != meas_seqs:
        raise ModelingError(f"whitebox_measured_binding_mismatch:{cid}:sequences_block_table_conflict")
    if step_seqs > 0:
        _assert_bind_eq(cid, "sequences", meas_seqs, step_seqs)
        _assert_bind_eq(cid, "block_table_sequences", bt_seqs, step_seqs)
    if step_tokens > 0:
        _assert_bind_eq(cid, "q_len", _opt_pos_int(actual.get("q_len"), actual.get("query_tokens"), nb.get("q_len"), nb.get("query_tokens")), step_tokens)
    loc = tuple(getattr(op, "local_shape", ()) or ())
    low_kv_len = _opt_pos_int(
        pgeo["kv_lengths"][0] if pgeo.get("kv_lengths") else None,
        loc[1] if len(loc) >= 2 else None,
    )
    _assert_bind_eq(cid, "kv_len", _opt_pos_int(actual.get("seq_len"), actual.get("kv_len"), nb.get("kv_len")), low_kv_len)
    low_pages = _opt_pos_int(pgeo["page_counts"][0] if pgeo.get("page_counts") else None)
    for mp in (_opt_pos_int(actual.get("page_count")), bt_pages, _opt_pos_int(nb.get("page_count"))):
        _assert_bind_eq(cid, "page_count", mp, low_pages)
    low_layout = str(nabi.get("query_layout") or "").strip().upper() or None
    meas_layout = str(actual.get("query_layout") or actual.get("input_layout") or nb.get("query_layout") or "").strip().upper() or None
    _assert_bind_eq(cid, "query_layout", meas_layout, low_layout)
    _assert_bind_eq(cid, "kv_heads", _opt_pos_int(actual.get("kv_heads"), nb.get("kv_heads")), max(1, local_kv))
    tp = max(1, int(getattr(sim_case.runtime.parallel, "tp", 1) or 1))
    _assert_bind_eq(cid, "heads", _opt_pos_int(actual.get("heads"), nb.get("heads")), max(1, int(getattr(sim_case.model, "num_heads", 1) or 1) // tp))


def _verify_gemm_binding(op: Any, nb: Mapping[str, Any], n_act: Mapping[str, Any], *, cid: str, oid: str) -> None:
    meta = dict(getattr(op, "metadata", ()) or ())
    _assert_bind_eq(cid, f"matrix_groups:{oid}", _opt_pos_int(n_act.get("matrix_groups"), n_act.get("e"), nb.get("matrix_groups"), nb.get("e")), max(1, int(meta.get("matrix_weight_bank_groups") or meta.get("matrix_groups") or 1)))
    loc = tuple(getattr(op, "local_shape", ()) or ())
    if len(loc) >= _MIN_GEMM_RANK:
        for idx, dk in enumerate(("m", "n", "k")):
            _assert_bind_eq(cid, f"weight_{dk}:{oid}", _opt_pos_int(n_act.get(dk), nb.get(dk)), int(loc[idx]))
    low_dt = str(getattr(op, "dtype", "") or "bf16").strip().lower()
    low_wdt = str(meta.get("weight_dtype") or low_dt).strip().lower()
    low_q = str(meta.get("quant_mode") or meta.get("quantization") or "none").strip().lower()
    act_dt = str(n_act.get("dtype") or n_act.get("input_dtype") or nb.get("dtype") or "").strip().lower() or None
    act_wdt = str(n_act.get("weight_dtype") or "").strip().lower() or None
    act_q = str(n_act.get("quant_mode") or n_act.get("quantization") or "").strip().lower()
    _assert_bind_eq(cid, f"dtype:{oid}", act_dt, low_dt or None)
    _assert_bind_eq(cid, f"weight_dtype:{oid}", act_wdt, low_wdt or None)
    if act_q and act_q not in {"", "none", "dense", low_q, low_dt}:
        raise ModelingError(f"whitebox_measured_binding_mismatch:{cid}:quant_mode:{oid}:measured={act_q}:lowered={low_q}")
    if (n_act.get("has_scales") is True or str(n_act.get("scale_dtype") or "").strip()) and low_q in {"", "none", "dense"}:
        raise ModelingError(f"whitebox_measured_binding_mismatch:{cid}:unexpected_quant_scales:{oid}")


def _verify_measured_binding(
    chosen_ops: tuple[Any, ...], sim_case: Any, payload: Mapping[str, Any], *,
    local_kv: int, expected_level: str, cid: str, step_tokens: int, step_seqs: int,
) -> None:
    binding = payload.get("measured_binding") if isinstance(payload.get("measured_binding"), Mapping) else {}
    actual = payload.get("actual_shape") if isinstance(payload.get("actual_shape"), Mapping) else {}
    if not binding and not actual and not payload.get("timed_nodes"):
        return
    ops_by_id = {str(op.op_id): op for op in chosen_ops}
    raw_timed = payload.get("timed_nodes")
    if expected_level in {"L2", "L3"} and isinstance(raw_timed, Sequence) and not isinstance(raw_timed, (str, bytes)) and raw_timed:
        timed_ops = [_norm_node_op_id(str(x), cid) for x in raw_timed]
        if set(timed_ops) != set(ops_by_id):
            raise ModelingError(f"whitebox_measured_binding_mismatch:{cid}:{expected_level.lower()}_op_set_mismatch")
    raw_ns = actual.get("node_shapes")
    node_shapes = {_norm_node_op_id(str(k), cid): v for k, v in raw_ns.items() if isinstance(v, Mapping)} if isinstance(raw_ns, Mapping) else {}
    nb_by_id = {str(nb["op_id"]): nb for nb in (binding.get("node_bindings") or ()) if isinstance(nb, Mapping) and nb.get("op_id")}
    for op in chosen_ops:
        oid = str(op.op_id)
        nb = nb_by_id.get(oid, binding if expected_level == "L1" else {})
        n_act = node_shapes.get(oid, actual if (expected_level == "L1" or not node_shapes) else {})
        if op.family == "attention":
            _verify_attention_binding(op, sim_case, nb, n_act, local_kv=local_kv, cid=cid, step_tokens=step_tokens, step_seqs=step_seqs)
        elif op.family in _GEMM_FAMILIES and (nb or n_act):
            _verify_gemm_binding(op, nb, n_act, cid=cid, oid=oid)
        elif op.family == "indexer":
            meta = dict(getattr(op, "metadata", ()) or ())
            sgs = meta.get("structure_subgraphs")
            tv = sgs[0].get("tensor_view") if isinstance(sgs, Sequence) and sgs and isinstance(sgs[0], Mapping) and isinstance(sgs[0].get("tensor_view"), Mapping) else {}
            low_tk = _opt_pos_int(tv.get("topk"))
            meas_tk = _opt_pos_int(n_act.get("sparse_count"), n_act.get("top_k"), n_act.get("topk"), nb.get("sparse_count"), nb.get("topk"), payload.get("topk"))
            if meas_tk is not None and low_tk is not None and meas_tk != low_tk:
                raise ModelingError(f"whitebox_measured_binding_mismatch:{cid}:topk:measured={meas_tk}:lowered={low_tk}")
            for dt_k in ("query_dtype", "cache_dtype"):
                ld, md = str(tv.get(dt_k) or "").strip().lower(), str(n_act.get(dt_k) or nb.get(dt_k) or "").strip().lower()
                if md and ld and md != ld:
                    raise ModelingError(f"whitebox_measured_binding_mismatch:{cid}:{dt_k}:measured={md}:lowered={ld}")
        elif op.family == "attention_merge" and (nb or n_act):
            dcp = max(1, int(getattr(sim_case.runtime.parallel, "dcp", 1) or 1))
            got_p = _opt_pos_int(n_act.get("partial_count"), nb.get("partial_count"))
            if dcp >= _MIN_DCP_PARTIALS and got_p is not None and got_p != dcp:
                raise ModelingError(f"whitebox_measured_binding_mismatch:{cid}:partial_count:{oid}:measured={got_p}:lowered={dcp}")
            got_dt = str(n_act.get("dtype") or nb.get("dtype") or "").strip().lower()
            low_dt = str(getattr(op, "dtype", "") or "fp32").strip().lower()
            if got_dt and low_dt and got_dt != low_dt:
                raise ModelingError(f"whitebox_measured_binding_mismatch:{cid}:dtype:{oid}:measured={got_dt}:lowered={low_dt}")


def _select_level_ops(all_ops: tuple[Any, ...], *, level: str, payload: Mapping[str, Any], cid: str) -> tuple[Any, ...]:
    if level == "L3":
        return all_ops
    if level == "L1":
        binding = payload.get("measured_binding") if isinstance(payload.get("measured_binding"), Mapping) else {}
        tid = str(payload.get("target_op_id") or binding.get("target_op_id") or "").strip()
        tfam = str(payload.get("target_family") or binding.get("family") or "").strip()
        matches = [o for o in all_ops if o.op_id == tid] if tid else ([o for o in all_ops if o.family == tfam] if tfam else ([all_ops[0]] if len(all_ops) == 1 else []))
        if not matches:
            raise ModelingError(f"whitebox_l1_target_not_found:{cid}:{tid or tfam}")
        chosen = matches[0]
        if tfam and chosen.family != tfam:
            raise ModelingError(f"whitebox_l1_family_mismatch:{cid}:expected={tfam}:got={chosen.family}")
        return (replace(chosen, deps=()),)
    if level == "L2":
        tids = payload.get("target_op_ids")
        if isinstance(tids, Sequence) and not isinstance(tids, (str, bytes)) and tids:
            wanted = {str(x) for x in tids}
            chosen_list = [o for o in all_ops if o.op_id in wanted]
            if len(chosen_list) != len(wanted):
                raise ModelingError(f"whitebox_l2_target_op_ids_missing:{cid}")
        elif payload.get("target_layer") is not None:
            chosen_list = [o for o in all_ops if int(o.layer) == int(payload["target_layer"])]
        else:
            raise ModelingError(f"whitebox_l2_requires_subgraph_target:{cid}")
        if not chosen_list or (len(chosen_list) >= len(all_ops) and len(all_ops) > 1):
            raise ModelingError(f"whitebox_l2_subgraph_invalid:{cid}")
        cids = {o.op_id for o in chosen_list}
        return tuple(replace(o, deps=tuple(d for d in o.deps if d in cids)) for o in chosen_list)
    raise ModelingError(f"whitebox_level_unsupported:{level}")


def _distribute_total_tokens(
    total_tokens: int, seqs: int, *, ctx: int, phase: str, cid: str,
) -> tuple[int, int, tuple[RequestView, ...]]:
    if total_tokens < seqs:
        raise ModelingError(f"whitebox_tokens_below_sequences:{cid}:tokens={total_tokens}:seqs={seqs}")
    base, rem = divmod(total_tokens, seqs)
    if rem == 0:
        return base, total_tokens, ()
    reqs = tuple(
        RequestView(f"req-{i}", base + (1 if i < rem else 0), ctx, True, phase)
        for i in range(seqs)
    )
    return base, total_tokens, reqs


def _resolve_step_dims(
    payload: Mapping[str, Any], *, local_kv: int, cid: str, phase: str = "decode",
) -> tuple[int, int, int, int, int, tuple[RequestView, ...]]:
    ctx = max(0, int(payload.get("context_len") or payload.get("context") or 0))
    batch = max(1, int(payload.get("batch_size") or payload.get("sequences") or 1))
    axis = str(payload.get("axis") or "").strip()
    bxkv_raw = payload.get("batch_x_kv_heads") if payload.get("batch_x_kv_heads") is not None else (payload.get("value") if axis == "batch_x_kv_heads" else None)
    kv_per = max(1, int(local_kv))
    if bxkv_raw is not None and not isinstance(bxkv_raw, bool):
        bxkv = int(bxkv_raw)
        if bxkv < 1 or bxkv % kv_per != 0:
            raise ModelingError(f"whitebox_batch_x_kv_heads_indivisible:{cid}:target={bxkv}:local_kv={kv_per}")
        seqs = bxkv // kv_per
    else:
        seqs, bxkv = batch, batch * kv_per
    q_raw, tok_raw = payload.get("query_tokens"), payload.get("tokens")
    q_val = int(q_raw) if (q_raw is not None and not isinstance(q_raw, bool)) else None
    tok_val = int(tok_raw) if (tok_raw is not None and not isinstance(tok_raw, bool)) else None
    if axis == "tokens":
        total = tok_val if (tok_val is not None and tok_val > 0) else max(1, int(payload.get("value") or 1))
        per_seq, total_tok, reqs = _distribute_total_tokens(total, seqs, ctx=ctx, phase=phase, cid=cid)
        return per_seq, ctx, seqs, bxkv, total_tok, reqs
    act = payload.get("actual_shape") if isinstance(payload.get("actual_shape"), Mapping) else {}
    mb = payload.get("measured_binding") if isinstance(payload.get("measured_binding"), Mapping) else {}
    eff_fam = str(mb.get("family") or payload.get("target_family") or "").strip()
    act_m = _opt_pos_int(act.get("m"))
    synth_q1 = (
        q_val == 1 and tok_val is not None and tok_val > 1
        and eff_fam in _GEMM_FAMILIES and act_m is not None and act_m >= tok_val and act_m % tok_val == 0
        and _ACTUAL_SEQ_SHAPE_KEYS.isdisjoint(act) and "tokens" not in act
        and "q_len" not in mb and "query_tokens" not in mb
    )
    if q_val is not None and not synth_q1:
        if q_val < 1:
            raise ModelingError(f"whitebox_query_tokens_invalid:{cid}:{q_val}")
        return q_val, ctx, seqs, bxkv, q_val * seqs, ()
    if tok_val is not None and tok_val > 0:
        per_seq, total_tok, reqs = _distribute_total_tokens(tok_val, seqs, ctx=ctx, phase=phase, cid=cid)
        return per_seq, ctx, seqs, bxkv, total_tok, reqs
    return 1, ctx, seqs, bxkv, seqs, ()


def eligible_replay_cases(cases: Sequence[GateCase]) -> list[GateCase]:
    return [c for c in cases if not c.is_counterfactual and c.has_matched_capture and (c.level not in {"L2", "L3"} or c.block_capture_present)]


class WhiteboxSpineRunner:
    """Level-specific L1/L2/L3 ``EvaluationRunner`` over one real workload ``Engine``."""

    def __init__(
        self, *, clock_operator: str = DEFAULT_CLOCK_OPERATOR, primitive_table: PrimitiveTable | None = None,
        profile_override: Mapping[str, Any] | None = None,
    ) -> None:
        if clock_operator not in VALID_CLOCK_OPERATORS:
            raise ModelingError(f"whitebox_clock_operator_invalid:{clock_operator}")
        self.clock_operator = clock_operator
        self.primitive_table = primitive_table
        self.profile_override = deepcopy(dict(profile_override)) if isinstance(profile_override, Mapping) else None

    @staticmethod
    def _apply_model_overrides(composed: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
        md = dict(composed["model"])
        if "layers" in overrides:
            nl = int(overrides["layers"])
            md["layers"], md["body_layers"], md["layer_plans"] = nl, nl, list(md["layer_plans"])[:nl]
        if "index_topk" in overrides:
            tk = int(overrides["index_topk"])
            md["mla"] = {**dict(md.get("mla") or {}), "index_topk": tk}
            if isinstance(md.get("layer_plans"), list):
                md["layer_plans"] = [
                    {**dict(lp), "index_topk": tk} if isinstance(lp, Mapping) and lp.get("index_topk") is not None else lp
                    for lp in md["layer_plans"]
                ]
        if "index_dtype" in overrides:
            md["quant"] = {**dict(md.get("quant") or {}), "index_dtype": str(overrides["index_dtype"]).strip()}
        return {**composed, "model": md}

    def _materialize_case(self, case: GateCase, payload: Mapping[str, Any]) -> Any:
        from mindie_sim.api import compose_case
        from mindie_sim.compose import compose_case as compose_case_dict
        from mindie_sim.prediction.structure import parse_case
        from mindie_sim.schema.case import Case

        raw = payload.get("sim_case") or {"compose_model": _DEFAULT_MODEL, "compose_hardware": case.cohort.sku}
        topo_str = str(payload.get("parallel_topology") or case.cohort.parallel_topology or "tp1").strip().lower()
        topo_req = {k: int(v) for k, v in _TOPO_TOKEN_RE.findall(topo_str)}
        with temporary_scaffold_profile(case.cohort.sku, self.profile_override):
            if isinstance(raw, Case):
                sim_case: Any = raw
            elif isinstance(raw, Mapping):
                check_no_service_regimes(raw, path=f"{case.case_id}.sim_case")
                if "compose_model" in raw:
                    opts = {**topo_req, **dict(raw.get("compose_options") or {})}
                    overrides = raw.get("model_overrides")
                    if isinstance(overrides, Mapping) and overrides:
                        d = compose_case_dict(model=str(raw["compose_model"]), hardware=str(raw.get("compose_hardware", case.cohort.sku)), **opts)
                        sim_case = parse_case(self._apply_model_overrides(d, overrides))
                    else:
                        sim_case = compose_case(raw["compose_model"], raw.get("compose_hardware", case.cohort.sku), options=opts or None)
                else:
                    sim_case = parse_case(dict(raw))
            else:
                raise ModelingError(f"whitebox_case_invalid_type:{case.case_id}")
        if str(sim_case.machine.sku).strip() != str(case.cohort.sku).strip():
            raise ModelingError(f"whitebox_materialized_sku_mismatch:{case.case_id}:expected={case.cohort.sku}:got={sim_case.machine.sku}")
        for axis_name, expected_val in topo_req.items():
            actual_val = int(getattr(sim_case.runtime.parallel, axis_name, 1) or 1)
            if actual_val != expected_val:
                raise ModelingError(
                    f"whitebox_materialized_parallel_mismatch:{case.case_id}:{axis_name}:expected={expected_val}:got={actual_val}"
                )
        if self.primitive_table is not None and self.primitive_table.sku != sim_case.machine.sku:
            raise ModelingError(f"primitive_table_sku_mismatch:table={self.primitive_table.sku}:case={sim_case.machine.sku}")
        return sim_case

    def _run_level(self, case: GateCase, *, expected_level: str) -> EngineInterval:
        from mindie_sim.prediction.results import LoweringRequest, StepRequest
        from mindie_sim.prediction.spine import default_spine

        payload = dict(case.payload)
        sim_case = self._materialize_case(case, payload)
        local_kv = int(compute_local_kv_heads(int(getattr(sim_case.model, "num_kv_heads", 1) or 1), max(1, int(getattr(sim_case.runtime.parallel, "tp", 1) or 1))))
        phase = str(payload.get("phase") or case.cohort.phase or "decode")
        tokens, context, seqs, bxkv, total_tokens, step_reqs = _resolve_step_dims(payload, local_kv=local_kv, cid=case.case_id, phase=phase)
        step_req = StepRequest(phase=phase, tokens=tokens, context=context, sequences=seqs, requests=step_reqs, scope="body")  # type: ignore[arg-type]
        spine = default_spine()
        prepared = spine.lower(sim_case, LoweringRequest(step=step_req, tier="mainline"))
        graphs: dict[int, tuple[Any, ...]] = {}
        for entry in prepared.graph.executable:
            chosen = _select_level_ops(entry.ops, level=expected_level, payload=payload, cid=case.case_id)
            _verify_measured_binding(chosen, prepared.case, payload, local_kv=local_kv, expected_level=expected_level, cid=case.case_id, step_tokens=(0 if step_reqs else tokens), step_seqs=seqs)
            graphs[entry.replica] = chosen
        ext_in = {"analysis_input": 0.0} if any("analysis_input" in op.deps for ops in graphs.values() for op in ops) else None
        engine, start_ev, end_ev, elapsed_ms, busy_ms, obs = _run_graphs_on_real_engine(
            prepared.case, graphs, start_event_name=case.boundary.start_event, end_event_name=case.boundary.end_event,
            external_inputs=ext_in, family_registry=spine.families, sequences=seqs, context=context,
            primitive_table=self.primitive_table,
        )
        base_interval = bind_engine_interval(case, engine, start_ev, end_ev, source_level=expected_level)
        elapsed_ns = int(base_interval.elapsed_span_ns if base_interval.elapsed_span_ns is not None else round(elapsed_ms * NS_PER_MS))
        busy_ns = int(base_interval.busy_union_ns if base_interval.busy_union_ns is not None else min(elapsed_ns, int(round(busy_ms * NS_PER_MS))))
        bound_clock = str(base_interval.clock_operator)
        witness_clock = "busy_union" if (bound_clock == "busy_union" or str(payload.get("clock_operator") or "") == "busy_union") else bound_clock
        if elapsed_ns <= 0:
            raise ModelingError(f"whitebox_prediction_nonpositive:{case.case_id}")
        target_ns = busy_ns if witness_clock == "busy_union" else elapsed_ns
        if target_ns <= 0:
            raise ModelingError(f"whitebox_selected_clock_nonpositive:{case.case_id}:{witness_clock}")
        lp, ld = _build_executed_witness_program_dag(graphs, obs, target_ns=target_ns)
        diag = (
            ("clock_operator", bound_clock), ("busy_union_ns", busy_ns), ("elapsed_span_ns", elapsed_ns),
            ("replay_level", expected_level), ("engine_uid", str(engine.uid)),
            ("program_dag_sha256", program_dag_sha256(lp, ld)), ("lean_program", lp), ("lean_dag", ld),
            ("executed_op_families", sorted({str(op.family) for ops in graphs.values() for op in ops})),
            ("executed_op_ids", [str(op.op_id) for ops in graphs.values() for op in ops][:_MAX_DIAG_OP_IDS]),
            ("lowered_dimensions", {"materialized_sku": str(sim_case.machine.sku), "tokens": total_tokens, "context": context, "sequences": seqs, "batch_x_kv_heads": bxkv}),
        )
        return replace(base_interval, diagnostics=diag)

    def run_l1(self, case: GateCase) -> EngineInterval:
        return self._run_level(case, expected_level="L1")

    def run_l2(self, case: GateCase) -> EngineInterval:
        return self._run_level(case, expected_level="L2")

    def run_l3(self, case: GateCase) -> EngineInterval:
        return self._run_level(case, expected_level="L3")


def validate_disjoint_lineage(
    cases: Sequence[GateCase], *, external_train_roots: Sequence[str] = (), require_disjoint_level_roots: bool = True,
) -> dict[str, Any]:
    train_roots = {c.boundary.acquisition_uid for c in cases if c.split == "train"} | {str(r).strip() for r in external_train_roots if str(r).strip()}
    held_roots = {c.boundary.acquisition_uid for c in cases if c.split == "heldout"}
    overlap = sorted(train_roots & held_roots)
    if overlap:
        raise ModelingError(f"train_heldout_lineage_leakage:{','.join(overlap)}")
    by_level: dict[str, set[str]] = {}
    for c in cases:
        if c.split == "heldout":
            by_level.setdefault(c.level, set()).add(c.boundary.acquisition_uid)
    if require_disjoint_level_roots and len(by_level) > 1:
        lvls = sorted(by_level)
        for i, l1 in enumerate(lvls):
            for l2 in lvls[i + 1:]:
                if by_level[l1] & by_level[l2]:
                    raise ModelingError(f"multilevel_shared_acquisition_root:{l1}:{l2}")
    return {"train_roots": sorted(train_roots), "heldout_roots": sorted(held_roots), "heldout_roots_by_level": {k: sorted(v) for k, v in sorted(by_level.items())}}


def _resolve_row_clock_operator(
    row: Mapping[str, Any], plan_pt: Mapping[str, Any] | None, *, cid: str, default_clock_operator: str,
) -> str:
    b_raw = row.get("boundary")
    p_raw = row.get("payload")
    b_clock = str(b_raw.get("clock_operator") or "").strip() if isinstance(b_raw, Mapping) else ""
    r_clock = str(row.get("clock_operator") or "").strip()
    p_clock = str(p_raw.get("clock_operator") or "").strip() if isinstance(p_raw, Mapping) else ""
    plan_clock = str((plan_pt or {}).get("clock_operator") or "").strip()
    declared = [c for c in (b_clock, r_clock, p_clock) if c]
    if len(set(declared)) > 1:
        raise ModelingError(f"clock_operator_conflict:{cid}:declared={sorted(set(declared))}")
    clock_op = (declared[0] if declared else "") or plan_clock or default_clock_operator
    if clock_op not in VALID_CLOCK_OPERATORS:
        raise ModelingError(f"clock_operator_invalid:{cid}:{clock_op}")
    if plan_clock and plan_clock != clock_op:
        raise ModelingError(f"clock_operator_plan_mismatch:{cid}:plan={plan_clock}:got={clock_op}")
    return clock_op


def _validate_row_against_plan_point(
    row: Mapping[str, Any], plan_pt: Mapping[str, Any] | None, *, cid: str,
) -> None:
    if not plan_pt:
        return
    pl = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
    coh = row.get("cohort") if isinstance(row.get("cohort"), Mapping) else {}
    plan_topo = str(plan_pt.get("parallel_topology") or "").strip().lower()
    row_topo = str(row.get("parallel_topology") or pl.get("parallel_topology") or coh.get("parallel_topology") or "").strip().lower()
    if plan_topo and row_topo and row_topo != plan_topo:
        raise ModelingError(f"observation_plan_geometry_mismatch:{cid}:parallel_topology:expected={plan_topo}:got={row_topo}")
    axis = str(plan_pt.get("axis") or "").strip()
    if axis and plan_pt.get("value") is not None:
        exp_val = int(plan_pt["value"])
        if axis == "context_len":
            got_val = _opt_pos_int(row.get("context_len"), row.get("context"), pl.get("context_len"), pl.get("context"))
        else:
            got_val = _opt_pos_int(row.get(axis), pl.get(axis))
        if got_val != exp_val:
            raise ModelingError(f"observation_plan_geometry_mismatch:{cid}:{axis}:expected={exp_val}:got={got_val}")
    if plan_pt.get("batch_size") is not None and axis != "batch_x_kv_heads":
        exp_b = int(plan_pt["batch_size"])
        got_b = _opt_pos_int(row.get("batch_size"), row.get("sequences"), pl.get("batch_size"), pl.get("sequences"))
        if (got_b is not None and got_b != exp_b) or (got_b is None and exp_b > 1):
            raise ModelingError(f"observation_plan_geometry_mismatch:{cid}:batch_size:expected={exp_b}:got={got_b}")
    if plan_pt.get("query_tokens") is not None and axis != "tokens":
        got_q = _opt_pos_int(row.get("query_tokens"), pl.get("query_tokens"))
        if got_q is not None and got_q != int(plan_pt["query_tokens"]):
            raise ModelingError(f"observation_plan_geometry_mismatch:{cid}:query_tokens:expected={int(plan_pt['query_tokens'])}:got={got_q}")
    if plan_pt.get("topk") is not None:
        got_tk = _opt_pos_int(row.get("topk"), pl.get("topk"))
        if got_tk is not None and got_tk != int(plan_pt["topk"]):
            raise ModelingError(f"observation_plan_geometry_mismatch:{cid}:topk:expected={int(plan_pt['topk'])}:got={got_tk}")


def _row_to_gate_case(
    row: Mapping[str, Any], *, sku: str, idx: int, plan_pt: Mapping[str, Any] | None = None,
    default_clock_operator: str = DEFAULT_CLOCK_OPERATOR,
) -> GateCase:
    cid = str(row.get("case_id") or row.get("point_id") or f"obs-{idx}")
    for fk in FORBIDDEN_CALLER_PREDICTION_KEYS:
        if row.get(fk) is not None:
            raise ModelingError(f"caller_prediction_forbidden:{cid}:{fk}")
    check_no_service_regimes(row, path=cid)
    clock_op = _resolve_row_clock_operator(row, plan_pt, cid=cid, default_clock_operator=default_clock_operator)
    _validate_row_against_plan_point(row, plan_pt, cid=cid)
    plan_topo = str((plan_pt or {}).get("parallel_topology") or "").strip().lower()
    if isinstance(row.get("boundary"), Mapping) and isinstance(row.get("cohort"), Mapping):
        raw_copy = dict(row)
        raw_copy["boundary"] = {**dict(row["boundary"]), "clock_operator": clock_op}
        coh_copy = dict(row["cohort"])
        if plan_topo and not str(coh_copy.get("parallel_topology") or "").strip():
            coh_copy["parallel_topology"] = plan_topo
        raw_copy["cohort"] = coh_copy
        pl = {**dict(raw_copy.get("payload") or {}), "clock_operator": clock_op}
        for k in ("sim_case", "target_op_id", "target_family", "target_op_ids", "measured_binding", "actual_shape", "timed_nodes", "context_len", "context", "tokens", "batch_size", "sequences", "query_tokens", "batch_x_kv_heads", "parallel_topology", "topk"):
            if row.get(k) is not None and k not in pl:
                pl[k] = row[k]
        if plan_topo:
            pl.setdefault("parallel_topology", plan_topo)
        if plan_pt:
            pl.setdefault("axis", str(plan_pt.get("axis") or ""))
            if plan_pt.get("value") is not None:
                pl.setdefault("value", int(plan_pt["value"]))
        raw_copy["payload"] = pl
        gc = parse_gate_case(raw_copy)
        if gc.cohort.sku != sku:
            raise ModelingError(f"observation_sku_mismatch:{cid}:expected={sku}:got={gc.cohort.sku}")
        if plan_topo:
            req_r = validate_parallel_topology(plan_topo)
            if len(set(gc.boundary.device_ranks)) != req_r:
                raise ModelingError(f"observation_participating_ranks_mismatch:{cid}:expected={req_r}:got={len(set(gc.boundary.device_ranks))}")
        return gc
    row_sku = str(row.get("sku") or "").strip()
    if row_sku and row_sku != sku:
        raise ModelingError(f"observation_sku_mismatch:{cid}:expected={sku}:got={row_sku}")
    level = str(row.get("level") or "L1").strip().upper()
    if level not in STANDARD_LEVEL_BOUNDARY_KINDS:
        raise ModelingError(f"observation_level_invalid:{cid}:{level}")
    split = str(row.get("split") or "heldout").strip().lower()
    acq = str(row.get("acquisition_uid") or row.get("acquisition_root") or "").strip()
    run_uid = str(row.get("run_uid") or "").strip()
    raw_sha = str(row.get("raw_sha256") or "").strip()
    if not acq or not run_uid or not raw_sha:
        raise ModelingError(f"observation_provenance_missing:{cid}")
    bkind = str(row.get("boundary_kind") or STANDARD_LEVEL_BOUNDARY_KINDS[level]).strip()
    is_delta = bkind == "paired_rung_delta"
    meas_ns = int(round(float(row["measured_ns"]))) if row.get("measured_ns") is not None else int(round(float(row.get("busy_union_ns" if clock_op == "busy_union" else "elapsed_span_ns") or 0)))
    if not is_delta and meas_ns <= 0:
        raise ModelingError(f"observation_measured_ns_nonpositive:{cid}")
    ranks = tuple(int(r) for r in (row.get("device_ranks") or (0,)))
    ev_ids = tuple(str(e) for e in (row.get("event_ids") or (f"{cid}-s", f"{cid}-e")))
    eff_topo = str(row.get("parallel_topology") or plan_topo or "tp1").strip().lower()
    if plan_topo:
        req_r = validate_parallel_topology(plan_topo)
        if len(set(ranks)) != req_r:
            raise ModelingError(f"observation_participating_ranks_mismatch:{cid}:expected={req_r}:got={len(set(ranks))}")
    payload = {k: row[k] for k in ("sim_case", "target_op_id", "target_family", "target_layer", "target_op_ids", "measured_binding", "actual_shape", "timed_nodes", "context_len", "context", "tokens", "batch_size", "sequences", "query_tokens", "batch_x_kv_heads", "parallel_topology", "topk", "phase") if row.get(k) is not None}
    payload["clock_operator"] = clock_op
    payload.setdefault("parallel_topology", eff_topo)
    if plan_pt:
        payload.setdefault("axis", str(plan_pt.get("axis") or ""))
        payload.setdefault("value", int(plan_pt["value"]))
    compact_doc: dict[str, Any] = {
        "case_id": cid, "level": level, "split": split,
        "debug_cross_level": bool(row.get("debug_cross_level", False)),
        "boundary": {
            "raw_sha256": raw_sha, "acquisition_uid": acq, "run_uid": sanitize_public_run_id(run_uid),
            "parent_digests": list(row.get("parent_digests") or ()), "event_ids": list(ev_ids),
            "device_ranks": list(ranks), "clock_domain": str(row.get("clock_domain") or "device_ns"),
            "start_event": str(row.get("start_event") or "start"), "end_event": str(row.get("end_event") or "end"),
            "boundary_kind": bkind, "unit": "ns",
            "capture_uri": sanitize_public_capture_uri(str(row.get("capture_uri") or ""), raw_sha256=raw_sha),
            "clock_operator": clock_op,
        },
        "input_digest": str(row.get("input_digest") or raw_sha),
        "stack_digest": str(row.get("stack_digest") or hashlib.sha256(b"mindie-sim-whitebox").hexdigest()),
        "implementation": str(row.get("implementation") or "whitebox_spine"),
        "cohort": {"level": level, "cohort_name": str(row.get("cohort_name") or "onboarding_cliff"), "sku": sku, "phase": str(row.get("phase") or "decode"), "mode": str(row.get("mode") or "eager"), "shape_regime": str(row.get("regime_tag") or "cliff"), "parallel_topology": eff_topo},
        "measured_ns": meas_ns, "payload": payload,
    }
    for opt_k in ("is_counterfactual", "has_matched_capture", "block_capture_present", "rung_arm_a_measured_ns", "rung_arm_b_measured_ns"):
        if opt_k in row:
            compact_doc[opt_k] = row[opt_k]
    return parse_gate_case(compact_doc)


def _verify_optional_capture_files(cases: Sequence[GateCase], capture_root: Path | None, require_capture_files: bool) -> bool:
    if capture_root is None:
        if require_capture_files:
            raise ModelingError("capture_file_missing:capture_root_not_provided")
        return False
    root = Path(capture_root)
    if not root.is_dir():
        raise ModelingError(f"capture_file_missing:{root.name}")
    for c in cases:
        sha = c.boundary.raw_sha256.lower()
        cand = root / f"{sha}.json"
        if not cand.is_file():
            raise ModelingError(f"capture_file_missing:{c.case_id}:{sha[:SHORT_DIGEST_LEN]}")
        raw_bytes = cand.read_bytes()
        if hashlib.sha256(raw_bytes).hexdigest().lower() != sha:
            raise ModelingError(f"capture_file_sha256_mismatch:{c.case_id}")
        id_err = verify_capture_file_identity(
            raw_bytes, cid=c.case_id, sku=c.cohort.sku, split=c.split, level=c.level,
            measured_ns=c.measured_ns, measured_p99_ns=None, clock_operator=c.boundary.clock_operator,
        )
        if id_err:
            raise ModelingError(id_err)
    return True


def score_sealed_gate_cases(
    rows: Sequence[Mapping[str, Any]], *, sku: str, runner: Any = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT, atol_ns: int = DEFAULT_PHYSICAL_ATOL_NS,
    clock_operator: str = DEFAULT_CLOCK_OPERATOR, primitive_table: PrimitiveTable | None = None,
    profile_override: Mapping[str, Any] | None = None, external_train_roots: Sequence[str] = (),
    manifest_lineage_edges: Sequence[Any] = (),
    fits_manifest: FrozenFitManifest | Mapping[str, Any] | str | Path | None = None,
    expected_points: Sequence[Mapping[str, Any]] = (), expected_train_points: Sequence[Mapping[str, Any]] = (),
    expected_point_ids: Sequence[str] = (), expected_levels: Mapping[str, str] | None = None,
    expected_family: str = "", require_disjoint_level_roots: bool = True,
    capture_root: str | Path | None = None, require_capture_files: bool = False,
) -> dict[str, Any]:
    """Score sealed silicon observations via real ``multilevel_gate.evaluate`` and ``WhiteboxSpineRunner``."""
    cap_pct = validate_strict_threshold_pct(threshold_pct)
    eff_atol = validate_strict_atol_ns(atol_ns)
    plan_by_id = {str(p["point_id"]): p for p in (*expected_train_points, *expected_points) if isinstance(p, Mapping) and p.get("point_id")}
    cases = [_row_to_gate_case(r, sku=sku, idx=i, plan_pt=plan_by_id.get(str(r.get("case_id") or r.get("point_id") or "")), default_clock_operator=clock_operator) for i, r in enumerate(rows)]
    if not cases:
        return {"ok": False, "status": "insufficient_evidence", "sku": sku, "threshold_pct": cap_pct, "atol_ns": eff_atol, "capture_files_verified": False, "fits_sha256": "", "lineage": {}, "missing_plan_points": sorted(set(expected_point_ids) | {str(p["point_id"]) for p in expected_points}), "missing_train_points": sorted(str(p["point_id"]) for p in expected_train_points), "over_budget_points": [], "failing_cases": [], "heldout_summary": None, "train_summary": None, "residuals": []}
    for c in cases:
        if expected_levels and c.case_id in expected_levels and c.level != expected_levels[c.case_id]:
            raise ModelingError(f"observation_plan_level_mismatch:{c.case_id}:expected={expected_levels[c.case_id]}:got={c.level}")
    lineage_summary = validate_disjoint_lineage(cases, external_train_roots=external_train_roots, require_disjoint_level_roots=require_disjoint_level_roots)
    m_edges = parse_lineage_edges(manifest_lineage_edges)
    if m_edges:
        lineage_summary["lineage_edges"] = [[p, c] for p, c in m_edges]
    manifest = Manifest(f"onboarding-{sanitize_public_alias(sku, 'sku')}", "mindie-multilevel-manifest/1", tuple(cases), lineage_edges=m_edges, strict_three_level_blind=require_disjoint_level_roots)
    fits, fits_sha = load_frozen_fit_manifest(fits_manifest, default_fit_id=f"onboarding-fits-{sanitize_public_alias(sku, 'sku')}")
    validate_lineage(manifest, fits)
    cap_verified = _verify_optional_capture_files(cases, Path(capture_root) if capture_root is not None else None, require_capture_files)
    active_runner = runner if runner is not None else WhiteboxSpineRunner(clock_operator=clock_operator, primitive_table=primitive_table, profile_override=profile_override)
    raw_by_id = {str(r.get("case_id") or r.get("point_id") or f"obs-{i}"): r for i, r in enumerate(rows)}
    case_by_id = {c.case_id: c for c in cases}
    diag_by_id: dict[str, dict[str, Any]] = {}

    class _TapRunner:
        def _run(self, fn_name: str, c: GateCase) -> EngineInterval:
            iv = getattr(active_runner, fn_name)(c)
            d = dict(iv.diagnostics)
            if expected_family and isinstance(d.get("executed_op_families"), (list, tuple)) and d["executed_op_families"] and expected_family not in d["executed_op_families"]:
                raise ModelingError(f"operator_family_replay_mismatch:{c.case_id}:expected={expected_family}:executed={list(d['executed_op_families'])}")
            diag_by_id[c.case_id] = d
            return iv

        def run_l1(self, c: GateCase) -> EngineInterval:
            return self._run("run_l1", c)

        def run_l2(self, c: GateCase) -> EngineInterval:
            return self._run("run_l2", c)

        def run_l3(self, c: GateCase) -> EngineInterval:
            return self._run("run_l3", c)

    eval_rows = list(evaluate_multilevel(manifest, fits, _TapRunner(), target_split="all"))
    residuals, failing, over_budget, held_apes, train_apes = [], [], [], [], []
    covered_heldout_ids: set[str] = set()
    covered_train_ids: set[str] = set()
    for rec in eval_rows:
        src, d, gc = raw_by_id.get(rec.case_id, {}), diag_by_id.get(rec.case_id, {}), case_by_id[rec.case_id]
        eff_clock = str(rec.clock_operator)
        if eff_clock not in VALID_CLOCK_OPERATORS:
            raise ModelingError(f"clock_operator_invalid:{rec.case_id}:{eff_clock}")
        if d.get("clock_operator") and str(d["clock_operator"]) != eff_clock:
            raise ModelingError(f"whitebox_clock_operator_mismatch:{rec.case_id}:expected={eff_clock}:got={d['clock_operator']}")
        is_delta = gc.boundary.boundary_kind == "paired_rung_delta"
        pred_ns = int(rec.predicted_ns)
        if not is_delta and rec.status in {"ok", "debug_only", "near_zero_delta"} and pred_ns <= 0:
            raise ModelingError(f"whitebox_selected_clock_missing:{rec.case_id}:{eff_clock}")
        meas_ns = int(rec.measured_ns)
        if is_delta:
            ape = round(abs(pred_ns - meas_ns) / float(abs(meas_ns)) * PERCENT_SCALE, DISPLAY_ROUND_DIGITS) if meas_ns != 0 else (0.0 if pred_ns == 0 else PERCENT_SCALE)
        else:
            ape = round(abs(pred_ns - meas_ns) / float(meas_ns) * PERCENT_SCALE, DISPLAY_ROUND_DIGITS) if (meas_ns > 0 and pred_ns > 0) else PERCENT_SCALE
        over = exceeds_physical_tolerance(pred_ns, meas_ns, threshold_pct=cap_pct, atol_ns=eff_atol, allow_signed=is_delta) if rec.status == "ok" else False
        residuals.append({
            "case_id": rec.case_id, "level": rec.level, "split": gc.split, "sku": sku,
            "boundary_kind": gc.boundary.boundary_kind,
            "status": rec.status, "excluded_reason": rec.excluded_reason,
            "measured_ns": meas_ns, "measured_p99_ns": int(src["measured_p99_ns"]) if src.get("measured_p99_ns") else None,
            "predicted_ns": pred_ns, "ape_pct": ape, "over_budget": over, "clock_operator": eff_clock,
            "engine_receipt": str(rec.engine_receipt or ""),
            "program_dag_sha256": str(d.get("program_dag_sha256") or ""),
            "executed_op_families": list(d.get("executed_op_families") or ()), "executed_op_ids": list(d.get("executed_op_ids") or ()),
            "raw_sha256": gc.boundary.raw_sha256, "acquisition_root": gc.boundary.acquisition_uid,
            "run_uid": gc.boundary.run_uid, "parent_digests": list(gc.boundary.parent_digests),
            "capture_uri": sanitize_public_capture_uri(str(src.get("capture_uri") or gc.boundary.capture_uri), raw_sha256=gc.boundary.raw_sha256),
            "implementation": str(src.get("implementation") or "whitebox_spine"), "target_family": str(src.get("target_family") or expected_family),
            "operator_id": str(src.get("operator_id") or ""), "upstream_abi": str(src.get("upstream_abi") or ""),
        })
        if rec.status != "ok":
            continue
        if gc.split == "heldout":
            covered_heldout_ids.add(rec.case_id)
            held_apes.append(ape)
            if over:
                over_budget.append(rec.case_id)
                failing.append(rec.case_id)
        else:
            covered_train_ids.add(rec.case_id)
            train_apes.append(ape)
    miss_ho = sorted((set(expected_point_ids) | {str(p["point_id"]) for p in expected_points}) - covered_heldout_ids)
    miss_tr = sorted({str(p["point_id"]) for p in expected_train_points} - covered_train_ids)

    def _summ(apes: list[float]) -> dict[str, Any] | None:
        if not apes:
            return None
        s = sorted(apes)
        return {"count": len(s), "max_ape_pct": s[-1], "p90_ape_pct": s[min(len(s) - 1, int(len(s) * _P90_QUANTILE))], "median_ape_pct": s[len(s) // 2]}
    ok = not miss_ho and not miss_tr and not failing and bool(held_apes)
    out = {
        "ok": ok, "status": "insufficient_evidence" if (miss_ho or miss_tr or not held_apes) else ("pass" if ok else "fail"),
        "sku": sku, "threshold_pct": cap_pct, "atol_ns": eff_atol, "capture_files_verified": cap_verified,
        "fits_sha256": fits_sha, "lineage": lineage_summary,
        "missing_plan_points": miss_ho, "missing_train_points": miss_tr, "over_budget_points": sorted(over_budget),
        "failing_cases": sorted(failing), "heldout_summary": _summ(held_apes), "train_summary": _summ(train_apes), "residuals": residuals,
    }
    assert_sanitized_payload(out, field_name="sealed_gate_score")
    return out
