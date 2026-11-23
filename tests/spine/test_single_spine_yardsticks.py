"""Single-spine equivalence yardsticks, negative controls, 93L drift lock, and RSS probe.

[Role] Establish the canonical strict integer-nanosecond ``< 0.01%`` cohort
  equivalence gate (``10_000 * |cand - ref| < ref`` on total and every
  module/bucket key), prove all six negative controls fail loudly (exact
  ``0.01%`` boundary, cancelling module drift, missing point, fabricated or
  uncomputed ``predicted_ns``, mismatched ``PrimitiveTable`` digest, and
  cross-layer auxiliary-stream tail exposure drift), lock the empirical
  93-layer Kimi-K3.1 ``layers=3`` fold error as a strict xfail ready for H4,
  and validate the isolated Linux child-process VmRSS/VmHWM probe.
[Invariants] Strict integer-ns arithmetic only; zero silent fallbacks; zero
  caller-supplied prediction bypasses; zero local host pollution.
[Timeouts & Locks] Tier 1: runs literal 93-layer Engine replay and one bounded
  child Python process on Linux; no locks or network access.
[Side-Effect Policy] Read-only over repository assets; no persistent writes.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import sys
from typing import Any

import pytest

from _support.axes import GEN5_DT_SKU, REPRESENTATIVE_SKUS
from _support.rss_probe import (
    ChildRssMeasurement,
    RssBudgetError,
    assert_rss_budget,
    measure_child_rss,
)
from mindie_sim.compose import compose_case
from mindie_sim.observations.multilevel import EngineInterval, GateCase, bind_engine_interval
from mindie_sim.onboarding.contracts import (
    FORBIDDEN_CALLER_PREDICTION_KEYS,
    NS_PER_MS,
    exceeds_physical_tolerance,
)
from mindie_sim.onboarding.whitebox_runner import score_sealed_gate_cases
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.sweep import pd as pd_mod

pytestmark = pytest.mark.tier1

COHORT_TOLERANCE_DENOMINATOR = 10_000  # strict < 0.01% == 1 / 10_000
KIMI_EXPECTED_TARGET_LAYERS = 93
A2_SKU = REPRESENTATIVE_SKUS["a2"]


class CohortEquivalenceError(ValueError):
    """Raised when candidate execution violates strict < 0.01% cohort equivalence."""


def _require_nonneg_int_ns(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise CohortEquivalenceError(f"cohort_ns_invalid:{label}:{value!r}")
    return value


def _check_strict_ns_bound(cand_ns: int, ref_ns: int, label: str) -> None:
    if ref_ns == 0:
        if cand_ns != 0:
            raise CohortEquivalenceError(
                f"cohort_zero_ref_mismatch:{label}:cand_ns={cand_ns}:ref_ns=0"
            )
        return
    delta_ns = abs(cand_ns - ref_ns)
    # Exact 0.01% equality (10_000 * delta == ref) MUST fail.
    if COHORT_TOLERANCE_DENOMINATOR * delta_ns >= ref_ns:
        err_pct = (delta_ns * 100.0) / float(ref_ns)
        raise CohortEquivalenceError(
            f"cohort_equivalence_exceeded:{label}:"
            f"cand_ns={cand_ns}:ref_ns={ref_ns}:delta_ns={delta_ns}:err_pct={err_pct:.6f}%"
        )


def _validate_engine_receipt(receipt: Mapping[str, Any] | None, side: str) -> None:
    if receipt is None:
        return
    for forbidden in FORBIDDEN_CALLER_PREDICTION_KEYS:
        if forbidden in receipt and receipt[forbidden] is not None:
            raise CohortEquivalenceError(
                f"cohort_fabricated_prediction_key:{side}:{forbidden}"
            )
    if str(receipt.get("source") or "") != "engine":
        raise CohortEquivalenceError(
            f"cohort_non_engine_source:{side}:{receipt.get('source')!r}"
        )
    if not str(receipt.get("engine_uid") or "").strip():
        raise CohortEquivalenceError(f"cohort_missing_engine_uid:{side}")


def _validate_provenance_pair(
    cand_val: str | None,
    ref_val: str | None,
    *,
    code: str,
) -> None:
    if cand_val is None and ref_val is None:
        return
    c_str = str(cand_val or "").strip()
    r_str = str(ref_val or "").strip()
    if not c_str or not r_str or c_str != r_str:
        raise CohortEquivalenceError(f"{code}:cand={c_str!r}:ref={r_str!r}")


def assert_cohort_equivalence_ns(
    cand_ns: int,
    ref_ns: int,
    cand_buckets_ns: Mapping[str, int],
    ref_buckets_ns: Mapping[str, int],
    *,
    expected_keys: Sequence[str] | None = None,
    cand_table_digest: str | None = None,
    ref_table_digest: str | None = None,
    cand_clock_operator: str | None = None,
    ref_clock_operator: str | None = None,
    cand_receipt: Mapping[str, Any] | None = None,
    ref_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strict integer-ns ``< 0.01%`` equivalence gate across total and all buckets."""
    _validate_engine_receipt(cand_receipt, "cand")
    _validate_engine_receipt(ref_receipt, "ref")
    _validate_provenance_pair(
        cand_table_digest if cand_table_digest is not None else (
            cand_receipt.get("primitive_table_digest") if cand_receipt is not None else None
        ),
        ref_table_digest if ref_table_digest is not None else (
            ref_receipt.get("primitive_table_digest") if ref_receipt is not None else None
        ),
        code="cohort_primitive_table_mismatch",
    )
    _validate_provenance_pair(
        cand_clock_operator if cand_clock_operator is not None else (
            cand_receipt.get("clock_operator") if cand_receipt is not None else None
        ),
        ref_clock_operator if ref_clock_operator is not None else (
            ref_receipt.get("clock_operator") if ref_receipt is not None else None
        ),
        code="cohort_clock_operator_mismatch",
    )

    c_total = _require_nonneg_int_ns(cand_ns, "total.cand")
    r_total = _require_nonneg_int_ns(ref_ns, "total.ref")
    if not isinstance(cand_buckets_ns, Mapping) or not isinstance(ref_buckets_ns, Mapping):
        raise CohortEquivalenceError("cohort_buckets_mapping_required")

    cand_keys = frozenset(cand_buckets_ns)
    ref_keys = frozenset(ref_buckets_ns)
    if expected_keys is not None:
        if len(expected_keys) != len(set(expected_keys)):
            raise CohortEquivalenceError("cohort_duplicate_expected_keys")
        exp_keys = frozenset(expected_keys)
        if missing_ref := sorted(exp_keys - ref_keys):
            raise CohortEquivalenceError(f"cohort_missing_reference_keys:{','.join(missing_ref)}")
        if missing_exp := sorted(exp_keys - cand_keys):
            raise CohortEquivalenceError(f"cohort_missing_expected_keys:{','.join(missing_exp)}")
        if extra_ref := sorted(ref_keys - exp_keys):
            raise CohortEquivalenceError(f"cohort_unexpected_reference_keys:{','.join(extra_ref)}")
    elif r_total > 0 and not ref_keys:
        raise CohortEquivalenceError("cohort_empty_buckets_for_positive_reference")

    if missing_cand := sorted(ref_keys - cand_keys):
        raise CohortEquivalenceError(f"cohort_missing_candidate_keys:{','.join(missing_cand)}")
    if extra_cand := sorted(cand_keys - ref_keys):
        raise CohortEquivalenceError(f"cohort_unexpected_candidate_keys:{','.join(extra_cand)}")

    _check_strict_ns_bound(c_total, r_total, "total")
    max_bucket_err_pct = 0.0
    for key in sorted(ref_keys):
        c_val = _require_nonneg_int_ns(cand_buckets_ns[key], f"bucket[{key}].cand")
        r_val = _require_nonneg_int_ns(ref_buckets_ns[key], f"bucket[{key}].ref")
        _check_strict_ns_bound(c_val, r_val, f"bucket[{key}]")
        if r_val > 0:
            max_bucket_err_pct = max(max_bucket_err_pct, abs(c_val - r_val) * 100.0 / float(r_val))

    total_err_pct = (abs(c_total - r_total) * 100.0 / float(r_total)) if r_total > 0 else 0.0
    return {
        "cand_ns": c_total,
        "ref_ns": r_total,
        "delta_ns": abs(c_total - r_total),
        "total_err_pct": total_err_pct,
        "max_bucket_err_pct": max_bucket_err_pct,
        "bucket_count": len(ref_keys),
    }


def _sealed_gate_row(case_id: str, *, measured_ns: int = 10_000, clock_operator: str = "busy_union") -> dict[str, Any]:
    raw_sha = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    return {
        "case_id": case_id,
        "point_id": case_id,
        "sku": A2_SKU,
        "split": "heldout",
        "level": "L1",
        "acquisition_root": f"acq-{case_id}",
        "run_uid": f"run-{case_id}",
        "raw_sha256": raw_sha,
        "event_ids": [f"{case_id}-s", f"{case_id}-e"],
        "device_ranks": [0],
        "measured_ns": measured_ns,
        "measured_p99_ns": measured_ns + 100,
        "clock_operator": clock_operator,
        "phase": "decode",
        "context_len": 64,
        "query_tokens": 1,
        "batch_size": 1,
    }


def _engine_stub_runner(
    duration_ns: int,
    *,
    diagnostics: tuple[tuple[str, Any], ...],
) -> Any:
    class _Runner:
        def _exec(self, case: GateCase) -> EngineInterval:
            eng = Engine([Resource("device", 1_000_000.0, "ns", "fifo")], trace_limit=32)
            s_ev = eng.submit(case.boundary.start_event, {"device": 100.0})
            e_ev = eng.submit(case.boundary.end_event, {"device": float(duration_ns)}, after=(s_ev,))
            eng.advance(until=(e_ev,))
            return bind_engine_interval(case, eng, s_ev, e_ev, diagnostics=diagnostics)

        run_l1 = run_l2 = run_l3 = _exec

    return _Runner()


def test_cohort_equivalence_rejects_exact_boundary_step() -> None:
    """Negative control (a): exact 0.01% step (10_000 * delta == ref) MUST fail."""
    ref_ns = 1_000_000
    # Contrast with the non-strict silicon tolerance helper where exact equality passes.
    assert exceeds_physical_tolerance(1_000_100, ref_ns, threshold_pct=0.01, atol_ns=0) is False

    # Just inside the strict bound passes.
    ok = assert_cohort_equivalence_ns(1_000_099, ref_ns, {"step": 1_000_099}, {"step": ref_ns})
    assert ok["delta_ns"] == 99 and ok["total_err_pct"] < 0.01

    # Exact +0.01%, exact -0.01%, and >0.01% all fail loudly.
    for bad_ns in (1_000_100, 999_900, 1_000_101):
        with pytest.raises(CohortEquivalenceError, match="cohort_equivalence_exceeded:total"):
            assert_cohort_equivalence_ns(bad_ns, ref_ns, {"step": bad_ns}, {"step": ref_ns})

    # Zero-work reference requires exact zero equality.
    assert assert_cohort_equivalence_ns(0, 0, {"zero": 0}, {"zero": 0})["delta_ns"] == 0
    with pytest.raises(CohortEquivalenceError, match="cohort_zero_ref_mismatch"):
        assert_cohort_equivalence_ns(1, 0, {"zero": 1}, {"zero": 0})


def test_cohort_equivalence_rejects_cancelling_module_drift() -> None:
    """Negative control (b): +0.5% / -0.5% module drift cancelling in total MUST fail."""
    ref_buckets = {"attention": 10_000_000, "moe": 10_000_000, "norm": 2_000_000}
    cand_buckets = {"attention": 10_050_000, "moe": 9_950_000, "norm": 2_000_000}
    assert sum(cand_buckets.values()) == sum(ref_buckets.values()) == 22_000_000
    with pytest.raises(CohortEquivalenceError, match=r"cohort_equivalence_exceeded:bucket\[attention\]"):
        assert_cohort_equivalence_ns(22_000_000, 22_000_000, cand_buckets, ref_buckets)


def test_cohort_equivalence_rejects_missing_grid_or_module_point() -> None:
    """Negative control (c): missing bucket/grid point MUST fail cohort and sealed gates."""
    ref_buckets = {"attn": 5_000_000, "ffn": 5_000_000, "comm": 1_000_000}
    with pytest.raises(CohortEquivalenceError, match="cohort_missing_candidate_keys:comm"):
        assert_cohort_equivalence_ns(11_000_000, 11_000_000, {"attn": 5_000_000, "ffn": 5_000_000}, ref_buckets)
    with pytest.raises(CohortEquivalenceError, match="cohort_unexpected_candidate_keys:extra"):
        assert_cohort_equivalence_ns(
            11_000_000, 11_000_000, {**ref_buckets, "extra": 0}, ref_buckets,
        )
    with pytest.raises(CohortEquivalenceError, match="cohort_missing_reference_keys:lm_head"):
        assert_cohort_equivalence_ns(
            11_000_000, 11_000_000, ref_buckets, ref_buckets,
            expected_keys=("attn", "ffn", "comm", "lm_head"),
        )

    # Real sealed gate also rejects missing expected grid points.
    row_a = _sealed_gate_row("grid-a", clock_operator="elapsed_span")
    scored = score_sealed_gate_cases(
        [row_a],
        sku=A2_SKU,
        clock_operator="elapsed_span",
        expected_point_ids=("grid-a", "grid-b-missing"),
        runner=_engine_stub_runner(
            10_000,
            diagnostics=(("clock_operator", "elapsed_span"), ("elapsed_span_ns", 10_000), ("busy_union_ns", 10_000)),
        ),
    )
    assert scored["ok"] is False
    assert scored["status"] == "insufficient_evidence"
    assert scored["missing_plan_points"] == ["grid-b-missing"]


def test_cohort_equivalence_and_whitebox_reject_fabricated_or_missing_predicted_ns() -> None:
    """Negative control (d): fabricated/non-Engine predicted_ns & missing clock MUST fail."""
    good_ref = {"source": "engine", "engine_uid": "eng-ref-01"}
    with pytest.raises(CohortEquivalenceError, match="cohort_fabricated_prediction_key:cand:predicted_ns"):
        assert_cohort_equivalence_ns(
            1_000_000, 1_000_000, {"step": 1_000_000}, {"step": 1_000_000},
            cand_receipt={"source": "engine", "engine_uid": "eng-c1", "predicted_ns": 1_000_000},
            ref_receipt=good_ref,
        )
    with pytest.raises(CohortEquivalenceError, match="cohort_non_engine_source:cand"):
        assert_cohort_equivalence_ns(
            1_000_000, 1_000_000, {"step": 1_000_000}, {"step": 1_000_000},
            cand_receipt={"source": "formula_substitute", "engine_uid": "fake"},
            ref_receipt=good_ref,
        )

    # Caller-injected predicted_ns is rejected at sealed-gate ingress.
    row_busy = _sealed_gate_row("wb-missing-clock", clock_operator="busy_union")
    with pytest.raises(ModelingError, match="caller_prediction_forbidden"):
        score_sealed_gate_cases(
            [{**row_busy, "predicted_ns": 10_000}],
            sku=A2_SKU,
            clock_operator="busy_union",
            runner=_engine_stub_runner(
                10_000,
                diagnostics=(("clock_operator", "busy_union"), ("busy_union_ns", 10_000), ("elapsed_span_ns", 10_000)),
            ),
        )

    # Missing or zero selected clock in runner diagnostics MUST NOT fall back to rec.predicted_ns.
    for bad_diag in (
        (("clock_operator", "busy_union"), ("elapsed_span_ns", 10_000)),
        (("clock_operator", "busy_union"), ("busy_union_ns", 0), ("elapsed_span_ns", 10_000)),
        (),
    ):
        with pytest.raises(ModelingError, match="whitebox_selected_clock_missing:wb-missing-clock:busy_union_ns"):
            score_sealed_gate_cases(
                [row_busy],
                sku=A2_SKU,
                clock_operator="busy_union",
                runner=_engine_stub_runner(10_000, diagnostics=bad_diag),
            )


def test_cohort_equivalence_rejects_mismatched_primitive_table_fingerprint() -> None:
    """Negative control (e): swapped or rescaled PrimitiveTable fingerprint MUST fail."""
    base_case = parse_case(compose_case("qwen3-32b", GEN5_DT_SKU, layers=1, tp=1))
    perturbed_case = parse_case(
        compose_case("qwen3-32b", GEN5_DT_SKU, layers=1, tp=1, hbm_peak_bytes_per_s=1.6e12)
    )
    tbl_base = EvaluationContext.create(base_case).bindings.primitive_table
    tbl_perturbed = EvaluationContext.create(perturbed_case).bindings.primitive_table
    assert tbl_base is not None and tbl_perturbed is not None
    fp_base = tbl_base.canonical_envelope_sha256()
    fp_perturbed = tbl_perturbed.canonical_envelope_sha256()
    assert fp_base and fp_perturbed and fp_base != fp_perturbed

    # Identical timings with mismatched PrimitiveTable digests MUST fail.
    with pytest.raises(CohortEquivalenceError, match="cohort_primitive_table_mismatch"):
        assert_cohort_equivalence_ns(
            5_000_000, 5_000_000, {"step": 5_000_000}, {"step": 5_000_000},
            cand_table_digest=fp_perturbed, ref_table_digest=fp_base,
        )
    # Missing/disabled table (primitive_table=False -> empty digest) against calibrated table MUST fail.
    with pytest.raises(CohortEquivalenceError, match="cohort_primitive_table_mismatch"):
        assert_cohort_equivalence_ns(
            5_000_000, 5_000_000, {"step": 5_000_000}, {"step": 5_000_000},
            cand_table_digest="", ref_table_digest=fp_base,
        )


def test_cohort_equivalence_rejects_cross_layer_aux_stream_tail_drift() -> None:
    """Negative control (f): cross-layer aux-stream tail exposure drift MUST fail."""
    def _run_layers(layer_count: int, *, drain_between_layers: bool) -> tuple[int, dict[str, int]]:
        eng = Engine(
            [
                Resource("cube", 1_000_000.0, "ns", "fifo"),
                Resource("hbm", 1_000_000.0, "ns", "fair"),
            ],
            backend="python",
            trace_limit=64,
        )
        prev_main: tuple[Any, ...] = ()
        prev_aux: tuple[Any, ...] = ()
        all_tails: list[Any] = []
        layer_main_ends: dict[str, Any] = {}
        layer_aux_ends: dict[str, Any] = {}
        for idx in range(layer_count):
            deps = (*prev_main, *prev_aux) if drain_between_layers else prev_main
            fork_ev = eng.submit(f"l{idx}_fork", {"cube": 300.0}, after=deps)
            main_ev = eng.submit(f"l{idx}_main", {"cube": 700.0}, after=(fork_ev,))
            # Aux-stream tail extends 400 ns past main_ev and overlaps next layer's cube work.
            aux_ev = eng.submit(f"l{idx}_aux_tail", {"hbm": 1100.0}, after=(fork_ev, *prev_aux))
            prev_main = (main_ev,)
            prev_aux = (aux_ev,)
            all_tails.extend((main_ev, aux_ev))
            layer_main_ends[f"layer{idx}_main"] = main_ev
            layer_aux_ends[f"layer{idx}_aux"] = aux_ev
        done = eng.boundary("step_done", after=tuple(all_tails))
        eng.advance(until=(done,))
        total_ns = int(round(float(done.time) * NS_PER_MS))
        buckets = {
            **{k: int(round(float(ev.time) * NS_PER_MS)) for k, ev in layer_main_ends.items()},
            **{k: int(round(float(ev.time) * NS_PER_MS)) for k, ev in layer_aux_ends.items()},
            "step_end": total_ns,
        }
        return total_ns, buckets

    ref_ns, ref_buckets = _run_layers(2, drain_between_layers=False)
    drained_ns, drained_buckets = _run_layers(2, drain_between_layers=True)
    one_layer_ns, _ = _run_layers(1, drain_between_layers=False)
    folded_2x_ns = one_layer_ns * 2
    assert ref_ns == 2500 and drained_ns == 2800 and folded_2x_ns == 2800
    with pytest.raises(CohortEquivalenceError, match="cohort_equivalence_exceeded"):
        assert_cohort_equivalence_ns(drained_ns, ref_ns, drained_buckets, ref_buckets)
    with pytest.raises(CohortEquivalenceError, match="cohort_equivalence_exceeded:total"):
        assert_cohort_equivalence_ns(
            folded_2x_ns, ref_ns, {"step_end": folded_2x_ns}, {"step_end": ref_ns},
        )


@pytest.mark.xfail(
    strict=True,
    raises=CohortEquivalenceError,
    reason=(
        "H4 replaces mindie_sim/sweep/pd.py:1596 layers=3 fold with "
        "mindie_sim/prediction/execution/layer_runs.py:232 exact cohort"
    ),
)
def test_kimi_ninety_three_layer_fold_equivalence_against_literal_engine() -> None:
    """Empirical 93L Kimi-K3.1 reference: current 3-layer fold fails < 0.01% until H4."""
    raw = compose_case(
        "kimi-k3.1",
        GEN5_DT_SKU,
        dp=1,
        tp=1,
        ep=1,
        mtp_proposal_tokens=0,
        prompt_tokens=32,
        output_tokens=1,
        concurrency=1,
    )
    parsed = parse_case(raw)
    assert len(parsed.model.layer_plans) == KIMI_EXPECTED_TARGET_LAYERS

    eval_ctx = EvaluationContext.create(parsed)
    table = eval_ctx.bindings.primitive_table
    assert table is not None
    table_digest = table.canonical_envelope_sha256()
    assert table_digest

    graphs, eff_case, _ = pd_mod._lower_and_run_graph(
        parsed, phase="decode", tokens=1, context=32, sequences=1, scope="model", run_graph=False,
    )
    ref_graph = graph_costs(
        eff_case,
        {0: graphs[0]},
        external_inputs=None,
        isolated=False,
        include_submission=False,
        evaluation=eval_ctx,
        fast_span_only=False,
        timeline_replicas=(0,),
    )
    ref_ns = int(round(float(ref_graph["duration_ms"]) * NS_PER_MS))
    assert ref_ns > 0 and int(ref_graph["operator_occurrences"]) > 1000

    cand_ms = pd_mod._archetype_weighted_step_ms(
        parsed,
        phase="decode",
        tokens=1,
        context=32,
        sequences=1,
        scope="model",
        evaluation_context=eval_ctx,
        reclaim_on_exit=False,
    )
    cand_ns = int(round(float(cand_ms) * NS_PER_MS))
    assert cand_ns > 0

    assert_cohort_equivalence_ns(
        cand_ns,
        ref_ns,
        {"model_step": cand_ns},
        {"model_step": ref_ns},
        expected_keys=("model_step",),
        cand_table_digest=table_digest,
        ref_table_digest=table_digest,
        cand_clock_operator="device_graph_duration",
        ref_clock_operator="device_graph_duration",
        cand_receipt={"source": "engine", "engine_uid": "sweep_archetype_fold"},
        ref_receipt={"source": "engine", "engine_uid": "literal_93l_engine"},
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc/self/status VmRSS/VmHWM required")
def test_isolated_child_rss_probe_measures_pre_rss_and_hwm_delta() -> None:
    """Isolated Linux child probe measures VmRSS, VmHWM, delta, and enforces RSS budgets."""
    measurement = measure_child_rss(
        "_support.rss_probe:tiny_engine_rss_fixture",
        {"context": 32, "hardware": GEN5_DT_SKU},
    )
    assert measurement.pre_rss_kib > 0
    assert measurement.post_hwm_kib >= measurement.pre_rss_kib
    assert measurement.hwm_mib >= measurement.pre_rss_mib > 0.0
    assert measurement.delta_hwm_mib >= 0.0
    assert measurement.result["ok"] is True
    assert measurement.result["span_ms"] > 0.0
    assert measurement.result["op_count"] > 0
    assert_rss_budget(measurement)

    # Negative controls on budget enforcement (ready for H4/H5 32-die < 250 MiB gate).
    over_delta = ChildRssMeasurement(
        pre_rss_kib=100 * 1024,
        pre_hwm_kib=100 * 1024,
        post_rss_kib=350 * 1024,
        post_hwm_kib=350 * 1024,
        result=None,
    )
    with pytest.raises(RssBudgetError, match="rss_delta_exceeded"):
        assert_rss_budget(over_delta)
    over_hwm = ChildRssMeasurement(
        pre_rss_kib=400 * 1024,
        pre_hwm_kib=400 * 1024,
        post_rss_kib=520 * 1024,
        post_hwm_kib=520 * 1024,
        result=None,
    )
    with pytest.raises(RssBudgetError, match="rss_hwm_exceeded"):
        assert_rss_budget(over_hwm)
