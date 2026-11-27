"""Multi-core MapReduce solver governance, PD stage factorization, and export polish tests.

[Role] Verify:
  (1) ``MINDIE_GRANTED_WORKERS`` resolution, ``[1, 64]`` clamping, automatic
      clamp to ``1`` under ``PYTEST_CURRENT_TEST`` / ``PYTEST_XDIST_WORKER``,
      explicit ``granted_workers=N`` override, and zombie-free pool cleanup;
  (2) Bit-for-bit identical ``solve_deployment`` / ``solve_optimal_deployment``
      outputs across ``granted_workers=1`` and ``granted_workers=4`` on
      single-pool (``pd_decode``, ``colocated``) and ``pd_disaggregated`` with
      ``|P| + |D|`` stage factorization + closed-form ``(n_p, n_d)`` join;
  (3) Zero-gap 4-stage waterfall Stage-0 anchoring when pairing ``merged``
      Before with ``expected_ideal`` After across bundle, comparison plan, and
      XLSX Summary;
  (4) Accurate ``mxfp4`` display ``dtype`` and tensor contract on MXFP4 routed
      expert operators without regressing MXFP8 or W4A8 precision tags.
[Invariants] Uses ``monkeypatch`` and ``tmp_path`` exclusively (zero host env or
  filesystem pollution); leaves zero active child worker processes on exit.
"""
from __future__ import annotations

from dataclasses import replace
import math
import multiprocessing
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import zipfile

import pytest

from mindie_sim.compose import compose_case
from mindie_sim.export import (
    align_before_after_operators,
    build_export_bundle,
    compose_export_case,
    export_comparison_xlsx,
)
from mindie_sim.export.action_plan import build_comparison_plan
from mindie_sim.export.ooxml import NS_MAIN
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.results import StepRequest
from mindie_sim.prediction.streams import MAIN_STREAM, StreamCoreBudget, StreamEvent, StreamLane
from mindie_sim.rewrite.apply import apply_plan
from mindie_sim.rewrite.catalog_replay import _catlass_proposals, _replay_ops
from mindie_sim.rewrite.oracle import (
    MAX_OVERLAP_TRACE_EVENTS,
    ScenarioRequest,
    _bind_replay,
    _report,
    finalize_plan,
    prepare_scenario,
    refine_layer_cohort_plan,
)
from mindie_sim.rewrite.schema import Edit, OpRef, StreamSchedule, empty_plan
from mindie_sim.rewrite.verify import _evaluation
from mindie_sim.schema.common import ModelingError
from mindie_sim.solve import solve_deployment
from mindie_sim.sweep.mapreduce import (
    GRANTED_WORKERS_ENV,
    MAX_GRANTED_WORKERS,
    MIN_GRANTED_WORKERS,
    BoundedMapPool,
    resolve_granted_workers,
)
from mindie_sim.sweep.pd import (
    _build_pd_ratio_rows,
    _mark_pd_pareto,
    _solve_optimal_pd_ratio_closed_form,
    solve_optimal_deployment,
)
from mindie_sim.templates import catlass_lower
from mindie_sim.templates import registry as template_registry

pytestmark = pytest.mark.tier1

_WATERFALL_TOL_US = 1e-6
_CLOSED_FORM_TOL = 1e-10
_PARITY_REL_TOL = 1e-11
_PARITY_ABS_TOL_MS = 1e-9
_TEST_PROMPT_TOKENS = 1024
_COHORT_TEST_LAYERS = 12
_DP_TEST_REPLICAS = 8
_DP_MIN_SPEEDUP = 5.0
_CATLASS_PROPOSAL_SLICE = 20
_TEST_OUTPUT_TOKENS = 128
_TEST_TPOT_SLA_MS = 50.0
_TEST_TTFT_SLA_MS = 2000.0
_TEST_NUM_DEVICES = 8
_TEST_LAYERS = 4


def _probe_worker_env_and_value(item: int) -> tuple[int, str]:
    """Top-level pickle-safe worker verifying child worker env pinning."""
    return int(item) * int(item), str(os.environ.get(GRANTED_WORKERS_ENV, ""))


def _read_xlsx_sheet_cells(xlsx_path: Path, sheet_index: int = 1) -> dict[str, str]:
    ns = {"m": NS_MAIN}
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sst.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//m:t", ns)))
        root = ET.fromstring(zf.read(f"xl/worksheets/sheet{sheet_index}.xml"))
    cells: dict[str, str] = {}
    for cell in root.findall(".//m:c", ns):
        ref = cell.attrib.get("r", "")
        is_node = cell.find("m:is/m:t", ns)
        if is_node is not None and is_node.text is not None:
            cells[ref] = is_node.text
            continue
        v_node = cell.find("m:v", ns)
        if v_node is not None and v_node.text is not None:
            if cell.attrib.get("t") == "s":
                idx = int(v_node.text)
                cells[ref] = shared[idx] if 0 <= idx < len(shared) else v_node.text
            else:
                cells[ref] = v_node.text
    return cells


def test_granted_workers_resolution_clamping_and_pytest_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(GRANTED_WORKERS_ENV, "16")
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    # Default implicit resolution auto-clamps to 1 under pytest/xdist.
    assert resolve_granted_workers() == MIN_GRANTED_WORKERS
    # Explicit caller override bypasses the pytest guard while still clamping [1, 64].
    assert resolve_granted_workers(4) == 4
    assert resolve_granted_workers(0) == MIN_GRANTED_WORKERS
    assert resolve_granted_workers(-5) == MIN_GRANTED_WORKERS
    assert resolve_granted_workers(128) == MAX_GRANTED_WORKERS
    assert resolve_granted_workers(8, task_count=3) == 3

    with pytest.raises(ModelingError, match="invalid_granted_workers"):
        resolve_granted_workers(True)  # type: ignore[arg-type]
    with pytest.raises(ModelingError, match="invalid_granted_workers"):
        resolve_granted_workers("4")  # type: ignore[arg-type]

    # Outside pytest env guards, MINDIE_GRANTED_WORKERS is honored and clamped.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    assert resolve_granted_workers() == 16
    monkeypatch.setenv(GRANTED_WORKERS_ENV, "256")
    assert resolve_granted_workers() == MAX_GRANTED_WORKERS
    monkeypatch.setenv(GRANTED_WORKERS_ENV, "0")
    assert resolve_granted_workers() == MIN_GRANTED_WORKERS
    monkeypatch.setenv(GRANTED_WORKERS_ENV, "not_an_int")
    assert resolve_granted_workers() == MIN_GRANTED_WORKERS
    monkeypatch.delenv(GRANTED_WORKERS_ENV, raising=False)
    assert resolve_granted_workers() == MIN_GRANTED_WORKERS


def test_bounded_map_pool_order_child_pin_and_zero_zombie_cleanup() -> None:
    inputs = list(range(8))
    with BoundedMapPool(granted_workers=4, max_tasks=len(inputs)) as pool:
        assert pool.effective_workers == 4
        out = pool.map_ordered(_probe_worker_env_and_value, inputs)
    assert [val for val, _ in out] == [x * x for x in inputs]
    assert all(env_val == "1" for _, env_val in out)
    assert multiprocessing.active_children() == []


def test_single_pool_solver_bit_for_bit_deterministic_across_worker_counts() -> None:
    common_kw = dict(
        num_devices=_TEST_NUM_DEVICES,
        prompt_tokens=_TEST_PROMPT_TOKENS,
        output_tokens=_TEST_OUTPUT_TOKENS,
        tpot_slo_ms=_TEST_TPOT_SLA_MS,
        deployment_mode="pd_decode",
        mtp_tokens=0,
        max_sequences_per_dp=4,
        layers=2,
        mode="simulation",
    )
    sol_1 = solve_deployment("qwen3-235b-a22b", "a5", granted_workers=1, **common_kw)
    sol_4 = solve_deployment("qwen3-235b-a22b", "a5", granted_workers=4, **common_kw)

    assert sol_1["feasible"] is True
    assert sol_4 == sol_1
    assert sol_4["recommended_deployment"]["best_configuration"] == sol_1["recommended_deployment"]["best_configuration"]
    assert sol_4["recommended_deployment"]["runner_ups"] == sol_1["recommended_deployment"]["runner_ups"]
    assert sol_4["recommended_deployment"]["search_coverage"] == sol_1["recommended_deployment"]["search_coverage"]
    assert multiprocessing.active_children() == []

    opt_1 = solve_optimal_deployment(
        "qwen3-235b-a22b", "a5",
        num_devices=_TEST_NUM_DEVICES,
        prompt_tokens=_TEST_PROMPT_TOKENS,
        output_tokens=_TEST_OUTPUT_TOKENS,
        ttft_sla_ms=_TEST_TTFT_SLA_MS,
        tpot_sla_ms=_TEST_TPOT_SLA_MS,
        deployment_mode="colocated",
        mtp_proposal_tokens=0,
        max_sequences_per_dp=4,
        layers=2,
        granted_workers=1,
    )
    opt_4 = solve_optimal_deployment(
        "qwen3-235b-a22b", "a5",
        num_devices=_TEST_NUM_DEVICES,
        prompt_tokens=_TEST_PROMPT_TOKENS,
        output_tokens=_TEST_OUTPUT_TOKENS,
        ttft_sla_ms=_TEST_TTFT_SLA_MS,
        tpot_sla_ms=_TEST_TPOT_SLA_MS,
        deployment_mode="colocated",
        mtp_proposal_tokens=0,
        max_sequences_per_dp=4,
        layers=2,
        granted_workers=4,
    )
    assert opt_4["best_configuration"] == opt_1["best_configuration"]
    assert opt_4["candidates"] == opt_1["candidates"]
    assert opt_4["search_coverage"] == opt_1["search_coverage"]
    assert multiprocessing.active_children() == []


def test_pd_disaggregated_stage_factorization_and_deterministic_parity() -> None:
    common_kw = dict(
        num_devices=_TEST_NUM_DEVICES,
        prompt_tokens=_TEST_PROMPT_TOKENS,
        output_tokens=_TEST_OUTPUT_TOKENS,
        ttft_slo_ms=_TEST_TTFT_SLA_MS,
        tpot_slo_ms=_TEST_TPOT_SLA_MS,
        deployment_mode="pd_disaggregated",
        mtp_tokens=0,
        max_sequences_per_dp=4,
        layers=2,
        mode="simulation",
    )
    sol_1 = solve_deployment("qwen3-235b-a22b", "a5", granted_workers=1, **common_kw)
    sol_4 = solve_deployment("qwen3-235b-a22b", "a5", granted_workers=4, **common_kw)

    assert sol_1["feasible"] is True
    assert sol_4 == sol_1
    cov = sol_4["recommended_deployment"]["search_coverage"]
    assert cov["stage_p_evaluated"] >= 1
    assert cov["stage_d_evaluated"] >= 1
    assert cov["stage_p_evaluated"] + cov["stage_d_evaluated"] < cov["exact_evaluated"]
    assert multiprocessing.active_children() == []


def test_closed_form_pd_join_matches_exhaustive_ratio_grid() -> None:
    scenarios = (
        (16, 2, 4, 2.5, 0.9, 10.0, None, 256, "layerwise_overlap_d2d"),
        (32, 4, 4, 3.1, 1.4, 8.0, 5.0, 256, "pooled_dram_ssd"),
        (32, 2, 8, 1.7, 2.2, 4.0, None, 512, "direct_blocking_d2d"),
        (64, 4, 8, 5.5, 1.3, 6.2, 3.5, 128, "pooled_dram_ssd"),
        (16, 4, 2, 1.8, 0.0, 4.0, None, 1, "layerwise_overlap_d2d"),
    )
    for total, p_dies, d_dies, rps_p, rps_d, p_kv, d_kv, output, mode in scenarios:
        st = {
            "infeasible": None,
            "rps_p": rps_p,
            "rps_d": rps_d,
            "raw_rps_d": rps_d,
            "p_kv_rps_per_inst": p_kv,
            "d_kv_rps_per_inst": d_kv,
            "decode_tps_per_die": 100.0,
            "decode_step": 10.0,
            "effective_tpot": 10.0,
            "first_itl": 12.0,
            "decode_budget": 15.0,
        }
        closed = _solve_optimal_pd_ratio_closed_form(
            st, total=total, p_dies=p_dies, d_dies=d_dies, output=output, mode=mode,
        )
        rows = _build_pd_ratio_rows(
            st, total=total, p_dies=p_dies, d_dies=d_dies, output=output, mode=mode,
        )
        front = _mark_pd_pareto(rows)
        brute = max(
            front,
            key=lambda r: (float(r["e2e_tps_per_die"]), float(r["sustainable_tps"]), -int(r["used_dies"])),
            default=None,
        )
        assert closed is not None and brute is not None
        assert closed["n_prefill"] * p_dies + closed["n_decode"] * d_dies <= total
        assert math.isclose(
            float(closed["e2e_tps_per_die"]), float(brute["e2e_tps_per_die"]), abs_tol=_CLOSED_FORM_TOL,
        )


def _compose_4layer_export(model_id: str, hardware: str, *, track: str) -> dict:
    raw, _ = compose_export_case(
        model_id, hardware, dp=8, tp=1, ep=8, pp=1,
        prompt_tokens=_TEST_PROMPT_TOKENS, output_tokens=_TEST_OUTPUT_TOKENS,
        concurrency=8, feature_track=track,
    )
    model_cfg = dict(raw["model"])
    model_cfg["layers"] = _TEST_LAYERS
    if isinstance(model_cfg.get("layer_plans"), list):
        model_cfg["layer_plans"] = model_cfg["layer_plans"][:_TEST_LAYERS]
    return {**raw, "model": model_cfg}


def test_anchored_four_stage_waterfall_telescopes_with_zero_gap(tmp_path: Path) -> None:
    req = StepRequest(phase="decode", tokens=1, context=_TEST_PROMPT_TOKENS, sequences=8, scope="model")
    raw_before = _compose_4layer_export("qwen3.8-flash-next", "a5", track="shipping")
    raw_after = _compose_4layer_export("qwen3.8-flash-next", "a5", track="target")
    b_before = build_export_bundle(raw_before, request=req, apply_target_envelope=False)
    b_after = build_export_bundle(raw_after, request=req, apply_target_envelope=True)
    assert b_after.engine_step_receipt is not None
    assert len(b_after.stage_waterfall) == 4

    aligned = align_before_after_operators(b_before, b_after)
    plan = build_comparison_plan(b_before, b_after, aligned)
    wf = plan.stage_waterfall
    assert len(wf) == 4
    assert [row["stage_id"] for row in wf] == [
        "before_merged", "stage1_ops_config", "stage2_unmerged_pr", "stage3_target_after",
    ]
    assert int(wf[0]["span_ns"]) == b_before.step_span_ns
    assert int(wf[-1]["span_ns"]) == b_after.step_span_ns
    assert sum(int(row["delta_ns"]) for row in wf) == b_before.step_span_ns - b_after.step_span_ns
    assert plan.waterfall.reconciliation_error_ns == 0

    out_xlsx = tmp_path / "anchored_waterfall.xlsx"
    export_comparison_xlsx(b_before, out_xlsx, after_bundle=b_after)
    cells = _read_xlsx_sheet_cells(out_xlsx, sheet_index=1)
    h14_note = cells.get("H14", "")
    assert "4-Stage:" in h14_note
    assert f"Before {b_before.step_span_ns / 1_000_000.0:.3f}" in h14_note
    assert f"[3]算子/After {b_after.step_span_ns / 1_000_000.0:.3f}" in h14_note


def test_mxfp4_expert_operator_dtype_and_tensor_contract_surface_precision_abi() -> None:
    req = StepRequest(phase="decode", tokens=1, context=_TEST_PROMPT_TOKENS, sequences=8, scope="model")
    for model_id in ("kimi-k3.1", "qwen3.8-flash-next"):
        raw_shipping = _compose_4layer_export(model_id, "a5", track="shipping")
        bun_shipping = build_export_bundle(raw_shipping, request=req, apply_target_envelope=False)
        expert_rows = [
            op for op in bun_shipping.operators
            if op.family in {"grouped_gemm", "grouped_moe"} or "megamoe" in op.recipe.lower()
        ]
        assert expert_rows, f"expected routed expert rows on {model_id}"
        assert all(op.dtype == "mxfp4" for op in expert_rows)
        assert all("MXFP4" in op.tensor_contract for op in expert_rows)

        raw_target = _compose_4layer_export(model_id, "a5", track="target")
        bun_target = build_export_bundle(raw_target, request=req, apply_target_envelope=True)
        target_experts = [
            op for op in bun_target.operators
            if op.family in {"grouped_gemm", "grouped_moe"} or "megamoe" in op.recipe.lower()
        ]
        assert target_experts, f"expected routed expert rows on {model_id} target"
        assert all(op.dtype == "mxfp4" for op in target_experts)
        assert all("MXFP4" in op.tensor_contract for op in target_experts)

        norm_rows = [op for op in bun_shipping.operators if op.family in {"rmsnorm", "add_rmsnorm"}]
        assert norm_rows and all(op.dtype != "mxfp4" for op in norm_rows)

    # Control checks: MXFP8 and W4A8 experts preserve their native display dtypes.
    raw_mxfp8 = _compose_4layer_export("glm-5.3-flash", "a5", track="shipping")
    bun_mxfp8 = build_export_bundle(raw_mxfp8, request=req, apply_target_envelope=False)
    mxfp8_experts = [op for op in bun_mxfp8.operators if op.family in {"grouped_gemm", "grouped_moe"}]
    assert mxfp8_experts and all(op.dtype == "fp8_e4m3" for op in mxfp8_experts)

    raw_w4a8 = _compose_4layer_export("deepseek-v4.1-flash", "a3", track="shipping")
    bun_w4a8 = build_export_bundle(raw_w4a8, request=req, apply_target_envelope=False)
    w4a8_experts = [op for op in bun_w4a8.operators if op.family in {"grouped_gemm", "grouped_moe"}]
    assert w4a8_experts and all(op.dtype == "int8" for op in w4a8_experts)


def test_oracle_refined_layer_cohort_plan_matches_literal_execution() -> None:
    case = compose_case(
        model="qwen3-32b", hardware="a3", tp=8, dp=1, pp=1, ep=1,
        quant="w8a8", layers=_COHORT_TEST_LAYERS,
        prompt_tokens=_TEST_PROMPT_TOKENS, output_tokens=_TEST_OUTPUT_TOKENS, concurrency=16,
    )
    step = StepRequest(phase="decode", tokens=16, context=_TEST_PROMPT_TOKENS, sequences=16)
    dossier = prepare_scenario(ScenarioRequest(case=case, step=step))
    mainline_seed = dossier.handle._mainline
    m_case = mainline_seed.scenario.atomic_step.case
    m_graphs = {entry.replica: _replay_ops(entry.ops) for entry in mainline_seed.scenario.atomic_step.graph.executable}

    refined = refine_layer_cohort_plan(m_case, m_graphs[0])
    cohort_shapes = [(c.representative_layer, c.layer_indices) for c in refined.cohorts]
    assert cohort_shapes == [
        (0, (0,)),
        (1, tuple(range(1, _COHORT_TEST_LAYERS - 1))),
        (_COHORT_TEST_LAYERS - 1, (_COHORT_TEST_LAYERS - 1,)),
    ]

    lit_report = graph_costs(
        m_case, m_graphs, recipes=mainline_seed.book, external_inputs={"analysis_input": 0.0},
        evaluation=dossier.handle.evaluation_context, isolated=False,
        include_submission=False, trace_limit=0,
    )
    assert math.isclose(
        mainline_seed.evaluation.span_ms, float(lit_report["duration_ms"]),
        rel_tol=_PARITY_REL_TOL, abs_tol=_PARITY_ABS_TOL_MS,
    )
    coh_serviced = mainline_seed.evaluation.execution_evidence["serviced"]
    lit_serviced = lit_report["execution"]["serviced"]
    assert set(coh_serviced) == set(lit_serviced)
    for res_name, expected in lit_serviced.items():
        assert math.isclose(
            float(coh_serviced[res_name]), float(expected),
            rel_tol=_PARITY_REL_TOL, abs_tol=_PARITY_ABS_TOL_MS,
        )


def test_oracle_dp_catalog_memo_preserves_per_replica_die_isolation_and_speedup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = compose_case(
        model="qwen3-235b-a22b", hardware="a3", tp=1, dp=_DP_TEST_REPLICAS, pp=1, ep=_DP_TEST_REPLICAS,
        layers=2, prompt_tokens=_TEST_PROMPT_TOKENS, output_tokens=_TEST_OUTPUT_TOKENS,
        concurrency=_DP_TEST_REPLICAS,
    )
    step = StepRequest(phase="decode", tokens=_DP_TEST_REPLICAS, context=_TEST_PROMPT_TOKENS, sequences=_DP_TEST_REPLICAS)
    dossier = prepare_scenario(ScenarioRequest(case=case, step=step))
    scenario = dossier.handle.scenario
    ops0 = scenario.atomic_ops[0].ops
    datapath = scenario.datapath
    proposals = list(_catlass_proposals(ops0[:_CATLASS_PROPOSAL_SLICE], datapath, datapath.aic_count, datapath.aiv_count))
    assert proposals
    op_edit = proposals[0][0][0]
    plan = replace(
        empty_plan(scenario.base_graph_sha256, scenario.framing_sha256),
        edits=(Edit("e0", operation=op_edit),),
    )
    applied = apply_plan(plan, scenario)

    lower_calls: list[int] = []
    orig_lower = template_registry.lower_catlass_spec

    def _counting_lower(*args, **kwargs):
        lower_calls.append(int(kwargs.get("die", 0)))
        return orig_lower(*args, **kwargs)

    catlass_lower._PURE_LOWER_CACHE.clear()
    monkeypatch.setattr(template_registry, "lower_catlass_spec", _counting_lower)
    graphs, book, summaries, bound_streams, _, _ = _bind_replay(
        scenario, applied, dossier.handle.recipe_book, dossier.handle,
    )
    assert len(lower_calls) == 1

    recipe_ids = {
        replica: next(op.recipe for op in ops if op.op_id == op_edit.target.id)
        for replica, ops in graphs.items()
    }
    assert len(set(recipe_ids.values())) == 1

    summary_sites = sorted(
        (s.semantic_work[0].replica, s.semantic_work[0].die, s.read_obligations[0].die)
        for s in summaries
    )
    assert summary_sites == [(r, r, r) for r in range(_DP_TEST_REPLICAS)]

    monkeypatch.setenv("MINDIE_SIM_DP_DEDUP", "1")
    t0 = time.perf_counter()
    rep_dedup = _report(
        scenario.atomic_step.case, graphs, dossier.handle.evaluation_context, book,
        dossier.handle._replay, streams=bound_streams,
    )
    dedup_s = time.perf_counter() - t0

    monkeypatch.setenv("MINDIE_SIM_DP_DEDUP", "0")
    t1 = time.perf_counter()
    rep_full = _report(
        scenario.atomic_step.case, graphs, dossier.handle.evaluation_context, book,
        dossier.handle._replay, streams=bound_streams,
    )
    full_s = time.perf_counter() - t1
    monkeypatch.setenv("MINDIE_SIM_DP_DEDUP", "1")

    assert math.isclose(
        float(rep_dedup["duration_ms"]), float(rep_full["duration_ms"]),
        rel_tol=_PARITY_REL_TOL, abs_tol=_PARITY_ABS_TOL_MS,
    )
    assert full_s / max(dedup_s, _PARITY_ABS_TOL_MS) >= _DP_MIN_SPEEDUP

    full_serviced = rep_full["execution"]["serviced"]
    hbm_read_rows = sorted(k for k in full_serviced if k.endswith(":r") and "memory@r" in k)
    assert len(hbm_read_rows) == _DP_TEST_REPLICAS
    full_unit_peaks = rep_full["execution"]["peak_allocated_per_unit"]
    for replica in range(_DP_TEST_REPLICAS):
        assert any(f"@r{replica}:s0:" in key and val > 0 for key, val in full_unit_peaks.items())

    for field_name in ("serviced", "resource_busy_ms"):
        dedup_map = rep_dedup["execution"][field_name]
        full_map = rep_full["execution"][field_name]
        assert set(dedup_map) == set(full_map)
        for key, expected in full_map.items():
            assert math.isclose(
                float(dedup_map[key]), float(expected),
                rel_tol=_PARITY_REL_TOL, abs_tol=_PARITY_ABS_TOL_MS,
            )

    same_backend = rep_dedup.get("engine_backend") == rep_full.get("engine_backend")
    for field_name in ("peak_allocated", "peak_allocated_per_unit"):
        dedup_map = rep_dedup["execution"][field_name]
        full_map = rep_full["execution"][field_name]
        assert set(dedup_map) == set(full_map)
        for key, expected in full_map.items():
            if same_backend:
                assert math.isclose(
                    float(dedup_map[key]), float(expected),
                    rel_tol=_PARITY_REL_TOL, abs_tol=_PARITY_ABS_TOL_MS,
                )
            elif expected > 0:
                assert float(dedup_map[key]) > 0.0
            if "@r0:" in key:
                for peer in range(1, _DP_TEST_REPLICAS):
                    peer_key = key.replace("@r0:", f"@r{peer}:", 1)
                    assert math.isclose(
                        float(dedup_map[peer_key]), float(dedup_map[key]),
                        rel_tol=_PARITY_REL_TOL, abs_tol=_PARITY_ABS_TOL_MS,
                    )
                    assert math.isclose(
                        float(full_map[peer_key]), float(full_map[key]),
                        rel_tol=_PARITY_REL_TOL, abs_tol=_PARITY_ABS_TOL_MS,
                    )

    monkeypatch.setattr(template_registry, "lower_catlass_spec", orig_lower)
    full_eval = finalize_plan(dossier.handle, plan)
    assert full_eval.verification.passed
    assert math.isclose(
        float(full_eval.step_eval.span_ms), float(rep_full["duration_ms"]),
        rel_tol=_PARITY_REL_TOL, abs_tol=_PARITY_ABS_TOL_MS,
    )


def test_oracle_multi_replica_symmetric_stream_cohort_matches_literal() -> None:
    dp_pair = 2
    case = compose_case(
        model="qwen3-235b-a22b", hardware="a3", tp=1, dp=dp_pair, pp=1, ep=dp_pair,
        layers=_COHORT_TEST_LAYERS, prompt_tokens=_TEST_PROMPT_TOKENS,
        output_tokens=_TEST_OUTPUT_TOKENS, concurrency=dp_pair,
    )
    step = StepRequest(phase="decode", tokens=dp_pair, context=_TEST_PROMPT_TOKENS, sequences=dp_pair)
    dossier = prepare_scenario(ScenarioRequest(case=case, step=step))
    scenario = dossier.handle.scenario
    ops0 = scenario.atomic_ops[0].ops
    by_id = {op.op_id: op for op in ops0}
    target = next(
        op for op in ops0
        if op.layer == _COHORT_TEST_LAYERS - 1 and op.vector_ops > 0 and op.flops == 0 and not op.link_bytes
    )
    events = [
        StreamEvent(f"ev_in_{idx}", MAIN_STREAM, dep, "aux", (target.op_id,))
        for idx, dep in enumerate(target.deps) if dep in by_id and by_id[dep].stage == target.stage
    ]
    consumers = tuple(op.op_id for op in ops0 if target.op_id in op.deps and op.stage == target.stage)
    if consumers:
        events.append(StreamEvent("ev_out", "aux", target.op_id, MAIN_STREAM, consumers))
    sched = StreamSchedule(
        lanes=(
            StreamLane(MAIN_STREAM, StreamCoreBudget(isolation="shared_cap")),
            StreamLane("aux", StreamCoreBudget(aic=1, aiv=2, isolation="shared_cap")),
        ),
        assignments=((OpRef("atom", target.op_id), "aux"),),
        events=tuple(events),
    )
    plan = replace(empty_plan(scenario.base_graph_sha256, scenario.framing_sha256), edits=(Edit("e_stream", operation=sched),))
    applied = apply_plan(plan, scenario)
    graphs, book, _, bound_streams, _, _ = _bind_replay(scenario, applied, dossier.handle.recipe_book, dossier.handle)
    replay_graphs = {r: _replay_ops(ops) for r, ops in graphs.items()}

    coh = _report(
        scenario.atomic_step.case, graphs, dossier.handle.evaluation_context, book,
        dossier.handle._replay, trace_limit=MAX_OVERLAP_TRACE_EVENTS, streams=bound_streams,
    )
    lit = graph_costs(
        scenario.atomic_step.case, replay_graphs, recipes=book, external_inputs={"analysis_input": 0.0},
        evaluation=dossier.handle.evaluation_context, isolated=False,
        include_submission=False, trace_limit=MAX_OVERLAP_TRACE_EVENTS, stream_plans=bound_streams,
    )
    assert math.isclose(float(coh["duration_ms"]), float(lit["duration_ms"]), rel_tol=_PARITY_REL_TOL, abs_tol=_PARITY_ABS_TOL_MS)
    assert set(coh["execution"]["serviced"]) == set(lit["execution"]["serviced"])
    for key, expected in lit["execution"]["serviced"].items():
        assert math.isclose(float(coh["execution"]["serviced"][key]), float(expected), rel_tol=_PARITY_REL_TOL, abs_tol=_PARITY_ABS_TOL_MS)
    coh_eval = _evaluation(scenario, graphs, coh, applied.plan_sha256, "should_achieve")
    for rep in range(dp_pair):
        assert coh_eval.service_intervals.get((rep, target.op_id))
        assert any(rep == r and oid.startswith("s0.l5.") for r, oid in coh_eval.service_intervals)
    assert all(row.start_ms + _PARITY_ABS_TOL_MS >= row.ready_ms for row in coh_eval.operators)
