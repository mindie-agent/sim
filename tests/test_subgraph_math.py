"""Fusion-invariant subgraph math: conservation, completeness and evaluation.

[Role]
  Establish the algebraic invariants of ``mindie_sim.subgraph_math`` against a
  synthetic resource ledger and against real declared exports.

[Invariants]
  * A selected native fusion is charged exactly once; a partially selected
    fusion has unknown realized work instead of invented attribution.
  * Unavailable algebraic/boundary ledgers stay ``None`` with explicit reasons;
    physical work is never renamed into an algebraic counter.
  * Boundary byte unions count overlapping intervals once and require a
    complete cold-HBM ledger.
  * Efficiency evaluation preserves a valid observed zero and scales complete
    invocations; mixed dtypes use their own peaks.
  * Structural math is attached before the content digest, so the exported
    ``graph_id`` re-hashes exactly.

[Timeouts & Locks]
  Bounded synthetic records and at most two four-layer exports; no waits.

[Side-Effect Policy]
  In-memory records only; no filesystem writes.
"""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from mindie_sim.compose import compose_case
from mindie_sim.structure import export_structure
from mindie_sim.subgraph_math import (
    ALGEBRAIC_WORK_UNAVAILABLE,
    BOUNDARY_LEDGER_UNAVAILABLE,
    SEMANTIC_FAMILIES,
    SubgraphMathSummary,
    attach_subgraph_math,
    boundary_bytes,
    build_subgraph_math_rollup,
    effective_physical_work,
    evaluate_subgraph_efficiency,
    exclusive_family_members,
    semantic_family_of_group,
)

WORK = {"hbm_bytes": 0.0, "flops": 0.0, "vector_ops": 0.0, "link_bytes": 0.0}


def _work(hbm=0.0, flops=0.0, vector=0.0, link=0.0):
    return {"hbm_bytes": hbm, "flops": flops, "vector_ops": vector, "link_bytes": link}


def _node(node_id, work, boundary=None, metadata=None):
    return {"node_id": node_id, "physical_work": work,
            "resource_boundary": boundary, "operator": {"metadata": metadata or {}}}


def _fused_boundary(resource_id):
    return {"kind": "internal_to_selected_native_fusion", "operator_ids": [resource_id]}


def _report(fused: bool):
    nodes = [
        _node("a", _work(hbm=10.0, flops=100.0)),
        _node("b", _work(hbm=20.0, flops=200.0)),
    ]
    resources = []
    if fused:
        nodes.extend([
            _node("c", _work(hbm=0.0), _fused_boundary("res:0")),
            _node("d", _work(hbm=0.0), _fused_boundary("res:0")),
        ])
        resources.append(_node("res:0", _work(hbm=40.0, flops=400.0, link=8.0)))
    else:
        nodes.extend([
            _node("c", _work(hbm=30.0, flops=300.0)),
            _node("d", _work(hbm=10.0, flops=100.0)),
        ])
    groups = [
        {"subgraph_id": "sg:ffn", "node_ids": ["a", "b"], "role": "router_selection",
         "family": "moe", "phase": "decode", "replica": 0, "stage": 0, "structure": {}},
        {"subgraph_id": "sg:attn", "node_ids": ["c", "d"], "role": "mla_prolog",
         "family": "mla", "phase": "decode", "replica": 0, "stage": 0, "structure": {}},
    ]
    return {"nodes": nodes, "resource_operators": resources, "subgraphs": groups,
            "model": {"quant": {"kv_dtype": "bf16", "weight_format": "int8",
                                "activation_format": "int8"}},
            "execution": {"phase": "decode", "software": {"parallel": {"tp": 8}}}}


def test_effective_physical_work_charges_fusion_once_and_detects_partial():
    unfused, fused = _report(fused=False), _report(fused=True)
    first, reasons = effective_physical_work(unfused, ["a", "b"])
    assert reasons == ()
    assert first == _work(hbm=30.0, flops=300.0)
    # The fused region's semantic members carry no second charge.
    second, reasons = effective_physical_work(fused, ["c", "d"])
    assert reasons == ()
    assert second == _work(hbm=40.0, flops=400.0, link=8.0)
    # Selecting only part of a fused resource cannot invent proportional work.
    partial, why = effective_physical_work(fused, ["c"])
    assert partial is None
    assert why == ("partial_selected_fusion:res:0",)
    with pytest.raises(ValueError, match="unknown_subgraph_member"):
        effective_physical_work(fused, ["missing"])


def test_effective_physical_work_reports_incomplete_ledgers():
    broken = _report(fused=False)
    broken["nodes"][0]["physical_work"] = {**WORK, "hbm_bytes": None}
    totals, reasons = effective_physical_work(broken, ["a"])
    assert totals is None
    assert reasons == ("physical_work_incomplete:a",)
    dangling = _report(fused=True)
    dangling["resource_operators"] = []
    with pytest.raises(ValueError, match="dangling_resource_boundary"):
        effective_physical_work(dangling, ["c", "d"])


def test_algebraic_work_is_fusion_invariant_even_when_unavailable():
    unfused, fused = _report(fused=False), _report(fused=True)
    left = [summary for summary in _summaries(unfused) if summary.subgraph_id == "sg:attn"]
    right = [summary for summary in _summaries(fused) if summary.subgraph_id == "sg:attn"]
    # The same semantic program under a resource-fusion change: algebraic
    # counters are unavailable on both sides, but never differ or become
    # physical counters.
    for summary in (*left, *right):
        assert summary.algebraic_cube_flops is None
        assert summary.algebraic_vector_ops is None
        assert summary.algebraic_cube_flops_by_dtype is None
        assert ALGEBRAIC_WORK_UNAVAILABLE in summary.unavailable_reasons
        assert summary.minimal_boundary_hbm_bytes is None
        assert BOUNDARY_LEDGER_UNAVAILABLE in summary.unavailable_reasons
        assert summary.fusion_hbm_overhead_bytes is None
    assert [s.semantic_family for s in left] == ["attention"]
    assert [s.semantic_family for s in right] == ["attention"]
    assert left[0].realized_hbm_bytes == right[0].realized_hbm_bytes == 40.0


def _summaries(report):
    attach_subgraph_math(report)
    return [SubgraphMathSummary(**{
        key: value for key, value in group["math_summary"].items()
        if key in SubgraphMathSummary.__dataclass_fields__
    }) for group in report["subgraphs"]]


def test_boundary_bytes_unions_overlaps_and_requires_complete_cold_ledger():
    accesses = [
        ("alloc:kv", "read", 0, 100),
        ("alloc:kv", "read", 50, 150),
        ("alloc:weight", "read", 0, 10),
        ("alloc:out", "write", 0, 20),
    ]
    assert boundary_bytes(accesses, complete=True, hbm_cold=True) == 180
    assert boundary_bytes(accesses, complete=False, hbm_cold=True) is None
    assert boundary_bytes(accesses, complete=True, hbm_cold=False) is None
    with pytest.raises(ValueError, match="invalid_boundary_access"):
        boundary_bytes([("alloc", "sideways", 0, 1)], complete=True, hbm_cold=True)
    with pytest.raises(ValueError, match="invalid_boundary_access"):
        boundary_bytes([("alloc", "read", 5, 1)], complete=True, hbm_cold=True)


def test_exclusive_family_members_unions_and_rejects_overlap():
    groups = [
        {"node_ids": ["a", "b"], "role": "router_selection"},
        {"node_ids": ["b", "c"], "role": "grouped_gate_up_activation_quant"},
        {"node_ids": ["d"], "role": "unknown_role", "family": "unknown"},
    ]
    families = exclusive_family_members(groups, semantic_family_of_group)
    assert families["moe_ffn"] == {"a", "b", "c"}
    assert semantic_family_of_group(groups[-1]) is None
    overlap = [
        {"node_ids": ["a"], "role": "router_selection"},
        {"node_ids": ["a"], "role": "mla_prolog"},
    ]
    with pytest.raises(ValueError, match="overlapping_semantic_family_ownership"):
        exclusive_family_members(overlap, semantic_family_of_group)


def _summary(**overrides):
    payload = dict(
        subgraph_id="sg", subgraph_kind="role", semantic_family="attention",
        phase="decode", layer_multiplicity=1, parameterization={},
        algebraic_cube_flops=None, algebraic_cube_flops_by_dtype=None,
        algebraic_vector_ops=None, minimal_boundary_hbm_bytes=None,
        realized_hbm_bytes=None, unavailable_reasons=(),
    )
    payload.update(overrides)
    return SubgraphMathSummary(**payload)


PEAKS = {
    "compute_peaks": {"bf16": 1.0e15, "fp8": 2.0e15},
    "vector_peak_ops_per_s": 1.0e14,
    "hbm_peak_bytes_per_s": 1.0e12,
}


def test_evaluate_subgraph_efficiency_scales_and_uses_per_dtype_peaks():
    summary = _summary(
        algebraic_cube_flops_by_dtype={"bf16": 1.0e12, "fp8": 5.0e11},
        algebraic_vector_ops=2.0e11, minimal_boundary_hbm_bytes=1.0e8,
        realized_hbm_bytes=2.0e8,
    )
    report = evaluate_subgraph_efficiency(
        summary, duration_s=1.0e-3, hardware_peaks=PEAKS, invocations=2)
    assert report["algebraic_cube_mfu"] == pytest.approx(2.0 + 0.5)
    assert report["algebraic_vector_mfu"] == pytest.approx(4.0)
    assert report["realized_mbu"] == pytest.approx(0.4)
    assert report["minimal_boundary_mbu"] == pytest.approx(0.2)
    assert report["fusion_headroom_bytes"] == pytest.approx(2.0e8)
    assert report["byte_accounting_consistent"] is True
    # A mixed dtype without a declared peak stays unavailable rather than
    # borrowing an arbitrary peak.
    unknown = _summary(algebraic_cube_flops_by_dtype={"hif8": 1.0})
    assert evaluate_subgraph_efficiency(
        unknown, duration_s=1.0, hardware_peaks=PEAKS)["algebraic_cube_mfu"] is None


def test_evaluate_subgraph_efficiency_preserves_observed_zero():
    summary = _summary(realized_hbm_bytes=2.0e8, minimal_boundary_hbm_bytes=1.0e8)
    report = evaluate_subgraph_efficiency(
        summary, duration_s=1.0, hardware_peaks=PEAKS, observed_hbm_bytes=0.0)
    assert report["realized_mbu"] == 0.0
    assert report["fusion_headroom_bytes"] == -1.0e8
    assert report["byte_accounting_consistent"] is False
    with pytest.raises(ValueError, match="duration_s_must_be_positive_finite"):
        evaluate_subgraph_efficiency(summary, duration_s=0.0, hardware_peaks=PEAKS)
    with pytest.raises(ValueError, match="complete_invocation_count_required"):
        evaluate_subgraph_efficiency(
            summary, duration_s=1.0, hardware_peaks=PEAKS, invocations=0)
    with pytest.raises(ValueError, match="invalid_work"):
        evaluate_subgraph_efficiency(
            _summary(algebraic_vector_ops=-1.0), duration_s=1.0, hardware_peaks=PEAKS)


def _export(model, hardware, tp, phase, tokens, context, layers, *, fuse=False):
    case = compose_case(
        model, hardware, tp=tp, prompt_tokens=tokens if phase == "prefill" else 8,
        output_tokens=4, layers=layers)
    if fuse:
        case["runtime"]["kernels"].setdefault("lowerings", {})["native_fusion"] = {
            "implementation": "declared_native_fusion",
            "options": {"roles": ["mla_prolog"]},
        }
    return export_structure(
        case, phase=phase, tokens=tokens, context=context, sequences=1, scope="body")


def test_export_attaches_math_before_digest_and_reports_unavailable_ledgers():
    report = _export("qwen3-30b-a3b", "a3", 8, "prefill", 32, 0, 1)
    payload = {key: value for key, value in report.items()
               if key not in {"graph_id", "source", "native_calls",
                              "native_call_coverage"}}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           allow_nan=False)
    assert report["graph_id"] == "sha256:" + sha256(canonical.encode()).hexdigest()
    assert report["subgraph_math_by_family"]["totals_additive"] is False
    assert report["subgraph_math_by_family"]["scope"] == (
        "declared_semantic_subgraph_union")
    assert report["subgraph_math_by_family"]["semantic_families"] == list(
        SEMANTIC_FAMILIES)
    assert all(group["math_summary"] for group in report["subgraphs"])
    by_id = {node["node_id"]: node for node in report["nodes"]}
    assert report["resource_operators"] == []
    for group in report["subgraphs"]:
        summary = group["math_summary"]
        member_sum = sum(by_id[node_id]["physical_work"]["hbm_bytes"]
                         for node_id in group["node_ids"])
        assert summary["realized_hbm_bytes"] == pytest.approx(member_sum)
        assert summary["minimal_boundary_hbm_bytes"] is None
        assert BOUNDARY_LEDGER_UNAVAILABLE in summary["unavailable_reasons"]
    rollup = build_subgraph_math_rollup(report)
    assert set(rollup["families"]) <= set(SEMANTIC_FAMILIES)
    assert rollup["uncovered_node_ids"] == []


def test_selected_fusion_charges_resource_not_members_twice():
    baseline = _export("deepseek-v3.2", "a3", 8, "prefill", 128, 0, 2)
    fused = _export("deepseek-v3.2", "a3", 8, "prefill", 128, 0, 2, fuse=True)
    assert baseline["resource_operators"] == []
    assert fused["resource_operators"]
    target = next(group for group in fused["subgraphs"]
                  if group.get("resource_boundary"))
    summary = target["math_summary"]
    resource = fused["resource_operators"][0]
    assert summary["realized_hbm_bytes"] == pytest.approx(
        resource["physical_work"]["hbm_bytes"])
    assert summary["algebraic_cube_flops"] is None
    assert summary["fusion_hbm_overhead_bytes"] is None
    # Work is conserved by the selected resource boundary; no member is added twice.
    assert target["node_ids"]
    assert all(not any(node_id == resource["node_id"] for node_id in target["node_ids"])
               for group in fused["subgraphs"])
