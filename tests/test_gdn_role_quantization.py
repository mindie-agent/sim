"""GDN projection roles share representation across frontend, Engine and residency.

[Role] Verify explicit mixed-precision roles against real Qwen catalog geometry.
[Invariants] Role overrides alter representation, not matrix FLOPs or model
  structure. Parent exclusions survive; no observed latency enters predictions.
[Timeouts & Locks] One real layer and finite local Engine execution; no locks.
[Side-Effect Policy] Read a retained case; no subprocess, remote or device work.
"""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, operator_costs, target_graphs
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.structure import parse_case


ROLES = {
    "linear_qkvz": ("gdn.qkvz", "gdn_qkvz"),
    "linear_beta_alpha": ("gdn.beta_alpha", "gdn_ba"),
    "linear_output": ("gdn.o", "gdn_o"),
}
BF16 = {"weight_format": "bf16", "activation_format": "bf16"}


def case_for(*, quantization="w8a8", overrides=None, source_prefill=False):
    path = Path(__file__).parents[1] / "examples/cases/qwen36-a3-tp4.json"
    raw = json.loads(path.read_text())
    # Preserve all first-layer dimensions; layer count isolates a component,
    # not a claim about full-model latency or memory capacity.
    raw["model"].update(layers=1, quantization=quantization)
    raw["runtime"]["kernels"] = {"programs": {"matmul": {"kind": "steady_matmul"}}}
    if quantization == "w8a8_mxfp8" or source_prefill:
        raw["machine"] = {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 4}
    if source_prefill:
        raw["runtime"]["kernels"]["lowerings"] = {"gdn_prefill": "gdn_prefill_arch35"}
    case = parse_case(raw)
    if overrides:
        raw["model"]["quant"] = replace(case.model.quant, category_overrides=overrides).to_dict()
        case = parse_case(raw)
    return case


def projections(case, phase):
    graphs = target_graphs(case, phase=phase, tokens=4, context=4096, scope="body")
    ops = graphs[0]
    selected = {op.recipe: op for op in ops if op.recipe in ROLES}
    assert len(selected) == 3
    return graphs, selected


def check_formats_and_producers(case, graphs, selected, expected):
    by_id = {op.op_id: op for op in graphs[0]}
    terms = {term.name: term for term in layer_weight_terms(case.model.layer_plans[0], case.model)}
    for recipe, op in selected.items():
        category, term = ROLES[recipe]
        metadata = dict(op.metadata)
        rows, n, k = op.local_shape
        fmt = expected[recipe]
        assert metadata["quant_category"] == category
        assert metadata["weight_format"] == metadata["activation_format"] == op.dtype == fmt
        assert metadata["packed_weight_bytes"] == term_bytes(
            terms[term], case.model.quant, case.runtime.parallel)
        assert op.flops == 2 * rows * n * k
        if fmt == "bf16":
            assert metadata["packed_weight_bytes"] == 2 * n * k
            assert not metadata.get("activation_scale_bytes")
            assert metadata.get("activation_producer") != "explicit_quantization"
            assert all(by_id[dep].family not in {"quantize", "rmsnorm_quant"}
                       for dep in op.deps if dep in by_id)
        else:
            producer = by_id[op.deps[0]]
            assert producer.family in {"quantize", "rmsnorm_quant"}
            assert metadata["activation_producer"] == "explicit_quantization"
            assert metadata["activation_scale_bytes"] == 4 * rows
            assert producer.mte3_bytes == rows * k + 4 * rows
    priced = operator_costs(case, tuple(selected.values()), input_ready_ms=3)
    peaks = dict(case.machine.dies[0].compute_peaks)
    for row in priced["operators"]:
        op = next(op for op in selected.values() if op.recipe == row["operator"]["recipe"])
        matrix = sum(value for key, value in row["declared_service"].items() if key.startswith("matrix@"))
        assert matrix == pytest.approx(1000 * op.flops / peaks[op.dtype])
        assert not row["service_before_input"] and not row["live_tasks"]


@pytest.mark.parametrize("phase,source", [("decode", False), ("prefill", False), ("prefill", True)])
def test_explicit_native_gate_and_output_keep_quantized_qkv_producer(phase, source):
    case = case_for(overrides={"gdn.beta_alpha": BF16, "gdn.o": BF16}, source_prefill=source)
    graphs, selected = projections(case, phase)
    # Exact TP4 shapes established by the retained Qwen3.6 q4 caller capture.
    assert {name: op.local_shape for name, op in selected.items()} == {
        "linear_qkvz": (4, 4096, 5120), "linear_beta_alpha": (4, 24, 5120),
        "linear_output": (4, 5120, 1536)}
    check_formats_and_producers(case, graphs, selected, {
        "linear_qkvz": "int8", "linear_beta_alpha": "bf16", "linear_output": "bf16"})
    result = graph_costs(case, graphs, external_inputs={"analysis_input": 0}, isolated=False)
    assert result["duration_ms"] > 0
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0


def test_parent_gdn_override_and_more_specific_child_follow_existing_precedence():
    case = case_for(overrides={"gdn": BF16,
                              "gdn.qkvz": {"weight_format": "int8", "activation_format": "int8"}})
    graphs, selected = projections(case, "decode")
    check_formats_and_producers(case, graphs, selected, {
        "linear_qkvz": "int8", "linear_beta_alpha": "bf16", "linear_output": "bf16"})


@pytest.mark.parametrize("phase,source", [("decode", False), ("prefill", False), ("prefill", True)])
def test_qwen38_mxfp8_parent_exclusion_still_keeps_every_gdn_projection_native(phase, source):
    case = case_for(quantization="w8a8_mxfp8", source_prefill=source)
    # Qwen3.8 uses the same GDN geometry; exercise the actual Qwen3.8 catalog,
    # preserving the tested hardware and source selection.
    raw = case.to_dict()
    raw["model"] = {"preset": "qwen3.8-27b", "layers": 1, "quantization": "w8a8_mxfp8"}
    case = parse_case(raw)
    graphs, selected = projections(case, phase)
    check_formats_and_producers(case, graphs, selected, {name: "bf16" for name in ROLES})


@pytest.mark.parametrize("quantization", ("w8a8", "w8a8_mxfp8"))
def test_undeclared_roles_retain_the_old_parent_category_physical_work(quantization):
    case = case_for(quantization=quantization)
    _, selected = projections(case, "decode")
    for op in selected.values():
        m, n, k = op.local_shape
        old = gemm_op(op_id=op.op_id, deps=(), stage=op.stage, layer=op.layer,
                      requests=op.requests, m=m, n=n, k=k, quant=case.model.quant,
                      category="gdn", phase=op.phase, recipe=op.recipe, family=op.family)
        for field in ("dtype", "flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes"):
            assert getattr(op, field) == getattr(old, field)
        metadata, previous = dict(op.metadata), dict(old.metadata)
        for field in ("packed_weight_bytes", "weight_format", "activation_format",
                      "activation_scale_bytes", "matrix_epilogue"):
            assert metadata.get(field) == previous.get(field)
