"""Source helper inheritance preserves explicit software and physical work."""
from copy import deepcopy

import pytest

from mindie_sim import graph_costs, operator_costs, predict, target_graphs, target_operators
from mindie_sim.prediction.body_defaults import attention_defaults, consume_target_declaration
from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure import export_structure
from test_body_gdn_defaults import case


GEOMETRY = dict(scope="body", phase="decode", tokens=4, sequences=2, context=128)
CALLER = "vllm_ascend_gdn_decode"


def source_case(**options):
    raw = case("Ascend950DT_9582")
    raw["runtime"]["kernels"] = {"lowerings": {"gdn_decode": {
        "implementation": CALLER, "options": options}}}
    return raw


def gating(raw, ops=None):
    ops = target_operators(raw, **GEOMETRY) if ops is None else ops
    return next(row for row in operator_costs(raw, ops)["operators"]
                if row["operator"]["recipe"] == "ascend_gdn_decode_gating")


def test_partial_source_binding_executes_same_store_and_survives_graph_roundtrip():
    raw = source_case(recurrent_implementation="arch35_fp32")
    before = deepcopy(raw)
    baseline = predict(case("Ascend950DT_9582"), **GEOMETRY)
    body = predict(raw, **GEOMETRY)
    assert body["body_forward_ms"] == baseline["body_forward_ms"]
    assert body["timeline"] == baseline["timeline"]
    assert body["execution"]["serviced"] == baseline["execution"]["serviced"]
    graphs = deepcopy(target_graphs(raw, **GEOMETRY))
    resolved = consume_target_declaration(parse_case(raw), graphs)
    binding = resolved.runtime.kernels["lowerings"]["gdn_decode"]
    assert binding == graphs.effective_kernels["lowerings"]["gdn_decode"]
    options = binding["options"]
    gated = next(op for op in graphs[0] if op.recipe == "ascend_gdn_decode_gating")
    assert dict(gated.metadata)["beta_store"] == {
        key: value for key, value in options["gating_beta_store"].items()
        if key != "bytes_per_element"}
    exported = export_structure(raw, **GEOMETRY)
    assert exported["execution"]["software"]["kernels"]["lowerings"]["gdn_decode"] == binding
    graph = graph_costs(raw, graphs, external_inputs={"analysis_input": 0}, isolated=False)
    assert graph["duration_ms"] == body["body_forward_ms"]
    assert graph["execution"]["serviced"] == body["execution"]["serviced"]
    assert raw == before


def test_empty_store_is_dma_opt_out_and_custom_queue_ignores_hardware_response():
    implicit = source_case(recurrent_implementation="analytical")
    dma = source_case(recurrent_implementation="analytical", gating_beta_store={})
    queue = {"resource": "scalar_store_issue", "queue": {
        "outstanding": 1, "granule_bytes": 128, "response_ms": .001}}
    custom = source_case(recurrent_implementation="analytical", gating_beta_store=queue)
    inherited, plain, explicit = gating(implicit), gating(dma), gating(custom)
    assert inherited["duration_ms"] > plain["duration_ms"]
    assert explicit["duration_ms"] > inherited["duration_ms"]
    scalar = "scalar_store_issue@r0:s0"
    assert scalar not in plain["serviced"]
    assert explicit["serviced"][scalar] == inherited["serviced"][scalar] == 48
    for resource in ("memory@r0:s0", "vector@r0:s0"):
        assert explicit["serviced"][resource] == inherited["serviced"][resource] == plain["serviced"][resource]
    ops = target_operators(custom, **GEOMETRY)
    assert ops.effective_kernels["lowerings"]["gdn_decode"]["options"]["gating_beta_store"] == queue
    changed = deepcopy(custom)
    changed["machine"]["device"] = {"store_queues": {"scalar_store_issue": {"response_ms": .01}}}
    assert gating(changed, ops)["duration_ms"] == explicit["duration_ms"]
    assert dict(next(op for op in ops if op.family == "recurrent_vector").metadata)[
        "recurrence_selection"] == "analytical"
    family = source_case(recurrent_implementation="analytical")
    family["runtime"]["kernels"]["families"] = {"gdn_gating": {
        "kind": "tiled_pointwise", "input_bytes_per_element": 4,
        "outputs": [{"bytes_per_element": 4}, {"bytes_per_element": 2, **queue}]}}
    assert gating(family)["duration_ms"] == explicit["duration_ms"]


def test_inherited_hardware_reference_reprices_existing_explicit_source_graph():
    raw = source_case(recurrent_implementation="arch35_fp32")
    ops = target_operators(raw, **GEOMETRY)
    changed = deepcopy(raw)
    changed["machine"]["device"] = {"store_queues": {"scalar_store_issue": {"response_ms": .001}}}
    before, reused, fresh = gating(raw, ops), gating(changed, ops), gating(changed)
    assert reused["duration_ms"] == fresh["duration_ms"] > before["duration_ms"]
    assert reused["serviced"] == pytest.approx(before["serviced"], rel=1e-12)


def test_hardware_without_store_queue_does_not_gain_one_from_caller_selection():
    raw = source_case(recurrent_implementation="analytical")
    raw["machine"] = parse_case(raw).machine.to_dict()
    for device in raw["machine"]["devices"]:
        device["store_queues"] = {}
    row = gating(raw)
    assert "scalar_store_issue@r0:s0" not in row["serviced"]
    assert "gating_beta_store" not in target_operators(raw, **GEOMETRY).effective_kernels[
        "lowerings"]["gdn_decode"]["options"]
    a3 = source_case(recurrent_implementation="analytical")
    a3["machine"] = case("Ascend910_9362")["machine"]
    assert "scalar_store_issue@r0:s0" not in gating(a3)["serviced"]
    fp16 = source_case(recurrent_implementation="analytical")
    fp16["model"]["activation_dtype"] = "fp16"
    selected, _ = attention_defaults(parse_case(fp16), "decode", tokens=4)
    assert "gating_beta_store" not in selected.runtime.kernels["lowerings"]["gdn_decode"]["options"]


def test_policy_analytic_opt_out_and_source_leaf_keep_independent_semantics():
    raw = source_case(recurrent_implementation="analytical")
    binding = raw["runtime"]["kernels"]["lowerings"]["gdn_decode"]
    raw["runtime"]["kernels"]["lowerings"]["gdn_decode"] = {
        "cases": [{"domain": {"max_query_tokens": {"max": 1}}, "binding": binding}],
        "otherwise": "analytical"}
    analytic = target_operators(raw, **GEOMETRY)
    assert not any(op.recipe == "ascend_gdn_decode_gating" for op in analytic)
    native = target_operators(raw, **{**GEOMETRY, "tokens": 1})
    assert next(dict(op.metadata)["beta_store"] for op in native
                if op.recipe == "ascend_gdn_decode_gating")["from_hardware"]
    selected = consume_target_declaration(parse_case(raw), native)
    assert selected.runtime.kernels == native.effective_kernels
    source = next(op for op in native if op.family == "recurrent_vector")
    assert dict(source.metadata)["recurrence_selection"] == "analytical"
