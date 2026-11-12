"""A2 compressed-branch source gate and shared-service envelope.

The V4.1 source declares 38 layers with a real `gated_ring` compressor and a
long-KV owner, and 2 SWA-only layers with `compress_ratio=0`,
`compressor_kind=none`, and no long-KV owner.  The ordinary build must keep the
window branch for the latter and must not synthesize the compressed branch,
merge, K1 floors or aux fork.  Genuine compressed layers keep their exact jobs,
work and shared-service envelope.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mindie_sim.prediction.piecewise.compiler import (
    CompiledOperator, ImplementationPolicy, PipeSupply,
)
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import ModelPipeline
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, instantiate_model
from mindie_sim.prediction.subgraphs.state import StateView
from mindie_sim.schema.common import ModelingError

CASE = (Path(__file__).parents[1] / "examples/cases/longctx-decode-20261030"
        / "v41-a3-tp4-dp4-ep16.json")


def _case():
    return parse_case(json.loads(CASE.read_text()))


def _compressed_source(layer) -> bool:
    return (int(layer.compress_ratio or 0) > 0
            and str(layer.compressor_kind or "").lower() not in {"", "none"}
            and (0 <= int(layer.kv_owner_layer_id) < layer.index
                 or bool(layer.owns_long_kv)))


def _pipeline(case, flag: bool) -> ModelPipeline:
    features = FeatureConfig.from_runtime(
        replace(case.runtime, multistream_window_compressed=flag), case.model)
    supply = PipeSupply.from_hardware(case.machine, dtype="bf16")
    return ModelPipeline(
        case.model, features, supply,
        ImplementationPolicy(cores=max(1, case.machine.dies[0].compute_units)))


def _a2_attention(instance, layer: int):
    return instance.find("A2", layer=layer, role="attention")


def test_v41_source_gate_and_genuine_service_envelope():
    case = _case()
    features = FeatureConfig.from_runtime(
        replace(case.runtime, multistream_window_compressed=True), case.model)
    instantiated = instantiate_model(case.model, features)
    genuine = [plan.index for plan in case.model.layer_plans if _compressed_source(plan)]
    window_only = [plan.index for plan in case.model.layer_plans if not _compressed_source(plan)]
    assert len(genuine) == 38 and len(window_only) == 2
    assert len(genuine) + len(window_only) == len(case.model.layer_plans)

    for layer in window_only:
        a2 = _a2_attention(instantiated, layer)
        assert a2.variant == "window_only"
        assert tuple(op.name for op in a2.template.operators) == (
            "kv_read", "window_attention", "grouped_output_a", "grouped_output_b",
            "output_gate", "tp_allreduce")
        assert all(op.stream == "main" for op in a2.template.operators)
        assert "compressed_kv" not in {port.name for port in a2.state_in}
        k1 = instantiated.find("K1", layer=layer)
        assert tuple(op.name for op in k1.template.operators) == ("compress",)
        assert int(k1.operator("compress").theta.find("row_bytes", -1)) == 0

    for layer in genuine:
        a2 = _a2_attention(instantiated, layer)
        assert a2.variant == "window_compressed"
        names = tuple(op.name for op in a2.template.operators)
        assert names[:4] == ("kv_read", "window_attention",
                             "compressed_attention", "merge")
        streams = {op.name: op.stream for op in a2.template.operators}
        assert streams["compressed_attention"] == "aux"
        assert streams["merge"] == "main"
        k1 = instantiated.find("K1", layer=layer)
        assert tuple(op.name for op in k1.template.operators) == (
            "compress", "kpool", "tail_store", "state_store")

    geometry = WorkloadGeometry(
        (RequestGeometry(1, 126000, phase="decode", replica=0),), phase="decode")
    off = _pipeline(case, False).forward(geometry, StateView())
    on = _pipeline(case, True).forward(geometry, StateView())
    for layer in genuine:
        delta = float(on.layers[layer].duration_ms - off.layers[layer].duration_ms)
        assert abs(delta + 1 / 200) < 1e-12
        envelope = on.layers[layer].contention
        assert envelope is not None
        assert dict(envelope.branches).keys() == {"main", "aux"}
        assert envelope.shared_pipes
        assert envelope.makespan_ms < envelope.isolated_sum_ms
    for layer in window_only:
        assert abs(float(on.layers[layer].duration_ms
                        - off.layers[layer].duration_ms)) < 1e-12
        assert on.layers[layer].contention is None
        a2 = _a2_attention(instantiated, layer)
        names = tuple(op.name for op in a2.template.operators)
        assert "compressed_attention" not in names and "merge" not in names
        k1 = next(item for item in on.layers[layer].subgraphs if item.kind == "K1")
        assert k1.duration_ms == 0
    assert round(on.total_ms - off.total_ms, 10) == -0.19
    assert round(on.total_ms - off.total_ms, 10) == -len(genuine) * 0.005


def test_v41_zero_k1_does_not_advance_compression_state():
    case = _case()
    pipeline = _pipeline(case, False)
    layers = pipeline.instantiated.layers
    genuine = [plan.index for plan in case.model.layer_plans if _compressed_source(plan)]
    window_only = [plan.index for plan in case.model.layer_plans if not _compressed_source(plan)]
    active = WorkloadGeometry(
        (RequestGeometry(1, 126000, phase="decode", replica=0),), phase="decode")
    active_next = WorkloadGeometry(
        (RequestGeometry(1, 126001, phase="decode", replica=0),), phase="decode")
    owner, consumer = genuine[0], genuine[1]

    state = StateView()
    for layer in window_only:
        state = pipeline._advance_layer_state(
            layers[layer], state, active,
            request_ids=("r0",), commit_tokens=(1,))
        assert all(record_owner != layer for record_owner, _ in state.compression)

    state = pipeline._advance_layer_state(
        layers[owner], state, active, request_ids=("r0",), commit_tokens=(1,))
    owner_record = dict(state.compression)[owner]
    assert owner_record.ratio == 2 and owner_record.rows > 0
    assert owner_record.consumed > 0

    # The next layer consumes the owner's real state; the window-only layers
    # still have no invented cursor.
    state_next = pipeline._advance_layer_state(
        layers[consumer], state, active, request_ids=("r0",), commit_tokens=(1,))
    assert dict(state_next.compression)[owner].ratio == 2
    assert dict(state_next.compression)[owner].rows == owner_record.rows
    assert all(record_owner != layer for layer in window_only
               for record_owner, _ in state_next.compression)

    # Next phase on the same owner keeps the source ratio and real rows.
    state_phase2 = pipeline._advance_layer_state(
        layers[owner], state_next, active_next,
        request_ids=("r0",), commit_tokens=(1,))
    owner_next = dict(state_phase2.compression)[owner]
    assert owner_next.ratio == 2
    assert owner_next.consumed > owner_record.consumed
    assert owner_next.rows >= owner_record.rows
    assert all(record_owner != layer for layer in window_only
               for record_owner, _ in state_phase2.compression)


def test_compressed_attention_without_positive_ratio_is_refused():
    case = _case()
    features = FeatureConfig.from_runtime(case.runtime, case.model)
    instance = instantiate_model(case.model, features).find(
        "A2", layer=2, role="attention")
    supply = PipeSupply.from_hardware(case.machine, dtype="bf16")
    policy = ImplementationPolicy(cores=max(1, case.machine.dies[0].compute_units))
    compiled = CompiledOperator(
        instance.operator("compressed_attention"), supply, policy,
        kind="A2", layer=2)
    bound = compiled.operator
    bad_theta = replace(bound.theta, extras=tuple(
        (key, 0 if key == "compress_ratio" else value)
        for key, value in bound.theta.extras))
    with pytest.raises(ModelingError,
                       match="a2_compressed_branch_requires_positive_ratio"):
        CompiledOperator(replace(bound, theta=bad_theta), supply, policy,
                         kind="A2", layer=2)


@pytest.mark.parametrize("relative", (
    "examples/cases/longctx-decode-20261030/qwen38-a3-tp4.json",
    "examples/cases/glm53-a5.json",
))
def test_raw_flag_is_inert_for_incompatible_sources(relative):
    path = Path(__file__).parents[1] / relative
    raw = json.loads(path.read_text())
    base = parse_case(raw)
    explicit = parse_case({**raw, "runtime": {
        **raw.get("runtime", {}), "multistream_window_compressed": False}})
    fb = FeatureConfig.from_runtime(base.runtime, base.model)
    fe = FeatureConfig.from_runtime(explicit.runtime, explicit.model)
    assert fb == fe and fb.multistream_window_compressed is False
    ib = instantiate_model(base.model, fb)
    ie = instantiate_model(explicit.model, fe)
    assert tuple(item.performance_key() for item in ib.subgraphs) == tuple(
        item.performance_key() for item in ie.subgraphs)
    assert all(op.stream == "main" for item in ib.subgraphs
               for op in item.template.operators)
