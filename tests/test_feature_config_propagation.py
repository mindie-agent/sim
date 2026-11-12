"""FeatureConfig propagation through the ordinary piecewise front.

[Role]
  Verify declared runtime capture/mode, machine/model identity and derived
  parallel experts reach FeatureConfig and compiled feature records.
[Invariants]
  * Full capture uses graph_capture_keys (fallback graph_buckets); piecewise
    uses piecewise_capture_keys; eager carries no capture sizes.
  * project() without a graph_bucket argument preserves the existing capture
    set; it does not silently erase it.
  * ETP derives from TP/DP/EP when not separately declared, and KV dtype
    falls back to the model when quantization omits it.
  * The actual case.machine/model identity reaches compiled feature metadata.
[Timeouts & Locks] One-layer compiled controls; no device or I/O.
[Side-Effect Policy] Pure in-memory records; no source-case writes.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

from mindie_sim.prediction.piecewise.geometry import WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig
from mindie_sim.schema.case import Software
from mindie_sim.schema.model import ModelSpec
from test_piecewise_subgraph_model_pipeline import FEATURES, POLICY, SUPPLY, dense_model, moe_model


def _model():
    return dense_model()


def _machine():
    return SimpleNamespace(hardware_key="Ascend910_9362", sku="Ascend910_9362",
                           name="Ascend910_9362", arch_family="A3")


def _runtime(**values):
    return Software.from_mapping({"parallel": {"tp": 1}, **values})


def test_capture_mode_selects_actual_bucket_set():
    model = _model()
    full = _runtime(graph_capture_mode="full", graph_capture_keys=(1, 2),
                    graph_buckets=(4, 8), piecewise_capture_keys=(16,))
    assert FeatureConfig.from_runtime(full, model).graph_buckets == (1, 2)
    piecewise = replace(full, graph_capture_mode="piecewise")
    assert FeatureConfig.from_runtime(piecewise, model).graph_buckets == (16,)
    eager = replace(full, graph_mode="eager")
    assert FeatureConfig.from_runtime(eager, model).graph_buckets == ()


def test_project_preserves_capture_set_when_not_asked_to_change():
    model = _model()
    features = FeatureConfig.from_model(model, graph_buckets=(4, 8))
    assert features.project(tp=2).graph_buckets == (4, 8)
    assert features.project(tp=2, graph_buckets=(16,)).graph_buckets == (16,)


def test_etp_derivation_and_kv_fallback():
    model = replace(_model(), kv_dtype="fp16", quant=None)
    runtime = Software.from_mapping({"parallel": {"tp": 4, "dp": 2, "ep": 2}})
    features = FeatureConfig.from_runtime(runtime, model)
    assert features.etp == 2
    assert features.project().etp == 2
    assert features.kv_dtype == "fp16"
    from_model = FeatureConfig.from_model(model, tp=4, dp=2, ep=2)
    assert from_model.etp == 2


def test_machine_identity_reaches_compiled_feature_record():
    model = _model()
    features = FeatureConfig.from_model(model, machine=_machine(), **FEATURES)
    pipe = assemble_model(model, supply=SUPPLY, policy=POLICY, features=features)
    instance = pipe.instantiated.find("A0", layer=0, role="attention")
    compiled = pipe._subgraph(instance)
    metadata = dict(compiled.feature_metadata)
    assert metadata["hardware_key"] == "Ascend910_9362"
    assert metadata["device"] == "Ascend910_9362"
    assert metadata["model_identifier"] == model.name
    assert dict(features.metadata)["arch_family"] == "A3"


def test_same_interpreter_switch_off_on_off_has_distinct_compiled_identity():
    model = _model()
    off = FeatureConfig.from_model(model, machine=_machine(), graph_buckets=(), **FEATURES)
    on = FeatureConfig.from_model(model, machine=_machine(), graph_buckets=(4, 8), **FEATURES)
    off_again = FeatureConfig.from_model(model, machine=_machine(), graph_buckets=(), **FEATURES)
    assert off.graph_buckets == off_again.graph_buckets == ()
    assert on.graph_buckets == (4, 8)
    p_off = assemble_model(model, supply=SUPPLY, policy=POLICY, features=off)
    p_on = assemble_model(model, supply=SUPPLY, policy=POLICY, features=on)
    assert p_off.features.graph_buckets == ()
    assert p_on.features.graph_buckets == (4, 8)
    c_off = p_off._subgraph(p_off.instantiated.find("A0", layer=0, role="attention"))
    c_on = p_on._subgraph(p_on.instantiated.find("A0", layer=0, role="attention"))
    assert c_off.operators[0].operator.theta.find("graph_buckets") == ()
    assert c_on.operators[0].operator.theta.find("graph_buckets") == (4, 8)


def test_etp_positive_changes_real_f3_expert_graph():
    model = moe_model()
    two = FeatureConfig.from_model(model, tp=4, ep=2)
    one = replace(two, etp=1)
    assert two.etp == 2 and one.etp == 1
    p_two = assemble_model(model, supply=SUPPLY, policy=POLICY, features=two)
    p_one = assemble_model(model, supply=SUPPLY, policy=POLICY, features=one)
    f3_two = p_two.instantiated.find("F3", layer=0)
    f3_one = p_one.instantiated.find("F3", layer=0)
    assert dict(f3_two.operator("expert_gate_up").theta.matrix)["N"] == 32
    assert dict(f3_one.operator("expert_gate_up").theta.matrix)["N"] == 64
    assert "etp_allreduce" in {op.name for op in f3_two.operators}
    assert "etp_allreduce" not in {op.name for op in f3_one.operators}
    geometry = WorkloadGeometry.homogeneous(2, 1, 16)
    assert p_two.forward(geometry).total_ms != p_one.forward(geometry).total_ms


def test_parsed_case_machine_identity_reaches_features():
    from mindie_sim.prediction.structure import parse_case
    raw = json.loads((Path(__file__).parent / "fixtures" /
                      "moe-a5-q30-e16k4-tp4ep4-structure.json").read_text())
    case = parse_case(raw)
    features = FeatureConfig.from_runtime(case.runtime, case.model)
    metadata = dict(features.metadata)
    assert metadata["hardware_key"] == (case.machine.hardware_key or case.machine.sku)
    assert metadata["device"] == case.machine.sku
    assert metadata["model_identifier"]


def test_actual_source_parallel_etp_cases():
    cases = (
        Path(__file__).parents[1] / "examples/cases/longctx-decode-20261030/"
        "a3-qwen35-frozen-ffn/qwen35-a3-tp4-frozen-ffn.json",
        Path(__file__).parents[1] / "examples/cases/high-concurrency-20261030/"
        "glm-source-complete/a5_glm53flash_tp8_c4_p8k.json",
        Path(__file__).parents[1] / "examples/cases/kimi-k3-a5-sp-right-pad-conditional.json",
        Path(__file__).parents[1] / "examples/cases/high-concurrency-20261030/"
        "new17/glm53flash-a3-tp16-c4.json",
    )
    from mindie_sim.prediction.structure import parse_case
    seen = set()
    for path in cases:
        case = parse_case(json.loads(path.read_text()))
        parallel = case.runtime.parallel
        features = FeatureConfig.from_runtime(case.runtime, case.model)
        assert features.etp == parallel.etp
        assert features.project().etp == parallel.etp
        seen.add((parallel.tp, parallel.ep, features.etp))
    assert (4, 1, 4) in seen or any(tp > 1 and ep == 1 and etp == tp
                                    for tp, ep, etp in seen)
