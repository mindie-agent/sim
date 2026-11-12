"""Declared KDA use_beta_sigmoid reaches the piecewise decode walk.

[Role]
  The native decode source removes linear_beta_sigmoid only when the declared
  option is true. The piecewise A4 template has no separate producer, so a
  declared false option must add that same elementwise launch.

[Invariants]
  * True publishes kda_use_beta_sigmoid=1 and adds no sigmoid row.
  * False publishes 0 and adds one linear_beta_sigmoid per KDA layer.
  * Each row's vector work is tokens * local value heads * 5, the
    Builder.vector arguments in recurrent.linear. Elapsed is whatever the
    shared elementwise compiler returns; the body delta is the sum of those
    rows, not a fitted total.
  * An undeclared option does not invent the metadata key.

[Timeouts & Locks]
  Two finite A3 body walks. No device, subprocess or lock.

[Side-Effect Policy]
  Reads the checked-in case. No filesystem writes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mindie_sim.prediction.body import body_forward
from mindie_sim.prediction.piecewise.compiler import PipeSupply
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig

ROOT = Path(__file__).resolve().parents[1]
A3_CASE = (ROOT / "examples/cases/high-concurrency-20261030"
           / "glm-kda-accepted-conditional/glm53flash-a3-b4-q6.json")


def _raw(option):
    payload = json.loads(A3_CASE.read_text(encoding="utf-8"))
    payload["runtime"].update({"tp": 16, "dp": 1, "ep": 16, "pp": 1, "graph_mode": "none"})
    kernels = payload["runtime"].get("kernels") or {}
    for key in ("service_regimes", "invocations", "primitives"):
        kernels.pop(key, None)
    if kernels:
        payload["runtime"]["kernels"] = kernels
    if option is None:
        payload["runtime"]["kernels"]["lowerings"]["kda"]["options"].pop("use_beta_sigmoid", None)
    else:
        payload["runtime"]["kernels"]["lowerings"]["kda"]["options"]["use_beta_sigmoid"] = option
    return payload


def _walk(option, *, native: bool = True):
    case = parse_case(_raw(option))
    native_result = (body_forward(case, phase="decode", tokens=6, context=4096, sequences=4)
                     if native else None)
    features = FeatureConfig.from_runtime(case.runtime, case.model)
    pipeline = assemble_model(case.model, supply=PipeSupply.from_hardware(case.machine),
                              features=features)
    geometry = WorkloadGeometry(
        tuple(RequestGeometry(6, 4096, phase="decode") for _ in range(4)),
        phase="decode")
    prediction = pipeline.forward(geometry)
    rows = tuple(row for layer in prediction.layers for row in layer.rows
                 if row.cost.name == "linear_beta_sigmoid")
    body_ms = float(sum(layer.duration_ms for layer in prediction.layers))
    return case, features, native_result, rows, body_ms


def _derived(cost, key):
    return dict(cost.derived)[key]


def test_declared_false_adds_one_external_beta_sigmoid_per_kda_layer():
    case_true, features_true, native_true, rows_true, body_true = _walk(True)
    case_false, features_false, native_false, rows_false, body_false = _walk(False)
    assert dict(features_true.metadata)["kda_use_beta_sigmoid"] == 1
    assert dict(features_false.metadata)["kda_use_beta_sigmoid"] == 0
    assert rows_true == ()

    plans = [plan for plan in case_false.model.layer_plans if plan.attention == "kda"]
    assert plans
    assert len(rows_false) == len(plans)
    tokens = 24
    tp = 16
    for plan, row in zip(plans, rows_false):
        heads = max(1, math.ceil(int(plan.linear_value_heads or plan.linear_heads) / tp))
        assert row.layer == plan.index
        assert float(_derived(row.cost, "elements")) == tokens * heads
        assert float(_derived(row.cost, "vector_ops")) == tokens * heads * 5
    sigmoid_ms = float(sum(row.cost.duration_ms for row in rows_false))
    assert sigmoid_ms > 0
    assert body_false - body_true == pytest.approx(sigmoid_ms, abs=1e-9)

    native_delta = (float(native_false["body_forward_ms"])
                    - float(native_true["body_forward_ms"]))
    piece_delta = body_false - body_true
    assert native_delta > 0
    assert piece_delta == pytest.approx(native_delta, rel=0.10)


def test_undeclared_kda_option_is_not_invented():
    _case, features, _native, rows, _body = _walk(None, native=False)
    assert "kda_use_beta_sigmoid" not in dict(features.metadata)
    assert rows == ()
