"""V0 vision and D0 MTP work on the normal ModelPipeline path.

[Role]
  Use the actual Qwen3.5 source model with its vision/MTP auxiliary records and
  an explicitly requested media token count.  Verify media creates V0 work only
  when requested and draft creates D0 work only when the caller enables it.
[Invariants]
  * A text-only/zero-media prefill has no V0 rows.
  * A media prefill has the V0 chain and a larger whole total.
  * include_draft=False has no D0 rows; include_draft=True executes the draft
    and target-verify chain once and changes the whole total.
  * Same-process off/on/off uses separate feature/state inputs; no cache clear.
[Timeouts & Locks] One small model/geometry; no I/O beyond the source case.
[Side-Effect Policy] Read-only source case; immutable records.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from mindie_sim.prediction.piecewise.compiler import ImplementationPolicy, PipeSupply
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import ModelPipeline
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig
from mindie_sim.prediction.subgraphs.state import StateView

CASE = (Path(__file__).parents[1] / "examples/cases/longctx-decode-20261030/"
        "a3-qwen35-frozen-ffn/qwen35-a3-tp4-frozen-ffn.json")


def _setup(media_tokens):
    raw = copy.deepcopy(json.loads(CASE.read_text()))
    raw["workload"] = {"requests": [{
        "request_id": "r0", "prompt_tokens": 8,
        "media_tokens": int(media_tokens), "output_tokens": 2}]}
    raw["runtime"]["parallel"] = {"tp": 4, "dp": 1, "ep": 1, "pp": 1}
    case = parse_case(raw)
    cores = max(1, case.machine.dies[0].compute_units)
    features = FeatureConfig.from_runtime(case.runtime, case.model).project(
        tp=4, dp=1, ep=1, cores=cores, graph_buckets=())
    supply = PipeSupply.from_hardware(case.machine, dtype="bf16")
    pipe = ModelPipeline(case.model, features, supply,
                         ImplementationPolicy(cores=cores))
    state = StateView()
    for layer in range(len(case.model.layer_plans)):
        state = state.mark_kv(layer).mark_index(layer).mark_selection(layer)
    return pipe, state


def _kinds(prediction, kind):
    return [row for row in prediction.rows if row.cost.kind == kind]


def test_v0_media_positive_and_zero_media_negative():
    pipe, _state = _setup(0)
    text = pipe.forward(WorkloadGeometry(
        (RequestGeometry(8, 0, phase="prefill"),), phase="prefill",
        media_tokens=0), include_draft=False, fusion=False)
    assert not _kinds(text, "V0")

    pipe, _state = _setup(4)
    media = pipe.forward(WorkloadGeometry(
        (RequestGeometry(8, 0, phase="prefill"),), phase="prefill",
        media_tokens=4), include_draft=False, fusion=False)
    assert _kinds(media, "V0")
    assert media.total_ms > text.total_ms


def test_d0_enabled_positive_disabled_negative_and_off_on_off():
    pipe, state = _setup(0)
    geometry = WorkloadGeometry((RequestGeometry(1, 8, phase="decode"),),
                                phase="decode")
    off = pipe.forward(geometry, state=state, include_draft=False, fusion=False)
    on = pipe.forward(geometry, state=state, include_draft=True, fusion=False)
    off_again = pipe.forward(geometry, state=state, include_draft=False, fusion=False)
    assert not _kinds(off, "D0") and not _kinds(off_again, "D0")
    assert _kinds(on, "D0")
    assert {row.cost.name for row in _kinds(on, "D0")} >= {
        "draft_projection", "target_verify", "accept_commit"}
    assert on.total_ms > off.total_ms
    assert off.total_ms == off_again.total_ms
