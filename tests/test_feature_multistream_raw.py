"""Raw Software spelling for the three implemented multistream controls."""
from __future__ import annotations
import json
from dataclasses import replace
from pathlib import Path

from mindie_sim.prediction.piecewise.compiler import ImplementationPolicy, PipeSupply
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import ModelPipeline
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, instantiate_model
from mindie_sim.schema.case import Software

CASE = Path(__file__).parents[1]


def _case(relative):
    return parse_case(json.loads((CASE / relative).read_text()))


def _aux_streams(model, runtime):
    features = FeatureConfig.from_runtime(runtime, model)
    instantiated = instantiate_model(model, features)
    found = []
    for layer in instantiated.layers:
        for subgraph in layer.subgraphs:
            if subgraph.kind not in {"A1", "A2", "A3", "A4", "K1"}:
                continue
            found.extend((subgraph.kind, op.name, op.stream)
                         for op in subgraph.template.operators if op.stream != "main")
    return features, tuple(found)


def _model_pipeline(case, flag):
    runtime = replace(case.runtime, **{flag: True})
    cores = max(1, case.machine.dies[0].compute_units)
    features = FeatureConfig.from_runtime(runtime, case.model).project(
        tp=4, dp=1, ep=16, cores=cores, graph_buckets=())
    supply = PipeSupply.from_hardware(case.machine, dtype="bf16")
    return ModelPipeline(case.model, features, supply,
                         ImplementationPolicy(cores=cores))


def test_three_exact_raw_fields_are_accepted_defaulted_and_restored():
    names = ("multistream_kda_proj", "multistream_attention_indexer",
             "multistream_window_compressed")
    default = Software.from_mapping({})
    assert all(getattr(default, name) is False for name in names)
    explicit = Software.from_mapping({name: True for name in names})
    assert all(getattr(explicit, name) is True for name in names)
    assert all(explicit.to_dict()[name] is True for name in names)
    restored = Software.from_mapping(explicit.to_dict())
    assert restored == explicit

    features = FeatureConfig.from_runtime(explicit)
    assert all(getattr(features, name) is True for name in names)
    assert set(names) <= set(dict(features.metadata))
    assert FeatureConfig.from_runtime(default) != features


def test_actual_qwen38_multistream_kda_proj_is_off_default_on_with_raw_field():
    case = _case("examples/cases/longctx-decode-20261030/a5-gdn-hccl-source/"
                 "a5-qwen38-tp2-mesh_oneshot.json")
    base, base_streams = _aux_streams(case.model, case.runtime)
    on, on_streams = _aux_streams(case.model, replace(
        case.runtime, multistream_kda_proj=True))
    assert not base.multistream_kda_proj and not base_streams
    assert on.multistream_kda_proj
    assert ("A3", "ba_projection", "aux") in on_streams
    assert on.metadata != base.metadata


def test_actual_v41_multistream_window_compressed_is_off_default_on():
    case = _case("examples/cases/longctx-decode-20261030/"
                 "v41-a3-tp4-dp4-ep16.json")
    base, base_streams = _aux_streams(case.model, case.runtime)
    on, on_streams = _aux_streams(case.model, replace(
        case.runtime, multistream_window_compressed=True))
    assert not base.multistream_window_compressed and not base_streams
    assert on.multistream_window_compressed
    assert ("A2", "compressed_attention", "aux") in on_streams


def test_actual_glm53_multistream_attention_indexer_changes_parent_schedule():
    case = _case("examples/cases/glm53-a5.json")
    geometry = WorkloadGeometry((RequestGeometry(8, 0, phase="prefill"),),
                                phase="prefill")
    cores = max(1, case.machine.dies[0].compute_units)
    supply = PipeSupply.from_hardware(case.machine, dtype="bf16")
    on_f = FeatureConfig.from_runtime(
        replace(case.runtime, multistream_attention_indexer=True), case.model)
    on = ModelPipeline(case.model, on_f, supply,
                       ImplementationPolicy(cores=cores))
    on_prediction = on.forward(geometry, fusion=True)
    assert on_f.multistream_attention_indexer
    off_f = FeatureConfig.from_runtime(case.runtime, case.model)
    off = ModelPipeline(case.model, off_f, supply,
                        ImplementationPolicy(cores=cores))
    off_prediction = off.forward(geometry, fusion=True)
    assert not off_f.multistream_attention_indexer
    assert on_prediction.total_ms != off_prediction.total_ms
    on_layer = on_prediction.layers[3]
    off_layer = off_prediction.layers[3]
    assert on_layer.duration_ms != off_layer.duration_ms
