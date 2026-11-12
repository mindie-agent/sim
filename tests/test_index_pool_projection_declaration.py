"""Analytical KPool projection roles, independent of the parity band.

[Role]
  The pooled indexer declares separate F.linear launches only when
  ``projection_quant`` can see the mode.  One compiled row must follow the
  existing one-owner MTE copy.  Context does not enter those shapes.

[Invariants]
  * No captured millisecond is an input.  The expected elapsed is the
    slowest owner's fp32 copy divided by one declared mte2_aic unit.
  * An explicit non-FP32 override and a CANN caller do not become the
    FP32 pair.  A missing quant record does not invent one.

[Timeouts & Locks]
  One A3 instantiation and a handful of single-operator evaluations.

[Side-Effect Policy]
  Reads the checked-in case.  Writes nothing.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.piecewise.compiler import PipeSupply
from mindie_sim.prediction.piecewise.families_recurrent_indexer import (
    index_pool_mode,
)
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.matrix_mapping import (
    mte_owner_unit_count, owner_tiles,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.builders_attention import _build_i0
from mindie_sim.prediction.subgraphs.builders_ffn import _build_r0
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, _Ctx
from mindie_sim.schema.subgraphs import BoundOperator

ROOT = Path(__file__).resolve().parents[1]
A3_CASE = (ROOT / "examples/cases/high-concurrency-20261030"
           / "glm-kda-accepted-conditional/glm53flash-a3-b4-q6.json")


def _definition() -> dict:
    payload = json.loads(A3_CASE.read_text(encoding="utf-8"))
    runtime = payload.setdefault("runtime", {})
    kernels = runtime.get("kernels") or {}
    for key in ("service_regimes", "invocations", "primitives"):
        kernels.pop(key, None)
    if kernels:
        runtime["kernels"] = kernels
    elif "kernels" in runtime:
        runtime.pop("kernels")
    runtime.update({"tp": 16, "dp": 1, "ep": 16, "pp": 1, "graph_mode": "none"})
    return payload


def _case():
    return parse_case(_definition())


def _ctx(case, layer):
    features = FeatureConfig.from_runtime(case.runtime, case.model)
    return _Ctx(case.model, layer, features, stage=0, layers=case.model.layers)


def _names(built):
    return [operator.name for operator in built.template.operators]


def _bound(operator, geometry, supply):
    from mindie_sim.prediction.piecewise.compiler import OperatorPiecewiseCompiler
    compiled = OperatorPiecewiseCompiler(supply).compile_operator(
        operator, kind="I0")
    cost = compiled.evaluate(geometry, readiness=False)
    units = mte_owner_unit_count(supply.resource_units)
    tokens = geometry.effective_tokens
    tile_m = int(cost.get("tile_m"))
    tile_n = int(cost.get("tile_n"))
    act = float(operator.theta.find("act_bytes"))
    weight = {"fp32": 4.0, "bf16": 2.0, "int8": 1.0}[operator.theta.find("weight_format")]
    issued = [act * extent_m * int(operator.theta.require("K"))
              + weight * extent_n * int(operator.theta.require("K"))
              for _c, extent_m, extent_n, _a in owner_tiles(
                  tokens, int(operator.theta.require("N")), tile_m, tile_n, units)]
    peak = max(issued) / (supply.rate("mte2") / units)
    return cost, peak, issued, tile_m, tile_n


def _operator(built, name):
    template = next(item for item in built.template.operators if item.name == name)
    return BoundOperator(template, built.signature(name))


def test_a3_declares_fused_fp32_pool_and_bf16_query():
    case = _case()
    layer = case.model.layer_plans[3]
    mode = index_pool_mode(
        layer, case.model.quant, hidden=case.model.hidden_size,
        activation_dtype=case.model.activation_dtype)
    assert mode.kind == "analytical" and mode.declaration.fused
    built = _build_i0(_ctx(case, layer))
    assert _names(built) == [
        "index_pool_hidden_cast", "index_q_projection", "index_k_projection",
        "index_pool_gate", "index_quant", "index_cache_store"]
    key = built.signature("index_k_projection")
    gate = built.signature("index_pool_gate")
    query = built.signature("index_q_projection")
    cast = built.signature("index_pool_hidden_cast")
    assert (int(key.require("K")), int(key.require("N"))) == (4096, 160)
    assert (int(gate.require("K")), int(gate.require("N"))) == (4096, 128)
    assert key.find("weight_format") == key.find("act_format") == "fp32"
    assert gate.find("weight_format") == gate.find("act_format") == "fp32"
    assert float(key.find("weight_payload")) == 160 * 4096 * 4
    assert float(gate.find("weight_payload")) == 128 * 4096 * 4
    assert key.find("projection_packing") == "source_joint_key_weights"
    assert int(key.find("key_slice_end")) == 128
    assert (int(key.find("weights_slice_begin")), int(key.find("weights_slice_end"))) == (128, 160)
    assert (int(query.require("K")), int(query.require("N"))) == (1536, 32 * 128)
    assert query.find("weight_format") == "bf16"
    assert query.find("category") == "indexer_fp8"
    assert query.find("k_source") == "q_lora_rank"
    assert int(cast.require("elements")) == 4096
    assert int(cast.find("read_bytes")) == 4096 * 2
    assert int(cast.find("write_bytes")) == 4096 * 4


def test_owner_copy_moves_with_tokens_and_kv_width_not_context_or_heads():
    case = _case()
    features_case = _case()
    supply = PipeSupply.from_hardware(features_case.machine)
    layer = case.model.layer_plans[3]
    built = _build_i0(_ctx(case, layer))
    wide = _build_i0(_ctx(case, replace(layer, index_heads=96)))
    narrow = _build_i0(_ctx(case, replace(layer, index_dim=64)))
    ordinary = WorkloadGeometry(
        tuple(RequestGeometry(6, 4096, phase="decode") for _ in range(4)),
        phase="decode")
    empty = WorkloadGeometry(
        tuple(RequestGeometry(6, 0, phase="decode") for _ in range(4)),
        phase="decode")
    short = WorkloadGeometry(
        tuple(RequestGeometry(1, 4096, phase="decode") for _ in range(4)),
        phase="decode")
    assert ordinary.effective_tokens == 24
    assert short.effective_tokens == 4
    key = _operator(built, "index_k_projection")
    gate = _operator(built, "index_pool_gate")
    key_cost, key_peak, key_issued, tile_m, tile_n = _bound(key, ordinary, supply)
    gate_cost, gate_peak, gate_issued, _, _ = _bound(gate, ordinary, supply)
    assert tile_m == 128 and tile_n == 128
    assert key_issued and gate_issued
    widths = [extent_n for _c, _m, extent_n, _a in owner_tiles(24, 160, tile_m, tile_n, 20)]
    assert 128 in widths and max(widths) == 128
    assert float(key_cost.duration_ms) == pytest.approx(key_peak, rel=0.10)
    assert float(gate_cost.duration_ms) == pytest.approx(gate_peak, rel=0.10)
    assert key_cost.dominant_pipe == gate_cost.dominant_pipe == "mte2"
    assert float(key_cost.duration_ms) > 0.010
    empty_cost, _, _, _, _ = _bound(key, empty, supply)
    assert float(empty_cost.duration_ms) == pytest.approx(float(key_cost.duration_ms), rel=1e-9)
    short_cost, short_peak, _, _, _ = _bound(key, short, supply)
    assert float(short_cost.duration_ms) == pytest.approx(short_peak, rel=0.10)
    assert short_peak < key_peak
    wide_key = _operator(wide, "index_k_projection")
    assert int(wide_key.theta.require("N")) == 224
    wide_cost, wide_peak, _, _, _ = _bound(wide_key, ordinary, supply)
    assert max(extent_n for _c, _m, extent_n, _a in owner_tiles(
        24, 224, 128, 128, 20)) == 128
    assert float(wide_cost.duration_ms) == pytest.approx(wide_peak, rel=0.10)
    assert wide_peak == pytest.approx(key_peak, rel=1e-9)
    narrow_gate = _operator(narrow, "index_pool_gate")
    assert int(narrow_gate.theta.require("N")) == 64
    narrow_cost, narrow_peak, _, narrow_tm, narrow_tn = _bound(narrow_gate, ordinary, supply)
    assert narrow_tn == 64
    assert float(narrow_cost.duration_ms) == pytest.approx(narrow_peak, rel=0.10)
    assert narrow_peak < gate_peak
    del narrow_tm


def test_mode_stays_off_without_fp32_evidence():
    case = _case()
    layer = case.model.layer_plans[3]
    hidden = case.model.hidden_size
    act = case.model.activation_dtype
    assert index_pool_mode(
        replace(layer, index_kpool=1), case.model.quant, hidden=hidden,
        activation_dtype=act).kind == "not_pooled"
    assert index_pool_mode(
        replace(layer, index_kpool_compress=False), case.model.quant,
        hidden=hidden, activation_dtype=act).kind == "not_pooled"
    assert index_pool_mode(
        layer, None, hidden=hidden, activation_dtype=act).kind == "quant_undistinguished"
    assert index_pool_mode(
        layer, case.model.quant, hidden=hidden, activation_dtype=act,
        cann_caller=True).kind == "cann_caller"
    parent = replace(case.model.quant, category_overrides={
        "indexer": {"weight_format": "int8", "activation_format": "int8"}})
    parent_mode = index_pool_mode(layer, parent, hidden=hidden, activation_dtype=act)
    assert parent_mode.kind == "analytical" and parent_mode.declaration.fused
    parent_built = _build_i0(_ctx(replace(case, model=replace(case.model, quant=parent)), layer))
    key = parent_built.signature("index_k_projection")
    assert key.find("weight_format") == key.find("act_format") == "int8"
    assert int(key.require("N")) == 160
    split = replace(case.model.quant, category_overrides={
        "indexer.weights": {"weight_format": "bf16", "activation_format": "bf16"}})
    split_mode = index_pool_mode(layer, split, hidden=hidden, activation_dtype=act)
    assert split_mode.kind == "analytical" and not split_mode.declaration.fused
    split_built = _build_i0(_ctx(replace(case, model=replace(case.model, quant=split)), layer))
    assert "index_weights_projection" in _names(split_built)
    assert split_built.signature("index_k_projection").find("weight_format") == "fp32"
    assert int(split_built.signature("index_k_projection").require("N")) == 128
    weights = split_built.signature("index_weights_projection")
    assert weights.find("weight_format") == "bf16"
    assert int(weights.require("N")) == 32
    plain = _build_i0(_ctx(case, replace(layer, index_kpool=1)))
    assert "index_pool_gate" not in _names(plain)
    assert "index_rope" in _names(plain)
    assert plain.signature("index_k_projection").find("weight_format") != "fp32"


def test_hc_pre_adds_cast_and_affine_without_replacing_inv_rms():
    case = _case()
    layer = next(item for item in case.model.layer_plans if item.hyper_connections)
    built = _build_r0(_ctx(case, layer))
    assert _names(built)[:4] == [
        "hc_values_fp32", "inv_rms", "hc_projection", "hc_scale_bias_gate"]
    cast = built.signature("hc_values_fp32")
    gate = built.signature("hc_scale_bias_gate")
    norm = built.signature("inv_rms")
    mult = int(layer.hc_mult or 1)
    streams = mult * case.model.hidden_size
    assert int(cast.find("read_bytes")) == streams * 2
    assert int(cast.find("write_bytes")) == streams * 4
    assert cast.find("native_recipe") == "hc_values_fp32"
    mixed = (2 + mult) * mult
    assert int(gate.require("elements")) == mixed
    assert int(gate.find("read_bytes")) == (mixed + 1) * 4
    assert int(gate.find("write_bytes")) == mixed * 4
    assert gate.find("native_recipe") == "hc_normalize_scale_bias_gate"
    assert int(gate.find("static_parameter_bytes")) == (mixed + 3) * 4
    assert norm.role == "norm"
    assert int(norm.require("elements")) == streams


def test_assembled_a3_i0_matches_the_declaration():
    case = _case()
    features = FeatureConfig.from_runtime(case.runtime, case.model)
    supply = PipeSupply.from_hardware(case.machine)
    pipeline = assemble_model(case.model, supply=supply, features=features)
    found = None
    for layer in pipeline.instantiated.layers:
        if layer.layer != 3:
            continue
        for instance in layer.subgraphs:
            if instance.kind == "I0":
                found = instance
                break
        if found:
            break
    assert found is not None
    names = [operator.name for operator in found.operators]
    assert names == [
        "index_pool_hidden_cast", "index_q_projection", "index_k_projection",
        "index_pool_gate", "index_quant", "index_cache_store"]
    key = next(operator for operator in found.operators if operator.name == "index_k_projection")
    assert int(key.theta.require("N")) == 160
    assert key.theta.find("act_format") == "fp32"
    assert "pool_key_indexer" not in dict(features.metadata)
