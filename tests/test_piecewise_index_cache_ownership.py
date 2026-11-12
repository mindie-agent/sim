"""Index-cache producer/consumer ownership across shared V4.1 layers.

[Role]
  Pin the source contract for the index K cache and the top-k selection:
  V4.1 layers 2/8/14/20 own and write the cache; layers 24/28/32/36 are
  indexers that reuse owner 20's cache while owning their own top-k selection.
  The top-k consumer must not require an index cache at its own layer.
[Invariants]
  * The owner layer's I0 contains exactly one index_cache_store; a shared
    non-owner I0 contains none.
  * A shared consumer's index_score/filter requires the declared cache owner,
    while its topk declares its own selection owner.
  * Full ordinary V4.1 assembly and forward reaches every layer without a
    missing index port.
  * An authentic plan with an indexer consumer but no cache producer raises
    state_port_not_ready:index_cache and does not mark the missing port ready.
[Timeouts & Locks] One ordinary 40-layer assembly plus small synthetic models;
  no I/O, device or locks.
[Side-Effect Policy] Read-only in-memory catalog and synthetic plans.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.piecewise.compiler import (
    ImplementationPolicy, PipeSupply,
)
from mindie_sim.prediction.piecewise.composition import evaluate_serial_split
from mindie_sim.prediction.piecewise.geometry import WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig
from mindie_sim.prediction.subgraphs.state import CompressionState, StateView
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec

SUPPLY = PipeSupply((("cube", 1_000_000.0), ("vector", 1_000_000.0),
                     ("hbm", 1_000_000.0), ("mte1", 2_000_000.0),
                     ("mte2", 2_000_000.0), ("mte3", 2_000_000.0),
                     ("fixpipe", 2_000_000.0), ("link", 1_000_000.0)))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=2, moe_row_tile=4,
                              split_target=1000, max_splits=4)
FEATURES = dict(token_tile=16, kv_tile=16, cores=2, moe_row_tile=4,
                split_target=1000, max_splits=4)


def _pipe(model):
    return assemble_model(
        model, supply=SUPPLY, policy=POLICY,
        features=FeatureConfig.from_model(model, **FEATURES))


def _index_plan(*, owns_index_k: bool) -> dict:
    return dict(
        index=0, attention="gqa", ffn="dense", num_heads=4,
        num_kv_heads=2, head_dim=16, intermediate_size=128,
        cache_kind="window_compressed", window=8, compress_ratio=4,
        compressor_kind="weighted_pool", indexer=True, index_heads=2,
        index_dim=8, index_topk=4, index_kpool=1, owns_long_kv=True,
        owns_index_k=owns_index_k, index_k_owner_layer_id=0,
        topk_owner_layer_id=0, owns_topk=True)


def _index_model(*, owns_index_k: bool) -> ModelSpec:
    return ModelSpec.from_mapping(dict(
        name="index-cache-control", hidden_size=64, num_heads=4,
        num_kv_heads=2, head_dim=16, intermediate_size=128,
        vocab_size=128, layers=1,
        layer_plans=[_index_plan(owns_index_k=owns_index_k)]))


def _subgraph(pipe, layer: int, kind: str):
    return next(item for item in pipe.instantiated.layers[layer].subgraphs
                if item.kind == kind)


def test_v41_shared_indexer_consumes_owner_cache_but_owns_selection():
    model = load_model("deepseek-v4.1-flash")
    pipe = _pipe(model)
    prediction = pipe.forward(WorkloadGeometry.homogeneous(2, 4, 32))
    assert prediction.total_ms > 0

    owner_i0 = _subgraph(pipe, 20, "I0")
    shared_i0 = _subgraph(pipe, 24, "I0")
    assert "index_cache_store" in {op.name for op in owner_i0.operators}
    assert "index_cache_store" not in {op.name for op in shared_i0.operators}

    shared_i1 = _subgraph(pipe, 24, "I1")
    score = shared_i1.operator("index_score")
    topk = shared_i1.operator("topk")
    candidate = shared_i1.operator("candidate_filter")
    assert score.theta.find("require_index") == 20
    assert topk.theta.find("owner_layer") == 24
    assert candidate.theta.find("require_index") == 20
    assert candidate.theta.find("owner_layer") == 24

    # The next shared group (28/32) reuses owner 20 and keys its own selection.
    later_i1 = _subgraph(pipe, 28, "I1")
    assert later_i1.operator("index_score").theta.find("require_index") == 20
    assert later_i1.operator("topk").theta.find("owner_layer") == 28


def test_missing_cache_producer_raises_while_present_owner_passes():
    present = _pipe(_index_model(owns_index_k=True))
    assert present.forward(WorkloadGeometry.homogeneous(2, 1, 8)).total_ms > 0
    owner_i0 = _subgraph(present, 0, "I0")
    assert "index_cache_store" in {op.name for op in owner_i0.operators}

    missing = _pipe(_index_model(owns_index_k=False))
    assert "index_cache_store" not in {
        op.name for op in _subgraph(missing, 0, "I0").operators}
    with pytest.raises(ModelingError, match=r"state_port_not_ready:index_cache:layer=0"):
        missing.forward(WorkloadGeometry.homogeneous(2, 1, 8))


def test_a2_requires_the_real_selection_edge_not_just_cache_and_compression():
    model = load_model("deepseek-v4.1-flash")
    pipe = _pipe(model)

    def compressed(layer: int):
        instance = next(
            item for item in pipe.instantiated.layers[layer].subgraphs
            if item.kind == "A2"
            and any(op.name == "compressed_attention" for op in item.operators))
        return pipe._subgraph(instance)

    # Shared indexer and its non-indexer siblings: cache owner 20 is separate
    # from the actual top-k selection owners 24/28/36.
    for layer, cache_owner, selection_owner in (
            (24, 20, 24), (25, None, 24), (28, 20, 28), (36, 20, 36)):
        compiled = compressed(layer)
        theta = next(op.operator.theta for op in compiled.operators
                     if op.name == "compressed_attention")
        assert theta.find("require_index", None) == cache_owner
        assert theta.find("require_selection", None) == selection_owner

    geometry = WorkloadGeometry.homogeneous(1, 4, 8)
    for layer, selection_owner in ((24, 24), (36, 36)):
        compiled = compressed(layer)
        base = StateView().mark_kv(20).mark_index(20)
        with pytest.raises(ModelingError, match=rf"index_selection:layer={selection_owner}"):
            compiled.evaluate(geometry, base)
        with pytest.raises(ModelingError, match=rf"index_selection:layer={selection_owner}"):
            compiled.evaluate(geometry, base.mark_selection(20))
        with pytest.raises(ModelingError, match=rf"index_selection:layer={selection_owner}"):
            evaluate_serial_split(compiled, geometry, base)
        with pytest.raises(ModelingError, match=r"index_cache:layer=20"):
            compiled.evaluate(geometry, StateView().mark_kv(20).mark_selection(selection_owner))

        good = replace(
            base.mark_selection(selection_owner),
            compression=((20, CompressionState.start(1, 8)),))
        direct = compiled.evaluate(geometry, good)
        serial = evaluate_serial_split(compiled, geometry, good)
        assert direct.total_ms > 1.0
        assert serial.total_ms > 1.0
