"""Homogeneous proof, replica rebinding and bounded semantic graph cache."""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.prediction.cohort_reuse import (
    prepare_cohort_graphs, prove_homogeneous_cohort, rebind_replica_graph,
)
from mindie_sim.prediction.graph_cache import SemanticGraphCache, semantic_graph_key
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from _support.builders import dense_case


def op(op_id="layer@r0", flops=100.0, *, deps=()):
    return OpSpec(op_id, "gemm", "decode", 0, 0, ("r0",), (8,), (8,), (8,), (8,),
                  flops=flops, hbm_bytes=flops, deps=deps)


def test_homogeneous_proof_and_asymmetry_fallback():
    case = dense_case(dp=2)
    plan = (op("a@r0", 100.0), op("b@r0", 200.0, deps=("a@r0",)))
    proof = prove_homogeneous_cohort(case, {0: plan, 1: plan})
    assert proof.homogeneous is True and proof.template_reused is True
    asymmetric = (op("a@r0", 100.0), op("b@r0", 999.0, deps=("a@r0",)))
    failed = prove_homogeneous_cohort(case, {0: plan, 1: asymmetric})
    assert failed.homogeneous is False
    assert "asymmetric_replica_graph" in failed.reasons
    prepared = prepare_cohort_graphs(case, {0: plan, 1: asymmetric}, reuse="auto")
    assert prepared.template_reused is False
    assert prepared.fallback_reason
    with pytest.raises(ModelingError, match="proof_required"):
        prepare_cohort_graphs(case, {0: plan, 1: asymmetric}, reuse="require")


def test_template_reuse_rebinds_rank_and_dependency_identity():
    case = dense_case(dp=2)
    template = (op("a@r0"), op("b@r0", deps=("a@r0",)))
    rebound = rebind_replica_graph(
        template, replica=1, placement={0: 1}, request_binding={"r0": "r1"},
        state_bindings={})
    assert [item.op_id for item in rebound] == ["a@r1", "b@r1"]
    assert rebound[1].deps == ("a@r1",)
    assert rebound[0].requests == ("r1",)
    prepared = prepare_cohort_graphs(case, {0: template, 1: template}, reuse="auto")
    assert prepared.template_reused is True
    assert set(prepared.graphs) == {0, 1}
    assert prepared.graphs[0] == tuple(template)
    assert prepared.graphs[1][0].op_id == "a@r1"


def test_graph_cache_is_a_bounded_lru_and_skips_oversize_entries():
    cache = SemanticGraphCache(max_bytes=4096, max_entries=2)
    key = lambda name: semantic_graph_key(
        dense_case(), None, scope=name, selection_signature=())
    assert cache.get_or_build(key("one"), lambda: ("a" * 100,)) == ("a" * 100,)
    assert cache.get_or_build(key("one"), lambda: ("miss",)) == ("a" * 100,)
    assert cache.hits == 1 and cache.misses == 1
    cache.get_or_build(key("two"), lambda: ("b" * 100,))
    cache.get_or_build(key("three"), lambda: ("c" * 100,))
    assert cache.entries == 2 and cache.evictions >= 1
    oversized = cache.get_or_build(key("huge"), lambda: ("x" * 10000,))
    assert oversized == ("x" * 10000,)
    assert cache.oversized == 1
    cache.clear()
    assert cache.stats()["entries"] == 0


def test_semantic_key_invalidates_on_frontend_structure():
    case = dense_case()
    base = semantic_graph_key(case, None, scope="body")
    changed = semantic_graph_key(
        replace(case, runtime=replace(case.runtime,
                                      parallel=replace(case.runtime.parallel,
                                                       linear_proj_tp=2))),
        None, scope="body")
    assert base != changed
    assert base == semantic_graph_key(case, None, scope="body")
