"""Physical shared-prefix capacity union, COW tails and DP replica ownership."""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.catalog.cache_ownership import (
    SHARING_PRIVATE, batch_cache_footprint, cache_component_ranges,
    draft_cache_components,
)
from mindie_sim.schema.case import Software
from mindie_sim.schema.parallel import ParallelSpec
from issue34_helpers import dense_case, dense_model, linear_model, view


def shared_case(*, dp=1):
    case = dense_case(dp=dp, layers=2)
    return replace(case, runtime=replace(
        case.runtime, prefix_cache_mode="shared"))


def test_dense_physical_union_counts_shared_prefix_once_per_replica():
    case = shared_case()
    requests = (view("a", 1, 200, prefix=128), view("b", 1, 200, prefix=128))
    footprint = batch_cache_footprint(
        case, requests=requests, replica=0, phase="decode", padded=False)
    row = 2 * 2 * 16 * 2
    assert footprint.shared_once_total == pytest.approx(2 * 128 * row)
    assert footprint.private_total == pytest.approx(2 * 2 * (201 - 128) * row)
    assert footprint.total == pytest.approx(
        footprint.shared_once_total + footprint.private_total)
    duplicate = batch_cache_footprint(
        case, requests=requests, replica=1, phase="decode", padded=False)
    assert duplicate.total == footprint.total


def test_private_mode_and_p0_identity():
    case = dense_case(layers=2)
    request = view("a", 1, 200)
    footprint = batch_cache_footprint(
        case, requests=(request,), replica=0, phase="decode", padded=False)
    expected = sum(sum(value for value in layer_cache_components(
        case.model, layer, case.runtime.parallel, request.kv_len,
        capacity_padded=False).values()) for layer in case.model.layer_plans)
    assert footprint.shared_once_total == 0
    assert footprint.total == pytest.approx(expected)


def test_partial_prefix_block_becomes_a_private_cow_tail():
    case = shared_case()
    aligned = cache_component_ranges(
        case.model, case.model.layer_plans[0], case.runtime.parallel,
        sequence_tokens=201, prefix_tokens=128, interleave=128, rank=None,
        capacity_padded=False)
    partial = cache_component_ranges(
        case.model, case.model.layer_plans[0], case.runtime.parallel,
        sequence_tokens=201, prefix_tokens=129, interleave=128, rank=None,
        capacity_padded=False)
    shared_a = [item for item in aligned if item.sharing != SHARING_PRIVATE]
    shared_b = [item for item in partial if item.sharing != SHARING_PRIVATE]
    assert shared_a[0].valid_bytes == shared_b[0].valid_bytes == 128 * 128
    private_a = sum(item.valid_bytes for item in aligned
                    if item.sharing == SHARING_PRIVATE)
    private_b = sum(item.valid_bytes for item in partial
                    if item.sharing == SHARING_PRIVATE)
    assert private_b == pytest.approx(private_a)


def test_different_keys_never_pool_and_window_keeps_only_live_rows():
    case = shared_case()
    different = batch_cache_footprint(
        case, requests=(view("a", 1, 200, prefix=128),
                        replace(view("b", 1, 200, prefix=128),
                                prefix_cache_key="other")),
        replica=0, phase="decode", padded=False)
    same = batch_cache_footprint(
        case, requests=(view("a", 1, 200, prefix=128),
                        view("b", 1, 200, prefix=128)),
        replica=0, phase="decode", padded=False)
    assert different.shared_once_total == pytest.approx(2 * same.shared_once_total)
    windowed = replace(dense_model(layers=2),
                       layer_plans=tuple(
                           replace(layer, window=64)
                           for layer in dense_model(layers=2).layer_plans))
    case = replace(case, model=windowed)
    footprint = batch_cache_footprint(
        case, requests=(view("a", 1, 400, prefix=256),),
        replica=0, phase="decode", padded=False)
    assert footprint.private_total <= 2 * 64 * 64 * 2


def test_linear_state_stays_private_and_draft_geometry_is_inherited():
    case = dense_case(layers=1)
    case = replace(case, model=linear_model("kda", layers=1))
    runtime = replace(case.runtime, recover_ssm=True,
                      metadata=(("draft_proposal_tokens", 5),))
    footprint = batch_cache_footprint(
        replace(case, runtime=runtime),
        requests=(view("a", 1, 200, prefix=128),),
        replica=0, phase="decode", padded=False)
    assert footprint.state_total > 0
    assert footprint.shared_once_total == 0
    from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
    draft = DraftSpec(kind="mtp", layers=1, hidden_size=32, intermediate_size=64,
                      num_heads=4, num_kv_heads=2, head_dim=16, q_width=64,
                      kv_width=32, vocab_size=128, attention_kind="gqa",
                      kv_cache_elements=32, target_shared_embedding=True,
                      target_shared_lm_head=True)
    model = replace(dense_model(layers=1), auxiliary=AuxiliarySpec(mtp=draft))
    draft_components = draft_cache_components(
        model, ParallelSpec(tp=1), draft_kind="mtp", sequence_tokens=8)
    assert draft_components["draft_kv"] > 0
    assert any(name.startswith("draft.") for name in draft_components)
