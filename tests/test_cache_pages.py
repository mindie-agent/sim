"""Independent allocation, valid-payload and admission counterexamples."""
from dataclasses import replace

import pytest

from mindie_sim import Case, predict
from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.catalog.cache_pages import CachePageGroup
from mindie_sim.prediction.capacity import Memory
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from test_prediction_deployment import pd_case
from test_prediction_system import case, linear_work


def page_case():
    raw = case(prompt=4, output=1)
    raw["runtime"].update(kv_block_size=8, gpu_memory_utilization=1, memory={
        "derive_workspace": False,
        "cache_page_groups": [
            {"layers": [0], "tokens_per_page": 3, "bytes_per_layer_page": 192},
            {"layers": [1], "tokens_per_page": 5, "bytes_per_layer_page": 320},
        ],
    })
    return raw


def test_independent_page_sizes_drive_real_admission_without_global_rounding():
    raw = page_case()
    weights = Memory(Case.from_mapping(raw)).weights[0]
    # Four valid rows allocate two 3-row pages and one 5-row page.
    raw["machine"]["device"]["memory_bytes"] = weights + 2 * 192 + 320
    memory = Memory(Case.from_mapping(raw))
    assert memory.cache_bytes(4) == (704,)
    assert memory.cache_bytes(4, padded=False) == (2 * 4 * 64,)
    assert memory.resize(0, "r", 4)
    assert not memory.resize(0, "r", 7)
    assert memory.live == [[704]]
    memory.release(0, "r")
    assert memory.live == [[0]]
    assert predict(raw, lower=linear_work)["status"] == "modeled"
    raw["runtime"]["kv_block_size"] = 16
    assert Memory(Case.from_mapping(raw)).cache_bytes(4) == (704,)


def test_pd_moves_valid_tensors_even_when_both_allocators_have_different_pages():
    raw = pd_case()
    for pool in raw["pools"].values():
        pool["runtime"].update(page_case()["runtime"])
    raw["pools"]["decode"]["runtime"]["memory"]["cache_page_groups"] = [
        {"layers": [0, 1], "tokens_per_page": 7, "bytes_per_layer_page": 448}]
    result = predict(raw, lower=linear_work)
    # 3 committed tokens * 2 layers * 64 bytes, independent of 3/5/7-row pages.
    assert result["execution"]["handoff_bytes"] == 384
    assert result["execution"]["serviced"]["prefill:memory@r0:s0"] == 384
    assert result["execution"]["serviced"]["decode:memory@r0:s0"] == 384
    assert result["requests"][0]["finish_ms"] == 8


def compressed_case():
    parsed = Case.from_mapping(case())
    owner = replace(
        parsed.model.layer_plans[0], attention="mla", cache_kind="window_compressed", window=5,
        compress_ratio=4, kv_lora_rank=8, head_dim=8, compressor_width=1,
        compressor_state_rows=2, owns_long_kv=True, kv_owner_layer_id=0,
        q_lora_rank=8, qk_nope_head_dim=8, mla_variant="latent_mqa",
    )
    return replace(parsed, model=replace(parsed.model, layer_plans=(
        owner, replace(owner, index=1, owns_long_kv=False))))


def test_component_groups_share_pages_without_erasing_other_cache_components():
    parsed = compressed_case()
    groups = [
        {"layers": [0, 1], "components": ["window"], "tokens_per_page": 3,
         "bytes_per_group_page": 96, "resident_tokens": 5},
        {"layers": [0], "components": ["compressor_state"], "fixed_pages": 1,
         "bytes_per_group_page": 160},
    ]
    parsed = replace(parsed, runtime=replace(
        parsed.runtime, kv_block_size=8, memory={
            "derive_workspace": False, "cache_page_groups": groups}))
    memory = Memory(parsed)
    # One shared two-layer SWA group, one state ring, one owner's C4 payload.
    assert memory.cache_bytes(9) == (2 * 96 + 160 + 2 * 16,)
    assert memory.cache_bytes(9, padded=False) == (2 * 5 * 16 + 2 * 16 + 2 * 2 * 8 * 4,)
    # Before C4 completes its first row, no long-cache page exists.
    assert memory.cache_bytes(1) == (96 + 160,)
    assert memory.cache_bytes(1, padded=False) == (2 * 16 + 2 * 8 * 4,)


def test_component_rounding_happens_after_window_and_compression_geometry():
    parsed = compressed_case()
    owner = parsed.model.layer_plans[0]
    one = layer_cache_components(parsed.model, owner, ParallelSpec(), 1, interleave=8)
    nine = layer_cache_components(parsed.model, owner, ParallelSpec(), 9, interleave=8)
    assert one["window"] == 8 * 16
    assert one["compressed_kv"] == 0
    assert nine["window"] == 8 * 16
    assert nine["compressed_kv"] == 2 * 16
    assert one["compressor_state"] == nine["compressor_state"] == 2 * 2 * 8 * 4
    payload = layer_cache_components(
        parsed.model, owner, ParallelSpec(), 9, capacity_padded=False, interleave=8)
    assert payload["window"] == 5 * 16
    assert payload["compressed_kv"] == 2 * 16


def test_sparse_page_group_can_mix_producer_components_across_layers():
    parsed = compressed_case()
    owner, consumer = parsed.model.layer_plans
    model = replace(
        parsed.model, quant=replace(parsed.model.quant, index_dtype="int8", index_scale_dtype="fp16"),
        layer_plans=(
            replace(owner, owns_index_k=True, index_dim=4, index_heads=2, index_input="compressed"),
            replace(consumer, owns_long_kv=True, kv_owner_layer_id=1)))
    runtime = replace(parsed.runtime, kv_block_size=8, memory={
        "derive_workspace": False,
        "cache_page_groups": [{
            "members": [
                {"layer": 0, "component": "compressed_kv"},
                {"layer": 0, "component": "index_k"},
                {"layer": 1, "component": "compressed_kv"},
            ],
            "tokens_per_page": 8, "bytes_per_group_page": 100,
        }],
    })
    memory = Memory(replace(parsed, model=model, runtime=runtime))
    # Two common pages cover both long caches and one index plane. Windows
    # and compressor rings retain their own allocations on both producers.
    assert memory.cache_bytes(9) == (2 * 100 + 2 * 8 * 16 + 2 * 2 * 2 * 8 * 4,)
    assert memory.cache_bytes(9, padded=False) == (
        2 * 5 * 16 + 2 * 2 * 16 + 2 * 2 * 2 * 8 * 4 + 2 * (4 + 2),)


def test_window_flight_span_and_recycling_pages_preserve_historical_geometry():
    group = CachePageGroup(
        layers=(0,), tokens_per_page=128, bytes_per_group_page=462848,
        resident_tokens=128, max_in_flight_tokens=8192, recycling_pages=1,
        allocation="max_model_len",
    )
    # The old engine dump's 66/129 pages come from declared allocator geometry.
    assert group.pages(1, 16384) == (128 + 8192) // 128 + 1 == 66
    assert replace(group, max_in_flight_tokens=16384).pages(1, 16384) == 129
    assert replace(group, fixed_pages=1).pages(16384, 16384) == 1
    assert group.pages(0, 16384) == 0
    # Live allocation grows only as the owned request needs pages.
    assert replace(group, allocation="live", max_in_flight_tokens=0).pages(1, 16384) == 2


def test_allocator_ownership_cannot_double_charge_a_component_or_share_across_stages():
    raw = page_case()
    raw["runtime"]["memory"]["cache_page_groups"].append(
        {"layers": [0], "tokens_per_page": 3, "bytes_per_layer_page": 192})
    with pytest.raises(ModelingError, match="duplicate_owner"):
        Memory(Case.from_mapping(raw))
    raw = case(pp=2)
    raw["runtime"]["memory"] = {"cache_page_groups": [
        {"layers": [0, 1], "tokens_per_page": 4, "bytes_per_group_page": 512}]}
    with pytest.raises(ModelingError, match="spans_pipeline_stages"):
        Memory(Case.from_mapping(raw))
    # Independent per-layer allocations can span stages without sharing bytes.
    raw["runtime"]["memory"]["cache_page_groups"] = [
        {"layers": [0, 1], "tokens_per_page": 4, "bytes_per_layer_page": 256}]
    assert Memory(Case.from_mapping(raw)).cache_bytes(3) == (256, 256)
