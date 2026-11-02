"""Tests and A5 hardware calibration verification for HiSparse / Sparse KV Offload."""
from __future__ import annotations

from copy import deepcopy
import math
import pytest

from mindie_sim import predict
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.sparse_kv_offload import (
    copy_sfa_pool_capacity,
    sparse_kv_offload_memory_budget,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import OpSpec


def _a5_case():
    return parse_case({
        "machine": {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 1},
        "model": {
            "name": "a5-memfabric-bench",
            "layers": 1,
            "hidden_size": 512,
            "num_heads": 8,
            "num_kv_heads": 8,
            "head_dim": 64,
            "intermediate_size": 1024,
            "vocab_size": 1024,
        },
        "workload": {"requests": [{"prompt_tokens": 16, "output_tokens": 1}]},
        "runtime": {},
    })


def _glm_sparse_case(*, context_tokens: int = 8192, offloading: str | dict | None = None, tp: int = 1):
    lowerings = {}
    if offloading is not None:
        lowerings["sparse_kv_offload"] = offloading
    return {
        "machine": {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": tp},
        "model": {
            "name": "glm5-sparse-mla-mini",
            "layers": 2,
            "hidden_size": 1024,
            "num_heads": 16,
            "num_kv_heads": 16,
            "head_dim": 64,
            "intermediate_size": 2048,
            "vocab_size": 4096,
            "attention": "mla",
            "mla": {
                "q_lora_rank": 512,
                "kv_lora_rank": 512,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
            },
            "layer_plans": [
                {
                    "index": 0,
                    "attention": "mla",
                    "ffn": "dense",
                    "architecture": "mla",
                    "num_heads": 16,
                    "num_kv_heads": 16,
                    "head_dim": 64,
                    "intermediate_size": 2048,
                    "q_lora_rank": 512,
                    "kv_lora_rank": 512,
                    "qk_nope_head_dim": 128,
                    "qk_rope_head_dim": 64,
                    "v_head_dim": 128,
                    "mla_variant": "latent_mla",
                    "index_heads": 32,
                    "index_dim": 128,
                    "index_topk": 2048,
                    "indexer": True,
                    "owns_index_k": True,
                    "owns_topk": True,
                    "index_query_per_layer": True,
                    "kv_owner_layer_id": 0,
                    "index_k_owner_layer_id": 0,
                    "topk_owner_layer_id": 0,
                },
                {
                    "index": 1,
                    "attention": "mla",
                    "ffn": "dense",
                    "architecture": "mla",
                    "num_heads": 16,
                    "num_kv_heads": 16,
                    "head_dim": 64,
                    "intermediate_size": 2048,
                    "q_lora_rank": 512,
                    "kv_lora_rank": 512,
                    "qk_nope_head_dim": 128,
                    "qk_rope_head_dim": 64,
                    "v_head_dim": 128,
                    "mla_variant": "latent_mla",
                    "index_heads": 32,
                    "index_dim": 128,
                    "index_topk": 2048,
                    "indexer": False,
                    "owns_index_k": False,
                    "owns_topk": False,
                    "index_query_per_layer": False,
                    "kv_owner_layer_id": 1,
                    "index_k_owner_layer_id": 0,
                    "topk_owner_layer_id": 0,
                },
            ],
        },
        "workload": {"requests": [{"prompt_tokens": context_tokens, "output_tokens": 2}]},
        "runtime": {
            "tp": tp,
            "max_model_len": 32768,
            "max_num_seqs": 16,
            "kv_block_size": 128,
            "kernels": {"lowerings": lowerings} if lowerings else {},
        },
    }


@pytest.mark.parametrize(
    "tokens,observed_us",
    [
        (1, 21.37),
        (32, 20.69),
        (512, 21.54),
        (1024, 24.31),
        (2048, 36.72),
        (4096, 56.81),
    ],
)
def test_a5_memfabric_sparse_copy_d2h_matches_live_ascend950dt_9582_calibration(tokens, observed_us):
    case = _a5_case()
    payload = float(tokens * 1152)
    entries = 2 * tokens
    op = OpSpec(
        "d2h", "kv_offload_d2h", "decode", 0, 0, ("r0",),
        (tokens, 576), (tokens, 576), (tokens, 576), (tokens, 576),
        recipe="memfabric_sparse_copy_d2h", dtype="bf16", source_boundary="kernel",
        hbm_bytes=payload, mte2_bytes=payload, mte3_bytes=payload,
        metadata=(("memfabric_call", {"direction": "d2h", "entries": entries, "payload_bytes": payload}),),
    )
    res = graph_costs(case, (op,), isolated=False)
    pred_us = res["duration_ms"] * 1000.0
    assert pred_us == pytest.approx(observed_us, rel=0.05)


@pytest.mark.parametrize(
    "miss_tokens,observed_us",
    [
        (1, 20.51),
        (32, 20.47),
        (512, 21.38),
        (1024, 27.04),
        (2048, 39.09),
        (4096, 62.09),
        (8192, 113.89),
    ],
)
def test_a5_memfabric_sparse_copy_h2d_matches_live_ascend950dt_9582_calibration(miss_tokens, observed_us):
    case = _a5_case()
    payload = float(miss_tokens * 1152)
    entries = 2 * miss_tokens
    op = OpSpec(
        "h2d", "kv_offload_h2d", "decode", 0, 0, ("r0",),
        (miss_tokens, 576), (miss_tokens, 576), (miss_tokens, 576), (miss_tokens, 576),
        recipe="memfabric_sparse_copy_h2d", dtype="bf16", source_boundary="kernel",
        hbm_bytes=payload, mte2_bytes=payload, mte3_bytes=payload,
        metadata=(("memfabric_call", {"direction": "h2d", "entries": entries, "payload_bytes": payload}),),
    )
    res = graph_costs(case, (op,), isolated=False)
    pred_us = res["duration_ms"] * 1000.0
    assert pred_us == pytest.approx(observed_us, rel=0.04)


@pytest.mark.parametrize(
    "direction,pages,observed_us",
    [
        ("h2d", 1, 20.55),
        ("h2d", 8, 23.23),
        ("h2d", 16, 36.69),
        ("h2d", 64, 113.97),
        ("h2d", 128, 224.57),
        ("d2h", 1, 20.50),
        ("d2h", 8, 20.81),
        ("d2h", 16, 30.90),
        ("d2h", 64, 99.07),
        ("d2h", 128, 198.09),
    ],
)
def test_a5_memfabric_kv_exchange_uses_last_2_aivs_and_matches_calibration(direction, pages, observed_us):
    case = _a5_case()
    payload = float(pages * 128 * 1152)
    op = OpSpec(
        "kv_ex", "kv_offload_exchange", "prefill", 0, 0, ("r0",),
        (pages, 128, 576), (pages, 128, 576), (pages, 128, 576), (pages, 128, 576),
        recipe="memfabric_kv_exchange", dtype="bf16", source_boundary="kernel",
        hbm_bytes=payload, mte2_bytes=payload, mte3_bytes=payload,
        metadata=(("memfabric_call", {"direction": direction, "pages": pages, "payload_bytes": payload}),),
    )
    res = graph_costs(case, (op,), isolated=True, trace_limit=64)
    pred_us = res["duration_ms"] * 1000.0
    assert pred_us == pytest.approx(observed_us, rel=0.05)
    exec_meta = dict(res["isolated"]["operators"][0]["operator"]["metadata"])["program_execution"]
    # On Ascend950DT_9582 (64 AIVs), only AIV 62 and 63 execute kv_exchange_copy.
    assert tuple(exec_meta["active_aiv_owners"]) == (62, 63)


@pytest.mark.parametrize(
    "rows,observed_us",
    [
        (64, 21.10),
        (512, 21.61),
        (1024, 24.78),
        (2048, 35.52),
        (4096, 51.17),
        (8192, 93.58),
    ],
)
def test_a5_memfabric_entry_gather_matches_live_ascend950dt_9582_calibration(rows, observed_us):
    case = _a5_case()
    payload = float(rows * 1024)
    op = OpSpec(
        "eg", "kv_offload_gather", "decode", 0, 0, ("r0",),
        (rows, 512), (rows, 512), (rows, 512), (rows, 512),
        recipe="memfabric_entry_gather", dtype="bf16", source_boundary="kernel",
        hbm_bytes=payload, mte2_bytes=payload, mte3_bytes=payload,
        metadata=(("memfabric_call", {"rows": rows, "payload_bytes": payload}),),
    )
    res = graph_costs(case, (op,), isolated=True)
    pred_us = res["duration_ms"] * 1000.0
    assert pred_us == pytest.approx(observed_us, rel=0.06)
    exec_meta = dict(res["isolated"]["operators"][0]["operator"]["metadata"])["program_execution"]
    assert exec_meta["ub_bytes_per_owner"] == 122880


def test_glm5_full_vs_shared_indexer_layers_and_short_vs_long_decode_regimes():
    assert copy_sfa_pool_capacity(16) == 16
    assert copy_sfa_pool_capacity(17) == 32

    # 1. Short sequence (context=1024 < topk_buffer_size=2048 -> request_state=-3):
    #    No LIM or H2D miss copy is emitted; only D2H writeback of the new token.
    short_raw = _glm_sparse_case(
        context_tokens=1024,
        offloading={"implementation": "hisparse_kv_offload", "options": {"miss_rate": 0.2}},
    )
    short_report = predict(short_raw, scope="body", phase="decode", tokens=1, context=1024, sequences=[1])
    short_recipes = [op["recipe"] for op in short_report["timeline"]]
    assert short_recipes.count("memfabric_sparse_copy_d2h") == 2
    assert "memfabric_sparse_copy_h2d" not in short_recipes
    assert "fused_lightning_indexer_manage" not in short_recipes

    # 2. Long sequence (context=8192 >= topk_buffer_size=2048 -> request_state=-2):
    #    - Layer 0 ("full" Indexer layer) runs Indexer + LIM (skip_topk=False) + H2D miss gather.
    #    - Layer 1 ("shared" Indexer layer) skips Indexer AND skips LIM (skip_topk=True,
    #      shared_indexer_reuse=True), reusing Layer 0's topk and miss plan!
    long_raw = _glm_sparse_case(
        context_tokens=8192,
        offloading={"implementation": "vllm_ascend_sparse_kv_offload", "options": {"miss_rate": 0.2}},
    )
    long_report = predict(long_raw, scope="body", phase="decode", tokens=1, context=8192, sequences=[1])
    long_recipes = [op["recipe"] for op in long_report["timeline"]]
    assert long_recipes.count("memfabric_sparse_copy_d2h") == 2
    assert long_recipes.count("fused_lightning_indexer_manage") == 1
    assert long_recipes.count("memfabric_sparse_copy_h2d") == 2

    long_ops = target_graphs(
        parse_case(long_raw), phase="decode", tokens=1, context=8192, sequences=[1], scope="body"
    )[0]
    h2d_ops = [op for op in long_ops if op.recipe == "memfabric_sparse_copy_h2d"]
    l0_meta = dict(h2d_ops[0].metadata)
    l1_meta = dict(h2d_ops[1].metadata)
    expected_miss = math.ceil(2048 * 0.2)
    assert l0_meta["memfabric_call"]["tokens"] == expected_miss
    assert l0_meta["shared_indexer_reuse"] is False
    assert l1_meta["shared_indexer_reuse"] is True
    assert l1_meta["skip_topk"] is True
    assert l0_meta["hardware_support"]["memfabric_acc_offload"] == "native_verified_on_a5"
    assert l0_meta["hardware_support"]["copy_sfa_attention"] == "a2_a3_aclnn_binary_analytical_on_a5"


def test_sparse_kv_offload_memory_budget_and_host_dram_capacity_enforcement():
    base_raw = _glm_sparse_case(context_tokens=4096, offloading=None)
    offload_raw = _glm_sparse_case(
        context_tokens=4096,
        offloading={
            "implementation": "vllm_ascend_sparse_kv_offload",
            "options": {"dram_size_per_dp_gb": 0.02, "keep_device_kv_cache": False},
        },
    )
    base_mem = Memory(parse_case(base_raw))
    offload_mem = Memory(parse_case(offload_raw))

    # Without offload, 4096 tokens allocate both mla_kv (2 layers * 1152B) and index_k (1 layer * 130B FP8+scale) in NPU HBM.
    # With offload (keep_device_kv_cache=False), per-request NPU HBM only allocates index_k (1 layer * 130B)!
    npu_base_4k = sum(base_mem.cache_bytes(4096))
    npu_offload_4k = sum(offload_mem.cache_bytes(4096))
    assert npu_offload_4k == 4096 * 130
    assert npu_base_4k == 4096 * (2 * 1152 + 130)
    assert offload_mem.host_dram_cache_bytes(4096) == 4096 * 2 * 1152

    # With dram_size_per_dp_gb=0.02 (~20.48 MiB) minus 12 MiB alignment reserve (6 MiB * 2 layers),
    # usable DRAM is ~8.48 MiB (~4.24 MiB/layer = ~27 blocks = 3456 tokens), so 4096 tokens exceeds Host DRAM!
    assert offload_mem.resize(0, "req_fits", 2048) is True
    assert offload_mem.resize(0, "req_overflows_dram", 4096) is False
    summary = offload_mem.to_dict()["sparse_kv_offload"]
    assert summary["limiting_factor"] == "host_dram"
    assert summary["peak_host_dram_bytes_per_replica"][0] == 2048 * 2 * 1152


def test_prefill_layerwise_offload_and_pd_sfa_remote_d2h_connector_handoff():
    prefill_raw = _glm_sparse_case(
        context_tokens=1024,
        offloading={
            "implementation": "vllm_ascend_sparse_kv_offload",
            "options": {"layerwise_kv_offload": True, "num_shared_layers": 1},
        },
    )
    prefill_report = predict(prefill_raw, scope="body", phase="prefill", tokens=1024, context=0, sequences=[1024])
    kv_ex_ops = [op for op in prefill_report["timeline"] if op["recipe"] == "memfabric_kv_exchange"]
    assert len(kv_ex_ops) == 2
    assert all(op["stream"] == "offload" for op in kv_ex_ops)

    # Disaggregated P/D with tp=4: SfaRemoteD2HConnector sends mla_kv 1x (RD2H to shared Host DRAM)
    # instead of 4x across TP ranks when use_fused_copy_sfa=False (use_fused_overlap=True).
    base_pd = _glm_sparse_case(context_tokens=1024, offloading=None, tp=4)
    base_pd["pools"] = {
        "prefill": {"machine": deepcopy(base_pd["machine"]), "runtime": deepcopy(base_pd["runtime"])},
        "decode": {"machine": deepcopy(base_pd["machine"]), "runtime": deepcopy(base_pd["runtime"])},
    }
    base_pd["connector"] = {"bytes_per_s": 50e9, "latency_us": 5.0}

    offload_pd = deepcopy(base_pd)
    offload_pd["pools"]["decode"]["runtime"]["kernels"] = {
        "lowerings": {
            "sparse_kv_offload": {
                "implementation": "vllm_ascend_sparse_kv_offload",
                "options": {"use_fused_overlap": True, "dram_size_per_dp_gb": 64.0},
            }
        }
    }
    res_base = predict(base_pd)
    res_offload = predict(offload_pd)
    assert res_base["status"] == "modeled"
    assert res_offload["status"] == "modeled"
    # Base handoff replicates both mla_kv and index_k 4x (tp=4): 4 * 1024 * (2*1152 + 130) = 9,969,664 B
    # Offload handoff replicates index_k 4x (D2D) + mla_kv 1x (RD2H to shared Host DRAM):
    #   4 * 1024 * 130 + 1 * 1024 * 2 * 1152 = 2,891,776 B!
    assert res_base["execution"]["handoff_bytes"] == 4 * 1024 * (2 * 1152 + 130)
    assert res_offload["execution"]["handoff_bytes"] == 4 * 1024 * 130 + 1 * 1024 * (2 * 1152)
    assert res_offload["execution"]["handoff_bytes"] < res_base["execution"]["handoff_bytes"]
