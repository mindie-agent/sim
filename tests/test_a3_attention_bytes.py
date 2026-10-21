"""Tier 0 tests for the attention memory model in the served graph.

A served attention kernel is fused: QK^T, the softmax and PV execute on chip, so
the score/probability matrix is never written to HBM and K/V are streamed once per
**query tile**, not once per (query, key) pair.  The old accounting charged
``pairs * kv_bytes_per_token`` (KV re-read for every query row) plus a full
probability matrix to QK, PV and the softmax, which made a 4096-token Qwen2.5-7B
prefill ``attention_qk.hbm_bytes = 18.15 GB`` per layer -- memory-bound at
1.2 TB/s for ~15 ms per layer, i.e. a ~3.5x over-prediction of long-prompt TTFT.

These tests pin the physics, not a measured duration: tiled KV traffic, no
probability HBM traffic, unchanged causal FLOPs, and a compute-bound long-prompt
attention term.
"""
from __future__ import annotations

import math

import pytest

from mindie_sim.a3_cost import PredictiveCostModel
from mindie_sim.a3_types import MachineSpec, ModelSpec, RuntimeSpec
from mindie_sim.a3_workload import (
    ATTENTION_QUERY_TILE, IterationPlan, RequestView, infer_iteration,
)

BF16 = 2.0


def machine(**overrides) -> MachineSpec:
    """Declared A3-class part: 20 AIC at 1650 MHz, 1.2 TB/s HBM, one HCCS wire."""
    payload = {
        "name": "attn-9362",
        "hardware_key": "Ascend910_9362",
        "arch_family": "A3",
        "dies_per_package": 2,
        "num_devices": 8,
        "ranks_per_node": 8,
        "device": {
            "hbm_gib": 64,
            "cube_flops_per_s": 20 * 8192 * 1.65e9,
            "vector_ops_per_s": 40 * 128 * 1.65e9,
            "hbm_bw_bytes_per_s": 1.2e12,
            "mte_bw_bytes_per_s": 2.4e12,
            "launch_us": 20.0,
            "l2_bytes": 201326592,
            "aic_count": 20,
            "aiv_count": 40,
        },
        "compute_peaks": {"bf16": 20 * 8192 * 1.65e9},
        "links": [{
            "name": "d2d", "channel": "d2d_intra_node", "capacity_gbps": 196,
            "latency_us": 2.5, "members": list(range(8)),
        }],
    }
    payload.update(overrides)
    return MachineSpec.from_mapping(payload)


def gqa_model(**overrides) -> ModelSpec:
    """Qwen2.5-7B layer geometry: 28 heads / 4 KV heads, head_dim 128.

    One layer by default: these tests are about one attention kernel's bytes.  The
    layer count only enters the whole-network prediction, not the per-op physics.
    """
    payload = {
        "name": "qwen2.5-7b", "architecture": "gqa", "layers": 1,
        "hidden_size": 3584, "num_heads": 28, "num_kv_heads": 4,
        "head_dim": 128, "intermediate_size": 18944, "vocab_size": 152064,
        "activation_dtype": "bf16", "weight_dtype": "bf16",
        "kv_dtype": "bf16", "compute_dtype": "bf16",
    }
    payload.update(overrides)
    return ModelSpec.from_mapping(payload)


def runtime(**overrides) -> RuntimeSpec:
    payload = {
        "max_num_batched_tokens": 8192, "max_num_seqs": 1, "max_model_len": 8192,
        "tp": 1, "ep": 1, "dp": 1, "pp": 1, "graph_mode": "eager",
    }
    payload.update(overrides)
    return RuntimeSpec.from_mapping(payload)


def attention_ops(
    q_len: int = 4096, context: int = 0, *,
    model: ModelSpec | None = None, runtime_spec: RuntimeSpec | None = None,
    phase: str = "prefill",
):
    model = model or gqa_model()
    runtime_spec = runtime_spec or runtime()
    parallel = runtime_spec.parallel
    view = RequestView("r0", q_len, context, True, phase)
    work = infer_iteration(
        IterationPlan((view,)), model, runtime_spec, parallel,
    )
    return [op for op in work.ops if op.family.startswith("attention")]


def expected_bytes(q_len: int, kv_len: int, *, heads: int = 28,
                   kv_heads: int = 4, head_dim: int = 128) -> dict[str, float]:
    """Q read + tiled K/V read + O write, with no probability matrix."""
    tiles = math.ceil(q_len / ATTENTION_QUERY_TILE)
    q_read = q_len * heads * head_dim * BF16
    out_write = q_len * heads * head_dim * BF16
    k_read = tiles * kv_len * kv_heads * head_dim * BF16
    v_read = tiles * kv_len * kv_heads * head_dim * BF16
    return {
        "tiles": tiles, "q_read": q_read, "out_write": out_write,
        "k_read": k_read, "v_read": v_read,
        "kv_read": k_read + v_read,
        "total": q_read + k_read + v_read + out_write,
    }


# ---------------------------------------------------------------------------
# 1. One fused kernel, no probability matrix in HBM
# ---------------------------------------------------------------------------
def test_prefill_attention_is_one_fused_op_without_probability_hbm():
    ops = attention_ops(q_len=4096)
    assert [op.family for op in ops] == ["attention_fused"]
    op = ops[0]
    expected = expected_bytes(4096, 4096)
    assert op.hbm_bytes == pytest.approx(expected["total"], rel=1e-12)
    assert op.mte2_bytes == pytest.approx(
        expected["q_read"] + expected["kv_read"], rel=1e-12,
    )
    assert op.mte3_bytes == pytest.approx(expected["out_write"], rel=1e-12)
    assert op.link_bytes == 0.0
    # The score matrix alone would be q*heads*kv*bf16 = 0.94 GB; the fused kernel
    # never materialises it.
    probability_matrix = 4096 * 28 * 4096 * BF16
    assert op.hbm_bytes < probability_matrix
    assert op.hbm_bytes < 0.4e9, "long-prompt attention must not be an 18 GB HBM pass"
    metadata = dict(op.metadata)
    assert metadata["attention_fused"] is True
    assert metadata["probability_hbm_bytes"] == 0.0
    assert metadata["attention_kv_tiles"] == 32
    assert metadata["q_tile_traffic_bytes"] == pytest.approx(expected["kv_read"])


def test_kv_traffic_scales_with_query_tiles_not_with_query_key_pairs():
    def kv_bytes(q_len: int) -> float:
        op = attention_ops(q_len=q_len)[0]
        return float(dict(op.metadata)["q_tile_traffic_bytes"])

    # ceil(q/q_tile): 128 rows per tile, so the KV volume is linear in the number
    # of tiles and never quadratic in q.
    assert kv_bytes(128) == pytest.approx(1 * 128 * 4 * 256 * BF16)
    assert kv_bytes(256) == pytest.approx(2 * 256 * 4 * 256 * BF16)
    assert kv_bytes(2048) == pytest.approx(16 * 2048 * 4 * 256 * BF16)
    assert kv_bytes(4096) == pytest.approx(32 * 4096 * 4 * 256 * BF16)
    # Doubling q doubles the tiles (and the KV read); the visible-pair count -- the
    # old multiplier -- grows ~4x.
    pairs = {q: q * (q + 1) / 2 for q in (2048, 4096)}
    assert pairs[4096] / pairs[2048] == pytest.approx(4.0, rel=0.01)
    assert kv_bytes(4096) / kv_bytes(2048) == pytest.approx(4.0, rel=1e-12)
    # A single query tile reads the cache once, whatever the tile size.
    for q_len in (1, 2, 127, 128):
        op = attention_ops(q_len=q_len)[0]
        assert dict(op.metadata)["attention_kv_tiles"] == 1
        assert dict(op.metadata)["q_tile_traffic_bytes"] == pytest.approx(
            max(1, q_len) * 4 * 256 * BF16, rel=1e-12,
        )


def test_decode_attention_still_reads_the_kv_cache_once_per_step():
    # q = 1 is one tile: the decode KV read is unchanged by the tiling fix (the
    # old per-pair formula happened to be right at q = 1).
    context = 4096
    op = attention_ops(q_len=1, context=context, phase="decode")[0]
    expected = expected_bytes(1, context + 1)
    assert dict(op.metadata)["attention_kv_tiles"] == 1
    assert op.hbm_bytes == pytest.approx(expected["total"], rel=1e-12)
    assert dict(op.metadata)["q_tile_traffic_bytes"] == pytest.approx(
        (context + 1) * 4 * 256 * BF16, rel=1e-12,
    )


def test_attention_flops_keep_the_causal_pair_count():
    # Memory changed; the work did not: a causal prefill chunk is triangular.
    q_len, heads, head_dim = 4096, 28, 128
    op = attention_ops(q_len=q_len)[0]
    pairs = q_len * (q_len + 1) / 2
    assert op.flops == pytest.approx(4.0 * heads * head_dim * pairs, rel=1e-12)
    # Prior context is counted once per query row.
    contextual = attention_ops(q_len=1024, context=3072)[0]
    assert contextual.flops == pytest.approx(
        4.0 * heads * head_dim * (1024 * 3072 + 1024 * 1025 / 2), rel=1e-12,
    )
    # The softmax's vector work stays inside the fused kernel.
    assert op.vector_ops == pytest.approx(
        q_len * heads * q_len * 5.0, rel=1e-12,
    )


def test_long_prompt_attention_is_compute_bound_not_memory_bound():
    op = attention_ops(q_len=4096)[0]
    cost = PredictiveCostModel(machine(), compute_dtype="bf16").cost(op)
    demands = cost.demand_map()
    capacities = PredictiveCostModel(machine(), compute_dtype="bf16").capacity_map()
    cube_ms = demands["cube"] / capacities["cube"]
    hbm_ms = demands["hbm"] / capacities["hbm"]
    assert cube_ms > hbm_ms, (cube_ms, hbm_ms)
    # ~0.47 ms of cube work per layer at the declared 20-AIC peak, not 15 ms of
    # HBM streaming.
    assert cost.service_ms == pytest.approx(cube_ms, rel=0.9)
    assert cost.estimated_ms < 1.0


# ---------------------------------------------------------------------------
# 2. The declared decomposed lowering keeps the same physics
# ---------------------------------------------------------------------------
def test_decomposed_attention_halves_read_only_their_own_operand():
    ops = attention_ops(
        q_len=4096,
        runtime_spec=runtime(metadata={"attention_decomposition": "qk_softmax_pv"}),
    )
    assert [op.family for op in ops] == [
        "attention_qk", "attention_softmax", "attention_pv",
    ]
    qk, softmax, pv = ops
    expected = expected_bytes(4096, 4096)
    # QK reads Q and K, writes nothing to HBM (scores stay on chip).
    assert qk.hbm_bytes == pytest.approx(
        expected["q_read"] + expected["k_read"], rel=1e-12,
    )
    assert qk.mte3_bytes == 0.0
    # PV reads V and writes O; the probability matrix is never an HBM operand.
    assert pv.hbm_bytes == pytest.approx(
        expected["v_read"] + expected["out_write"], rel=1e-12,
    )
    assert pv.mte2_bytes == pytest.approx(expected["v_read"], rel=1e-12)
    # The softmax is an on-chip vector pass: no HBM, MTE2 or MTE3 traffic.
    assert softmax.hbm_bytes == 0.0
    assert softmax.mte2_bytes == 0.0
    assert softmax.mte3_bytes == 0.0
    assert softmax.vector_ops == pytest.approx(4096 * 28 * 4096 * 5.0, rel=1e-12)
    # Summed, the halves move exactly the fused kernel's bytes.
    assert qk.hbm_bytes + softmax.hbm_bytes + pv.hbm_bytes == pytest.approx(
        expected["total"], rel=1e-12,
    )
    # And the causal FLOPs are the fused op's.
    fused = attention_ops(q_len=4096)[0]
    assert qk.flops + pv.flops == pytest.approx(fused.flops, rel=1e-12)


def test_tensor_parallel_attention_uses_local_heads_and_kv_heads():
    ops = attention_ops(q_len=1024, runtime_spec=runtime(tp=4))
    op = ops[0]
    expected = expected_bytes(1024, 1024, heads=7, kv_heads=1)
    assert op.hbm_bytes == pytest.approx(expected["total"], rel=1e-12)
    assert op.vector_ops == pytest.approx(1024 * 7 * 1024 * 5.0, rel=1e-12)
