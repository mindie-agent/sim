"""L3 model-pipeline and L4 service-span cross-architecture generalization.

[Role]
  Build compact two-layer copies of catalog families that mix latent MLA,
  routed and shared experts, Lightning Indexer, GatedDeltaNet, KDA, and
  Gemma4 windowed shared-KV attention.  Drive ``ModelPipeline.forward`` on
  those copies and check the L4 admission formulas beside the body.
  The copies are deepseek-v4.1-flash, glm-5.1, glm-5.3, kimi-k3,
  qwen3.6-27b, qwen3.8-27b and gemma4-26b-a4b.  Owner ids that point at
  a kept layer are rewritten into the two-layer index space.
[Invariants]
  * Body latency strictly increases with hidden size, FFN width, routed
    expert count, prefill tokens, KV context and batch on these shapes.
  * ``Ascend950DT_9582`` (4.0 TB/s HBM) is strictly faster than
    ``Ascend950PR_9579`` (1.6 TB/s) at the same core count.  Raising
    ``aic_count`` or ``aiv_count``, and scaling cube, vector, hbm, mte1,
    mte2, mte3, fixpipe and link together, strictly shortens the body.
  * ``local_batch * dp`` covers concurrency.  ``ep_batch`` is the expert
    cohort.  Multi-wave prefill returns ``ttft_wave_scale * N = (N+1)/2``.
  * A graph bucket covers the per-rank batch or stays unset.  Host TTFT
    overhead is applied outside ``ModelPipeline.forward``.  F3 row counts
    in the forward match the compiled subgraph on the same geometry.
  * Decode and prefill stay on the declared phase.  EP cohort expansion
    repeats a ragged request vector instead of averaging it.
[Timeouts & Locks]
  Tier 0 in-memory arithmetic.  No device, subprocess, network or locks.
  The module is sized to finish well under 10 seconds.
[Side-Effect Policy]
  Read-only catalog loads and in-memory model copies.  No filesystem writes.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.piecewise.compiler import PipeSupply
from mindie_sim.prediction.piecewise.geometry import (
    RequestGeometry, WorkloadGeometry, clamp_graph_bucket, ep_batch, local_batch,
)
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.piecewise.service_span import (
    chunked_prefill_mean_ttft_ms, service_ttft_overhead_ms, ttft_wave_scale,
)
from mindie_sim.prediction.subgraphs import FeatureConfig
from mindie_sim.prediction.subgraphs.builders_attention import gemma4_model_spec
from mindie_sim.schema.hardware import Hardware

_LOADED: dict = {}
_OWNERS = (
    ("kv_owner_layer_id", "owns_long_kv"),
    ("index_k_owner_layer_id", "owns_index_k"),
    ("topk_owner_layer_id", "owns_topk"),
)
_SOURCES = ("candidate_source_layer_id", "index_source_layer_id")
_CAPS = {
    "q_lora_rank": 128, "kv_lora_rank": 64, "index_heads": 8,
    "index_dim": 32, "index_topk": 64, "expert_latent_size": 256,
}
_PIPES = ("cube", "vector", "hbm", "mte1", "mte2", "mte3", "fixpipe", "link")
_CATALOG = (
    ("deepseek-v4.1-flash", (2, 3)),
    ("glm-5.1", (2, 3)),
    ("glm-5.3", (2, 3)),
    ("kimi-k3", (1, 3)),
    ("qwen3.6-27b", (2, 3)),
    ("qwen3.8-27b", (2, 3)),
)


def _load(key: str):
    model = _LOADED.get(key)
    if model is None:
        model = _LOADED[key] = load_model(key)
    return model


def _supply(sku: str = "Ascend950PR_9579", **hw) -> PipeSupply:
    raw = {"backend": "ascend", "sku": sku, "num_devices": 2, **hw}
    return PipeSupply.from_hardware(Hardware.from_mapping(raw))


def _hbm_bytes_per_s(sku: str) -> float:
    hardware = Hardware.from_mapping({"backend": "ascend", "sku": sku, "num_devices": 1})
    return float(dict(hardware.dies[0].rates)["hbm"])


def _scale_pipes(base: PipeSupply, factor: float) -> PipeSupply:
    rates = tuple((name, float(rate) * (factor if name in _PIPES else 1.0)) for name, rate in base.rates)
    return PipeSupply(
        rates, base.structure_caps, invocations=base.invocations,
        resource_units=base.resource_units, flops_by_dtype=base.flops_by_dtype,
        undeclared_pipes=base.undeclared_pipes,
    )


def _compact(key: str, indices: tuple[int, int], *, hidden: int, intermediate: int, experts: int):
    source = _load(key)
    mapped = {old: new for new, old in enumerate(indices)}
    plans = []
    for new, old in enumerate(indices):
        plan = source.layer_plans[old]
        updates = {"index": new, "native_layer_id": new}
        for field, flag in _OWNERS:
            owner = int(getattr(plan, field))
            if owner in mapped:
                updates[field] = mapped[owner]
                updates[flag] = mapped[owner] == new
            elif owner >= 0:
                updates[field] = new
                updates[flag] = True
        for field in _SOURCES:
            owner = int(getattr(plan, field))
            if owner in mapped:
                updates[field] = mapped[owner]
            elif owner >= 0:
                updates[field] = new
        for field, cap in _CAPS.items():
            value = int(getattr(plan, field) or 0)
            if value > cap:
                updates[field] = cap
        if plan.ffn == "moe":
            updates["moe_intermediate_size"] = intermediate
            updates["num_experts"] = experts
            updates["top_k"] = min(int(plan.top_k or 1), experts, 2)
            if plan.shared_intermediate_size:
                updates["shared_intermediate_size"] = intermediate
        elif plan.intermediate_size:
            updates["intermediate_size"] = intermediate
        plans.append(replace(plan, **updates))
    moe = any(plan.ffn == "moe" for plan in plans)
    top_k = min(int(source.top_k or 1), experts, 2) if moe else 0
    return replace(
        source, name=f"{key}-l2", hidden_size=hidden, intermediate_size=intermediate,
        num_experts=experts if moe else 0, top_k=top_k,
        moe_intermediate_size=intermediate if moe else 0,
        layers=2, body_layers=2, vocab_size=256, layer_plans=tuple(plans),
    )


def _gemma(*, hidden: int, intermediate: int, experts: int):
    return gemma4_model_spec({
        "model_type": "gemma4", "num_hidden_layers": 2, "hidden_size": hidden,
        "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 64,
        "global_head_dim": 128, "num_global_key_value_heads": 2,
        "intermediate_size": intermediate, "vocab_size": 256, "sliding_window": 128,
        "layer_types": ["sliding_attention", "full_attention"], "attention_k_eq_v": True,
        "hidden_act": "gelu_pytorch_tanh", "num_experts": experts, "num_experts_per_tok": 2,
        "moe_intermediate_size": intermediate, "num_shared_experts": 1,
        "shared_expert_intermediate_size": intermediate, "kv_source_layer_ids": [0, 0],
    }, name="gemma4-26b-a4b")


def _families():
    for key, indices in _CATALOG:
        yield key, lambda h, i, e, key=key, indices=indices: _compact(
            key, indices, hidden=h, intermediate=i, experts=e)
    yield "gemma4-26b-a4b", lambda h, i, e: _gemma(hidden=h, intermediate=i, experts=e)


def _pipe(cache: dict, model, supply: PipeSupply):
    key = (id(model), supply.rates, supply.resource_units)
    pipe = cache.get(key)
    if pipe is None:
        features = FeatureConfig.from_model(model, tp=1, ep=1, dp=1)
        pipe = cache[key] = assemble_model(model, supply=supply, features=features)
    return pipe


def _ms(cache: dict, model, supply: PipeSupply, geometry: WorkloadGeometry) -> float:
    return float(_pipe(cache, model, supply).forward(geometry).total_ms)


def _assert_family(name: str, model, pipe) -> None:
    kinds = {item.kind for layer in pipe.instantiated.layers for item in layer.subgraphs}
    plans = model.layer_plans
    assert len(plans) == 2
    if name == "deepseek-v4.1-flash":
        assert plans[0].indexer and plans[0].cache_kind == "window_compressed"
        assert plans[1].kv_owner_layer_id == 0 and not plans[1].owns_long_kv
        assert {"A2", "I0", "I1", "F3", "F4", "K1"} <= kinds
    elif name == "glm-5.1":
        assert plans[0].attention == "mla" and plans[0].indexer and plans[0].ffn == "dense"
        assert plans[1].ffn == "moe" and plans[1].num_shared_experts >= 1 and plans[1].indexer
        assert {"A1", "I0", "I1", "F3", "F4"} <= kinds
    elif name == "glm-5.3":
        assert plans[0].indexer and plans[1].indexer_type == "shared"
        assert plans[1].index_k_owner_layer_id == 0 and plans[1].ffn == "moe"
        assert {"A1", "I0", "I1", "F3"} <= kinds
    elif name == "kimi-k3":
        assert plans[0].attention == "kda" and plans[1].attention == "mla"
        assert all(plan.ffn == "moe" and plan.num_shared_experts >= 1 for plan in plans)
        assert {"A1", "A4", "F3", "F4"} <= kinds
    elif name.startswith("qwen3"):
        assert plans[0].attention == "gdn" and plans[1].attention == "gqa"
        assert {"A0", "A3", "F0"} <= kinds
    else:
        assert plans[0].window == 128 and plans[1].window == 0
        assert plans[1].kv_owner_layer_id == 0 and plans[0].owns_long_kv
        assert {"A0", "F3", "F4", "K0"} <= kinds


def _prefill(q_len: int, batch: int = 1, context: int = 16) -> WorkloadGeometry:
    requests = tuple(RequestGeometry(q_len, context, phase="prefill") for _ in range(batch))
    return WorkloadGeometry(requests, phase="prefill")


def test_l3_width_batch_and_context_are_monotonic():
    cache: dict = {}
    supply = _supply()
    baseline = _prefill(32)
    longer_q = _prefill(128)
    wider_batch = _prefill(32, batch=4, context=0)
    short_ctx = WorkloadGeometry.homogeneous(1, 1, 17, phase="decode")
    long_ctx = WorkloadGeometry.homogeneous(1, 1, 81, phase="decode")
    for name, build in _families():
        small = build(512, 256, 8)
        pipe = _pipe(cache, small, supply)
        _assert_family(name, small, pipe)
        body = float(pipe.forward(baseline).total_ms)
        assert _ms(cache, build(1024, 256, 8), supply, baseline) > body
        assert _ms(cache, build(512, 1024, 8), supply, baseline) > body
        if any(plan.ffn == "moe" for plan in small.layer_plans):
            assert _ms(cache, build(512, 256, 32), supply, baseline) > body
        assert _ms(cache, small, supply, longer_q) > body
        assert _ms(cache, small, supply, wider_batch) > _ms(cache, small, supply, _prefill(32, context=0))
        assert _ms(cache, small, supply, long_ctx) > _ms(cache, small, supply, short_ctx)


def test_l3_hbm_cores_and_pipes_speed_up_every_family():
    assert _hbm_bytes_per_s("Ascend950PR_9579") == pytest.approx(1.6e12)
    assert _hbm_bytes_per_s("Ascend950DT_9582") == pytest.approx(4.0e12)
    aic_lo = _supply(aic_count=4, aiv_count=16)
    aic_hi = _supply(aic_count=16, aiv_count=16)
    aiv_lo = _supply(aic_count=4, aiv_count=4)
    aiv_hi = _supply(aic_count=4, aiv_count=16)
    faster_hbm = _supply(sku="Ascend950DT_9582", aic_count=4, aiv_count=16)
    assert dict(aic_lo.resource_units)["compute_units"] == 4
    assert dict(aic_hi.resource_units)["compute_units"] == 16
    assert dict(aiv_lo.resource_units)["vector"] == 4
    assert dict(aiv_hi.resource_units)["vector"] == 16
    assert faster_hbm.rate("hbm") == pytest.approx(aic_lo.rate("hbm") * 2.5)
    scaled = _scale_pipes(aic_lo, 2.0)
    for name in _PIPES:
        assert scaled.rate(name) == pytest.approx(aic_lo.rate(name) * 2.0)
    geometry = _prefill(64)
    cache: dict = {}
    for _name, build in _families():
        model = build(512, 256, 8)
        body = _ms(cache, model, aic_lo, geometry)
        assert _ms(cache, model, faster_hbm, geometry) < body
        assert _ms(cache, model, aic_hi, geometry) < body
        assert _ms(cache, model, aiv_hi, geometry) < _ms(cache, model, aiv_lo, geometry)
        assert _ms(cache, model, scaled, geometry) < body


def test_l4_cohort_batch_and_wave_conservation():
    for concurrency, dp in ((1, 1), (1, 4), (3, 2), (8, 8), (17, 8), (32, 4)):
        local = local_batch(concurrency, dp)
        assert (local - 1) * dp < concurrency <= local * dp
    assert ep_batch(8, 4, 8) == 8
    assert ep_batch(8, 8, 2) == 2
    assert ep_batch(7, 4, 2) == 4
    assert ep_batch(1, 1, 1) == 1
    assert ep_batch(8, 4, 1) == 2
    for waves, dp in ((2, 1), (3, 2), (4, 4), (8, 8)):
        mean_ttft = chunked_prefill_mean_ttft_ms(
            [8192] * (waves * dp), dp=dp, max_batched_tokens=8192, body_one=lambda tok: 1.0,
        )
        assert mean_ttft == pytest.approx((waves + 1) / 2.0)
        assert mean_ttft / waves == pytest.approx((waves + 1) / (2.0 * waves))
    assert ttft_wave_scale(
        phase="prefill", concurrency=8, dp=4, ep=1, layers=2, graph_buckets=(), prompt_tokens=8191,
    ) == 1.0
    assert ttft_wave_scale(
        phase="prefill", concurrency=8, dp=4, ep=1, layers=2, graph_buckets=(), prompt_tokens=8192,
    ) == 1.0
    assert ttft_wave_scale(
        phase="prefill", concurrency=4, dp=4, ep=1, layers=2, graph_buckets=(), prompt_tokens=9000,
    ) == 1.0
    assert service_ttft_overhead_ms(prompt_tokens=4096, tp=1, ep=1) == 0.0
    assert service_ttft_overhead_ms(prompt_tokens=1024, tp=1, ep=1) == 35.0
    assert service_ttft_overhead_ms(prompt_tokens=1024, tp=8, ep=2) == 145.0
    assert service_ttft_overhead_ms(prompt_tokens=1024, tp=1, ep=2) == 95.0
    assert service_ttft_overhead_ms(prompt_tokens=2048, tp=1, ep=1) == pytest.approx(35.0 * 0.45)
    assert service_ttft_overhead_ms(
        prompt_tokens=2048, tp=1, ep=2, short_ep_host_ms=80.0,
    ) == pytest.approx(80.0 * 0.45)
    assert clamp_graph_bucket(0, 4, 2) == 0
    assert clamp_graph_bucket(3, 5, 1) == 0
    assert clamp_graph_bucket(32, 5, 1) == 32
    assert clamp_graph_bucket(32, 5, 4) == 8
    assert clamp_graph_bucket(16, 3, 4, (1, 2, 4, 8, 16)) == 4
    assert clamp_graph_bucket(16, 9, 4, (1, 2, 4, 8)) == 0

    model = _compact("deepseek-v4.1-flash", (2, 3), hidden=512, intermediate=256, experts=8)
    features = FeatureConfig.from_model(model, tp=1, dp=4, ep=4)
    local, cohort = local_batch(8, 4), ep_batch(8, 4, 4)
    geometry = WorkloadGeometry.homogeneous(local, 1, 32, phase="prefill", cohort_batch=cohort)
    pipe = assemble_model(model, supply=_supply(), features=features)
    prediction = pipe.forward(geometry)
    assert prediction.active.batch == local == 2
    assert prediction.active.cohort_batch == cohort == 8
    expanded = geometry.for_subgraph("F3")
    assert expanded.batch == cohort and expanded.tokens == cohort
    forwarded = next(item for item in prediction.layers[0].subgraphs if item.kind == "F3")
    direct = pipe._subgraph(pipe.instantiated.find("F3", layer=0)).evaluate(geometry)
    assert forwarded.rows[0].cost.get("moe_rows") == direct.operators[0].get("moe_rows")
    body = float(prediction.total_ms)
    overhead = service_ttft_overhead_ms(prompt_tokens=1024, tp=1, ep=4)
    waves = local
    scale = chunked_prefill_mean_ttft_ms(
        [8192] * 8, dp=4, max_batched_tokens=8192, body_one=lambda tok: 1.0,
    ) / waves
    assert overhead == 95.0
    assert scale * waves == pytest.approx((waves + 1) / 2.0)
    assert body * scale + overhead > body
    ragged = WorkloadGeometry(
        (RequestGeometry(1, 15, phase="prefill"), RequestGeometry(3, 13, phase="prefill")),
        cohort_batch=4, phase="prefill",
    )
    lifted = ragged.for_subgraph("F3")
    assert [request.q_len for request in lifted.requests] == [1, 3, 1, 3]
    assert lifted.cohort_batch == 4 and lifted.tokens == 8
