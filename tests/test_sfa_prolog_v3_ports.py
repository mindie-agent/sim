"""Sparse SFA C8 MX prolog ports.

[Role] Check the fused prolog publishes the absorbed query, the quantized
  norm, and the selected index slot.
[Invariants] Selection needs the indexer, a positive index top-k, an FP8 or
  INT8 cache, and MX FP8 on the MLA category. Equal lora and head shapes do
  not select. Unfused execution still has the seven semantic kernels. The
  packed cache capacity is not a port or a traffic term. Cube and vector work
  of the seven members is unchanged.
[Timeouts & Locks] Pure CPU. No device or network.
[Side-Effect Policy] No filesystem writes.
"""
from __future__ import annotations

from fractions import Fraction

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.mx_dynamic_quant import producer_contract
from mindie_sim.prediction.piecewise.composition import evaluate_serial_split, lower_fusion
from mindie_sim.prediction.piecewise.geometry import WorkloadGeometry
from mindie_sim.prediction.subgraphs.state import StateView
from mindie_sim.schema.common import dtype_bytes

from test_piecewise_subgraph_model_pipeline import SUPPLY, make_model, pipeline

MX = QuantSpec(
    "sfa_mxfp8", weight_format="fp8_e4m3", activation_format="fp8_e4m3",
    weight_block=(1, 32), scale_format="fp8_e8m0", scale_bytes=1,
    activation_group_size=32, activation_scale_format="fp8_e8m0",
    activation_scale_pack=2, kv_dtype="fp8_e4m3",
)
HEADS = 4
Q_LORA = 128
KV_LORA = 512
NOPE = 128
ROPE = 64
TOKENS = 8
MEMBERS = {"q_down", "q_norm", "q_up", "kv_down", "kv_norm", "rotary", "kv_store"}
CAPACITY = 4080 * 128 * 656


def _plan(**changes):
    plan = dict(
        index=0, attention="mla", ffn="dense", num_heads=HEADS, num_kv_heads=1,
        head_dim=NOPE + ROPE, intermediate_size=128, q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA, qk_nope_head_dim=NOPE, qk_rope_head_dim=ROPE,
        v_head_dim=128, mla_variant="latent_mla", indexer=True, index_topk=32,
    )
    plan.update(changes)
    return plan


def _pipe(plan, *, quant=MX, kv_dtype="fp8_e4m3", source_r64=False):
    model = make_model(
        plans=[plan], hidden_size=256, num_heads=HEADS, num_kv_heads=1,
        head_dim=NOPE + ROPE, quant=quant, kv_dtype=kv_dtype,
    )
    overrides = {"kv_dtype": kv_dtype}
    if source_r64:
        overrides["metadata"] = (("mx_quant_source_revision", "r64"),)
    return pipeline(model, feature_overrides=overrides)


def _geometry():
    return WorkloadGeometry.homogeneous(1, TOKENS, TOKENS, phase="prefill")


def _a1(pipe):
    return pipe.instantiated.find("A1", layer=0, role="attention")


def _lower(pipe, supply=SUPPLY):
    instance = _a1(pipe)
    compiled = pipe._subgraph(instance)
    evaluation = compiled.evaluate(_geometry())
    result = lower_fusion(instance, evaluation.operators, "mla_prolog", supply=supply)
    return instance, compiled, evaluation, result


def test_sfa_c8_mx_publishes_absorbed_ports_and_keeps_internal_work():
    instance, compiled, evaluation, result = _lower(_pipe(_plan()))
    assert "sfa_prolog_v3" in dict(instance.template.contract)
    assert MEMBERS <= set(instance.template.fusions[0].members)
    kept = dict(result.kept)
    scale = int(producer_contract(Q_LORA)["scale_bytes"])
    assert kept == {
        "ql_nope": Fraction(TOKENS * HEADS * KV_LORA * 2),
        "q_pe": Fraction(TOKENS * HEADS * ROPE * 2),
        "q_c": Fraction(TOKENS * Q_LORA),
        "q_c_scale": Fraction(TOKENS * scale),
        "cache_index": Fraction(TOKENS * 8),
    }
    assert "query" not in kept and "kv_cache" not in kept and "q_activation" not in kept
    assert CAPACITY not in set(kept.values())
    work = dict(result.fused.work)
    assert CAPACITY not in set(work.values())
    member_work = {}
    for cost in result.semantic:
        for pipe, value in cost.work:
            member_work[pipe] = member_work.get(pipe, Fraction(0)) + value
    assert work["cube"] == member_work["cube"] > 0
    assert work["vector"] == member_work["vector"] > 0
    unabsorbed = Fraction(TOKENS * HEADS * (NOPE + ROPE) * 2)
    assert dict(result.removed)["q_activation"] == unabsorbed
    store_row = (KV_LORA + ROPE) * dtype_bytes("fp8_e4m3")
    packed = KV_LORA + ROPE * 2 + (KV_LORA // 128) * 4
    visible = kept["ql_nope"] + kept["q_pe"] + kept["q_c"] + kept["q_c_scale"]
    delta = Fraction(TOKENS * (packed - store_row))
    stages = [dict(stage) for stage in result.fused.stage_work]
    assert any(stage.get("mte3") == delta and stage.get("hbm") == delta for stage in stages)
    assert any(stage.get("mte2") == kept["cache_index"] for stage in stages)
    assert any(stage.get("mte3") == visible for stage in stages)
    kv_down = next(cost for cost in result.semantic if cost.name == "kv_down")
    assert work["mte2"] >= dict(kv_down.work)["mte2"]
    state = StateView().mark_index(0).mark_selection(0)
    serial = evaluate_serial_split(compiled, _geometry(), state, fusion=("mla_prolog",))
    names = {cost.name for cost in serial.costs}
    assert "fused:mla_prolog" in names
    assert not (MEMBERS & names)
    bare = {cost.name for cost in evaluation.operators}
    assert MEMBERS <= bare
    q_up = next(cost for cost in evaluation.operators if cost.name == "q_up")
    assert dict(q_up.materialization)["q_activation"] == unabsorbed


def test_same_shape_without_indexer_or_mx_or_c8_keeps_query_and_kv_cache():
    cases = (
        _pipe(_plan(indexer=False, index_topk=0)),
        _pipe(_plan(), kv_dtype="bf16"),
        _pipe(_plan(), quant=None, kv_dtype="fp8_e4m3"),
    )
    for pipe in cases:
        instance, _, _, result = _lower(pipe)
        assert "sfa_prolog_v3" not in dict(instance.template.contract)
        kept = dict(result.kept)
        assert {"query", "key", "value", "kv_cache"} <= set(kept)
        assert kept["kv_cache"] > 0 and kept["query"] > 0
        assert "ql_nope" not in kept
        row = dict(instance.operator("kv_store").theta.matrix)["row_bytes"]
        assert row == (KV_LORA + ROPE) * dtype_bytes(pipe.features.kv_dtype)


def test_r64_primary_selects_fourth_wuk_for_indexer_and_nonindexer():
    base = _plan(num_heads=64, q_lora_rank=2048, qk_nope_head_dim=192,
                 head_dim=256, v_head_dim=512)
    indexer = _pipe(base, source_r64=True)
    nonindexer = _pipe(dict(base, indexer=False, index_topk=0), source_r64=True)
    for pipe, flag in ((indexer, 1), (nonindexer, 0)):
        instance, _, evaluation, result = _lower(pipe)
        native = dict(instance.template.contract).get("mla_prolog_v3")
        assert native is not None
        assert "W_UK=64,192,512" in native
        assert native.endswith(f"query_norm={flag}")
        member_cube = sum((dict(cost.work).get("cube", Fraction(0))
                           for cost in result.semantic), Fraction(0))
        raw = Fraction(2 * TOKENS * 64 * 192 * 512)
        assert dict(result.fused.work)["cube"] - member_cube == raw


def test_wuk_uses_typed_bf16_peak_without_scaling_raw_work():
    from dataclasses import replace
    base = _pipe(_plan(num_heads=64, q_lora_rank=2048, qk_nope_head_dim=192,
                       head_dim=256, v_head_dim=512), source_r64=True)
    fast = replace(SUPPLY, compute_peaks=(("bf16", 1_000_000_000_000.0),
                                          ("fp8_e4m3", 1_000_000_000_000.0)))
    slow = replace(SUPPLY, compute_peaks=(("bf16", 500_000_000_000.0),
                                          ("fp8_e4m3", 1_000_000_000_000.0)))
    r_fast = _lower(base, fast)[3]
    r_slow = _lower(base, slow)[3]
    assert dict(r_fast.fused.work)["cube"] == dict(r_slow.fused.work)["cube"]
    raw = Fraction(2 * TOKENS * 64 * 192 * 512)
    # Only the BF16 W_UK stage changes service; the three FP8 members do not.
    delta = dict(r_slow.fused.times)["cube"] - dict(r_fast.fused.times)["cube"]
    assert delta == raw / Fraction(500_000_000_000.0) - raw / Fraction(1_000_000_000_000.0)
    wuk_1024 = 2 * 1024 * 64 * 192 * 512
    assert wuk_1024 == 12_884_901_888
    assert abs(wuk_1024 / 432_537_600_000 * 1e3 - 29.789091) < 1e-5
    assert 25_769_803_776 + 7_247_757_312 + 68_719_476_736 + wuk_1024 == 114_621_939_712
