"""Structural, arithmetic and ownership tests for the nine catalog models.

These tests assert actual tensor geometry, packed bytes, cache/state ownership,
routing/communication widths, vision/media expansion, prepare eligibility and
public predict/should behavior.  They intentionally do not assert only model
names, enum strings or total layer counts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim.a3_catalog_cache import (
    attention_rows, candidate_valid_rows, completed_rows, index_selection_cost,
    new_completed_rows, request_cache_breakdown, window_rows,
)
from mindie_sim.a3_catalog_media import lower_media
from mindie_sim.a3_catalog_prepare import catalog_prepare_candidates
from mindie_sim.a3_catalog_weights import (
    body_head_terms, draft_terms, engram_weight_terms, layer_weight_terms,
    stage_weight_bytes, term_bytes, vision_weight_terms,
)
from mindie_sim.a3_memory import memory_plan, workload_lifetime_tokens, workload_required_tokens
from mindie_sim.a3_types import (
    A3InputError, MachineSpec, ParallelSpec, RequestSpec, RuntimeSpec,
    WorkloadSpec,
)
from mindie_sim.a3_workload import IterationPlan, RequestView, infer_iteration
from mindie_sim.model_catalog import list_models, load_model

NINE = [
    "deepseek-v4-flash-0731", "deepseek-v4-pro-0813", "deepseek-v4-10t",
    "deepseek-v4.1-flash", "glm-5.3", "kimi-k3", "kimi-k3-dspark",
    "qwen3.8-2.4t-a95b", "qwen3.8-27b",
]
BODY_COUNTS = {
    "deepseek-v4-flash-0731": 43, "deepseek-v4-pro-0813": 61,
    "deepseek-v4-10t": 61, "deepseek-v4.1-flash": 40, "glm-5.3": 78,
    "kimi-k3": 93, "kimi-k3-dspark": 5, "qwen3.8-2.4t-a95b": 92,
    "qwen3.8-27b": 64,
}
FIXTURES = Path("/tmp/mindie-a3-dsh.DNIfZY/se-tools-fixtures")


def _runtime(tp=1, ep=1, dp=1, pp=1, **extra):
    payload = {
        "tp": tp, "ep": ep, "dp": dp, "pp": pp,
        "max_model_len": 512, "max_num_batched_tokens": 128,
        "max_num_seqs": 16, "graph_mode": "eager",
    }
    payload.update(extra)
    return RuntimeSpec.from_mapping(payload)


def _views(*specs):
    return IterationPlan(tuple(
        RequestView(f"r{i}", q, c, s, phase)
        for i, (q, c, s, phase) in enumerate(specs)
    ))


def _machine(devices=8):
    return MachineSpec.from_mapping({
        "name": f"a3-{devices}", "arch_family": "A3", "dies_per_package": 2,
        "num_devices": devices, "ranks_per_node": 8,
        "device": {
            "hbm_gib": 64, "cube_flops_per_s": 2.70336e14,
            "vector_ops_per_s": 1.6896e13, "hbm_bw_bytes_per_s": 1.2e12,
            "mte_bw_bytes_per_s": 2.4e12,
        },
    })


def _op_map(work):
    return {op.op_id: op for op in work.ops}


def _ancestors(ops, op_id):
    seen = set()
    stack = list(ops[op_id].deps)
    while stack:
        dep = stack.pop()
        if dep in seen:
            continue
        seen.add(dep)
        if dep in ops:
            stack.extend(ops[dep].deps)
    return seen


def test_body_schedules_match_independent_fixtures():
    for key in NINE:
        model = load_model(key)
        assert model.layers == BODY_COUNTS[key]
        assert len(model.layer_plans) == BODY_COUNTS[key]
        assert all(plan.scope == "body" for plan in model.layer_plans)
        assert all(not plan.mtp for plan in model.layer_plans)
        assert model.parameter_count() > 0
        assert model.resident_weight_bytes_override > 0
        if FIXTURES.exists():
            fixture = json.loads((FIXTURES / f"{key}.json").read_text())
            expected = fixture["expected"]
            assert [p.index for p in model.layer_plans] == list(range(BODY_COUNTS[key]))
            kinds = {}
            for plan in model.layer_plans:
                kinds[plan.architecture] = kinds.get(plan.architecture, 0) + 1
            ffn = {}
            for plan in model.layer_plans:
                ffn[plan.ffn] = ffn.get(plan.ffn, 0) + 1
            # Fixture attention labels are family aliases; compare the exact
            # per-family counts where names are directly comparable.
            if key == "kimi-k3":
                assert kinds.get("kda") == 69 and kinds.get("mla") == 24
                assert ffn == {"dense": 1, "moe": 92}
            elif key == "glm-5.3":
                assert ffn == {"dense": 3, "moe": 75}
                assert sum(1 for p in model.layer_plans if p.indexer) == 21
            elif key == "qwen3.8-2.4t-a95b":
                assert kinds == {"gdn": 69, "gated_gqa": 23}
            elif key == "qwen3.8-27b":
                assert kinds == {"gdn": 48, "gated_gqa": 16}
                assert all(p.ffn == "dense" for p in model.layer_plans)


def test_kimi_one_based_layers_final_mla_and_state():
    model = load_model("kimi-k3")
    assert model.layer_plans[3].native_layer_id == 4
    assert model.layer_plans[3].attention == "mla"
    assert model.layer_plans[92].native_layer_id == 93
    assert model.layer_plans[92].attention == "mla"
    assert model.layer_plans[0].attention == "kda"
    kda = model.layer_plans[0]
    assert kda.recurrent_state_elements == 96 * 128 * 128
    assert kda.recurrent_state_dtype == "fp32"
    assert kda.conv_state_elements == (3 * 96 * 128) * 3
    assert kda.conv_state_dtype == "bf16"
    assert kda.state_bytes_total == pytest.approx(96 * 128 * 128 * 4 + 36864 * 3 * 2)
    # TP8 state is exactly 54,263,808 bytes per request across 69 KDA layers.
    work = infer_iteration(
        _views((1, 128, False, "decode")), model, _runtime(tp=8, ep=8),
        ParallelSpec(tp=8, ep=8),
    )
    states = [op for op in work.ops if op.family == "recurrent_state"]
    assert len(states) == 69
    assert states[0].hbm_bytes == pytest.approx(54263808 / 69 * 2)
    # Two independent requests each own a state endpoint.
    work2 = infer_iteration(
        _views((1, 128, False, "decode"), (1, 128, False, "decode")),
        model, _runtime(tp=8, ep=8), ParallelSpec(tp=8, ep=8),
    )
    state2 = next(op for op in work2.ops if op.family == "recurrent_state")
    assert state2.hbm_bytes == pytest.approx(2 * 54263808 / 69 * 2)


def test_kimi_latent_moe_router_and_dispatch_width():
    model = load_model("kimi-k3")
    work = infer_iteration(
        _views((4, 0, True, "prefill")), model, _runtime(tp=8, ep=8),
        ParallelSpec(tp=8, ep=8),
    )
    ops = _op_map(work)
    router = next(op for op in work.ops if op.op_id.endswith("router") and ".l1." in op.op_id)
    assert router.launch_shape[-2] == 896
    assert router.launch_shape[-1] == 7168
    dispatch = next(op for op in work.ops if op.op_id.endswith("ep_dispatch") and ".l1." in op.op_id)
    assert dict(dispatch.metadata)["payload_width"] == 3584
    assert dispatch.semantic_shape[-1] == 3584
    gate = next(op for op in work.ops if op.op_id.endswith("expert_gate_up0") and ".l1." in op.op_id)
    assert gate.launch_shape[-1] == 3584
    assert gate.launch_shape[-2] == 2 * 3072  # EP8 => ETP1
    ep1 = infer_iteration(
        _views((1, 128, False, "decode")), model, _runtime(tp=8, ep=1),
        ParallelSpec(tp=8, ep=1),
    )
    gate_etp8 = next(op for op in ep1.ops
                     if op.op_id.endswith("expert_gate_up0") and ".l1." in op.op_id)
    assert gate_etp8.launch_shape[-2] == 2 * (3072 // 8)
    shared = next(op for op in work.ops if op.op_id.endswith("shared_gate_up") and ".l1." in op.op_id)
    assert shared.launch_shape[-1] == 7168
    # Routing control survives the latent transform.
    anc = _ancestors(ops, gate.op_id)
    assert any("router_score" in item for item in anc)
    assert any("latent_norm" in item for item in anc)


def test_dspark_standalone_target_features_and_markov_widths():
    model = load_model("kimi-k3-dspark")
    draft = model.auxiliary.dspark
    assert model.layers == 5 and draft.layers == 5
    assert draft.target_layer_ids == (7, 23, 51, 67, 83)
    assert draft.input_width == 5 * 7168
    assert draft.q_width == 4096 and draft.kv_width == 1024
    assert draft.block_size == 7 and draft.markov_rank == 256
    assert draft.confidence_input_width == 7168 + 256
    work = infer_iteration(
        _views((1, 0, True, "prefill")), model,
        _runtime(tp=1, ep=1, max_model_len=64, max_num_batched_tokens=8),
        ParallelSpec(tp=1, ep=1),
    )
    inp = next(op for op in work.ops if op.op_id == "dspark.input_proj")
    assert inp.launch_shape == (7, 7168, 35840)
    assert any(op.op_id == "dspark.base_logits" for op in work.ops)
    assert any(op.op_id.startswith("dspark.markov_lookup") for op in work.ops)
    assert any(op.op_id.startswith("dspark.markov_bias") for op in work.ops)
    assert any(op.op_id.startswith("dspark.sample") for op in work.ops)
    assert not any(op.op_id == "dspark.confidence" for op in work.ops)
    qk = next(op for op in work.ops if op.op_id.endswith(".qk") and "attn" in op.op_id)
    # 7-token noncausal draft block -> 49 query pairs, not causal 28.
    local_heads = 64
    head_dim = 64
    assert qk.flops == 2 * local_heads * 49 * head_dim
    assert work.padded_tokens >= 7
    # No fictional full-hidden token embedding/LM head on a target-feature draft.
    heads = body_head_terms(model) + draft_terms(draft, model, standalone_heads_only=True)
    assert not any(term.name in {"embedding", "lm_head", "draft_embedding", "draft_lm_head"}
                   for term in heads)


def test_qwen_local_gated_widths_gdn_heads_and_dense_ffn():
    for key, expected in (("qwen3.8-2.4t-a95b", 4608), ("qwen3.8-27b", 2048)):
        model = load_model(key)
        full = next(plan for plan in model.layer_plans if plan.attention == "gqa")
        gdn = next(plan for plan in model.layer_plans if plan.attention == "gdn")
        assert full.q_output_gate is True
        assert gdn.linear_key_heads == 16 and gdn.linear_value_heads == (
            128 if key.endswith("a95b") else 48)
        assert gdn.linear_head_dim == 128
        work = infer_iteration(
            _views((4, 0, True, "prefill")), model,
            _runtime(tp=8, ep=1), ParallelSpec(tp=8, ep=1),
        )
        qkv = next(op for op in work.ops
                   if op.op_id.endswith("qkv") and f".l{full.index}." in op.op_id
                   and ".attn." not in op.op_id and ".qkvz" not in op.op_id)
        assert qkv.launch_shape[-2] == expected
        qkvz = next(op for op in work.ops
                    if op.op_id.endswith("qkvz") and f".l{gdn.index}." in op.op_id)
        local_key = max(1, 16 // 8) * 128
        local_value = max(1, gdn.linear_value_heads // 8) * 128
        assert qkvz.launch_shape[-2] == 2 * local_key + 2 * local_value
        ba = next(op for op in work.ops
                  if op.op_id.endswith(".ba") and f".l{gdn.index}." in op.op_id)
        assert ba.launch_shape[-2] == 2 * max(1, gdn.linear_value_heads // 8)
        if key == "qwen3.8-27b":
            assert not any("router" in op.op_id for op in work.ops if f".l{full.index}." in op.op_id)


def test_glm_full_shared_indexer_and_mtp_scopes():
    model = load_model("glm-5.3")
    assert model.layers == 78
    assert sum(1 for p in model.layer_plans if p.ffn == "dense") == 3
    assert sum(1 for p in model.layer_plans if p.ffn == "moe") == 75
    full = [p for p in model.layer_plans if p.indexer]
    shared = [p for p in model.layer_plans if p.indexer_type == "shared"]
    assert len(full) == 21 and len(shared) == 57
    assert model.auxiliary.mtp is not None and model.auxiliary.mtp.layers == 1
    assert model.layers == 78  # MTP is outside the body schedule.
    sample = model.layer_plans[0]
    assert sample.qk_nope_head_dim == 192 and sample.qk_rope_head_dim == 64
    assert sample.v_head_dim == 256 and sample.head_dim == 256
    assert sample.index_source_status == "exact_native_full_schedule"
    assert shared[0].index_source_status == "derived_conservative_nearest_full"
    work = infer_iteration(
        _views((4, 0, True, "prefill")), model, _runtime(tp=1, ep=1),
        ParallelSpec(tp=1, ep=1),
    )
    stores = [op for op in work.ops if "index_k_store" in op.op_id]
    assert len(stores) == 21
    # Shared layers still pay their own query/selection read cost.
    shared_layer = shared[0].index
    assert any(f".l{shared_layer}.index_topk" in op.op_id for op in work.ops)


def test_v4_boundary_rows_windows_and_c4_thresholds():
    for key, owners in (("deepseek-v4-flash-0731", (21, 20)),
                        ("deepseek-v4-pro-0813", (30, 31)),
                        ("deepseek-v4-10t", (30, 31))):
        model = load_model(key)
        c4 = [p for p in model.layer_plans if p.compress_ratio == 4]
        c128 = [p for p in model.layer_plans if p.compress_ratio == 128]
        local = [p for p in model.layer_plans if p.compress_ratio == 0]
        assert (len(c4), len(c128)) == owners
        expected = {
            127: len(c4) * 31 + len(c128) * 0,
            128: len(c4) * 32 + len(c128) * 1,
            129: len(c4) * 32 + len(c128) * 1,
        }
        for seq, rows in expected.items():
            assert sum(completed_rows(seq, p.compress_ratio) for p in c4 + c128) == rows
        assert window_rows(127, 128) == 127
        assert window_rows(128, 128) == 128
        assert completed_rows(2047, 4) == 511
        assert completed_rows(2048, 4) == 512
        assert completed_rows(2049, 4) == 512
    pro_c4 = next(p for p in load_model("deepseek-v4-pro-0813").layer_plans
                  if p.compress_ratio == 4)
    assert completed_rows(4095, 4) == 1023
    assert completed_rows(4096, 4) == 1024
    assert completed_rows(4097, 4) == 1024
    # Exact per-request cache boundary.
    model = load_model("deepseek-v4-flash-0731")
    b127 = request_cache_breakdown(model, ParallelSpec(), 127)["by_component"]
    b128 = request_cache_breakdown(model, ParallelSpec(), 128)["by_component"]
    b129 = request_cache_breakdown(model, ParallelSpec(), 129)["by_component"]
    row = 512 * 2
    assert b127["long_kv"] == 651 * row
    assert b128["long_kv"] == 692 * row
    assert b129["long_kv"] == 692 * row
    assert b127["swa"] == 43 * 127 * row
    assert b128["swa"] == 43 * 128 * row
    assert b129["swa"] == 43 * 128 * row


def test_v41_owner_boundaries_candidate_and_c2_parity():
    model = load_model("deepseek-v4.1-flash")
    assert [p.index for p in model.layer_plans if p.owns_long_kv] == [2, 8, 14, 20]
    assert [p.index for p in model.layer_plans if p.owns_index_k] == [2, 8, 14, 20]
    assert [p.index for p in model.layer_plans if p.owns_topk] == [2, 8, 14, 20, 24, 28, 32, 36]
    assert [p.index for p in model.layer_plans if p.owns_candidate_blocks] == [20]
    layer26 = model.layer_plans[26]
    assert layer26.kv_owner_layer_id == 20
    assert layer26.index_k_owner_layer_id == 20
    assert layer26.topk_owner_layer_id == 24
    assert layer26.candidate_source_layer_id == 20
    assert layer26.candidate_filter is True
    assert new_completed_rows(127, 1, 2) == 1   # 127->128 completes a pair
    assert new_completed_rows(127, 2, 2) == 1   # 127->129 completes one pair
    assert new_completed_rows(128, 1, 2) == 0   # 128->129 leaves an odd residual
    assert new_completed_rows(0, 1, 2) == 0
    assert 3 * completed_rows(127, 2) + completed_rows(127, 1) == 316
    assert 3 * completed_rows(128, 2) + completed_rows(128, 1) == 320
    assert 3 * completed_rows(129, 2) + completed_rows(129, 1) == 321
    # C2 ring capacity and valid rows are distinct.
    ring = request_cache_breakdown(model, ParallelSpec(), 1)["by_component"]["c2_ring"]
    assert ring == 3 * 32 * 2 * 512 * 4
    assert candidate_valid_rows(16385) == 16384
    assert candidate_valid_rows(16385, first_block=1, blocks=2048) == 16377
    assert candidate_valid_rows(1) == 1
    assert candidate_valid_rows(8) == 8
    assert candidate_valid_rows(128) == 128
    total = 0
    for plan in model.layer_plans:
        cost = index_selection_cost(model, plan, 262144)
        total += cost.scored_rows
    assert total == 720896  # conservative block0-inclusive candidate set
    explicit = 0
    for plan in model.layer_plans:
        if not (plan.indexer or plan.owns_topk):
            continue
        if plan.candidate_source_layer_id >= 0:
            explicit += candidate_valid_rows(262144, first_block=1, blocks=2048)
        else:
            explicit += completed_rows(262144, model.layer_plans[plan.index_k_owner_layer_id].compress_ratio)
    assert explicit == 720896  # at N262144 selected blocks carry 16384 full rows
    # Long-read multiplicity retains every consumer.
    rows = sum(attention_rows(model, plan, 262144)[1] for plan in model.layer_plans)
    assert rows == 38 * 512
    swa = sum(attention_rows(model, plan, 262144)[2] for plan in model.layer_plans)
    assert swa == 40 * 128


def test_quantization_packs_scales_and_preserves_source_dtypes():
    qwen = load_model("qwen3.8-27b")
    layer = next(p for p in qwen.layer_plans if p.attention == "gqa")
    k_term = next(t for t in layer_weight_terms(layer, qwen) if t.name == "k_proj")
    assert term_bytes(k_term, qwen.quant, ParallelSpec(tp=8), 8) == 256 * 5120 * 2
    v41 = load_model("deepseek-v4.1-flash")
    # 2304x5120 FP4 group32 + E8M0 scales.
    assert v41.quant.matrix_bytes(2304, 5120, "routed_expert") == 6266880
    mhc = next(t for t in layer_weight_terms(v41.layer_plans[0], v41) if t.name == "hc_attn_fn")
    assert term_bytes(mhc, v41.quant, ParallelSpec(), source=True) == 24 * 4 * 5120 * 4
    kimi = load_model("kimi-k3")
    kda = next(p for p in kimi.layer_plans if p.attention == "kda")
    conv = next(t for t in layer_weight_terms(kda, kimi) if t.name == "kda_q_conv")
    assert term_bytes(conv, kimi.quant, ParallelSpec(), source=True) == 12288 * 4 * 4
    q27 = load_model("qwen3.8-27b")
    gdn = next(p for p in q27.layer_plans if p.attention == "gdn")
    for name in ("gdn_conv", "gdn_A_log", "gdn_dt_bias"):
        term = next(t for t in layer_weight_terms(gdn, q27) if t.name == name)
        assert term_bytes(term, q27.quant, ParallelSpec(), source=True) == term.rows * term.cols * 2
    table = next(t for t in engram_weight_terms(v41) if t.name.startswith("engram_table"))
    assert term_bytes(table, v41.quant, ParallelSpec(), source=True) == (
        table.rows * table.cols + table.rows * math.ceil(table.cols / 32)
    )
    # FP4 expert compute is BF16 unless core declares an FP4 peak.
    work = infer_iteration(
        _views((4, 0, True, "prefill")), kimi, _runtime(tp=8, ep=8),
        ParallelSpec(tp=8, ep=8),
    )
    gate = next(op for op in work.ops if op.op_id.endswith("expert_gate_up0") and ".l1." in op.op_id)
    assert gate.dtype == "bf16"
    assert dict(gate.metadata)["weight_format"] in {"mxfp4", "fp4", "fp4_e2m1"}


def test_vision_blocks_media_expansion_and_text_only_skip():
    v41 = load_model("deepseek-v4.1-flash")
    assert v41.vision is not None and v41.vision.layers == 32
    assert v41.vision.qkv_width == 3 * 1024 and v41.vision.head_dim == 64
    q27 = load_model("qwen3.8-27b")
    assert q27.vision.layers == 27 and q27.vision.qkv_width == 3 * 1152
    assert q27.vision.head_dim == 72
    kimi = load_model("kimi-k3")
    assert kimi.vision.layers == 27 and kimi.vision.qkv_width == 3 * 1536
    assert kimi.vision.head_dim == 128
    view = RequestView("r", 4, 0, True, "prefill")
    exp = lower_media(
        model=v41, runtime=SimpleNamespace(media_tokens=4, metadata=()),
        parallel=ParallelSpec(), quant=v41.quant, mode="prefill", views=(view,),
    )
    assert exp.text_positions == 4 and exp.patch_tokens == 36
    assert len([op for op in exp.ops if op.op_id.startswith("vision.l")]) >= 32 * 8
    qkv = next(op for op in exp.ops if op.op_id == "vision.l0.qkv")
    assert qkv.launch_shape[-2] == 3072
    assert qkv.launch_shape[-1] == 1024
    # Decode reuses merged embeddings and does not re-run the tower.
    dec = lower_media(
        model=v41, runtime=SimpleNamespace(media_tokens=4, metadata=()),
        parallel=ParallelSpec(), quant=v41.quant, mode="decode",
        views=(RequestView("r", 1, 16, True, "decode"),),
    )
    assert dec.ops == [] and dec.text_positions == 0
    # One-token prefill query grows by media positions; context does not.
    from mindie_sim.a3_graph_lowering import _effective_views
    effective, total = _effective_views((RequestView("r", 1, 0, True, "prefill"),), 4)
    assert effective[0].q_len == 5 and effective[0].context == 0 and total == 5
    work = infer_iteration(
        _views((1, 0, True, "prefill")), v41,
        _runtime(tp=8, ep=8, media_tokens=4), ParallelSpec(tp=8, ep=8),
    )
    assert work.valid_tokens == 5
    assert any(op.op_id.startswith("vision.l31") for op in work.ops)
    text_work = infer_iteration(
        _views((4, 0, True, "prefill")), v41,
        _runtime(tp=8, ep=8), ParallelSpec(tp=8, ep=8),
    )
    assert not any(op.op_id.startswith("vision.") for op in text_work.ops)


def test_memory_largest_stage_request_peak_and_state_ownership():
    v41 = load_model("deepseek-v4.1-flash")
    parallel = ParallelSpec(tp=8, dp=8, ep=64, pp=2)
    memory = memory_plan(v41, _runtime(tp=8, dp=8, ep=64, pp=2, max_model_len=128), _machine(128))
    assert memory.status == "fits"
    assert memory.weights_bytes == max(memory.stage_weights_bytes)
    assert memory.largest_weight_stage == memory.stage_weights_bytes.index(max(memory.stage_weights_bytes))
    assert memory.engram_weight_bytes > 0
    assert memory.ring_bytes_per_request == 3 * 32 * 2 * 512 * 4
    assert memory.state_bytes_per_request == 0.0
    kimi = load_model("kimi-k3")
    kmem = memory_plan(kimi, _runtime(tp=8, dp=8, ep=64, pp=3, max_model_len=128), _machine(192))
    assert kmem.status == "fits"
    assert kmem.state_bytes_per_request > 50e6
    assert kmem.state_reserve_bytes > 0
    assert kmem.request_peak_tokens == 0  # no workload supplied
    wl = WorkloadSpec((
        RequestSpec("a", 0, 100, 20),
        RequestSpec("b", 1, 200, 30),
    ))
    assert workload_required_tokens(wl) == 230
    assert workload_lifetime_tokens(wl) == 350
    mem = memory_plan(kimi, _runtime(tp=8, dp=8, ep=64, pp=3, max_model_len=256),
                      _machine(192), workload=wl)
    assert mem.request_peak_tokens == 230
    assert mem.per_request_cache and max(item["sequence_tokens"] for item in mem.per_request_cache) == 230


def test_optional_draft_is_joint_and_does_not_rerun_target_body():
    model = load_model("qwen3.8-27b")
    plain = infer_iteration(
        _views((4, 0, True, "prefill")), model, _runtime(tp=8, ep=1),
        ParallelSpec(tp=8, ep=1),
    )
    assert not any(op.op_id.startswith("draft.") or op.op_id.startswith("verify.")
                   for op in plain.ops)
    draft_rt = _runtime(tp=8, ep=1, metadata={
        "draft_mode": "mtp", "draft_accept_tokens": 0,
    })
    drafted = infer_iteration(
        IterationPlan((RequestView(
            "r0", 1, 16, True, "decode", draft_verify=True,
            draft_proposal_tokens=1, draft_accepted_tokens=0,
        ),)), model, draft_rt, ParallelSpec(tp=8, ep=1),
    )
    assert next(op for op in drafted.ops if op.op_id == "embedding").local_shape[0] == 2
    assert not any(
        op.op_id.startswith("verify.s0.")
        and not dict(op.metadata).get("zero_work_alias", False)
        for op in drafted.ops
    )
    assert next(op for op in drafted.ops
                if op.op_id == "s0.l0.gate_up").local_shape[0] == 2
    verify = next(op for op in drafted.ops if op.op_id == "verify.embedding")
    assert verify.local_shape[0] == 1
    commit = next(op for op in drafted.ops if op.op_id == "draft.state_commit")
    assert dict(commit.metadata)["accepted_tokens"] == 0
    assert dict(commit.metadata)["target_body_reused"] is True

def test_prepare_candidates_are_model_derived_and_known_at_classified():
    v41 = load_model("deepseek-v4.1-flash")
    runtime = _runtime(tp=8, ep=8, max_model_len=128, max_num_batched_tokens=32,
                       media_tokens=4)
    work = infer_iteration(_views((4, 0, True, "prefill")), v41, runtime,
                           ParallelSpec(tp=8, ep=8))
    wl = WorkloadSpec((RequestSpec("r0", 0, 4, 2),))
    candidates = catalog_prepare_candidates(v41, runtime, ParallelSpec(tp=8, ep=8),
                                            work=work, workload=wl)
    assert {c.kind for c in candidates} >= {
        "quant_scale_layout", "compressor_buffer_capacity",
        "engram_id_metadata", "vision_position_table",
    }
    engram_ids = [c for c in candidates if c.kind == "engram_id_metadata"]
    prompt_ids = [c for c in engram_ids if ".engram_ids." in c.candidate_id]
    assert prompt_ids and all(c.known_at == "request_arrival" for c in prompt_ids)
    assert prompt_ids[0].bytes == 4 * 24 * 8
    assert prompt_ids[0].request_id == "r0"
    assert prompt_ids[0].reusable is False
    assert prompt_ids[0].replace_op_ids and prompt_ids[0].consumer_op_ids
    assert "row_values_not_precomputed" in dict(prompt_ids[0].metadata)
    assert prompt_ids[0].host_rates_required == ("cpu_ops_per_s",)
    decode_ids = [c for c in engram_ids if "decode" in c.candidate_id]
    assert decode_ids and all(c.known_at == "previous_forward" for c in decode_ids)
    assert all(c.iteration == 1 and not c.consumer_op_ids for c in decode_ids)
    ring = [c for c in candidates if c.kind == "compressor_buffer_capacity"]
    assert len(ring) == 3 and all(c.bytes == 0 for c in ring)
    assert all(dict(c.resources)["hbm_reserved_bytes"] > 0 for c in ring)
    assert all(c.service_bytes == 0 and c.capacity_bytes > 0 for c in ring)
    assert all(c.request_id == "r0" and c.reusable is False for c in ring)
    assert all(c.consumer_op_ids for c in candidates
               if c.kind not in {"engram_id_metadata", "vision_position_table"})
    kimi = load_model("kimi-k3")
    krt = _runtime(tp=8, ep=8, pp=3)
    kwork = infer_iteration(_views((4, 0, True, "prefill")), kimi, krt,
                            ParallelSpec(tp=8, ep=8, pp=3))
    kc = catalog_prepare_candidates(kimi, krt, ParallelSpec(tp=8, ep=8, pp=3),
                                    work=kwork)
    conv = next(c for c in kc if c.kind == "conv_kernel_layout")
    assert conv.known_at == "model_load"
    assert dict(conv.metadata)["source_dtype"] == "fp32"
    assert dict(conv.metadata)["prepared_dtype"] == "bf16"
    assert conv.bytes == dict(conv.metadata)["source_bytes_preserved"]
    assert conv.read_bytes and conv.write_bytes and conv.service_bytes
    fp4 = [c for c in kc if c.kind == "quant_scale_layout"
           and "fp4" in c.candidate_id]
    assert fp4 and len({c.owner_stage for c in fp4}) == 3
    assert all(c.capacity_bytes < 64 * 1024 ** 3 for c in fp4)
    assert all(c.owner_stage >= 0 for c in kc)


def test_all_nine_iteration_graphs_have_unique_ids_and_resolved_dependencies():
    topologies = {
        "deepseek-v4-flash-0731": (8, 1, 64, 64),
        "deepseek-v4-pro-0813": (8, 1, 64, 64),
        "deepseek-v4-10t": (8, 1, 64, 64),
        "deepseek-v4.1-flash": (8, 2, 64, 128),
        "glm-5.3": (8, 2, 64, 128),
        "kimi-k3": (8, 3, 64, 192),
        "kimi-k3-dspark": (1, 1, 1, 8),
        "qwen3.8-2.4t-a95b": (8, 4, 64, 256),
        "qwen3.8-27b": (8, 1, 1, 8),
    }
    for key, (tp, pp, ep, devices) in topologies.items():
        model = load_model(key)
        parallel = ParallelSpec(tp=tp, dp=1, ep=ep, pp=pp)
        work = infer_iteration(
            _views((2, 0, True, "prefill")), model, _runtime(tp=tp, ep=ep, pp=pp),
            parallel,
        )
        ids = [op.op_id for op in work.ops]
        assert len(ids) == len(set(ids))
        assert work.padded_tokens >= 2
        assert all(op.deps is not None for op in work.ops)
        known = set(ids)
        for op in work.ops:
            for dep in op.deps:
                assert dep in known, (key, op.op_id, dep)
        assert sum(op.flops for op in work.ops) > 0
        assert sum(op.hbm_bytes for op in work.ops) > 0


def test_public_predict_should_and_search_on_standalone_draft():
    from mindie_sim.a3_framework import predict
    from mindie_sim.a3_search import search
    from mindie_sim.a3_should import should_cost
    case = {
        "machine": {
            "name": "a3-8", "arch_family": "A3", "dies_per_package": 2,
            "num_devices": 8, "ranks_per_node": 8,
            "device": {
                "hbm_gib": 64, "cube_flops_per_s": 2.70336e14,
                "vector_ops_per_s": 1.6896e13,
                "hbm_bw_bytes_per_s": 1.2e12, "mte_bw_bytes_per_s": 2.4e12,
            },
        },
        "model": {"catalog_key": "kimi-k3-dspark"},
        "workload": {"requests": [{"prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {
            "max_num_batched_tokens": 8, "max_num_seqs": 1,
            "max_model_len": 64, "tp": 1, "dp": 1, "ep": 1, "pp": 1,
            "graph_mode": "eager",
        },
    }
    predicted = predict(case)
    assert predicted["status"] == "modeled"
    assert predicted["metrics"]["ttft_ms"]["p50"] is not None
    normative = should_cost(case, constraints={
        "tp": [1], "dp": [1], "ep": [1], "pp": [1],
        "fusion": [False], "graph_padding": [False],
    })
    assert normative["status"] == "modeled"
    assert normative["hardware_lower_bound_ms"] <= normative["feasible_target_ms"] + 1e-9
    searched = search(case, {
        "tp": [1], "dp": [1], "ep": [1], "pp": [1],
        "max_num_batched_tokens": [8], "fusion": [False], "max_candidates": 1,
    })
    assert searched["status"] in {"modeled", "no_feasible_candidate"}
    assert searched["candidates"]


def test_exact_query_visible_pair_oracles():
    from mindie_sim.a3_catalog_attention_common import _causal_pairs, _long_selected_pairs
    assert _causal_pairs(1, 128, 128) == 128
    assert _causal_pairs(2, 127, 128) == 256
    assert _causal_pairs(129, 0, 128) == 8384
    assert _long_selected_pairs(1, 127, 2, 512) + _causal_pairs(1, 127, 128) == 192
    assert _long_selected_pairs(3, 0, 2, 512) + _causal_pairs(3, 0, 128) == 8
    assert _long_selected_pairs(2, 1024, 2, 512) + _causal_pairs(2, 1024, 128) == 1280


def test_compressor_input_compute_and_ring_execute_without_new_rows():
    model = load_model("deepseek-v4.1-flash")
    work = infer_iteration(
        _views((1, 128, False, "decode")), model,
        _runtime(tp=8, ep=8), ParallelSpec(tp=8, ep=8),
    )
    ops = _op_map(work)
    compressor = ops["s0.l2.compressor_kv"]
    assert compressor.launch_shape[0] == 1
    assert compressor.flops > 0
    assert ops["s0.l2.compressor_gate"].flops > 0
    ring = ops["s0.l2.c2_ring"]
    assert ring.flops == 0 and ring.hbm_bytes > 0
    # No new completed long row is fabricated, but readiness remains a dep.
    assert "s0.l2.long_cache_ready" in ops
    assert "s0.l2.long_kv_store" not in ops


def test_routed_etp_reduction_precedes_latent_norm_and_shared_reduction():
    model = load_model("kimi-k3")
    work = infer_iteration(
        _views((1, 128, False, "decode")), model,
        _runtime(tp=8, ep=1), ParallelSpec(tp=8, ep=1),
    )
    ops = _op_map(work)
    norm = next(op for op in work.ops
                if op.op_id.endswith("latent_out_norm") and ".l1." in op.op_id)
    assert "s0.l1.routed_reduce" in norm.deps
    shared = next(op for op in work.ops
                  if op.op_id.endswith("shared_reduce") and ".l1." in op.op_id)
    assert shared.link_bytes > 0
    assert dict(shared.metadata)["shared_branch_independent"] is True


def test_only_declared_candidate_source_and_layer24_dependency():
    model = load_model("deepseek-v4.1-flash")
    work = infer_iteration(
        _views((1, 128, False, "decode")), model,
        _runtime(tp=1, ep=1), ParallelSpec(tp=1, ep=1),
    )
    ops = _op_map(work)
    candidates = [op.op_id for op in work.ops if "candidate_blocks" in op.op_id]
    assert candidates == ["s0.l20.candidate_blocks"]
    topk = next(op for op in work.ops
                if op.op_id.endswith("index_topk") and ".l24." in op.op_id)
    assert "s0.l20.candidate_blocks" in topk.deps


def test_per_category_precision_and_compressor_runtime_bytes():
    from mindie_sim.a3_catalog_ops import compute_dtype_for
    from mindie_sim.a3_catalog_weights import term_bytes
    v41 = load_model("deepseek-v4.1-flash")
    assert compute_dtype_for(v41.quant, "vision") == ("bf16", "bf16")
    assert compute_dtype_for(v41.quant, "projector") == ("bf16", "bf16")
    assert compute_dtype_for(v41.quant, "mhc") == ("fp32", "fp32")
    assert compute_dtype_for(v41.quant, "compressor_fp32_runtime") == ("fp32", "fp32")
    c2 = next(t for t in layer_weight_terms(v41.layer_plans[2], v41)
              if t.name == "compressor_wkv")
    assert term_bytes(c2, v41.quant, ParallelSpec(tp=8), 8) == 10_485_760
    assert term_bytes(c2, v41.quant, ParallelSpec(tp=8), 8, source=True) == 5_242_880
    qwen = load_model("qwen3.8-27b")
    assert compute_dtype_for(qwen.quant, "vision") == ("bf16", "bf16")
    assert compute_dtype_for(qwen.quant, "gdn") == ("bf16", "bf16")


def test_vocab_parallel_embedding_and_target_aliases():
    qwen = load_model("qwen3.8-27b")
    emb = next(t for t in body_head_terms(qwen) if t.name == "embedding")
    assert emb.shard == "embedding"
    assert term_bytes(emb, qwen.quant, ParallelSpec(tp=8), 8) == 248320 * 5120 * 2 // 8
    dspark = load_model("kimi-k3-dspark")
    heads = body_head_terms(dspark) + draft_terms(
        dspark.auxiliary.dspark, dspark, standalone_heads_only=True,
    )
    names = {t.name for t in heads}
    assert "embedding" not in names and "lm_head" not in names
    assert "target_embedding_alias" in names and "target_lm_head_alias" in names
    alias = next(t for t in heads if t.name == "target_embedding_alias")
    assert alias.shard == "replicated"  # shared target owner, counted once


def test_selected_draft_residency_is_one_native_mode():
    v41 = load_model("deepseek-v4.1-flash")
    assert v41.auxiliary.dspark is None and v41.auxiliary.mtp is not None
    assert v41.auxiliary.mtp.layers == 3
    assert v41.auxiliary.mtp.num_experts == 128
    assert v41.auxiliary.mtp.top_k == 3
    parallel = ParallelSpec(tp=8, dp=8, ep=64, pp=2)
    body = stage_weight_bytes(v41, parallel, 8)
    with_draft = stage_weight_bytes(v41, parallel, 8, include_draft=True)
    deltas = [b - a for a, b in zip(body, with_draft)]
    assert max(deltas) > 0
    # Only the selected native MTP/DSpark module is charged, not two modes.
    assert max(deltas) < 500_000_000
    runtime = _runtime(tp=8, dp=8, ep=64, pp=2, metadata={"draft_mode": "mtp"})
    mem = memory_plan(v41, runtime, _machine(128))
    assert mem.optional_draft_weight_bytes == pytest.approx(max(deltas))


def test_media_per_request_once_and_no_empty_dependency():
    model = load_model("qwen3.8-27b")
    runtime = _runtime(
        tp=8, max_model_len=64,
        metadata={
            "media_tokens_by_request": '{"r0":4}',
            "media_positions_pre_expanded": "true",
        },
    )
    first = infer_iteration(_views((1, 0, True, "prefill")), model, runtime,
                            ParallelSpec(tp=8, ep=1))
    assert any(op.op_id.startswith("vision.") for op in first.ops)
    assert any(op.op_id == "media_text_join" for op in first.ops)
    assert not any(dep == "" for op in first.ops for dep in op.deps)
    chunk2 = infer_iteration(_views((1, 5, True, "prefill")), model, runtime,
                             ParallelSpec(tp=8, ep=1))
    assert not any(op.op_id.startswith("vision.") for op in chunk2.ops)
    assert not any(op.op_id == "media_text_join" for op in chunk2.ops)
    assert not any(dep == "" for op in chunk2.ops for dep in op.deps)
    # Decode with legacy media must not invent a media producer.
    decode = infer_iteration(
        _views((1, 16, False, "decode")), model,
        _runtime(tp=8, media_tokens=4), ParallelSpec(tp=8, ep=1),
    )
    assert not any(dep == "" for op in decode.ops for dep in op.deps)
    # Expansion happens before scheduling in RequestSpec.
    request = RequestSpec.from_mapping(
        {"prompt_tokens": 1, "media_tokens": 4, "output_tokens": 1}, 0,
    )
    assert request.prompt_tokens == 5 and request.text_prompt_tokens == 1


def test_verify_reuses_target_graph_per_request_and_rolls_back():
    model = load_model("qwen3.8-27b")
    runtime = _runtime(tp=8, ep=1, metadata={
        "draft_mode": "mtp", "draft_accept_tokens": 0,
    })
    one = infer_iteration(
        IterationPlan((RequestView(
            "a", 1, 16, True, "decode", draft_verify=True,
            draft_proposal_tokens=5, draft_accepted_tokens=0,
        ),)), model, runtime, ParallelSpec(tp=8, ep=1),
    )
    two = infer_iteration(
        IterationPlan((
            RequestView("a", 1, 16, True, "decode", draft_verify=True,
                        draft_proposal_tokens=5, draft_accepted_tokens=0),
            RequestView("b", 1, 16, True, "decode", draft_verify=True,
                        draft_proposal_tokens=5, draft_accepted_tokens=0),
        )), model, runtime, ParallelSpec(tp=8, ep=1),
    )
    assert next(op for op in one.ops if op.op_id == "embedding").local_shape[0] == 6
    assert next(op for op in two.ops if op.op_id == "embedding").local_shape[0] == 12
    assert next(op for op in one.ops if op.op_id == "verify.embedding").local_shape[0] == 5
    assert next(op for op in two.ops if op.op_id == "verify.embedding").local_shape[0] == 10
    assert not any(
        op.op_id.startswith("verify.s0.")
        and not dict(op.metadata).get("zero_work_alias", False)
        for op in one.ops
    )
    assert any(op.op_id == "verify.state_rollback" for op in one.ops)
    assert any(op.op_id == "verify.temporary_kv_discard" for op in one.ops)
    assert not any(op.op_id.startswith("verify.l") for op in one.ops)

def test_normative_weight_bytes_are_resident_once_not_per_token():
    from mindie_sim.a3_should import should_cost
    case = {
        "machine": {
            "name": "a3-8", "arch_family": "A3", "num_devices": 8,
            "ranks_per_node": 8,
            "device": {
                "hbm_gib": 64, "cube_flops_per_s": 2.70336e14,
                "vector_ops_per_s": 1.6896e13,
                "hbm_bw_bytes_per_s": 1.2e12, "mte_bw_bytes_per_s": 2.4e12,
            },
        },
        "model": {"catalog_key": "qwen3.8-27b"},
        "workload": {"requests": [{"prompt_tokens": 129, "output_tokens": 2}]},
        "runtime": {
            "max_num_batched_tokens": 256, "max_num_seqs": 2,
            "max_model_len": 256, "tp": 8, "dp": 1, "ep": 1, "pp": 1,
            "graph_mode": "eager",
        },
    }
    result = should_cost(case, constraints={
        "tp": [8], "dp": [1], "ep": [1], "pp": [1],
        "fusion": [False], "graph_padding": [False],
    })
    assert result["status"] == "modeled"
    assert result["semantic_work"]["weight_bytes"] <= (
        load_model("qwen3.8-27b").resident_weight_bytes_override * 2 * 1.01
    )
    assert result["hardware_lower_bound_ms"] < result["feasible_target_ms"]
    assert result["hardware_lower_bound_ms"] < 30.0


def test_campaign_records_expose_model_derived_precision_and_category():
    from mindie_sim.a3_catalog_campaign import iteration_campaign_records
    model = load_model("deepseek-v4.1-flash")
    work = infer_iteration(
        _views((2, 0, True, "prefill")), model, _runtime(tp=8, ep=8),
        ParallelSpec(tp=8, ep=8),
    )
    records = iteration_campaign_records(model, work)
    by_id = {record["op_id"]: record for record in records}
    attn = next(record for record in records
                if record["op_id"].endswith("q_b") and record["layer"] == 2)
    assert attn["primitive"] == "mla_q_b_local_heads"
    assert attn["category"].startswith("dsa_shared_compressed_2/moe")
    assert attn["weight_format"] == "fp8_e4m3"
    assert attn["compute_format"] == "fp8_e4m3"
    assert attn["semantic_shape"] and attn["launch_shape"]
    assert attn["packed_weight_bytes"] > 0
    compressor = by_id["s0.l2.compressor_kv"]
    assert compressor["quant_category"] == "compressor_fp32_runtime"
    assert compressor["compute_format"] == "fp32"
    assert compressor["weight_format"] == "fp32"
    expert = next(record for record in records
                  if record["op_id"].startswith("s0.l0.expert_gate_up"))
    assert expert["quant_category"] == "routed_expert"
    assert expert["compute_format"] == "bf16"
    assert expert["dequant_before_compute"] is True


def test_mandatory_weight_bytes_do_not_scale_with_prefill_tokens():
    from mindie_sim.a3_should import semantic_work
    from mindie_sim.a3_types import WorkloadSpec
    model = load_model("qwen3.8-27b")
    parallel = ParallelSpec(tp=8)
    small = WorkloadSpec((RequestSpec("a", 0, 1, 1),))
    large = WorkloadSpec((RequestSpec("b", 0, 129, 1),))
    ws = semantic_work(model, small, parallel)
    wl = semantic_work(model, large, parallel)
    # Same number of forward passes: active weight traffic is batch/prefill
    # independent; only KV/activation/attention work grows.
    # Active embedding traffic is exactly the newly-read prompt rows; model
    # weights are not charged per token.
    row = model.hidden_size * 2
    assert wl.weight_bytes - ws.weight_bytes == 128 * row
    assert wl.kv_bytes > ws.kv_bytes
    assert wl.attention_flops > ws.attention_flops
    two = semantic_work(model, WorkloadSpec((RequestSpec("c", 0, 129, 2),)), parallel)
    weight_per_forward = wl.weight_bytes - 129 * row
    assert two.weight_bytes == 2 * weight_per_forward + 130 * row


def test_two_requests_only_one_media_and_chunked_prefill_once():
    model = load_model("qwen3.8-27b")
    runtime = _runtime(
        tp=8, max_model_len=64,
        metadata={
            "media_tokens_by_request": '{"r0":4}',
            "media_positions_pre_expanded": "true",
        },
    )
    work = infer_iteration(
        _views((1, 0, True, "prefill"), (1, 0, True, "prefill")),
        model, runtime, ParallelSpec(tp=8, ep=1),
    )
    vision_layers = {op.layer for op in work.ops if op.op_id.startswith("vision.l")}
    assert vision_layers
    assert sum(op.op_id == "media_text_join" for op in work.ops) == 1
    assert not any(dep == "" for op in work.ops for dep in op.deps)
    # Chunk 2 of the same media request skips the tower and retains context.
    chunk = infer_iteration(
        _views((1, 2, True, "prefill"),),
        model, runtime, ParallelSpec(tp=8, ep=1),
    )
    # The media request already encoded on its first chunk; no re-run.
    assert not any(op.op_id.startswith("vision.") for op in chunk.ops)
    assert not any(op.op_id == "media_text_join" for op in chunk.ops)


def test_vision_merger_projector_normalization_and_bias_weights():
    qwen = load_model("qwen3.8-27b")
    terms = {t.name: t for t in vision_weight_terms(qwen)}
    assert term_bytes(terms["merger_w1"], qwen.quant, ParallelSpec(), 1) == 4608 * 4608 * 2
    assert term_bytes(terms["merger"], qwen.quant, ParallelSpec(), 1) == 5120 * 4608 * 2
    assert "merger_norm_bias" in terms and "vision_norm1_bias" in terms
    exp = lower_media(
        model=qwen, runtime=SimpleNamespace(media_tokens=4, metadata=()),
        parallel=ParallelSpec(), quant=qwen.quant, mode="prefill",
        views=(RequestView("r", 4, 0, True, "prefill"),),
    )
    assert {"vision.merger_norm", "vision.merger_w1", "vision.merger_gelu",
            "vision.merger"} <= {op.op_id for op in exp.ops}
    kimi = load_model("kimi-k3")
    kexp = lower_media(
        model=kimi, runtime=SimpleNamespace(media_tokens=4, metadata=()),
        parallel=ParallelSpec(), quant=kimi.quant, mode="prefill",
        views=(RequestView("r", 4, 0, True, "prefill"),),
    )
    assert {"vision.projector0", "vision.projector0_gelu",
            "vision.projector2", "vision.projector_post_norm"} <= {
        op.op_id for op in kexp.ops
    }
    v41 = load_model("deepseek-v4.1-flash")
    vterms = {t.name for t in vision_weight_terms(v41)}
    assert {"aligner_w1_bias", "aligner_w2_bias", "vision_qkv_bias",
            "vision_wo_bias", "vision_fc0_bias"} <= vterms


def test_optional_kimi_dspark_native_alias_markov_and_noncausal_block():
    from mindie_sim.a3_catalog_aux import lower_optional_draft
    model = load_model("kimi-k3")
    runtime = SimpleNamespace(
        metadata=(("draft_mode", "dspark"), ("draft_accept_tokens", 0))
    )
    result = lower_optional_draft(
        model=model, runtime=runtime, parallel=ParallelSpec(tp=1),
        quant=model.quant,
        views=(RequestView(
            "r", 1, 16, True, "decode", draft_verify=True,
            draft_proposal_tokens=5, draft_accepted_tokens=0,
        ),),
        context=17, mode="dspark",
    )
    ops = {op.op_id: op for op in result.ops}
    assert ops["draft.base_logits"].family == "logits"
    assert ops["draft.target_embedding_read"].hbm_bytes == 7 * 7168 * 2
    assert ops["draft.markov_bias0"].flops == 2 * 1 * 163840 * 256
    assert dict(ops["draft.markov_bias0"].metadata)["replicated"] is True
    assert ops["draft.markov_lookup1"].deps == ("draft.sample0",)
    assert "dspark.confidence" not in ops and "draft.confidence" not in ops
    qk = next(op for op in result.ops
              if op.op_id.endswith("draft_qk") and "draft.l0" in op.op_id)
    assert qk.flops == 2 * 64 * (7 * 17 + 49) * 64
    assert {"draft.l0.draft_context_kv", "draft.l0.draft_context_norm",
            "draft.l0.draft_context_rope",
            "draft.l0.draft_context_store"} <= set(ops)
    with pytest.raises(A3InputError):
        lower_optional_draft(
            model=model, runtime=runtime, parallel=ParallelSpec(tp=1, pp=2),
            quant=model.quant,
            views=(RequestView("r", 1, 16, True, "decode"),),
            context=17, mode="dspark",
        )


def test_engram_explicit_storage_modes_and_lookup_bounded_staging():
    from dataclasses import replace as dc_replace
    model = load_model("deepseek-v4.1-flash")
    table = next(t for t in engram_weight_terms(model) if t.name.startswith("engram_table"))
    assert model.engram.row_bytes == 512
    assert model.quant.engram_table_mode == "device_bf16"
    assert term_bytes(table, model.quant, ParallelSpec(), 1) == table.rows * 256 * 2
    assert term_bytes(table, model.quant, ParallelSpec(), 1, source=True) == (
        table.rows * 256 + table.rows * 8
    )
    int8 = dc_replace(model, quant=dc_replace(model.quant, engram_table_mode="device_int8_fp32"),
                      engram=dc_replace(model.engram, storage_mode="device_int8_fp32"))
    assert int8.engram.row_bytes == 288
    assert term_bytes(table, int8.quant, ParallelSpec(), 1) == table.rows * 288
    host = dc_replace(model, quant=dc_replace(model.quant, engram_table_mode="host_fp8_e8m0"),
                      engram=dc_replace(model.engram, storage_mode="host_fp8_e8m0"))
    assert host.engram.row_bytes == 264
    mem = memory_plan(model, _runtime(tp=16, dp=1, ep=16, pp=4, max_model_len=64),
                      _machine(64))
    assert mem.status == "fits"
    assert mem.engram_weight_bytes > 0


def test_verify_dependencies_logits_and_nonexecution_summaries():
    model = load_model("qwen3.8-27b")
    result = infer_iteration(
        IterationPlan((RequestView(
            "r0", 1, 16, True, "decode", draft_verify=True,
            draft_proposal_tokens=5, draft_accepted_tokens=0,
        ),)), model,
        _runtime(tp=8, ep=1, metadata={
            "draft_mode": "mtp", "draft_accept_tokens": "0",
        }), ParallelSpec(tp=8, ep=1),
    )
    by_id = {op.op_id: op for op in result.ops}
    assert by_id["verify.embedding"].deps
    assert any(dep.startswith("draft.") for dep in by_id["verify.embedding"].deps)
    ancestors = set()
    pending = list(by_id["verify.target_logits"].deps)
    while pending:
        current = pending.pop()
        if current in ancestors:
            continue
        ancestors.add(current)
        if current in by_id:
            pending.extend(by_id[current].deps)
    assert "s0.l63.ffn_residual" in ancestors
    assert "verify.l0.ffn" not in by_id

def test_encoded_media_keeps_positions_but_can_skip_tower():
    plain = RequestSpec.from_mapping(
        {"prompt_tokens": 1, "media_tokens": 4, "output_tokens": 1}, 0,
    )
    encoded = RequestSpec.from_mapping(
        {"prompt_tokens": 1, "media_tokens": 4, "media_encoded": True,
         "output_tokens": 1}, 0,
    )
    assert plain.prompt_tokens == encoded.prompt_tokens == 5
    assert encoded.text_prompt_tokens == 1


def test_dspark_batch_preserves_context_and_external_alias_ownership():
    from mindie_sim.a3_catalog_weights import (
        body_weight_bytes, external_shared_weight_bytes,
    )
    model = load_model("kimi-k3-dspark")
    work = infer_iteration(
        IterationPlan((
            RequestView("r0", 1, 16, False, "prefill"),
            RequestView("r1", 1, 16, False, "prefill"),
        )),
        model, _runtime(tp=1, ep=1, max_num_batched_tokens=16),
        ParallelSpec(tp=1, ep=1),
    )
    by_id = {op.op_id: op for op in work.ops}
    assert by_id["dspark.input_proj"].local_shape[0] == 14
    assert by_id["dspark.base_logits"].local_shape[0] == 14
    qk = next(op for op in work.ops if op.family == "attention_qk")
    assert qk.local_shape[1] == 23
    assert by_id["dspark.markov_bias0"].local_shape[:2] == (2, 163840)

    aliases = [t for t in draft_terms(
        model.auxiliary.dspark, model, standalone_heads_only=True,
    ) if t.shared_owner]
    assert {t.name for t in aliases} == {
        "target_embedding_alias", "target_lm_head_alias",
    }
    assert all(term_bytes(t, model.quant, ParallelSpec(), source=True) == 0
               for t in aliases)
    assert external_shared_weight_bytes(model, ParallelSpec()) == 4_697_620_480
    assert body_weight_bytes(model) < 5_000_000_000


def test_engram_staging_uses_one_global_token_budget_window():
    model = load_model("deepseek-v4.1-flash")
    memory = memory_plan(
        model,
        _runtime(tp=16, ep=16, pp=4, max_model_len=64,
                 max_num_batched_tokens=16384, max_num_seqs=256),
        MachineSpec.from_mapping({
            "arch_family": "A3", "num_devices": 64, "ranks_per_node": 8,
            "device": {
                "hbm_gib": 64, "cube_flops_per_s": 2.70336e14,
                "vector_ops_per_s": 1.6896e13, "hbm_bw_bytes_per_s": 1.2e12,
                "mte_bw_bytes_per_s": 2.4e12,
            },
        }),
    )
    assert memory.status == "fits"
    assert memory.engram_staging_bytes == 16384 * 48 * 520
    assert memory.engram_staging_bytes == 408_944_640


def test_prepare_c14_request_lifetime_consumer_and_stage_contract():
    model = load_model("deepseek-v4.1-flash")
    parallel = ParallelSpec(tp=16, ep=16, pp=4)
    runtime = _runtime(tp=16, ep=16, pp=4, max_model_len=256,
                       max_num_batched_tokens=32)
    workload = WorkloadSpec((
        RequestSpec("early", 0, 8, 2),
        RequestSpec("late", 100.0, 8, 1),
    ))
    work = infer_iteration(
        IterationPlan((
            RequestView("early", 8, 0, True, "prefill"),
            RequestView("late", 8, 0, True, "prefill"),
        )), model, runtime, parallel,
    )
    candidates = catalog_prepare_candidates(
        model, runtime, parallel, work=work, workload=workload,
    )
    prompt_ids = [c for c in candidates
                  if c.kind == "engram_id_metadata"
                  and ".engram_ids." in c.candidate_id]
    assert {c.request_id for c in prompt_ids} == {"early", "late"}
    assert all(not c.reusable for c in prompt_ids)
    assert all(c.consumer_op_ids and c.replace_op_ids for c in prompt_ids)
    assert all(c.known_at == "request_arrival" for c in prompt_ids)
    assert {c.arrival_ms for c in prompt_ids if c.request_id == "late"} == {100.0}
    assert all(c.service_bytes == 0 and c.capacity_bytes > 0
               for c in candidates if c.kind == "compressor_buffer_capacity")
    fp4 = [c for c in candidates if "fp4" in c.candidate_id]
    assert len(fp4) == 4 and {c.owner_stage for c in fp4} == {0, 1, 2, 3}
    assert all(c.capacity_bytes < 64 * 1024 ** 3 for c in fp4)
    conv = [c for c in candidates if c.kind == "conv_kernel_layout"]
    assert all(c.read_bytes and c.write_bytes for c in conv)


def test_semantic_shapes_precision_router_and_composite_contracts():
    v4 = load_model("deepseek-v4-10t")
    v4p = ParallelSpec(tp=8, ep=8)
    v4w = infer_iteration(
        _views((128, 0, True, "prefill")), v4,
        _runtime(tp=8, ep=8, max_num_batched_tokens=128), v4p,
    )
    v4ops = {op.op_id: op for op in v4w.ops}
    q_norm = v4ops["s0.l0.q_norm"]
    assert q_norm.local_shape[1] == v4.layer_plans[0].q_lora_rank
    assert dict(q_norm.metadata)["reduction_width"] == q_norm.local_shape[1]
    shared = v4ops["s0.l0.shared_act"]
    assert shared.local_shape[0] == 128
    assert 2 * shared.local_shape[1] == dict(shared.metadata)["input_width"]
    index_norm = v4ops["s0.l2.index_k_norm"]
    quant = v4ops["s0.l2.index_k_quant"]
    assert len(index_norm.local_shape) == 2
    assert quant.local_shape == (32, 128)
    qmeta = dict(quant.metadata)
    assert qmeta["input_format"] == "bf16" and qmeta["output_format"] == "int8"
    assert qmeta["scale_axis"] == "per_row" and qmeta["scale_bytes"] == 2.0
    assert qmeta["scale_format"] == "fp16"

    glm = load_model("glm-5.3")
    gp = ParallelSpec(tp=8, ep=8)
    gw = infer_iteration(
        _views((128, 0, True, "prefill")), glm,
        _runtime(tp=8, ep=8, max_num_batched_tokens=128), gp,
    )
    score = next(op for op in gw.ops if op.op_id == "s0.l3.router_score")
    sm = dict(score.metadata)
    assert sm["scoring_func"] == "sigmoid"
    assert sm["top_k"] == 8
    assert sm["normalize_topk"] is True
    assert sm["router_contract"] == "sigmoid_topk"

    qwen = load_model("qwen3.8-2.4t-a95b")
    qp = ParallelSpec(tp=8, ep=8)
    qw = infer_iteration(
        _views((128, 0, True, "prefill")), qwen,
        _runtime(tp=8, ep=8, max_num_batched_tokens=128), qp,
    )
    split = next(op for op in qw.ops if op.op_id == "s0.l0.split_norm")
    sm = dict(split.metadata)
    assert len(split.local_shape) == 2
    assert sm["composite"] is True
    assert sm["not_rmsnorm"] is True
    assert sm["subprimitives"] == ("split", "qk_l2norm", "mrope")


def test_prepare_hbm_service_conserves_read_plus_write_and_epsilon_identity():
    model = load_model("deepseek-v4.1-flash")
    parallel = ParallelSpec(tp=16, ep=16, pp=4)
    runtime = _runtime(tp=16, ep=16, pp=4, max_model_len=64,
                       max_num_batched_tokens=8)
    candidates = catalog_prepare_candidates(model, runtime, parallel)
    layouts = [c for c in candidates if c.kind == "quant_scale_layout"]
    assert layouts
    for candidate in layouts:
        resources = dict(candidate.resources)
        assert candidate.service_bytes == candidate.read_bytes + candidate.write_bytes
        assert resources["hbm_bytes"] == candidate.service_bytes
        assert resources["mte2_bytes"] == candidate.read_bytes
        assert resources["mte3_bytes"] == candidate.write_bytes

    v4 = load_model("deepseek-v4-10t")
    base = infer_iteration(
        _views((128, 0, True, "prefill")), v4,
        _runtime(tp=8, ep=8, max_num_batched_tokens=128),
        ParallelSpec(tp=8, ep=8),
    )
    norm = next(op for op in base.ops if op.op_id == "s0.l0.q_norm")
    assert dict(norm.metadata)["norm_eps"] == v4.norm_eps
    from dataclasses import replace as dc_replace
    changed = dc_replace(v4, norm_eps=1.25e-5)
    changed_work = infer_iteration(
        _views((128, 0, True, "prefill")), changed,
        _runtime(tp=8, ep=8, max_num_batched_tokens=128),
        ParallelSpec(tp=8, ep=8),
    )
    changed_norm = next(op for op in changed_work.ops if op.op_id == "s0.l0.q_norm")
    assert dict(changed_norm.metadata)["norm_eps"] == 1.25e-5
    from mindie_sim.a3_operator_campaign_conditions import _identity_features
    base_features = _identity_features({"metadata": dict(norm.metadata)})
    changed_features = _identity_features({"metadata": dict(changed_norm.metadata)})
    assert base_features != changed_features


def test_unique_head_and_embedding_row_mandatory_weight_accounting():
    from dataclasses import replace as dc_replace
    from mindie_sim.a3_graph_lowering import catalog_semantic_work
    from mindie_sim.a3_catalog_weights import (
        body_head_terms, layer_weight_terms, term_bytes,
    )
    model = load_model("qwen3.8-27b")
    parallel = ParallelSpec()
    layer_bytes = sum(
        term_bytes(t, model.quant, parallel, 1)
        for layer in model.layer_plans
        for t in layer_weight_terms(layer, model)
    )
    head_bytes = sum(
        term_bytes(t, model.quant, parallel, 1)
        for t in body_head_terms(model)
        if t.name != "embedding" and t.source_owned
    )
    row = model.hidden_size * 2
    one = catalog_semantic_work(
        model, WorkloadSpec((RequestSpec("r", 0, 1, 1),)), parallel,
    )
    assert one.weight_bytes == layer_bytes + head_bytes + row
    two = catalog_semantic_work(
        model, WorkloadSpec((RequestSpec("r", 0, 1, 2),)), parallel,
    )
    # Weights are re-read on the second forward; only the new decode input row
    # is added for embeddings.
    assert two.weight_bytes == 2 * (layer_bytes + head_bytes) + 2 * row
    prefix = catalog_semantic_work(
        model,
        WorkloadSpec((RequestSpec("r", 0, 1, 1, prefix_tokens=1),)),
        parallel,
    )
    assert prefix.weight_bytes == layer_bytes + head_bytes
    tied = dc_replace(model, tie_word_embeddings=True)
    tied_head = tied.vocab_size * tied.hidden_size * 2
    final_norm_bytes = sum(
        term_bytes(t, tied.quant, parallel, 1)
        for t in body_head_terms(tied) if t.name == "final_norm"
    )
    tied_one = catalog_semantic_work(
        tied, WorkloadSpec((RequestSpec("r", 0, 1, 1),)), parallel,
    )
    assert tied_one.weight_bytes == (
        layer_bytes + final_norm_bytes + tied_head + row
    )
    tied_two = catalog_semantic_work(
        tied, WorkloadSpec((RequestSpec("r", 0, 1, 2),)), parallel,
    )
    # Tied head is one unique physical read, while untied head is read per
    # forward just like the other head terms.
    assert tied_two.weight_bytes == (
        2 * (layer_bytes + final_norm_bytes) + tied_head + 2 * row
    )


def test_scheduled_draft_uses_per_view_counts_and_budget_fallback():
    from mindie_sim.a3_workload import IterationPlan, RequestView

    model = load_model("deepseek-v4.1-flash")
    parallel = ParallelSpec(tp=8, ep=8, pp=1)
    runtime = _runtime(tp=8, ep=8, pp=1, metadata={
        "draft_mode": "dspark", "draft_accept_tokens": 5,
    })

    def work_for(*views):
        return infer_iteration(IterationPlan(tuple(views)), model, runtime, parallel)

    b2 = work_for(RequestView(
        "r0", 1, 2, True, "decode", draft_verify=True,
        draft_proposal_tokens=1, draft_accepted_tokens=1,
    ))
    verify = next(op for op in b2.ops if op.op_id == "verify.embedding")
    assert verify.local_shape[0] == 1
    body_embedding = next(op for op in b2.ops if op.op_id == "embedding")
    assert body_embedding.local_shape[0] == 2
    q_a = next(op for op in b2.ops if op.op_id == "s0.l0.q_a")
    assert q_a.local_shape[0] == 2
    assert sum(op.op_id == "s0.l0.norm" for op in b2.ops) == 1
    assert not any(
        op.op_id.startswith("verify.s0.")
        and not dict(op.metadata).get("zero_work_alias", False)
        for op in b2.ops
    )
    # The configured block is emitted for the next iteration, not for the
    # current verify rows.
    assert sum(op.op_id.startswith("draft.next.draft.sample") for op in b2.ops) == 5
    assert any(op.op_id == "draft.next.proposals_ready" for op in b2.ops)
    commit = next(op for op in b2.ops if op.op_id == "draft.state_commit")
    meta = dict(commit.metadata)
    assert meta["scheduled_proposal_counts"] == (1,)
    assert meta["configured_proposal_tokens"] == 5
    assert meta["target_body_reused"] is True
    assert meta["emitted_candidate_cap"] == 5
    assert meta["unverified_candidate_tokens"] == 4
    assert meta["discarded_candidate_tokens"] == 4

    b6 = work_for(RequestView(
        "r0", 1, 2, True, "decode", draft_verify=True,
        draft_proposal_tokens=5, draft_accepted_tokens=5,
    ))
    verify = next(op for op in b6.ops if op.op_id == "verify.embedding")
    assert verify.local_shape[0] == 5
    body_embedding = next(op for op in b6.ops if op.op_id == "embedding")
    assert body_embedding.local_shape[0] == 6
    assert next(op for op in b6.ops
                if op.op_id == "s0.l0.q_a").local_shape[0] == 6
    assert sum(op.op_id == "s0.l0.norm" for op in b6.ops) == 1
    assert not any(
        op.op_id.startswith("verify.s0.")
        and not dict(op.metadata).get("zero_work_alias", False)
        for op in b6.ops
    )

    mixed = work_for(
        RequestView("long", 1, 2, True, "decode", draft_verify=True,
                    draft_proposal_tokens=5, draft_accepted_tokens=5),
        RequestView("short", 1, 2, True, "decode", draft_verify=True,
                    draft_proposal_tokens=1, draft_accepted_tokens=1),
    )
    verify = next(op for op in mixed.ops if op.op_id == "verify.embedding")
    assert verify.local_shape[0] == 6
    body_embedding = next(op for op in mixed.ops if op.op_id == "embedding")
    assert body_embedding.local_shape[0] == 8
    assert not any(
        op.op_id.startswith("verify.s0.")
        and not dict(op.metadata).get("zero_work_alias", False)
        for op in mixed.ops
    )
    commit = next(op for op in mixed.ops if op.op_id == "draft.state_commit")
    assert dict(commit.metadata)["scheduled_proposal_counts"] == (5, 1)
    assert dict(commit.metadata)["emitted_candidate_cap"] == 10
    assert dict(commit.metadata)["unverified_candidate_tokens"] == 4
    assert any(op.op_id == "draft.next.proposals_ready" for op in mixed.ops)

    fallback = work_for(RequestView(
        "r0", 1, 2, True, "decode", draft_verify=False,
        draft_proposal_tokens=0,
    ))
    assert not any(op.op_id.startswith("draft.") or op.op_id.startswith("verify.")
                   for op in fallback.ops)


def test_scheduled_draft_result_exposes_target_query_and_composition():
    from types import SimpleNamespace
    from mindie_sim.a3_catalog_aux import lower_optional_draft
    model = load_model("deepseek-v4.1-flash")
    result = lower_optional_draft(
        model=model,
        runtime=SimpleNamespace(metadata=(
            ("draft_mode", "dspark"), ("draft_accept_tokens", "5"),
        )),
        parallel=ParallelSpec(tp=8, ep=8, pp=1), quant=model.quant,
        views=(
            RequestView("a", 1, 2, True, "decode", draft_verify=True,
                        draft_proposal_tokens=1, draft_accepted_tokens=1),
            RequestView("b", 1, 2, True, "decode", draft_verify=True,
                        draft_proposal_tokens=5, draft_accepted_tokens=5),
        ),
        context=3, proposal_counts=(1, 5),
    )
    assert result.configured_proposal_tokens == 5
    assert result.scheduled_proposal_counts == (1, 5)
    assert result.emitted_proposal_counts == (5, 5)
    assert result.emitted_proposal_tokens == 10
    assert result.proposal_tokens == 6
    assert result.ordinary_query_rows == 2
    assert result.target_query_rows == 2
    assert not any(op.op_id.startswith("verify.") for op in result.ops)


def test_prefill_sampling_generates_initial_next_proposals_once():
    from mindie_sim.a3_workload import IterationPlan, RequestView
    model = load_model("deepseek-v4.1-flash")
    parallel = ParallelSpec(tp=8, ep=8, pp=1)
    runtime = _runtime(tp=8, ep=8, pp=1, metadata={
        "draft_mode": "dspark", "draft_accept_tokens": 5,
    })
    view = RequestView(
        "r0", 2, 0, True, "prefill", draft_verify=False,
        draft_proposal_tokens=0,
    )
    work = infer_iteration(IterationPlan((view,)), model, runtime, parallel)
    assert work.valid_tokens == 2
    assert not any(op.op_id.startswith("verify.") for op in work.ops)
    assert not any(op.op_id.startswith("verify.s0.") for op in work.ops)
    assert sum(op.op_id.startswith("draft.next.draft.sample") for op in work.ops) == 5
    assert any(op.op_id == "draft.next.proposals_ready" for op in work.ops)


def test_public_joint_speculative_target_query_budget_b2_b6():
    from mindie_sim.a3_framework import predict
    case = json.loads((
        Path(__file__).resolve().parents[1]
        / "examples/model-reference/cases/deepseek-v4.1-flash.json"
    ).read_text())
    case["runtime"].update(
        tp=64, ep=64, pp=1, dp=1, max_num_seqs=2, kv_block_size=1,
        metadata={"draft_mode": "dspark", "draft_accept_tokens": 5},
    )
    case["workload"] = {"requests": [{
        "request_id": "r0", "prompt_tokens": 2, "output_tokens": 7,
    }]}
    expected = {2: ([2, 2, 2, 2], [0, 1, 1, 1]),
                6: ([2, 6], [0, 5])}
    for budget, (queries, drafts) in expected.items():
        case["runtime"]["max_num_batched_tokens"] = budget
        result = predict(case)
        assert result["status"] == "modeled"
        request = result["requests"][0]
        assert (request["generated_tokens"], request["computed_tokens"]) == (7, 8)
        assert [trace["valid_tokens"] for trace in result["traces"]] == queries
        assert any(op["op_id"].startswith("draft.next.draft.sample")
                   for op in result["traces"][0]["ops"])
        verify_rows = []
        for trace in result["traces"]:
            qkv = [op for op in trace["ops"] if op["op_id"] == "s0.l0.q_a"]
            assert qkv and qkv[0]["local_shape"][0] == trace["valid_tokens"]
            target_aliases = [
                op for op in trace["ops"]
                if op["op_id"].startswith("verify.s0.")
            ]
            assert len(target_aliases) <= 1
            verify = [op for op in trace["ops"]
                      if op["op_id"] == "verify.embedding"]
            verify_rows.append(verify[0]["local_shape"][0] if verify else 0)
        assert verify_rows == drafts


def test_draft_next_counts_suppresses_initial_candidate_network():
    from mindie_sim.a3_workload import IterationPlan, RequestView
    model = load_model("deepseek-v4.1-flash")
    parallel = ParallelSpec(tp=8, ep=8, pp=1)
    runtime = _runtime(tp=8, ep=8, pp=1, metadata={
        "draft_mode": "dspark", "draft_accept_tokens": 5,
        "draft_next_counts": '{"r0": 0}',
    })
    work = infer_iteration(
        IterationPlan((RequestView(
            "r0", 2, 0, True, "prefill", draft_verify=False,
            draft_proposal_tokens=0,
        ),)), model, runtime, parallel,
    )
    assert not any(op.op_id.startswith("draft.next.") for op in work.ops)


def test_packed_swiglu_explicit_read_write_traffic_and_vision_gating():
    from mindie_sim.a3_catalog_media import lower_media
    from mindie_sim.a3_types import dtype_bytes

    def check_packed(op, *, gate_op):
        m, n, _ = gate_op.launch_shape[-3:]
        width = n // 2
        b = dtype_bytes(op.dtype)
        elements = m * width
        assert op.semantic_shape == (m, width)
        assert op.mte2_bytes == 2 * elements * b
        assert op.mte3_bytes == elements * b
        assert op.hbm_bytes == 3 * elements * b

    qwen = load_model("qwen3.8-27b")
    work = infer_iteration(
        _views((8, 0, True, "prefill")), qwen,
        _runtime(tp=8, ep=1, max_num_batched_tokens=8),
        ParallelSpec(tp=8, ep=1),
    )
    by_id = {op.op_id: op for op in work.ops}
    check_packed(by_id["s0.l0.act"], gate_op=by_id["s0.l0.gate_up"])

    v41 = load_model("deepseek-v4.1-flash")
    media = lower_media(
        model=v41, runtime=SimpleNamespace(media_tokens=4, metadata=()),
        parallel=ParallelSpec(tp=1), quant=v41.quant, mode="prefill",
        views=(RequestView("r", 4, 0, True, "prefill"),),
    )
    media_by_id = {op.op_id: op for op in media.ops}
    check_packed(media_by_id["vision.l0.act"], gate_op=media_by_id["vision.l0.fc0"])

    qmedia = lower_media(
        model=qwen, runtime=SimpleNamespace(media_tokens=4, metadata=()),
        parallel=ParallelSpec(tp=1), quant=qwen.quant, mode="prefill",
        views=(RequestView("r", 4, 0, True, "prefill"),),
    )
    qact = next(op for op in qmedia.ops if op.op_id == "vision.l0.act")
    qb = dtype_bytes(qact.dtype)
    qelements = qact.semantic_shape[0] * qact.semantic_shape[1]
    # Non-gated GELU keeps the ordinary one-read/one-write contract.
    assert qact.mte2_bytes == qelements * qb
    assert qact.mte3_bytes == qelements * qb
    assert qact.hbm_bytes == 2 * qelements * qb
