"""Regression tests for MLA replicated-group SP closure and GQA EAGLE drafting."""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.catalog.attention_common import _group_plan
from mindie_sim.catalog.ops import comm_op, gemm_op
from mindie_sim.compose import compose_case, prepare_speculation, resolve_hardware_sku
from mindie_sim.export.memory_ledger import compute_memory_quant_ledger
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.frontend.body_spec import body_forward
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.sharding import sequence_parallel
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import QuantSpec
from mindie_sim.schema.workload import RequestView

pytestmark = pytest.mark.tier0

CONTEXT_128K = 131_072
OUTPUT_1K = 1024
EXTENDED_MAX_MODEL_LEN = CONTEXT_128K + OUTPUT_1K
NATIVE_PROMPT_32K = 32_768
PREFILL_REPLAY_TOKENS = 1024
MTP_PROPOSALS = 5
MTP_ACCEPT_LEN = 3.5
DECODE_VERIFY_TOKENS = 1 + MTP_PROPOSALS

V41_LAYERS = 40
V41_HIDDEN_SIZE = 5120
V41_NUM_HEADS = 64
V41_HEAD_DIM = 512
V41_O_GROUPS = 8
V41_TP_REPLICATED = 16
V41_EP_REPLICATED = 16
V41_TP_SHARDED = 8
V41_EP_SHARDED = 8
V41_REPLAY_LAYERS = 2

QWEN_TP = 4
QWEN_REPLAY_LAYERS = 2
QWEN_NUM_KV_HEADS = 8
QWEN_HEAD_DIM = 128

UNIT_TP = 4
UNIT_WIDTH = 32
UNIT_DIVISIBLE_ROWS = 16
UNIT_UNPADDED_ROWS = 6
UNIT_PHYSICAL_ROWS = 8
UNIT_OVERFLOW_ROWS = 17
UNIT_K_WIDTH = 64


def test_mla_replicated_grouped_output_sp_closure_prefill_and_decode():
    """Grouped MLA output with o_groups < tp closes SP via local row shard without allreduce."""
    raw = compose_case(
        "deepseek-v4.1-flash", "a3",
        tp=V41_TP_REPLICATED, ep=V41_EP_REPLICATED, dp=1, sp=True,
        prompt_tokens=CONTEXT_128K, output_tokens=OUTPUT_1K,
        mtp_proposal_tokens=MTP_PROPOSALS, mtp_acceptance_length=MTP_ACCEPT_LEN,
    )
    case = parse_case(raw)
    assert case.machine.sku == resolve_hardware_sku("a3")
    assert case.model.layers == V41_LAYERS
    assert case.model.layer_plans[0].o_groups == V41_O_GROUPS
    group_parallel, groups = _group_plan(case.model.layer_plans[0], case.runtime.parallel)
    assert (group_parallel, groups) == (False, V41_O_GROUPS)

    act_bytes = dtype_bytes(case.model.activation_dtype)
    local_heads = V41_NUM_HEADS // V41_TP_REPLICATED

    for phase, q_tokens, context in (
        ("prefill", CONTEXT_128K, 0),
        ("decode", DECODE_VERIFY_TOKENS, CONTEXT_128K),
    ):
        graphs = target_graphs(
            case, phase=phase, tokens=q_tokens, context=context, sequences=1, scope="model",
        )
        ops = graphs[0]
        by_id = {op.op_id: op for op in ops}
        target_ops = [op for op in ops if not op.op_id.startswith("draft") and op.layer >= 0]
        shards = [op for op in target_ops if op.op_id.endswith(".grouped_output_b.sp_shard")]
        gathers = [op for op in target_ops if op.recipe == "output_gather"]
        attn_tp_ars = [
            op for op in target_ops
            if op.family == "allreduce"
            and dict(op.metadata).get("group_axis") == "tp"
            and dict(op.metadata).get("structure_block") == "attention"
        ]
        attn_rs = [
            op for op in target_ops
            if op.family == "reduce_scatter"
            and dict(op.metadata).get("structure_block") == "attention"
        ]
        assert len(shards) == V41_LAYERS
        assert len(gathers) == V41_LAYERS
        assert not attn_tp_ars
        assert not attn_rs

        expected_gather_bytes = (
            q_tokens * (local_heads * V41_HEAD_DIM) * act_bytes * (V41_TP_REPLICATED - 1)
        )
        assert gathers[0].link_bytes == pytest.approx(expected_gather_bytes)

        shard0 = shards[0]
        producer = by_id[shard0.deps[0]]
        assert producer.recipe == "grouped_output_b"
        assert dict(producer.metadata).get("tp_output_placement") == "replicated"
        shard_meta = dict(shard0.metadata)
        assert shard0.link_bytes == 0.0
        assert shard_meta["sequence_parallel_role"] == "output_shard"
        assert shard_meta["input_dependency"] == "replicated_tp_output"
        assert shard_meta["tp_output_placement"] == "replicated"

        if q_tokens % V41_TP_REPLICATED == 0:
            local_rows = q_tokens // V41_TP_REPLICATED
            assert shard0.family == "barrier"
            assert shard0.recipe == "sequence_parallel_view"
            assert shard0.local_shape == (local_rows, V41_HIDDEN_SIZE)
            assert (shard0.mte2_bytes, shard0.mte3_bytes, shard0.hbm_bytes) == (0.0, 0.0, 0.0)
        else:
            physical_rows = q_tokens + (-q_tokens) % V41_TP_REPLICATED
            local_rows = physical_rows // V41_TP_REPLICATED
            expected_read = float(q_tokens * V41_HIDDEN_SIZE * act_bytes)
            expected_write = float(physical_rows * V41_HIDDEN_SIZE * act_bytes)
            assert shard0.family == "pad"
            assert shard0.recipe == "sequence_parallel_right_pad"
            assert shard0.local_shape == (local_rows, V41_HIDDEN_SIZE)
            assert shard0.mte2_bytes == pytest.approx(expected_read)
            assert shard0.mte3_bytes == pytest.approx(expected_write)
            assert shard0.hbm_bytes == pytest.approx(expected_read + expected_write)
            assert shard0.vector_ops == (physical_rows - q_tokens) * V41_HIDDEN_SIZE

    raw_replay = compose_case(
        "deepseek-v4.1-flash", "a3",
        tp=V41_TP_REPLICATED, ep=V41_EP_REPLICATED, dp=1, sp=True, layers=V41_REPLAY_LAYERS,
        prompt_tokens=CONTEXT_128K, output_tokens=OUTPUT_1K,
        mtp_proposal_tokens=MTP_PROPOSALS, mtp_acceptance_length=MTP_ACCEPT_LEN,
    )
    dec_replay = body_forward(
        raw_replay, phase="decode", tokens=DECODE_VERIFY_TOKENS,
        context=CONTEXT_128K, sequences=1, scope="model",
    )
    pre_replay = body_forward(
        raw_replay, phase="prefill", tokens=PREFILL_REPLAY_TOKENS,
        context=0, sequences=1, scope="model",
    )
    for report in (dec_replay, pre_replay):
        assert report["body_forward_ms"] > 0.0
        assert report["execution"]["live_tasks"] == 0
        assert report["execution"]["live_programs"] == 0
        assert not any(report["execution"]["live_allocated"].values())
    assert dec_replay["included_components"]["draft_layers"] > 0


def test_sequence_parallel_close_output_replicated_vs_fail_fast_contracts():
    """close_output keeps TP-allreduce fail-fast strict and preserves sharded o_groups RS."""
    raw = compose_case(
        "deepseek-v4.1-flash", "a3",
        tp=V41_TP_SHARDED, ep=V41_EP_SHARDED, dp=1, sp=True, layers=V41_REPLAY_LAYERS,
        prompt_tokens=CONTEXT_128K, output_tokens=OUTPUT_1K,
        mtp_proposal_tokens=MTP_PROPOSALS, mtp_acceptance_length=MTP_ACCEPT_LEN,
    )
    sharded_case = parse_case(raw)
    sharded_ops = target_graphs(
        sharded_case, phase="decode", tokens=DECODE_VERIFY_TOKENS,
        context=CONTEXT_128K, sequences=1, scope="body",
    )[0]
    assert not any(op.op_id.endswith(".grouped_output_b.sp_shard") for op in sharded_ops)
    assert any(
        op.family == "reduce_scatter"
        and dict(op.metadata).get("structure_block") == "attention"
        for op in sharded_ops
    )

    quant = QuantSpec(name="bf16", weight_format="bf16", activation_format="bf16")
    model = load_model("qwen3-32b")
    rt_base = parse_case(compose_case(
        "qwen3.5-35b-a3b", "a3", tp=UNIT_TP, ep=UNIT_TP, dp=1, sp=True, layers=1,
    )).runtime
    div_views = (RequestView("r0", UNIT_DIVISIBLE_ROWS, 0, True, "prefill"),)
    pad_views = (RequestView("r0", UNIT_UNPADDED_ROWS, 0, True, "decode"),)
    div_plan = sequence_parallel.select(rt_base, model, div_views, tokens=UNIT_DIVISIBLE_ROWS)
    pad_plan = sequence_parallel.select(rt_base, model, pad_views, tokens=UNIT_UNPADDED_ROWS)
    assert div_plan is not None and pad_plan is not None
    assert pad_plan.physical_global == UNIT_PHYSICAL_ROWS

    plain_gemm = replace(
        gemm_op(
            op_id="o_proj", deps=("attn",), stage=0, layer=0, requests=("r0",),
            m=UNIT_DIVISIBLE_ROWS, n=UNIT_WIDTH, k=UNIT_K_WIDTH,
            quant=quant, category="attn.o", phase="prefill", family="gemm", recipe="output",
        ),
        stream="aux",
    )
    with pytest.raises(ModelingError, match="sequence_parallel_close_output_requires_tp_allreduce_tail"):
        sequence_parallel.close_output(
            (plain_gemm,), plain_gemm.op_id, plan=div_plan, width=UNIT_WIDTH, dtype="bf16",
        )

    ep_ar = comm_op(
        op_id="ep_reduce", deps=("attn",), stage=0, layer=0, requests=("r0",),
        family="allreduce", phase="prefill", recipe="ring_allreduce",
        link_bytes=1.0, shape=(UNIT_DIVISIBLE_ROWS, UNIT_WIDTH),
        metadata=(("group_axis", "ep"),),
    )
    with pytest.raises(ModelingError, match="sequence_parallel_close_output_requires_tp_allreduce_tail"):
        sequence_parallel.close_output(
            (ep_ar,), ep_ar.op_id, plan=div_plan, width=UNIT_WIDTH, dtype="bf16",
        )

    replicated = replace(
        plain_gemm,
        metadata=plain_gemm.metadata + (("tp_output_placement", "replicated"),),
    )
    div_ops, div_last = sequence_parallel.close_output(
        (replicated,), replicated.op_id, plan=div_plan, width=UNIT_WIDTH, dtype="bf16",
        input_rows=UNIT_DIVISIBLE_ROWS,
    )
    assert len(div_ops) == 2 and div_ops[0].op_id == replicated.op_id
    assert div_ops[-1].op_id == div_last and div_ops[-1].stream == "aux"
    assert div_ops[-1].family == "barrier" and div_ops[-1].link_bytes == 0.0
    assert div_ops[-1].local_shape == (UNIT_DIVISIBLE_ROWS // UNIT_TP, UNIT_WIDTH)

    pad_ops, _ = sequence_parallel.close_output(
        (replicated,), replicated.op_id, plan=pad_plan, width=UNIT_WIDTH, dtype="bf16",
        input_rows=UNIT_UNPADDED_ROWS,
    )
    act_b = dtype_bytes("bf16")
    assert pad_ops[-1].family == "pad" and pad_ops[-1].link_bytes == 0.0
    assert pad_ops[-1].local_shape == (UNIT_PHYSICAL_ROWS // UNIT_TP, UNIT_WIDTH)
    assert pad_ops[-1].mte2_bytes == pytest.approx(UNIT_UNPADDED_ROWS * UNIT_WIDTH * act_b)
    assert pad_ops[-1].mte3_bytes == pytest.approx(UNIT_PHYSICAL_ROWS * UNIT_WIDTH * act_b)

    with pytest.raises(ModelingError, match="sequence_parallel_close_output_duplicate_id"):
        sequence_parallel.close_output(
            (replicated,), replicated.op_id, plan=div_plan, width=UNIT_WIDTH, dtype="bf16",
            op_id=replicated.op_id,
        )
    with pytest.raises(ModelingError, match="sequence_parallel_reduce_scatter_rows_outside_allocation"):
        sequence_parallel.close_output(
            (replicated,), replicated.op_id, plan=div_plan, width=UNIT_WIDTH, dtype="bf16",
            input_rows=UNIT_OVERFLOW_ROWS,
        )


def test_gqa_explicit_mtp_synthesizes_eagle_draft_and_replays_at_long_context():
    """Explicit modeling opt-in on a GQA model builds a tagged 1-layer synthetic EAGLE3 draft."""
    raw_128k = compose_case(
        "qwen3-32b", "a3",
        tp=QWEN_TP, dp=1, layers=QWEN_REPLAY_LAYERS,
        prompt_tokens=CONTEXT_128K, output_tokens=OUTPUT_1K,
        max_model_len=EXTENDED_MAX_MODEL_LEN,
        mtp_proposal_tokens=MTP_PROPOSALS, mtp_acceptance_length=MTP_ACCEPT_LEN,
        feature_track="expected_ideal", allow_synthetic_gqa_draft=True,
    )
    case = parse_case(raw_128k)
    assert case.machine.sku == resolve_hardware_sku("a3")
    assert case.runtime.max_model_len == EXTENDED_MAX_MODEL_LEN
    assert case.runtime.mtp_acceptance_length == pytest.approx(MTP_ACCEPT_LEN)
    meta = dict(case.runtime.metadata)
    assert meta["draft_mode"] == "mtp"
    assert meta["draft_proposal_tokens"] == MTP_PROPOSALS

    assert case.model.auxiliary is not None and case.model.auxiliary.mtp is not None
    mtp = case.model.auxiliary.mtp
    assert mtp.kind == "eagle3"
    assert mtp.is_synthetic is True
    assert mtp.draft_provenance == "synthetic_eagle3_gqa"
    assert mtp.layers == 1
    assert mtp.attention_kind == "gqa"
    assert mtp.execution_mode == "eagle3_serial"
    assert mtp.kv_width == QWEN_NUM_KV_HEADS * QWEN_HEAD_DIM
    assert mtp.kv_cache_elements == 2 * QWEN_NUM_KV_HEADS * QWEN_HEAD_DIM
    assert mtp.qk_norm is True
    assert mtp.target_shared_lm_head is True
    assert len(mtp.layer_plans) == 1 and mtp.layer_plans[0].scope == "draft"

    # Memory receipt and export ledger must agree byte-for-byte on draft + target weights.
    mem = Memory(case)
    ledger = compute_memory_quant_ledger(case)
    assert mem.weights[0] == pytest.approx(ledger.weight_footprint_per_die_bytes)

    graphs = target_graphs(
        case, phase="decode", tokens=DECODE_VERIFY_TOKENS,
        context=CONTEXT_128K, sequences=1, scope="model",
    )
    draft_prefixes = sorted({
        op.op_id.split(".", 1)[0] for op in graphs[0] if op.op_id.startswith("draft")
    })
    assert draft_prefixes == [f"draft{idx}" for idx in range(MTP_PROPOSALS)]

    dec_report = body_forward(
        raw_128k, phase="decode", tokens=DECODE_VERIFY_TOKENS,
        context=CONTEXT_128K, sequences=1, scope="model",
    )
    pre_report = body_forward(
        raw_128k, phase="prefill", tokens=PREFILL_REPLAY_TOKENS,
        context=0, sequences=1, scope="model",
    )
    for report in (dec_report, pre_report):
        assert report["body_forward_ms"] > 0.0
        assert report["execution"]["live_tasks"] == 0
        assert report["execution"]["live_programs"] == 0
        assert not any(report["execution"]["live_allocated"].values())
    assert dec_report["included_components"]["draft_layers"] == 1

    # Native 32K horizon also composes cleanly with explicit modeling opt-in.
    raw_32k = compose_case(
        "qwen3-32b", "a3",
        tp=QWEN_TP, dp=1, layers=QWEN_REPLAY_LAYERS,
        prompt_tokens=NATIVE_PROMPT_32K, output_tokens=OUTPUT_1K,
        mtp_proposal_tokens=MTP_PROPOSALS, mtp_acceptance_length=MTP_ACCEPT_LEN,
        feature_track="expected_ideal", allow_synthetic_gqa_draft=True,
    )
    assert raw_32k["runtime"]["metadata"]["draft_proposal_tokens"] == MTP_PROPOSALS


def test_gqa_synthetic_draft_preserves_fail_fast_and_profile_opt_out_contracts():
    """Simulation never fabricates drafts; modeling requires explicit opt-in; horizon stays strict."""
    qwen_model = load_model("qwen3-32b")
    with pytest.raises(ModelingError, match="draft_structure_required_for_mtp_proposal_tokens"):
        prepare_speculation(qwen_model, MTP_PROPOSALS)

    with pytest.raises(ModelingError, match="draft_structure_required_for_mtp_proposal_tokens"):
        compose_case(
            "qwen3-32b", "a3", tp=QWEN_TP, dp=1, layers=QWEN_REPLAY_LAYERS,
            prompt_tokens=NATIVE_PROMPT_32K, output_tokens=OUTPUT_1K,
            mtp_proposal_tokens=MTP_PROPOSALS, mtp_acceptance_length=MTP_ACCEPT_LEN,
        )

    with pytest.raises(ModelingError, match="simulation_mode_forbids_synthetic_gqa_draft"):
        compose_case(
            "qwen3-32b", "a3", tp=QWEN_TP, dp=1, layers=QWEN_REPLAY_LAYERS,
            prompt_tokens=NATIVE_PROMPT_32K, output_tokens=OUTPUT_1K,
            mtp_proposal_tokens=MTP_PROPOSALS, mtp_acceptance_length=MTP_ACCEPT_LEN,
            allow_synthetic_gqa_draft=True,
        )

    with pytest.raises(ModelingError, match="draft_structure_required_for_mtp_proposal_tokens"):
        compose_case(
            "qwen3-32b", "a3", tp=QWEN_TP, dp=1, layers=QWEN_REPLAY_LAYERS,
            prompt_tokens=NATIVE_PROMPT_32K, output_tokens=OUTPUT_1K,
            mtp_proposal_tokens=MTP_PROPOSALS, mtp_acceptance_length=MTP_ACCEPT_LEN,
            feature_track="expected_ideal",
        )

    non_gqa = replace(qwen_model, attention="linear")
    with pytest.raises(ModelingError, match="draft_structure_required_for_mtp_proposal_tokens"):
        prepare_speculation(non_gqa, MTP_PROPOSALS, allow_synthetic_gqa=True)

    # Without explicit mtp_proposal_tokens, profile mode never fabricates a synthetic GQA draft.
    profile_raw = compose_case(
        "qwen3-32b", "a3", tp=QWEN_TP, dp=1, layers=QWEN_REPLAY_LAYERS,
        prompt_tokens=NATIVE_PROMPT_32K, output_tokens=OUTPUT_1K,
    )
    profile_aux = profile_raw["model"].get("auxiliary")
    assert profile_aux is None or (
        profile_aux.get("mtp") is None and profile_aux.get("dspark") is None
    )
    assert profile_raw["runtime"]["metadata"]["execution_policy"]["mtp_proposal_tokens"] == 0

    # Exceeding native max_position_embeddings without an explicit max_model_len override still fails.
    with pytest.raises(ModelingError, match="workload_exceeds_model_max_position_embeddings"):
        compose_case(
            "qwen3-32b", "a3", tp=QWEN_TP, dp=1, layers=QWEN_REPLAY_LAYERS,
            prompt_tokens=CONTEXT_128K, output_tokens=OUTPUT_1K,
            mtp_proposal_tokens=MTP_PROPOSALS, mtp_acceptance_length=MTP_ACCEPT_LEN,
            feature_track="expected_ideal", allow_synthetic_gqa_draft=True,
        )
