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

V4_FLASH_LAYERS = 43
V4_FLASH_HIDDEN_SIZE = 4096
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
    # V4.1 Engram has 24 lookup heads/table, which fails fast at TP=16 (24 % 16 != 0).
    raw_v41 = compose_case(
        "deepseek-v4.1-flash", "a3",
        tp=V41_TP_REPLICATED, ep=V41_EP_REPLICATED, dp=1, sp=True, layers=V41_REPLAY_LAYERS,
        prompt_tokens=CONTEXT_128K, output_tokens=OUTPUT_1K,
        mtp_proposal_tokens=MTP_PROPOSALS, mtp_acceptance_length=MTP_ACCEPT_LEN,
    )
    with pytest.raises(ModelingError, match="catalog_engram_heads_not_divisible:24%16"):
        target_graphs(parse_case(raw_v41), phase="decode", tokens=DECODE_VERIFY_TOKENS, context=CONTEXT_128K, sequences=1, scope="model")

    # V4-Flash has the same o_groups=8 < tp=16 grouped MLA output geometry without Engram.
    raw = compose_case(
        "deepseek-v4-flash-0731", "a3",
        tp=V41_TP_REPLICATED, ep=V41_EP_REPLICATED, dp=1, sp=True,
        prompt_tokens=CONTEXT_128K, output_tokens=OUTPUT_1K,
        mtp_proposal_tokens=MTP_PROPOSALS, mtp_acceptance_length=MTP_ACCEPT_LEN,
    )
    case = parse_case(raw)
    assert case.machine.sku == resolve_hardware_sku("a3")
    assert case.model.layers == V4_FLASH_LAYERS
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
        assert len(shards) == V4_FLASH_LAYERS
        assert len(gathers) == V4_FLASH_LAYERS
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
            assert shard0.local_shape == (local_rows, V4_FLASH_HIDDEN_SIZE)
            assert (shard0.mte2_bytes, shard0.mte3_bytes, shard0.hbm_bytes) == (0.0, 0.0, 0.0)
        else:
            physical_rows = q_tokens + (-q_tokens) % V41_TP_REPLICATED
            local_rows = physical_rows // V41_TP_REPLICATED
            expected_read = float(q_tokens * V4_FLASH_HIDDEN_SIZE * act_bytes)
            expected_write = float(physical_rows * V4_FLASH_HIDDEN_SIZE * act_bytes)
            assert shard0.family == "pad"
            assert shard0.recipe == "sequence_parallel_right_pad"
            assert shard0.local_shape == (local_rows, V4_FLASH_HIDDEN_SIZE)
            assert shard0.mte2_bytes == pytest.approx(expected_read)
            assert shard0.mte3_bytes == pytest.approx(expected_write)
            assert shard0.hbm_bytes == pytest.approx(expected_read + expected_write)
            assert shard0.vector_ops == (physical_rows - q_tokens) * V4_FLASH_HIDDEN_SIZE

    raw_replay = compose_case(
        "deepseek-v4-flash-0731", "a3",
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
        feature_track="target", allow_synthetic_gqa_draft=True,
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
        feature_track="target", allow_synthetic_gqa_draft=True,
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
            feature_track="target",
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
            feature_track="target", allow_synthetic_gqa_draft=True,
        )


def test_ngram_zero_neural_closure_partial_miss_and_293_step_conservation():
    """NGram has zero neural draft across consumers, honors partial/miss p_i, and conserves 293 steps."""
    from mindie_sim.catalog.draft_spec import NGRAM_CPU_PROVENANCE, SpeculativeExecutionContract
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.prediction.scheduler import Scheduler
    from mindie_sim.schema.model import RequestSpec
    from mindie_sim.schema.parallel import ParallelSpec
    from mindie_sim.schema.workload import IterationPlan

    # Both a no-draft model and a native-DSpark model bypass neural drafting under spec_method="ngram".
    for model_name in ("longcat-flash-lite", "deepseek-v4.1-flash"):
        raw = compose_case(
            model_name, "a3", tp=4, ep=4, dp=1, layers=2,
            prompt_tokens=256, output_tokens=32,
        )
        raw["runtime"]["metadata"] = {
            **raw["runtime"].get("metadata", {}),
            "spec_method": "ngram",
            "draft_mode": "ngram",
            "draft_proposal_tokens": MTP_PROPOSALS,
            "draft_accept_tokens": MTP_ACCEPT_LEN - 1.0,
        }
        case = parse_case(raw)
        view_hit = RequestView("r0", 1, 256, True, "decode", draft_verify=True, draft_proposal_tokens=5, draft_advanced_tokens=3)
        work = Frontend().program(IterationPlan((view_hit,)), case.model, case.runtime, case.machine, assembly="service").work
        # Zero neural draft ops; single verify expansion q_verify = 1 + 5 = 6 (never 11).
        assert not any(op.op_id.startswith("draft") and op.family != "sampling" for op in work.ops)
        emb = next(op for op in work.ops if op.family == "embedding")
        assert emb.local_shape[0] // case.model.hidden_size == 6

        # Partial hit (p=2 -> q=3) and miss (p=0 -> q=1, 0 RecoverSSM commits).
        view_partial = RequestView("r1", 1, 256, True, "decode", draft_verify=True, draft_proposal_tokens=2, draft_advanced_tokens=2)
        view_miss = RequestView("r2", 1, 256, True, "decode", draft_verify=True, draft_proposal_tokens=0, draft_advanced_tokens=1)
        attn_p = next(op for op in Frontend().program(IterationPlan((view_partial,)), case.model, case.runtime, case.machine).work.ops if op.family == "attention")
        attn_m = next(op for op in Frontend().program(IterationPlan((view_miss,)), case.model, case.runtime, case.machine).work.ops if op.family == "attention")
        assert attn_p.local_shape[0] == 3 and attn_m.local_shape[0] == 1
        miss_ops = Frontend().program(IterationPlan((view_miss,)), case.model, case.runtime, case.machine, assembly="service").work.ops
        assert not any("recover_ssm_commit" in op.op_id for op in miss_ops)

    # Dynamic spec widths under DCP=2 fail fast even for ngram (platform.py:2257).
    ngram_contract = SpeculativeExecutionContract(
        archetype="ngram", spec_method="ngram", num_speculative_tokens=5,
        requires_draft_spec=False, neural_draft_forwards=0, requires_draft_prepare=False,
        draft_provenance=NGRAM_CPU_PROVENANCE,
    )
    with pytest.raises(ModelingError, match="dynamic_spec_decode_incompatible_with_dcp"):
        ngram_contract.validate_dcp_spec_gate(None, ParallelSpec(tp=4, dcp=2), dynamic_spec_widths=True)

    # Exact 293-step conservation at O=1024, K=5, u_bar=3.5 (s_bar=2.5) ending (4,3), (1,2).
    raw_sched = compose_case("longcat-flash-lite", "a3", tp=4, ep=4, dp=1, layers=1, prompt_tokens=16, output_tokens=OUTPUT_1K)
    raw_sched["runtime"]["metadata"] = {
        **raw_sched["runtime"].get("metadata", {}),
        "spec_method": "ngram", "draft_mode": "ngram",
        "draft_proposal_tokens": 5, "draft_accept_tokens": 2.5,
    }
    sched_case = parse_case(raw_sched)
    sched = Scheduler(sched_case, [RequestSpec("req0", 0.0, 16, OUTPUT_1K)], Memory(sched_case))
    decode_steps: list[tuple[int, int]] = []
    # Prefill emits token 0.
    p_plan = sched.schedule(0.0)
    assert p_plan is not None and p_plan.requests[0].draft_prepare is False
    sched.commit(1.0)
    t_now = 2.0
    while not sched.done:
        d_plan = sched.schedule(t_now)
        assert d_plan is not None
        v = d_plan.requests[0]
        assert v.q_len == 1 and v.draft_prepare is False
        decode_steps.append((v.draft_proposal_tokens, v.draft_advanced_tokens))
        t_now += 1.0
        sched.commit(t_now)
    assert len(decode_steps) == 293
    assert decode_steps[-2:] == [(4, 3), (1, 2)]
    assert sum(p for p, _ in decode_steps) == 1460
    assert sum(1 + p for p, _ in decode_steps) == 1753
    assert 1 + sum(u for _, u in decode_steps) == OUTPUT_1K


def test_missing_draft_medusa_draft_model_and_eagle3_fail_fast():
    """Strict fail-fast on missing drafts (including MLA with nextn=0), medusa rejection, and zero silent EAGLE3 synthesis."""
    for name in ("longcat-flash-lite", "kimi-k2"):
        model = load_model(name)
        with pytest.raises(ModelingError, match="draft_structure_required_for_mtp_proposal_tokens"):
            prepare_speculation(model, 5)
        with pytest.raises(ModelingError, match="draft_structure_required_for_speculative_method:mtp"):
            prepare_speculation(model, 5, spec_method="mtp")
        with pytest.raises(ModelingError, match="unsupported_speculative_method_on_ascend_v2:medusa"):
            prepare_speculation(model, 5, spec_method="medusa")
        with pytest.raises(ModelingError, match="draft_structure_required_for_speculative_method:draft_model"):
            prepare_speculation(model, 5, spec_method="draft_model")
    qwen = load_model("qwen3-32b")
    with pytest.raises(ModelingError, match="draft_structure_required_for_speculative_method:eagle3"):
        prepare_speculation(qwen, 5, spec_method="eagle3", allow_synthetic_gqa=False)


def test_dspark_128k_zero_ffn_prepare_slot_mapping_none_and_rd2h_proration():
    """Exact integer 128K prepare FLOPs, zero FFN/attention, slot_mapping=None, and 2,064 B RD2H."""
    from mindie_sim.catalog.draft_spec import SpeculativeExecutionContract, dspark_rd2h_transfer_bytes

    expected_flops = {
        "deepseek-v4-flash-0731": 13_743_895_347_200,
        "deepseek-v4-pro-0813": 41_369_124_995_072,
        "deepseek-v4.1-flash": 22_677_427_322_880,
        "kimi-k3": 86_586_540_687_360,
        "kimi-k3.1": 87_187_836_108_800,
    }
    sc = SpeculativeExecutionContract(
        archetype="dspark", spec_method="dspark", num_speculative_tokens=5,
        requires_draft_spec=True, requires_draft_prepare=True,
    )
    for model_name, exp in expected_flops.items():
        raw = compose_case(
            model_name, "a5", tp=1, ep=1, dp=1,
            prompt_tokens=CONTEXT_128K, output_tokens=OUTPUT_1K,
            mtp_proposal_tokens=5, mtp_acceptance_length=3.5,
        )
        case = parse_case(raw)
        draft = case.model.auxiliary.dspark or case.model.auxiliary.mtp  # type: ignore[union-attr]
        assert sc.build_prepare_plan(CONTEXT_128K, draft_spec=draft).exact_gemm_flops == exp
        ops = target_graphs(case, phase="prefill", tokens=CONTEXT_128K, context=0, sequences=1, scope="model", draft_prepare=True)[0]
        d_ops = [op for op in ops if op.op_id.startswith("draft")]
        assert not any(op.family in ("grouped_gemm", "attention") for op in d_ops)
        gemm_flops = int(sum(op.flops for op in d_ops if op.family in ("gemm", "qkv_projection", "projection")))
        prolog_ops = [op for op in d_ops if op.recipe == "draft_context_mla_prolog_v3"]
        if prolog_ops:
            # L-B06: paired_sfa MLA DSpark prefill lowers context KV through fused PrologV3
            # (qa_kva matches exact qkv_a budget; total op.flops includes qb + w_uk).
            qa_kva_flops = int(sum(
                fl for op in prolog_ops for _, fl in dict(op.metadata).get("qa_kva_matrix_flops", ())
            ))
            qb_wuk_flops = int(sum(
                fl for op in prolog_ops for _, fl in dict(op.metadata).get("qb_wuk_matrix_flops", ())
            ))
            assert gemm_flops + qa_kva_flops == exp
            assert int(sum(op.flops for op in prolog_ops)) == qa_kva_flops + qb_wuk_flops
            assert qb_wuk_flops > 0
            # Contiguous draft cache opts out of PrologV3 and emits standalone qkv_a GEMMs == exp.
            raw_contig = compose_case(
                model_name, "a5", tp=1, ep=1, dp=1,
                prompt_tokens=CONTEXT_128K, output_tokens=OUTPUT_1K,
                mtp_proposal_tokens=5, mtp_acceptance_length=3.5,
            )
            raw_contig["runtime"]["metadata"] = {
                **raw_contig["runtime"].get("metadata", {}),
                "draft_cache_contiguous": True,
            }
            d_contig = [
                op for op in target_graphs(
                    parse_case(raw_contig), phase="prefill", tokens=CONTEXT_128K,
                    context=0, sequences=1, scope="model", draft_prepare=True,
                )[0]
                if op.op_id.startswith("draft")
            ]
            assert int(sum(op.flops for op in d_contig if op.family in ("gemm", "qkv_projection", "projection"))) == exp
        else:
            assert gemm_flops == exp

    # slot_mapping=None: V4.1 still runs context_wkv_proj (no cache scatter), K3.1 returns before KV proj.
    for model_name, expect_wkv in (("deepseek-v4.1-flash", True), ("kimi-k3.1", False)):
        raw = compose_case(model_name, "a5", tp=1, ep=1, dp=1, layers=2, prompt_tokens=64, output_tokens=8, mtp_proposal_tokens=5)
        raw["runtime"]["metadata"] = {**raw["runtime"].get("metadata", {}), "dspark_context_slot_mapping_none": True}
        case = parse_case(raw)
        d_ops = [op for op in target_graphs(case, phase="prefill", tokens=64, context=0, sequences=1, scope="model", draft_prepare=True)[0] if op.op_id.startswith("draft")]
        has_wkv = any(op.recipe == "draft_context_wkv_proj" for op in d_ops)
        has_scatter = any(op.family == "cache_store" for op in d_ops)
        assert has_wkv is expect_wkv and has_scatter is False

    # RD2H final-page proration: 129 rows * 16 B on 128-row page == 2,064 B (NOT 4,096 B).
    assert dspark_rd2h_transfer_bytes(129, page_tokens=128, row_bytes=16) == 2064


def test_dspark_decode_sp_collapse_padded_vocab_serial_markov_and_gqa_swa():
    """Q_phys=8, V4.1 71,680 B vs V4 401,408 B, padded vocab 128 cols/rank, 5 serial Markov, GQA SWA 650/783."""
    from mindie_sim.prediction.operators.attention.attention import AttentionPolicy, gqa

    # V4.1 pre-gather collapse (71,680 B) vs V4-Pro post-gather (401,408 B), 5 serial Markov with 0 AllGather.
    for model_name, exp_sp_bytes in (("deepseek-v4.1-flash", 71_680.0), ("deepseek-v4-pro-0813", 401_408.0)):
        raw = compose_case(
            model_name, "a3", tp=8, ep=8, dp=1, sp=True, layers=2,
            prompt_tokens=CONTEXT_128K, output_tokens=OUTPUT_1K,
            mtp_proposal_tokens=5, mtp_acceptance_length=3.5,
        )
        raw["runtime"]["metadata"] = {**raw["runtime"].get("metadata", {}), "draft_capture_bucket": 8}
        case = parse_case(raw)
        ops = target_graphs(case, phase="decode", tokens=6, context=CONTEXT_128K, sequences=1, scope="model")[0]
        sp_out = next(op for op in ops if op.op_id == "draft0.sp_output_gather")
        assert sp_out.local_shape[0] == 8 and sp_out.link_bytes == pytest.approx(exp_sp_bytes)
        markov_gemms = [op for op in ops if op.op_id.startswith("draft0.markov_gemm_")]
        assert len(markov_gemms) == 5
        assert not any("markov" in op.op_id and op.family == "allgather" for op in ops)

    # Padded base vocab (V_d=1000, TP=8, P_v=64 -> 128 cols/rank) + lmhead_tp padding to max_num_seqs * K.
    raw_v = compose_case(
        "deepseek-v4.1-flash", "a3", tp=8, ep=8, dp=1, sp=True, layers=2,
        prompt_tokens=128, output_tokens=16, mtp_proposal_tokens=5, max_num_seqs=4,
    )
    raw_v["runtime"]["metadata"] = {**raw_v["runtime"].get("metadata", {}), "lmhead_tp_enable": True}
    case_v = parse_case(raw_v)
    small_draft = replace(case_v.model.auxiliary.mtp, vocab_size=1000)  # type: ignore[union-attr]
    case_v = replace(case_v, model=replace(case_v.model, auxiliary=replace(case_v.model.auxiliary, mtp=small_draft)))  # type: ignore[arg-type]
    ops_v = target_graphs(case_v, phase="decode", tokens=6, context=128, sequences=1, scope="model")[0]
    d_logits = next(op for op in ops_v if op.op_id == "draft0.logits")
    assert d_logits.local_shape[1] == 128 and d_logits.local_shape[0] == 4 * 5

    # GQA noncausal SWA symmetric band (W-1, W-1) for window=128: exact 650 (q_d=5) and 783 (q_d=6) pairs/head.
    qwen_case = parse_case(compose_case("qwen3-32b", "a3", tp=1, dp=1, layers=1, prompt_tokens=256, output_tokens=16))
    swa_plan = replace(qwen_case.model.layer_plans[0], window=128, noncausal_attention=True)
    for q_d, exp_pairs in ((5, 650.0), (6, 783.0)):
        view = RequestView("r0", q_d, CONTEXT_128K, True, "decode")
        res = gqa(
            prefix="d0.l0", plan=swa_plan, views=(view,), stage=0, deps="in",
            model=qwen_case.model, parallel=qwen_case.runtime.parallel, quant=qwen_case.model.quant,
            mode="draft", padded_tokens=q_d, policy=AttentionPolicy(), runtime=qwen_case.runtime,
            machine=qwen_case.machine,
        )
        attn_op = next(op for op in res.ops if op.family == "attention")
        assert dict(attn_op.metadata)["cp_per_rank_pairs"] == (exp_pairs,)


def test_ascend_v2_mtp_disjoint_modes_pcp_downgrade_and_alltoall_filter():
    """Single-module, mixed tail, cyclic_nextn, multi_module_mtp, PCP downgrade & MoE ALLTOALL filter."""
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.prediction.frontend.speculation import resolve_ar_draft_capture_mode
    from mindie_sim.schema.workload import IterationPlan

    # (a) & (b) Single-module V2 MTP: B=2 mixed tail p=[4, 1] -> step 0 verify (5+2=7), steps 1..3 full B=2.
    raw = compose_case("glm-5.3-flash", "a5", tp=1, dp=1, ep=1, layers=2, prompt_tokens=128, output_tokens=16, mtp_proposal_tokens=4)
    case = parse_case(raw)
    v_mixed = (
        RequestView("r0", 1, 128, True, "decode", draft_verify=True, draft_proposal_tokens=4, draft_advanced_tokens=3),
        RequestView("r1", 1, 128, True, "decode", draft_verify=True, draft_proposal_tokens=1, draft_advanced_tokens=1),
    )
    work = Frontend().program(IterationPlan(v_mixed), case.model, case.runtime, case.machine).work
    d0_feat = next(op for op in work.ops if op.op_id == "draft0.features")
    d1_feat = next(op for op in work.ops if op.op_id == "draft1.features")
    d0_logits = next(op for op in work.ops if op.op_id == "draft0.logits")
    assert d0_feat.local_shape[0] == 7 and d1_feat.local_shape[0] == 2 and d0_logits.local_shape[0] == 2

    # (c) cyclic_nextn 3-block (K=5, B=2): per-req (7, 2, 1) -> batch (14, 4, 2), calls (2, 2, 1).
    cyc_draft = replace(case.model.auxiliary.mtp, layers=3, execution_mode="cyclic_nextn")  # type: ignore[union-attr]
    cyc_draft = replace(cyc_draft, layer_plans=tuple(replace(cyc_draft.layer_plans[0], index=i) for i in range(3)))
    cyc_model = replace(case.model, auxiliary=replace(case.model.auxiliary, mtp=cyc_draft))  # type: ignore[arg-type]
    v_b2 = (
        RequestView("r0", 1, 128, True, "decode", draft_verify=True, draft_proposal_tokens=5, draft_advanced_tokens=3),
        RequestView("r1", 1, 128, True, "decode", draft_verify=True, draft_proposal_tokens=5, draft_advanced_tokens=3),
    )
    rt_k5 = replace(
        case.runtime,
        metadata=tuple((k, 5 if k == "draft_proposal_tokens" else v) for k, v in case.runtime.metadata),
    )
    cyc_ops = Frontend().program(IterationPlan(v_b2), cyc_model, rt_k5, case.machine).work.ops
    cyc_feats = [op.local_shape[0] for op in cyc_ops if op.recipe == "draft_features"]
    assert cyc_feats == [12, 2, 2, 2, 2]
    assert [sum(cyc_feats[0:2]), sum(cyc_feats[2:4]), cyc_feats[4]] == [14, 4, 2]

    # (d) multi_module_mtp (L_d=3, K=3): packed (4,4,4) body rows & KV writes across all 3 modules; non-divisible fails fast.
    mm_draft = replace(cyc_draft, execution_mode="multi_module_mtp")
    mm_model = replace(case.model, auxiliary=replace(case.model.auxiliary, mtp=mm_draft))  # type: ignore[arg-type]
    rt_k3 = replace(
        case.runtime,
        metadata=tuple((k, 3 if k == "draft_proposal_tokens" else v) for k, v in case.runtime.metadata),
    )
    v_k3 = (RequestView("r0", 1, 128, True, "decode", draft_verify=True, draft_proposal_tokens=3, draft_advanced_tokens=2),)
    mm_ops = Frontend().program(IterationPlan(v_k3), mm_model, rt_k3, case.machine).work.ops
    assert sorted({op.op_id.split(".")[0] for op in mm_ops if op.op_id.startswith("draft")}) == ["draft0", "draft1", "draft2"]
    mm_attn_rows = [op.local_shape[0] for op in mm_ops if op.op_id.startswith("draft") and op.family == "attention"]
    mm_kv_rows = [op.local_shape[0] for op in mm_ops if op.op_id.startswith("draft") and op.recipe == "kv_store"]
    assert mm_attn_rows == [4, 4, 4]
    assert mm_kv_rows == [4, 4, 4]
    v_bad_k = (RequestView("r0", 1, 128, True, "decode", draft_verify=True, draft_proposal_tokens=5, draft_advanced_tokens=2),)
    with pytest.raises(ModelingError, match="multi_module_mtp_non_divisible_n_predict"):
        Frontend().program(IterationPlan(v_bad_k), mm_model, rt_k5, case.machine)

    # (e) PCP=2 downgrades step 0 (q > 1) to piecewise; MoE ALLTOALL filters width-1 decode only.
    rt_pcp = replace(
        case.runtime,
        graph_capture_mode="full_and_piecewise",
        parallel=replace(case.runtime.parallel, pcp=2, cp=1, cp_plan="pcp"),
    )
    m0, _ = resolve_ar_draft_capture_mode(rt_pcp, cyc_draft, step=0, decode_query_len=6)
    assert m0 == "piecewise"
    m1_w1, s1_w1 = resolve_ar_draft_capture_mode(
        replace(case.runtime, graph_capture_mode="full_decode_only"), cyc_draft,
        step=1, decode_query_len=1, decode_request_sizes=(1, 2, 4),
        moe_comm_selector=lambda sz: "alltoall" if sz < 4 else "mc2",
    )
    m1_w6, s1_w6 = resolve_ar_draft_capture_mode(
        replace(case.runtime, graph_capture_mode="full_decode_only"), cyc_draft,
        step=0, decode_query_len=6, decode_request_sizes=(1, 2, 4),
        moe_comm_selector=lambda sz: "alltoall",
    )
    assert m1_w1 == "full_decode_only" and s1_w1 == (4,) and m1_w6 == "full_decode_only" and s1_w6 == (1, 2, 4)


def test_speculative_prefill_bootstrap_continuing_width_ngram_and_eager_capture():
    """Verify prefill bootstrap (K=1/3 & DSpark), independent next-K, NGram-GPU homogeneous, eager draft, and per-layer slots."""
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.prediction.scheduler import Scheduler
    from mindie_sim.schema.model import RequestSpec
    from mindie_sim.schema.workload import IterationPlan

    fe = Frontend()
    # 1. Final service prefill seeds proposals for GLM MTP (K=1 and K=3) and DS-V4.1 DSpark (with Markov).
    for k_val in (1, 3):
        c_glm = parse_case(compose_case(
            "glm-5.3-flash", "a3", tp=8, ep=8, dp=1, layers=4,
            prompt_tokens=128, output_tokens=32, mtp_proposal_tokens=k_val, mtp_acceptance_length=1.5,
        ))
        p0 = Scheduler(c_glm, [RequestSpec("r0", 0.0, 128, 32)], Memory(c_glm)).schedule(0.0)
        ops0 = fe.program(p0, c_glm.model, c_glm.runtime, c_glm.machine, assembly="service").work.ops
        assert any(op.op_id.startswith("draft") and op.family == "logits" for op in ops0)

    c_v41 = parse_case(compose_case(
        "deepseek-v4.1-flash", "a3", tp=8, ep=8, dp=1, layers=2,
        prompt_tokens=128, output_tokens=32, mtp_proposal_tokens=3, mtp_acceptance_length=2.5,
    ))
    p0_v41 = Scheduler(c_v41, [RequestSpec("r0", 0.0, 128, 32)], Memory(c_v41)).schedule(0.0)
    ops0_v41 = fe.program(p0_v41, c_v41.model, c_v41.runtime, c_v41.machine, assembly="service").work.ops
    assert any("markov" in op.op_id for op in ops0_v41)

    # 2. Continuing p=0/p=1 decode under K=5 still emits all 5 steps; DSpark ragged next_k=(5,1) -> 6 logits.
    c_glm5 = parse_case(compose_case(
        "glm-5.3-flash", "a3", tp=8, ep=8, dp=1, layers=4,
        prompt_tokens=128, output_tokens=32, mtp_proposal_tokens=5, mtp_acceptance_length=3.5,
    ))
    for prev_p in (0, 1):
        vw = RequestView("r0", 1, 128, True, "decode", draft_verify=True, draft_proposal_tokens=prev_p, draft_advanced_tokens=1)
        ops_c = fe.program(IterationPlan((vw,)), c_glm5.model, c_glm5.runtime, c_glm5.machine, assembly="service").work.ops
        steps = {op.op_id.split(".")[0] for op in ops_c if op.op_id.startswith("draft") and op.op_id[5:6].isdigit()}
        assert len(steps) == 5

    c_v41_5 = parse_case(compose_case(
        "deepseek-v4.1-flash", "a3", tp=8, ep=8, dp=1, layers=2,
        prompt_tokens=128, output_tokens=32, mtp_proposal_tokens=5, mtp_acceptance_length=3.5,
    ))
    v_rag = (
        RequestView("r0", 1, 128, True, "decode", draft_verify=True, draft_proposal_tokens=5, draft_advanced_tokens=3, next_draft_proposal_tokens=5),
        RequestView("r1", 1, 128, True, "decode", draft_verify=True, draft_proposal_tokens=1, draft_advanced_tokens=1, next_draft_proposal_tokens=1),
    )
    ops_rag = fe.program(IterationPlan(v_rag), c_v41_5.model, c_v41_5.runtime, c_v41_5.machine, assembly="service").work.ops
    rag_logits = [op for op in ops_rag if op.op_id.startswith("draft") and op.family == "logits" and "markov" not in op.op_id]
    assert len(rag_logits) == 1 and rag_logits[0].local_shape[0] == 6

    # 3. NGram-GPU emits ngram_gpu_propose on BOTH 2L and 6L homogeneous Qwen across prefill, hit, and miss.
    for nlayers in (2, 6):
        raw_ng = compose_case("qwen3-32b", "a3", tp=4, ep=1, dp=1, layers=nlayers, prompt_tokens=128, output_tokens=32)
        raw_ng["runtime"]["metadata"] = {
            **raw_ng["runtime"].get("metadata", {}),
            "spec_method": "ngram_gpu", "draft_mode": "ngram_gpu",
            "draft_proposal_tokens": 5, "draft_accept_tokens": 2.5,
        }
        c_ng = parse_case(raw_ng)
        for vw in (
            RequestView("r0", 128, 0, True, "prefill"),
            RequestView("r0", 1, 128, True, "decode", draft_verify=True, draft_proposal_tokens=3, draft_advanced_tokens=2),
            RequestView("r0", 1, 128, True, "decode", draft_verify=True, draft_proposal_tokens=0, draft_advanced_tokens=1),
        ):
            ng_ops = fe.program(IterationPlan((vw,)), c_ng.model, c_ng.runtime, c_ng.machine, assembly="service").work.ops
            assert sum(1 for op in ng_ops if "ngram_gpu_propose" in op.op_id) == 1

    # 4. Incompatible spec_method fails fast in both prepare_speculation and frontend; draft_enforce_eager stays uncaptured.
    v41_model = load_model("deepseek-v4.1-flash")
    for bad_method in ("eagle3", "draft_model"):
        with pytest.raises(ModelingError, match=f"draft_structure_incompatible_for_speculative_method:{bad_method}:dspark"):
            prepare_speculation(v41_model, 3, spec_method=bad_method)
        rt_bad = replace(c_v41.runtime, metadata=c_v41.runtime.metadata + (("spec_method", bad_method),))
        with pytest.raises(ModelingError, match=f"draft_structure_incompatible_for_speculative_method:{bad_method}:dspark"):
            fe.program(IterationPlan((v_rag[0],)), c_v41.model, rt_bad, c_v41.machine, assembly="service")

    rt_eager = replace(c_glm5.runtime, metadata=c_glm5.runtime.metadata + (("draft_enforce_eager", True),))
    ops_eg = fe.program(IterationPlan((v_rag[0],)), c_glm5.model, rt_eager, c_glm5.machine, assembly="service").work.ops
    draft_eg = [op for op in ops_eg if op.op_id.startswith("draft")]
    assert draft_eg and not any(dict(op.metadata).get("graph_captured") for op in draft_eg)

    # 5. DSpark per-layer slot_mapping skips None layers and writes only valid slot>=0 rows at packed physical row pitch.
    rt_part = replace(
        c_v41.runtime,
        metadata=c_v41.runtime.metadata + (("dspark_layer_slot_mappings", {0: [0, -1, 2, -1], 1: None, 2: [0, 1, 2, 3]}),),
    )
    v_ds = RequestView("r0", 1, 128, True, "decode", draft_verify=True, draft_proposal_tokens=3, draft_advanced_tokens=2)
    ops_part = fe.program(IterationPlan((v_ds,)), c_v41.model, rt_part, c_v41.machine, assembly="service").work.ops
    kv_stores = [op for op in ops_part if "context_kv_store" in op.op_id]
    kv_norms = [op for op in ops_part if "context_kv_norm" in op.op_id]
    assert [op.local_shape[0] for op in kv_stores] == [2, 4]
    assert [op.local_shape[0] for op in kv_norms] == [4, 4]
    assert kv_stores[0].mte3_bytes == pytest.approx(2 * dict(kv_stores[0].metadata)["physical_row_bytes"])


def test_speculative_prefill_pp_ragged_ngram_validation_and_dspark_slots() -> None:
    """Speculative lowering invariants across MTP prefill/PP, DSpark ragged/slots, NGram, and graph coalescing."""
    from mindie_sim.catalog.draft_geometry import draft_layer_plans
    from mindie_sim.catalog.ops import vector_op
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.prediction.frontend.speculation import draft_program
    from mindie_sim.prediction.implementation_selection import LoweringRegion, lower_regions
    from mindie_sim.schema.case import Software
    from mindie_sim.schema.parallel import ParallelSpec
    from mindie_sim.schema.workload import IterationPlan

    fe = Frontend()
    a3 = parse_case(compose_case(
        "glm-5.3-flash", "a3-16p", tp=8, ep=8, num_devices=16, prompt_tokens=128, output_tokens=16,
    )).machine
    glm = load_model("glm-5.3-flash")
    kimi = load_model("kimi-k3")

    # P1-5: mixed finished+unfinished prefill samples ONLY finished request + step-1 MTP features + PP=2 transfer.
    rt_pp2 = Software(parallel=ParallelSpec(tp=8, pp=2, ep=8), metadata=(("spec_method", "mtp"), ("draft_proposal_tokens", 3)))
    v_mix = (RequestView("r_fin", 64, 0, True, "prefill"), RequestView("r_mid", 64, 0, False, "prefill"))
    regs_pf, _ = draft_program(fe, glm, rt_pp2, v_mix, "ready", prefill=True, machine=a3, assembly="service",
                               next_proposal_widths={"r_fin": 3, "r_mid": 0})
    pf_ops = [op for r in regs_pf for op in r.ops]
    pf_logits = [op for op in pf_ops if op.op_id == "draft_prefill.logits"]
    assert len(pf_logits) == 1 and pf_logits[0].requests == ("r_fin",) and pf_logits[0].local_shape[0] == 1
    assert any(op.op_id == "draft_prefill_step1.features" for op in pf_ops)
    assert any(op.op_id == "draft_proposal_transfer" and op.requests == ("r_fin",) for op in pf_ops)

    # P1-6: PP=2 transfer uses generated widths (5 IDs); DSpark ragged (5,1) Markov & prefill both total 6 rows.
    v_pp = (
        RequestView("r0", 1, 64, True, "decode", draft_verify=False, draft_proposal_tokens=0),
        RequestView("r1", 1, 64, True, "decode", draft_verify=True, draft_proposal_tokens=1),
    )
    regs_pp, _ = draft_program(fe, glm, rt_pp2, v_pp, "ready", prefill=False, machine=a3, assembly="service",
                               next_proposal_widths={"r0": 3, "r1": 2})
    xfer = next(op for r in regs_pp for op in r.ops if op.op_id == "draft_proposal_transfer")
    assert xfer.semantic_shape == (5,) and xfer.link_bytes == pytest.approx(5 * 8)

    rt_kimi = Software(parallel=ParallelSpec(tp=8, ep=8),
                       metadata=(("spec_method", "dspark"), ("draft_mode", "dspark"), ("draft_proposal_tokens", 5)))
    v_rag = (
        RequestView("r0", 1, 64, True, "decode", draft_verify=True, draft_proposal_tokens=5),
        RequestView("r1", 1, 64, True, "decode", draft_verify=True, draft_proposal_tokens=1),
    )
    regs_rag, _ = draft_program(fe, kimi, rt_kimi, v_rag, "ready", prefill=False, machine=a3, assembly="service",
                                next_proposal_widths={"r0": 5, "r1": 1})
    assert sum(op.local_shape[0] for r in regs_rag for op in r.ops if "markov_gemm_" in op.op_id) == 6
    v_rag_pf = (RequestView("r0", 64, 0, True, "prefill"), RequestView("r1", 64, 0, True, "prefill"))
    regs_rpf, _ = draft_program(fe, kimi, rt_kimi, v_rag_pf, "ready", prefill=True, machine=a3, assembly="service",
                                next_proposal_widths={"r0": 5, "r1": 1})
    rpf_ops = [op for r in regs_rpf for op in r.ops]
    assert sum(op.local_shape[0] for op in rpf_ops if op.op_id.endswith(".logits")) == 6
    assert sum(op.local_shape[0] for op in rpf_ops if "markov_gemm_" in op.op_id) == 6

    # multi_module_mtp validates configured K divisibility on BOTH EOS tail & prefill; EOS tail steps=1 runs module 0 only.
    mtp0 = draft_layer_plans(glm, "mtp")[0]
    glm_mm = replace(glm, auxiliary=replace(
        glm.auxiliary, mtp=replace(glm.auxiliary.mtp, layers=3, execution_mode="multi_module_mtp",
                                   layer_plans=tuple(replace(mtp0, index=i) for i in range(3))),
    ))
    rt_mm = Software(parallel=ParallelSpec(tp=8, ep=8),
                     metadata=(("spec_method", "mtp"), ("draft_proposal_tokens", 3), ("multi_module_mtp_n_predict", 3)))
    v_eos = (RequestView("r0", 1, 128, True, "decode", draft_verify=True, draft_proposal_tokens=3),)
    regs_tail, _ = draft_program(fe, glm_mm, rt_mm, v_eos, "ready", prefill=False, machine=a3, assembly="service",
                                 next_proposal_widths={"r0": 1})
    tail_ops = [op for r in regs_tail for op in r.ops]
    assert sorted({op.op_id.split(".")[0] for op in tail_ops if op.op_id.startswith("draft")}) == ["draft0"]
    rt_mm_bad = Software(parallel=ParallelSpec(tp=8, ep=8),
                         metadata=(("spec_method", "mtp"), ("draft_proposal_tokens", 4), ("multi_module_mtp_n_predict", 4)))
    with pytest.raises(ModelingError, match="multi_module_mtp_non_divisible_n_predict"):
        draft_program(fe, glm_mm, rt_mm_bad, v_eos, "ready", prefill=False, machine=a3, assembly="service", next_proposal_widths={"r0": 1})
    with pytest.raises(ModelingError, match="multi_module_mtp_non_divisible_n_predict"):
        draft_program(fe, glm_mm, rt_mm_bad, (RequestView("p0", 64, 0, True, "prefill"),), "ready", prefill=True, machine=a3, assembly="service", next_proposal_widths={"p0": 1})

    # NGram/NGram-GPU on Ascend MRV2: history-sized ngram_gpu_scan + propose, suppressed on output_tokens=1 EOS prefill.
    for ng_method in ("ngram", "ngram_gpu"):
        rt_ng = Software(parallel=ParallelSpec(tp=8, ep=8), metadata=(("spec_method", ng_method), ("draft_proposal_tokens", 5)))
        pf_ng = fe(IterationPlan((RequestView("r0", 32, 0, True, "prefill"),)), glm, rt_ng, a3).ops
        scan_op = next(op for op in pf_ng if op.recipe == "ngram_gpu_scan")
        assert scan_op.vector_ops > 0 and scan_op.mte2_bytes == pytest.approx(33 * 4)
        assert any(op.recipe == "ngram_gpu_propose" for op in pf_ng)
        eos_ng = fe(IterationPlan((RequestView("r0", 32, 0, True, "prefill"),), output_tokens=(("r0", 1),)), glm, rt_ng, a3).ops
        assert not any(op.recipe in {"ngram_gpu_scan", "ngram_gpu_propose"} for op in eos_ng)

    # spec_method="none" disables scheduling & draft lowering even with non-zero draft_proposal_tokens;
    # runtime.metadata["speculative"]["draft_len"] enforces FIA ceiling (5 passes, 20 fails fast).
    from mindie_sim.prediction.scheduler import Scheduler
    from mindie_sim.schema.model import RequestSpec
    raw_none = compose_case("glm-5.3-flash", "a3-16p", tp=8, ep=8, layers=4, prompt_tokens=64, output_tokens=16, mtp_proposal_tokens=3)
    raw_none["runtime"]["metadata"] = {**raw_none["runtime"].get("metadata", {}), "spec_method": "none"}
    c_none = parse_case(raw_none)
    sch_none = Scheduler(c_none, [RequestSpec("r0", 0.0, 64, 16)], Memory(c_none))
    sch_none.schedule(0.0)
    sch_none.commit(1.0)
    dec_none = sch_none.schedule(2.0)
    assert dec_none is not None and dec_none.requests[0].draft_proposal_tokens == 0 and dec_none.requests[0].draft_advanced_tokens == 1
    assert not any(op.op_id.startswith("draft") for op in fe.program(dec_none, c_none.model, c_none.runtime, c_none.machine, assembly="service").work.ops)

    rt_spec_ok = Software(parallel=ParallelSpec(tp=8, ep=8), metadata=(("speculative", {"draft_len": 5}),))
    assert fe.program(IterationPlan((RequestView("r0", 64, 0, True, "prefill"),)), glm, rt_spec_ok, a3, assembly="service").work.ops
    rt_spec_bad = Software(parallel=ParallelSpec(tp=8, ep=8), metadata=(("speculative", {"draft_len": 20}),))
    with pytest.raises(ModelingError, match="ascend_fia_speculative_tokens_exceeds_limit:20>15"):
        fe.program(IterationPlan((RequestView("r0", 64, 0, True, "prefill"),)), glm, rt_spec_bad, a3, assembly="service")

    raw_ng20 = compose_case("glm-5.3-flash", "a3-16p", tp=8, ep=8, prompt_tokens=128, output_tokens=16)
    raw_ng20["runtime"]["metadata"] = {
        **raw_ng20["runtime"].get("metadata", {}),
        "spec_method": "ngram_gpu", "draft_proposal_tokens": 20,
    }
    c_ng20 = parse_case(raw_ng20)
    with pytest.raises(ModelingError, match="ascend_fia_speculative_tokens_exceeds_limit:20>15"):
        fe.program(IterationPlan((RequestView("r0", 64, 0, True, "prefill"),)), c_ng20.model, c_ng20.runtime, c_ng20.machine, assembly="service")
    with pytest.raises(ModelingError, match="medusa"):
        draft_program(fe, glm, replace(rt_mm, metadata=(("spec_method", "medusa"), ("draft_proposal_tokens", 3))), v_eos, "ready", machine=a3)
    assert draft_program(fe, glm, replace(rt_mm, metadata=(("spec_method", "none"), ("draft_proposal_tokens", 3))), v_eos, "ready", machine=a3)[0] == []
    with pytest.raises(ModelingError, match="draft_tp_mismatch"):
        draft_program(fe, glm, replace(rt_mm, metadata=(("spec_method", "mtp"), ("draft_proposal_tokens", 3), ("draft_tp", 1))), v_eos, "ready", machine=a3)

    # Mixed PP=2 prefill+decode emits ONE aggregated draft_proposal_transfer (no duplicate op_id cycle);
    # Kimi DSpark prefill PP=2 emits draft_proposal_transfer.
    v_pp2_mix = (
        RequestView("p0", 64, 0, True, "prefill"),
        RequestView("d0", 1, 64, True, "decode", draft_verify=True, draft_proposal_tokens=3, draft_advanced_tokens=2),
    )
    mix_pp2_ops = fe.program(IterationPlan(v_pp2_mix), glm, rt_pp2, a3, assembly="service").work.ops
    xfers_mix = [op for op in mix_pp2_ops if op.op_id == "draft_proposal_transfer"]
    assert len(xfers_mix) == 1 and set(xfers_mix[0].requests) == {"p0", "d0"}
    rt_kimi_pp2 = Software(parallel=ParallelSpec(tp=8, pp=2, ep=8),
                           metadata=(("spec_method", "dspark"), ("draft_mode", "dspark"), ("draft_proposal_tokens", 5)))
    kimi_pp2_ops = fe.program(IterationPlan((RequestView("p0", 64, 0, True, "prefill"),)), kimi, rt_kimi_pp2, a3, assembly="service").work.ops
    assert sum(1 for op in kimi_pp2_ops if op.op_id == "draft_proposal_transfer") == 1

    # Eager TP1 DSpark with graph_capture_mode="none" & graph_capture_keys=(64,) executes 5 rows (not 64);
    # adjacent piecewise regions with different splitting_ops stay uncoalesced; partial Kimi slot preserves layer=1.
    rt_eg = Software(parallel=ParallelSpec(tp=1), graph_capture_mode="none", graph_capture_keys=(64,),
                     metadata=(("spec_method", "dspark"), ("draft_mode", "dspark"), ("draft_proposal_tokens", 5)))
    regs_eg, _ = draft_program(fe, kimi, rt_eg, (v_rag[0],), "ready", prefill=False, machine=a3, assembly="service")
    assert dict(regs_eg[0].ops[0].metadata)["dspark_physical_rows"] == 5

    calls = []
    class _SpyGraph:
        def apply(self, ops, scope, runtime):
            calls.append((tuple(op.op_id for op in ops), scope))
            return ops
    rt_full = Software(parallel=ParallelSpec(tp=8, ep=8), graph_capture_mode="full")
    o1 = vector_op(op_id="o1", deps=(), stage=0, layer=0, requests=("r0",), elements=16, ops_per_element=1, dtype="bf16", family="elementwise", phase="decode", recipe="add")
    o2 = vector_op(op_id="o2", deps=("o1",), stage=0, layer=0, requests=("r0",), elements=16, ops_per_element=1, dtype="bf16", family="elementwise", phase="decode", recipe="add")
    lower_regions((LoweringRegion((o1,), rt_full, "bf16", 1, (), replay_scope="full"),
                   LoweringRegion((o2,), rt_full, "bf16", 1, (), replay_scope="full")),
                  graph=_SpyGraph(), replay_scope="full", runtime=rt_full)
    assert len(calls) == 1 and calls[0] == (("o1", "o2"), "full")
    calls.clear()
    rt_pw1 = Software(parallel=ParallelSpec(tp=8, ep=8), graph_capture_mode="piecewise", metadata=(("splitting_ops", ("attention",)),))
    rt_pw2 = Software(parallel=ParallelSpec(tp=8, ep=8), graph_capture_mode="piecewise", metadata=(("splitting_ops", ("attention", "allreduce")),))
    lower_regions((LoweringRegion((o1,), rt_pw1, "bf16", 1, (), replay_scope="piecewise"),
                   LoweringRegion((o2,), rt_pw2, "bf16", 1, (), replay_scope="piecewise")),
                  graph=_SpyGraph(), replay_scope="piecewise", runtime=rt_pw1)
    assert len(calls) == 2

    kimi_mla = load_model("kimi-k3.1")
    rt_none_slots = replace(rt_kimi, metadata=rt_kimi.metadata + (("dspark_layer_slot_mappings", (None,) * 5),))
    regs_ns, _ = draft_program(fe, kimi_mla, rt_none_slots, v_rag_pf[:1], "ready", prefill=True, machine=a3, assembly="service")
    assert not any(op.recipe in {"kv_store", "draft_context_kv_scatter", "draft_context_mla_prolog_v3"} for r in regs_ns for op in r.ops)
    rt_part_kimi = replace(rt_kimi, metadata=rt_kimi.metadata + (("dspark_layer_slot_mappings", (None, tuple(range(32)), None, None, None)),))
    # L-B06: A3 component-major cache writes directly inside PrologV3 (layer=1, rank_owned_rows=32, zero kv_scatter).
    regs_pk, _ = draft_program(fe, kimi_mla, rt_part_kimi, v_rag_pf[:1], "ready", prefill=True, machine=a3, assembly="service")
    pk_prologs = [op for r in regs_pk for op in r.ops if op.recipe == "draft_context_mla_prolog_v3"]
    assert len(pk_prologs) == 1 and pk_prologs[0].layer == 1 and pk_prologs[0].op_id == "draft_prefill.context_kv1.mla_prolog_v3"
    assert dict(pk_prologs[0].metadata)["rank_owned_rows"] == 32
    assert dict(pk_prologs[0].metadata)["direct_cache_write"] is True
    assert not any(op.recipe in {"kv_store", "draft_context_kv_scatter"} for r in regs_pk for op in r.ops)
    # Token-interleaved cache stages through PrologV3 + standalone kv_scatter with layer=1 and 32 rows.
    rt_part_staged = replace(rt_part_kimi, metadata=rt_part_kimi.metadata + (("draft_cache_layout", "token_interleaved"),))
    regs_st, _ = draft_program(fe, kimi_mla, rt_part_staged, v_rag_pf[:1], "ready", prefill=True, machine=a3, assembly="service")
    pk_scatters = [op for r in regs_st for op in r.ops if op.recipe in {"kv_store", "draft_context_kv_scatter"}]
    assert len(pk_scatters) == 1 and pk_scatters[0].layer == 1 and pk_scatters[0].op_id == "draft_prefill.context_kv1.kv_scatter"
    assert pk_scatters[0].local_shape[0] == 32
