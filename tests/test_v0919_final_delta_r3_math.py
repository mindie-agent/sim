"""v0.9.19 physical routing, cache and caller conservation.

[Role] Verify the eleven R3 results on the public graph and recipe boundaries.
[Invariants] Expected values follow from row identities, placement and byte widths.
[Timeouts & Locks] Finite pure-Python checks, no locks.
[Side-Effect Policy] No accelerator, weights or persistent state.
"""
from dataclasses import replace
from itertools import combinations, product
from types import SimpleNamespace

import pytest

import test_v0919_final_delta_math as prior
import _support.operator_calibration_helpers as a
import _support.operator_calibration_helpers as b
from mindie_sim.catalog.mqsmla_layout import packed_row_parts
from mindie_sim.prediction import family_mqsmla, index_pool_native, latent_dcp, mqsmla
from mindie_sim.prediction.batch_prefix_attention import nested_prefix_union
from mindie_sim.prediction.rank_ledger import rank_ledger_for
from mindie_sim.prediction.index_pool import native_shard_sizes
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.pd_ranges import transfers_for_runs
from mindie_sim.schema.workload import RequestView


@pytest.mark.parametrize("dp,ep,pcp", [(4, 4, 2), (1, 2, 2), (1, 1, 2)])
def test_dbo_matches_unsplit_physical_assignments(dp, ep, pcp):
    from mindie_sim.public.errors import UnmodeledCapabilityError

    raw = prior.compose_case(
        "qwen3-30b-a3b", hardware="a5", tp=1, dp=dp, ep=ep, pcp=pcp,
        num_devices=dp * pcp, layers=1, enable_dbo=True,
        prompt_tokens=8, output_tokens=2, max_num_seqs=2)
    if ep > dp:
        raw["runtime"]["parallel"]["rank_table"] = [[0, 1]]
    def graph():
        return prior.target_graphs(prior.parse_case(raw), phase="prefill",
                                   tokens=8, context=0, sequences=2, scope="body")
    with pytest.raises(UnmodeledCapabilityError, match="unmodeled_in_op_dbo_pending_stage3"):
        graph()
    raw["runtime"]["enable_dbo"] = False
    whole = graph()
    for replica, ops in whole.items():
        assert not [op for op in ops if op.op_id.startswith("dbo")]
        packs = [op.local_shape[0] * op.local_shape[1] for op in ops if op.recipe == "route_pack"]
        gates = [op.local_shape[0] for op in ops if op.recipe == "grouped_gate_up"]
        unpack = [op.local_shape[0] * op.local_shape[1] for op in ops if op.recipe == "route_unpack"]
        assert packs == gates == unpack == [64]


def test_hca_runtime_decides_aliasing():
    views = tuple(RequestView(key, 1, 383, False, "decode", prefix_tokens=128,
                              prefix_cache_key="A") for key in ("a", "b"))
    for mode, total, union in (("private", 159296, 0), ("shared", 158688, 1)):
        fields = family_mqsmla._prefix_fields(views, SimpleNamespace(prefix_cache_mode=mode))
        shape = dict(sequences=((1, 383),) * 2, window=128, ratio=128, topk=0, **fields)
        assert mqsmla.quant_external_fill(shape, mqsmla.KV_QUANT_FP8) == (total, union)


def test_csa_nested_prefix_and_private_window():
    contexts = (4, 4, 4096)
    shape = dict(sequences=tuple((1, c) for c in contexts), prefix_tokens=contexts,
                 prefix_cache_keys=("A",) * 3, prefix_cache_mode="shared",
                 ratio=4, window=128, topk=512)
    assert mqsmla.quant_external_fill(shape, mqsmla.KV_QUANT_FP8) == (395808, 513)


def test_kimi_padded_producer_writes_the_entire_pack_input():
    case = prior.parse_case(prior.compose_case(
        "kimi-k3.1", hardware="a5", dp=1, tp=8, dcp=8, ep=8, num_devices=8,
        layers=4, quant="int8", prompt_tokens=8, output_tokens=2))
    model = replace(case.model, quant=replace(case.model.quant, kv_dtype="int8"))
    layer = model.layer_plans[3]
    views = (RequestView("r", 1, 127, False, "decode"),)
    builder = Builder("p", layer, 0, views, model, "decode")
    member = replace(a._c8_op(((1, 127),), 128, 96, 0), recipe="paired_mla_attention_analytic",
                     mte3_bytes=98304, hbm_bytes=98304, metadata=())
    builder.add(member)
    latent_dcp.attention_batch(builder, [member.op_id], views, plan=layer,
                               heads=96, local_heads=12, cp=8, rows=8)
    ops = {op.op_id.split(".")[-1]: op for op in builder.ops}
    assert ops["latent_attention_batch"].mte3_bytes == ops["dcp_output_pack"].mte2_bytes == 792576
    assert ops["dcp_output_pack"].mte3_bytes == ops["dcp_lse_merge"].mte2_bytes == 792576
    assert ops["dcp_output_alltoall"].link_bytes == 693504
    assert ops["dcp_lse_merge"].mte3_bytes == 98304


def test_keypool_native_gates_pack_before_exchange_and_counts_indices_once():
    plan, model = b._case_model(pool=16)
    model = replace(model, quant=replace(model.quant, index_scale_dtype="fp32"))
    runtime = b._Runtime(o_proj_tp=8)
    views = tuple(RequestView(str(i), 1, 131072, False, "decode") for i in range(8192))
    assert native_shard_sizes(views[:8], 16, 8) == (8,)
    shallow = (replace(views[0], q_len=8192, context=0),)
    assert native_shard_sizes(shallow, 16, 8) == (8192,)
    sizes = native_shard_sizes(views, 16, 8)
    assert sizes == (1023, 1024, 1024, 1024, 1024, 1024, 1024, 1025)
    count = 0
    for rank, owned in enumerate(sizes):
        builder = Builder("l", plan, 0, views, model, "decode")
        index_pool_native.selection(builder, query="q", weights="w", cache="k", views=views,
                                    plan=plan, quant=builder.quant, padded_tokens=8192,
                                    runtime=runtime, row_shard_rank=rank)
        ops = {o.op_id.split(".")[-1]: o for o in builder.ops}
        pack = ops["index_topk"]
        assert pack.deps == (ops["cann_pool_key_indexer"].op_id,)
        assert ops["index_exchange_stage"].deps == (pack.op_id,)
        wire = ops["index_kpool_row_shard_gather"]
        assert wire.deps == (ops["index_exchange_stage"].op_id,)
        assert ops["index_exchange_restore"].deps == (wire.op_id,)
        assert wire.link_bytes == 7 * 1025 * 2063 * 4
        assert pack.mte3_bytes == owned * 2063 * 4
        assert pack.mte2_bytes == owned * (2048 * 4 + 2 * 8)
        count += pack.mte2_bytes - owned * 2 * 8
    assert count == 67108864



def test_real_sp_bucket_preserves_every_rank_and_valid_row():
    import json
    from mindie_sim.prediction.frontend import Frontend
    raw = json.loads((prior.ROOT / "examples/cases/kimi-k3-a5-sp-right-pad-conditional.json").read_text())
    case = prior.parse_case(raw)
    counts = (10, 12, 14, 16)
    peers = {d: tuple(RequestView(f"{d}-{i}", 1, 128, False, "decode")
                      for i in range(n)) for d, n in enumerate(counts)}
    ledger = rank_ledger_for(Frontend(), case.model, case.runtime, peers[0], 16,
                             peers, 0, case.machine, 0)
    assert ledger is not None
    assert ledger.capacity == 16
    assert ledger.allocated == (2,) * 32 and ledger.physical == 64
    assert sum(ledger.valid) == sum(counts)


def test_forced_mask_marginals_match_inverse_and_expert_banks():
    from mindie_sim.prediction.source_force_eplb import ledger
    valid, allocated = (2, 2, 2, 1), (2, 2, 3, 3)
    result = ledger(32, 4, 8, 2, 7, 2, valid, allocated, 1)
    assert tuple(map(sum, result.peer_assignments)) == (16, 16, 16, 8)
    assert tuple(map(sum, zip(*result.peer_assignments))) == tuple(map(sum, result.rank_expert_rows))
    assert sum(result.expert_rows) == 56


def test_native_small_batch_has_no_gather_and_override_is_explicit():
    plan, model = b._case_model(pool=16)
    model = replace(model, quant=replace(model.quant, index_scale_dtype="fp32"))
    views = (RequestView("r", 8, 131072, False, "decode"),)
    for opts, expected in (({}, 0), ({"row_shard_tp": 8}, 1)):
        builder = Builder("l", plan, 0, views, model, "decode")
        index_pool_native.selection(builder, query="q", weights="w", cache="k", views=views,
                                    plan=plan, quant=builder.quant, padded_tokens=8,
                                    runtime=b._Runtime(o_proj_tp=8), **opts)
        wires = [op for op in builder.ops if op.recipe == index_pool_native.ROW_SHARD_GATHER]
        assert len(wires) == expected
        assert sum(op.link_bytes for op in wires) == expected * 57344



@pytest.mark.parametrize("groups", [((0, 2), (1, 3)), ((3, 0), (2, 1)), ((1, 0), (3, 2))])
def test_sp_source_is_the_tp_coordinate_even_if_ep_groups_cross_dp(groups):
    p = SimpleNamespace(tp=2, dp=2, pcp=1, pp=1, ep=2, rank_table=groups,
                        pcp_lane_policy="parked")
    runtime = SimpleNamespace(parallel=p, routing={}, kernels={"sequence_parallel": {"padding": "right_to_tp"}})
    frontend = SimpleNamespace(graph=SimpleNamespace(select=lambda views, rt: (None, "none")))
    peers = {d: tuple(RequestView(f"{d}-{i}", 1, 128, False, "decode")
                      for i in range(n)) for d, n in enumerate((4, 8))}
    for group in groups:
        rank = group[0]
        d = rank // 2
        ledger = rank_ledger_for(frontend, None, runtime, peers[d], len(peers[d]), peers, d, None, rank)
        expected = tuple((2, 4)[r // 2] for r in group)
        assert ledger is not None
        assert ledger.valid == ledger.allocated == expected and ledger.physical == sum(expected)



def test_glm_multi_request_chunk_kda_folds_in_kernel_qk_l2norm():
    raw = prior.compose_case(
        "glm-5.3-flash", hardware="a5", tp=1, dp=1, ep=1, num_devices=1,
        layers=4, prompt_tokens=64, output_tokens=2, max_num_seqs=2)
    raw["runtime"]["lowering_passes"] = ["kda_in_kernel_l2norm"]
    case = prior.parse_case(raw)
    ops = prior.target_graphs(case, phase="prefill", tokens=64, context=0,
                              sequences=2, scope="body")[0]
    assert not [op for op in ops if op.recipe == "qk_l2norm"]
    assert [op for op in ops if dict(op.metadata).get("in_kernel_qk_l2norm")]


def test_legacy_dsl_mqsmla_contract_rejects_cann_only_row_widths():
    import test_v0919_deep_audit_track_a as track_a
    from mindie_sim.observed_contracts.attention_contracts import (
        observed_attention_contract,
    )

    query = ((4, 64, 512), "bf16", "ND")
    empty = ((), None, None)
    for row_width in (608, 584):
        inputs = [query, empty, ((16, 1, 1, row_width), "uint8", "ND"),
                  ((1, 16), "int32", "ND"), empty, empty, empty]
        outputs = [query, empty]
        assert observed_attention_contract(
            track_a._observed_row("MqsmlaKernel___call__", inputs, outputs)) is None
        assert observed_attention_contract(
            track_a._observed_row("aclnnMixedQuantSparseFlashMla", inputs, outputs)) is not None


def test_glm_mixed_prefill_decode_kda_eliminates_standalone_qk_l2norm():
    from mindie_sim.prediction.attention import AttentionPolicy
    from mindie_sim.prediction.implementation_selection import resolve
    from mindie_sim.prediction.recurrent import linear
    from mindie_sim.prediction.transforms import lower_implementations

    raw = prior.compose_case(
        "glm-5.3-flash", hardware="a5", tp=1, dp=1, ep=1, num_devices=1,
        layers=4, prompt_tokens=64, output_tokens=2, max_num_seqs=2)
    raw["runtime"]["lowering_passes"] = ["kda_in_kernel_l2norm"]
    case = prior.parse_case(raw)
    plan = next(lp for lp in case.model.layer_plans if lp.attention == "kda")
    views = (RequestView("p", 64, 0, True, "prefill"),
             RequestView("d", 1, 128, True, "decode"))
    resolved_rt, _ = resolve(case.runtime, views, padded_tokens=65, replay_scope="none")
    res = linear(
        prefix="s0.l0", plan=plan, views=views, stage=0, deps="in",
        model=case.model, parallel=resolved_rt.parallel,
        quant=case.model.quant, mode="prefill",
        policy=AttentionPolicy(recurrent_chunk=64), runtime=resolved_rt,
        padded_tokens=65)
    lowered = lower_implementations(res.ops, resolved_rt, case.model.activation_dtype)
    assert not [op for op in lowered if op.recipe == "qk_l2norm"]
    folded = [op for op in lowered if dict(op.metadata).get("in_kernel_qk_l2norm")]
    assert len(folded) == 2


def test_legacy_dsl_mqsmla_lowering_rejects_cann_quant_modes():
    from dataclasses import replace
    from mindie_sim.prediction.operator_analysis import operator_costs
    from mindie_sim.schema.common import ModelingError

    for scale_dtype in ("bf16", "fp8_e8m0"):
        raw = prior.compose_case(
            "deepseek-v4.1-flash", hardware="a5", tp=4, dp=2, ep=8,
            num_devices=8, layers=4, quant="w8a8", prompt_tokens=16,
            output_tokens=2)
        packed = {"dtype": "fp8_e4m3", "group_size": 64, "scale_dtype": scale_dtype}
        raw["model"].setdefault("metadata", {})["cache_representations"] = {
            "window": dict(packed),
            "compressed_kv": dict(packed),
        }
        parsed = prior.parse_case(raw)
        cann_graph = prior.target_graphs(
            parsed, phase="decode", tokens=1, context=128, sequences=1, scope="body")[0]
        cann_op = next(op for op in cann_graph if op.recipe == "aclnn_mixed_quant_sparse_flash_mla")
        meta = dict(cann_op.metadata)
        geom = dict(meta["mqsmla_geometry"])
        geom.pop("implementation", None)
        meta["mqsmla_geometry"] = geom
        meta.pop("native_kind", None)
        crafted = replace(cann_op, recipe="cannbot_mixed_quant_sparse_flash_mla",
                          metadata=tuple(meta.items()))
        with pytest.raises(ModelingError):
            operator_costs(parsed, (crafted,), input_ready_ms=0)
        raw["runtime"].setdefault("metadata", {}).pop("automatic_lowerings", None)
        raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})[
            "mqsmla_attention"] = "cannbot_mixed_quant_sparse_flash_mla"
        with pytest.raises(ModelingError):
            prior.target_graphs(prior.parse_case(raw), phase="decode",
                                tokens=1, context=128, sequences=1,
                                scope="body", apply_body_defaults=False)


