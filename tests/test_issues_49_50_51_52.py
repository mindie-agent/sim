"""Regression tests for issues #49, #50, #51 (all five items) and #52.

[Role]
  Pin the shared-prefix compressed index accounting, the analytical online
  indexer TopK, the mHC coefficient sub-chain, the MC2 speculative capacity and
  orphan quantizer cleanup, model-scope speculative analysis, pure-SWA MQSMLA
  geometry, derived HBM service modes, uniform replica rebinding, the PLE host
  lookup boundary and the bounded two-stage streaming sweep.

[Invariants]
  * Every assertion is behavioral: conserved work, physical row geometry,
    dependency identity or a declared accounting boundary.
  * No device context, network access or measurement lookup is used.

[Timeouts & Locks]
  Tier-0 in-process cases with one bounded worker subprocess; no locks.

[Side-Effect Policy]
  Writes only under pytest's ``tmp_path``.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mindie_sim import compose_case
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.sweep_engine import SCREEN_BATCH


# --------------------------------------------------------------------------- #
# #51-(1a)/(4): shared-prefix compressed index-K accounting.
# --------------------------------------------------------------------------- #
def test_index_key_window_counts_physical_compressed_rows():
    from mindie_sim.prediction.operators.mla.generic_latent_attention import index_key_window

    rows, refills = index_key_window(((4, 128),), compression=32)
    assert (rows, refills) == (132 // 32, 132 // 32)
    logical, _ = index_key_window(((4, 128),))
    assert logical == 132
    assert rows == logical // 32


def test_unique_index_key_rows_deduplicates_only_the_shared_prefix():
    from mindie_sim.prediction.operators.mla.generic_latent_attention import unique_index_key_rows
    from mindie_sim.schema.workload import RequestView

    views = tuple(
        RequestView(f"r{i}", 1, 128, False, "decode", prefix_tokens=128,
                    prefix_cache_key="shared-prefix")
        for i in range(2)
    )
    union, refills = unique_index_key_rows(views, compression=32)
    # #53/#54 unify the shared prefix in the HBM union; every view still
    # pays its own precharged tile refill service.
    assert union == (128 + 1) // 32
    assert refills == 2 * union
    private, private_refills = unique_index_key_rows(
        views, compression=32, share_prefix=False)
    assert private == 2 * union
    assert private_refills == refills


def test_shared_prefix_planner_separates_component_and_row_bytes():
    from mindie_sim.prediction.prefix_reads import CacheAccess, plan_prefix_reads

    common = dict(context_tokens=4096, q_len=1, prefix_tokens=512,
                  prefix_key="p", layer=0, component="index_k", logical_rows=512)

    def access(name, row_bytes=128.0, component="index_k"):
        return CacheAccess(name, row_bytes=row_bytes, component=component,
                           **{key: value for key, value in common.items()
                              if key != "component"})

    by_row = plan_prefix_reads(
        (access("a"), access("b"), access("c", row_bytes=256.0)),
        mode="batch_shared", kernel_reuse="tile")
    assert by_row.read_reuse_applied is True
    # Two identical 128-byte accesses share one fill; the 256-byte access is a
    # different physical row and never inherits that share.
    assert by_row.unique_hbm_bytes == pytest.approx(
        512 * 128.0 + 512 * 256.0)
    by_component = plan_prefix_reads(
        (access("a"), access("b"),
         access("c", component="key"), access("d", component="key")),
        mode="batch_shared", kernel_reuse="tile")
    assert by_component.read_reuse_applied is True
    assert by_component.unique_hbm_bytes == pytest.approx(2 * 512 * 128.0)


def _indexer_op(rows, context, q_len, prefix_tokens, hbm, mte2):
    return OpSpec(
        op_id="s0.l0.index_scores", family="indexer", phase="decode", stage=0,
        layer=0, requests=("r0", "r1"), semantic_shape=(rows, 1, 1),
        local_shape=(rows, 1, 1), padded_shape=(rows, 1, 1),
        launch_shape=(rows, 1, 1), hbm_bytes=hbm, mte2_bytes=mte2,
        recipe="index_scores", dtype="int8")


def _prefix_views(context, prefix_tokens):
    from mindie_sim.schema.workload import RequestView

    return tuple(
        RequestView(f"r{i}", 1, context, False, "decode",
                    prefix_tokens=prefix_tokens, prefix_cache_key="key")
        for i in range(2)
    )


def test_apply_indexer_prefix_reads_compresses_declared_prefix_tokens():
    from mindie_sim.prediction.operators.sparse_indexer.batch_prefix_indexer import (
        apply_indexer_prefix_reads,
    )

    runtime = SimpleNamespace(prefix_cache_mode="shared", prefix_read_mode="auto")
    views = _prefix_views(context=4096, prefix_tokens=640)
    op = _indexer_op(128, 4096, 1, 640, hbm=1000.0, mte2=900.0)
    updated = apply_indexer_prefix_reads(
        op, views=views, rows_by_view=(128, 128), shared_rows_by_view=(128, 128),
        row_bytes=16.0, layer_index=0, runtime=runtime, compression_ratio=32,
        allow_recipe_rebind=False, hbm_component=1000.0, mte2_component=900.0)
    metadata = dict(updated.metadata)
    assert metadata["prefix_read_reuse_applied"] is True
    # 640 prefix tokens are 20 compressed index rows, not 640.
    private_rows = 128 - 20
    expected_hbm = 2 * private_rows * 16.0 + 20 * 16.0
    assert metadata["prefix_read_hbm_bytes"] == pytest.approx(expected_hbm)
    assert updated.mte2_bytes == pytest.approx(900.0)
    assert updated.hbm_bytes == pytest.approx(
        1000.0 - 1000.0 + metadata["prefix_read_hbm_bytes"])


def test_apply_indexer_prefix_reads_rejects_excess_components():
    from mindie_sim.prediction.operators.sparse_indexer.batch_prefix_indexer import (
        apply_indexer_prefix_reads,
    )
    from mindie_sim.schema.common import ModelingError

    runtime = SimpleNamespace(prefix_cache_mode="shared", prefix_read_mode="auto")
    views = _prefix_views(context=4096, prefix_tokens=640)
    op = _indexer_op(128, 4096, 1, 640, hbm=1000.0, mte2=900.0)
    with pytest.raises(ModelingError, match="prefix_read_component_exceeds"):
        apply_indexer_prefix_reads(
            op, views=views, rows_by_view=(128, 128),
            shared_rows_by_view=(128, 128), row_bytes=16.0, layer_index=0,
            runtime=runtime, hbm_component=2000.0, mte2_component=900.0)
    with pytest.raises(ModelingError, match="index_prefix_access_lengths"):
        apply_indexer_prefix_reads(
            op, views=views, rows_by_view=(128,), shared_rows_by_view=(1, 2),
            row_bytes=16.0, layer_index=0, runtime=runtime)


def test_native_quant_indexer_ignores_shared_prefix_pool():
    from mindie_sim.prediction.operator_analysis import target_operators

    def rows(mode):
        raw = compose_case("deepseek-v4-10t", hardware="a5", tp=1, ep=1,
                           num_devices=1, layers=4, prompt_tokens=640,
                           output_tokens=2, concurrency=1)
        parsed = parse_case(raw)
        kernels = {**dict(parsed.runtime.kernels),
                   "lowerings": {**dict(parsed.runtime.kernels.get("lowerings", {})),
                                 "quant_lightning_indexer":
                                     "cann_quant_lightning_indexer_v2_complete"}}
        case = replace(parsed, runtime=replace(
            parsed.runtime, prefix_cache_mode=mode, prefix_read_mode="auto",
            kernels=kernels))
        ops = target_operators(case, phase="decode", tokens=1, context=4096,
                               sequences=1, scope="body")
        return [op for op in ops if op.recipe == "quant_lightning_indexer_v2"]

    private, shared = rows("private"), rows("shared")
    assert private and len(private) == len(shared)
    for left, right in zip(private, shared):
        assert (left.hbm_bytes, left.mte2_bytes, left.mte3_bytes) == (
            right.hbm_bytes, right.mte2_bytes, right.mte3_bytes)
        assert dict(right.metadata).get("prefix_read_reuse_applied") is None
        assert dict(right.metadata).get("prefix_read_mode") is None


# --------------------------------------------------------------------------- #
# #51-(1b): analytical online indexer TopK.
# --------------------------------------------------------------------------- #
def _indexer_pair(flops=100.0, vector=10.0, metadata=()):
    score = OpSpec(
        op_id="l0.index_scores", family="indexer", phase="decode", stage=0,
        layer=0, requests=("r",), semantic_shape=(4, 2, 8),
        local_shape=(4, 2, 8), padded_shape=(4, 2, 8), launch_shape=(4, 2, 8),
        flops=flops, vector_ops=vector, hbm_bytes=100.0, mte2_bytes=40.0,
        mte3_bytes=16.0, deps=("q",), recipe="index_scores", dtype="int8",
        metadata=(("index_score_shards", 1), ("index_context_partitions", 1),
                  ("index_candidate_filter", False),
                  ("index_head_reduction", "complete_local"),
                  ("graph_segment", "capture")) + tuple(metadata))
    topk = OpSpec(
        op_id="l0.index_topk", family="topk", phase="decode", stage=0, layer=0,
        requests=("r",), semantic_shape=(4, 2), local_shape=(4, 2),
        padded_shape=(4, 2), launch_shape=(4, 2), flops=20.0, vector_ops=5.0,
        hbm_bytes=16.0, mte2_bytes=16.0, mte3_bytes=32.0, deps=(score.op_id,),
        recipe="index_topk", dtype="fp32",
        metadata=(("selection_output_dtype", "int32"),
                  ("graph_segment", "capture")))
    return score, topk


def test_online_indexer_topk_conserves_work_and_removes_roundtrip():
    from mindie_sim.prediction.rewrite_indexer_rules import online_indexer_topk

    score, topk = _indexer_pair()
    fused = online_indexer_topk([score, topk], "bf16")
    assert len(fused) == 1
    op = fused[0]
    assert op.recipe == "index_online_topk" and op.deps == ("q",)
    assert op.flops == score.flops + topk.flops
    assert op.vector_ops == score.vector_ops + topk.vector_ops
    assert dict(op.matrix_flops) == {"int8": score.flops,
                                     "fp32": topk.flops}
    assert sum(dict(op.matrix_flops).values()) == pytest.approx(op.flops)
    assert op.mte2_bytes == score.mte2_bytes
    assert op.mte3_bytes == topk.mte3_bytes
    assert op.hbm_bytes == (score.hbm_bytes + topk.hbm_bytes
                            - score.mte3_bytes - topk.mte2_bytes)
    metadata = dict(op.metadata)
    assert metadata["score_materialization"] == "analytical_online_ub"
    assert metadata["native_binary_claimed"] is False
    assert metadata["removed_score_roundtrip_bytes"] == 32.0


@pytest.mark.parametrize("override", [
    {"index_score_shards": 2},
    {"index_context_partitions": 2},
    {"index_candidate_filter": True},
    {"index_head_reduction": "partial"},
])
def test_online_indexer_topk_preserves_incomplete_reductions(override):
    from mindie_sim.prediction.rewrite_indexer_rules import online_indexer_topk

    score, topk = _indexer_pair()
    metadata = tuple((key, override.get(key, value))
                     for key, value in dict(score.metadata).items())
    score = replace(score, metadata=metadata)
    assert [op.recipe for op in online_indexer_topk([score, topk], "bf16")] == [
        "index_scores", "index_topk"]


def test_online_indexer_topk_requires_single_consumer_and_segment():
    from mindie_sim.prediction.rewrite_indexer_rules import online_indexer_topk

    score, topk = _indexer_pair()
    extra = replace(topk, op_id="l0.other", recipe="other")
    assert [op.recipe for op in
            online_indexer_topk([score, topk, extra], "bf16")] == [
        "index_scores", "index_topk", "other"]
    other_segment = replace(topk, metadata=(("graph_segment", "eager"),))
    assert [op.recipe for op in
            online_indexer_topk([score, other_segment], "bf16")] == [
        "index_scores", "index_topk"]


def test_indexer_online_mechanism_is_ideal_only():
    from mindie_sim.feature_mechanisms import (
        MECHANISM_CATALOG, MechanismContext,
    )
    from mindie_sim.stack_tracks import RUNTIME_STACKS, resolve_stack_track

    raw = compose_case("deepseek-v4.1-flash", hardware="a3", tp=8, ep=8,
                       num_devices=8, layers=2)
    model = parse_case(raw).model
    context = MechanismContext(
        model=model, sku="a3", arch="a3", parallel={"tp": 8, "ep": 8},
        caller_explicit=frozenset(), options={})
    mechanism = MECHANISM_CATALOG["indexer_online_ub_topk"]
    assert mechanism.eligible(context) is True
    runtime = {}
    mechanism.apply(runtime, context)
    assert "indexer_online_topk" in runtime["lowering_passes"]
    for track in ("merged", "unmerged_pr"):
        active = resolve_stack_track(
            RUNTIME_STACKS["vllm-ascend"], track=track,
            model_key="deepseek-v4.1-flash", ctx=context)["active_mechanisms"]
        assert "indexer_online_ub_topk" not in [
            item["id"] for item in active]
    ideal = resolve_stack_track(
        RUNTIME_STACKS["vllm-ascend"], track="expected_ideal",
        model_key="deepseek-v4.1-flash", ctx=context)["active_mechanisms"]
    assert "indexer_online_ub_topk" in [item["id"] for item in ideal]


# --------------------------------------------------------------------------- #
# #51-(2): mHC projection representation and resident coefficient region.
# --------------------------------------------------------------------------- #
def _v41_case(*, layers=2):
    return parse_case(compose_case(
        "deepseek-v4.1-flash", hardware="a3", tp=8, ep=8, num_devices=8,
        layers=layers))


def _with_passes(case, *extra, kernels=None):
    runtime = case.runtime
    current = tuple(name for name in runtime.lowering_passes
                    if name not in {"mhc_boundary_pr", "mhc_boundary_ideal"})
    runtime = replace(runtime, lowering_passes=(*current, *extra))
    if kernels is not None:
        runtime = replace(runtime, kernels={
            **dict(runtime.kernels),
            "lowerings": {**dict(runtime.kernels.get("lowerings", {})),
                          **kernels}})
    return replace(case, runtime=runtime)


def test_hc_projection_declares_fp32_input_and_output():
    from mindie_sim.prediction.operator_analysis import target_operators

    case = _v41_case()
    ops = target_operators(case, phase="decode", tokens=1, context=2048,
                           sequences=2, scope="body")
    projections = [op for op in ops if op.recipe == "hc_projection"]
    assert projections
    for op in projections:
        metadata = dict(op.metadata)
        assert metadata["input_format"] == "fp32"
        assert metadata["output_format"] == "fp32"


def test_mhc_ideal_fuses_the_closed_coefficient_chain():
    from mindie_sim.prediction.operator_analysis import target_operators

    case = _with_passes(_v41_case(), "mhc_boundary_ideal")
    plain = target_operators(_v41_case(), phase="decode", tokens=1,
                             context=2048, sequences=2, scope="body")
    ops = target_operators(case, phase="decode", tokens=1, context=2048,
                           sequences=2, scope="body")
    fused = [op for op in ops if op.recipe == "hc_coefficients_fused"]
    assert fused
    assert not [op for op in ops if op.recipe in {
        "hc_values_fp32", "hc_projection", "hc_inverse_rms",
        "hc_sinkhorn"}]
    chain = tuple(op for op in plain if op.recipe in {
        "hc_values_fp32", "hc_projection", "hc_inverse_rms",
        "hc_normalize_scale_bias_gate", "hc_sinkhorn"})
    # Four fused regions (two layers x two sublayers) replace 20 launches.
    assert len(fused) == 4 and len(chain) == 20
    first = fused[0]
    assert first.family == "gemm" and first.dtype == "fp32"
    assert first.returns == ("post", "comb")
    assert first.stream == "mhc_aux"
    assert first.flops == pytest.approx(
        sum(op.flops for op in chain[:5]))
    assert first.vector_ops == pytest.approx(
        sum(op.vector_ops for op in chain[:5]))
    values, projection, inv_rms, affine, sinkhorn = chain[:5]
    expected_read = (values.mte2_bytes
                     + max(0.0, projection.mte2_bytes - values.mte3_bytes)
                     + max(0.0, affine.mte2_bytes - projection.mte3_bytes
                           - inv_rms.mte3_bytes))
    assert first.mte2_bytes == pytest.approx(expected_read)
    assert first.mte3_bytes == pytest.approx(sinkhorn.mte3_bytes)
    assert first.hbm_bytes == pytest.approx(
        first.mte2_bytes + first.mte3_bytes)
    # Exactly the internal 3V + 2P + 2R + 2A round-trip disappears.
    eliminated = (3 * values.mte3_bytes + 2 * projection.mte3_bytes
                  + 2 * inv_rms.mte3_bytes + 2 * affine.mte3_bytes)
    assert (sum(op.hbm_bytes for op in chain[:5]) - first.hbm_bytes
            == pytest.approx(eliminated, rel=1e-9))


def test_mhc_pr_boundary_keeps_the_declared_coefficient_chain():
    from mindie_sim.prediction.operator_analysis import target_operators

    case = _with_passes(_v41_case(), "mhc_boundary_pr")
    ops = target_operators(case, phase="decode", tokens=1, context=2048,
                           sequences=2, scope="body")
    assert not [op for op in ops if op.recipe == "hc_coefficients_fused"]
    assert [op for op in ops if op.recipe == "hc_sinkhorn"]


def test_native_fusion_protects_the_source_hc_pre_subgraph():
    from mindie_sim.prediction.operator_analysis import target_operators

    binding = {"implementation": "declared_native_fusion",
               "options": {"roles": ["hc_pre"],
                           "traffic_profile": {
                               "hc_pre": "hc_pre_arch35_mk_175ef4d4"}}}
    case = _with_passes(_v41_case(), "mhc_boundary_ideal",
                        kernels={"native_fusion": binding})
    ops = target_operators(case, phase="decode", tokens=1, context=2048,
                           sequences=2, scope="body")
    assert not [op for op in ops if op.recipe == "hc_coefficients_fused"]
    assert [op for op in ops if op.recipe == "hc_pre.native"] or [
        op for op in ops if op.family == "native_fusion"]
    # A different selected role does not protect the subgraph.
    other = _with_passes(_v41_case(), "mhc_boundary_ideal",
                        kernels={"native_fusion": {
                            "implementation": "declared_native_fusion",
                            "options": {"roles": ["hc_post"]}}})
    assert [op for op in target_operators(
        other, phase="decode", tokens=1, context=2048, sequences=2,
        scope="body") if op.recipe == "hc_coefficients_fused"]


# --------------------------------------------------------------------------- #
# #51-(3): the qk_norm blocker is gone from residual normalization selection.
# --------------------------------------------------------------------------- #
def test_qk_norm_does_not_block_residual_normalization():
    from mindie_sim.feature_mechanisms import (
        MechanismContext, _apply_basic_norm_quant,
    )

    case = parse_case(compose_case(
        "qwen3-32b", hardware="a3", tp=1, ep=1, num_devices=1, layers=2))
    assert any(plan.qk_norm for plan in case.model.layer_plans)
    assert case.model.activation_dtype == "bf16"
    assert not (dict(case.model.catalog_facts).get("normalization") or {}
                ).get("norm_weight_offset", 0.0)
    assert "residual_norm" in case.runtime.lowering_passes
    assert dict(case.runtime.kernels["lowerings"])["add_rmsnorm"] == (
        "vllm_ascend_native_norm")
    runtime = {}
    context = MechanismContext(
        model=case.model, sku="a3", arch="a3", parallel={"tp": 1},
        caller_explicit=frozenset(), options={})
    _apply_basic_norm_quant(runtime, context)
    assert "residual_norm" in runtime["lowering_passes"]


# --------------------------------------------------------------------------- #
# #51-(5): MC2 speculative capacity and the orphan pack quantizer.
# --------------------------------------------------------------------------- #
def test_mc2_capacity_defaults_to_one_plus_draft_proposal_tokens():
    from mindie_sim.prediction.operators.mc2.moe_comm_selection import MoeCommConfig

    features = SimpleNamespace(tp=1, graph_buckets=(), metadata=(
        ("moe_comm_mode", "dynamic"), ("max_num_seqs", 2),
        ("draft_proposal_tokens", 3)))
    config = MoeCommConfig.from_features(features, ep=8, top_k=4, experts=32)
    assert config is not None
    assert config.uniform_decode_query_len == 4
    assert config.capacity == 8
    explicit = MoeCommConfig.from_features(
        SimpleNamespace(tp=1, graph_buckets=(), metadata=(
            ("moe_comm_mode", "dynamic"), ("max_num_seqs", 2),
            ("draft_proposal_tokens", 3), ("uniform_decode_query_len", 1))),
        ep=8, top_k=4, experts=32)
    assert explicit.uniform_decode_query_len == 1
    assert explicit.capacity == 2


def test_runtime_metadata_forwards_the_declared_dspark_width():
    from mindie_sim.prediction.frontend_features import _runtime_metadata
    from mindie_sim.schema.dspark import DsparkSpec

    dspark = DsparkSpec(draft_len=3, acceptance=(0.6, 0.4, 0.2))
    runtime = SimpleNamespace(metadata=(), dspark=dspark)
    assert dict(_runtime_metadata(runtime))["draft_proposal_tokens"] == 3
    declared = SimpleNamespace(metadata=(("draft_proposal_tokens", 5),),
                               dspark=dspark)
    assert dict(_runtime_metadata(declared))["draft_proposal_tokens"] == 5


WIDTH = 256
TOKENS = 8
TOPK = 4
EP = 8
RECEIVED = 16.0


def _mc2_op(op_id, family, recipe, deps=(), shape=(TOKENS, TOPK, WIDTH), **kw):
    return OpSpec(
        op_id=op_id, family=family, phase="decode", stage=0, layer=0,
        requests=("r",), semantic_shape=shape, local_shape=shape,
        padded_shape=shape, launch_shape=shape, deps=deps, recipe=recipe,
        dtype="bf16", **kw)


def _mc2_graph(activation_format="int8", scale_shape=(math.ceil(RECEIVED),),
               scale_bytes=RECEIVED * 4.0, with_quant=True):
    prefix = "s0.l0.ffn"
    pack = _mc2_op(prefix + ".pack", "routing", "route_pack")
    dispatch = _mc2_op(prefix + ".dispatch", "collective", "expert_dispatch",
                       deps=(pack.op_id,))
    combine = _mc2_op(prefix + ".combine", "collective", "expert_combine",
                      deps=(dispatch.op_id,))
    unpack = _mc2_op(prefix + ".unpack", "elementwise", "route_unpack",
                     deps=(combine.op_id,))
    rows = TOKENS * TOPK
    quant = OpSpec(
        op_id=prefix + ".pack.quantize", family="quantize", phase="decode",
        stage=0, layer=0, requests=("r",), semantic_shape=(rows, WIDTH),
        local_shape=(rows, WIDTH), padded_shape=(rows, WIDTH),
        launch_shape=(rows, WIDTH), deps=(pack.op_id,),
        recipe="dynamic_quantize", dtype="bf16", mte2_bytes=800.0,
        mte3_bytes=200.0, vector_ops=4000.0,
        metadata=(("input_format", "bf16"),
                  ("output_format", activation_format),
                  ("activation_group_size", WIDTH),
                  ("activation_scale_format", "fp32"),
                  ("activation_scale_shape", (rows,)),
                  ("activation_scale_bytes", rows * 4.0),
                  ("source_rows", float(rows)),
                  ("quantization_scope", "moe_dispatch_expert_input")))
    gate = OpSpec(
        op_id=prefix + ".expert_up", family="gemm", phase="decode", stage=0,
        layer=0, requests=("r",), semantic_shape=(TOKENS, 2 * WIDTH, WIDTH),
        local_shape=(TOKENS, 2 * WIDTH, WIDTH),
        padded_shape=(TOKENS, 2 * WIDTH, WIDTH),
        launch_shape=(TOKENS, 2 * WIDTH, WIDTH), deps=(dispatch.op_id,),
        recipe="expert_up", dtype="bf16",
        metadata=(("activation_format", activation_format),
                  ("activation_group_size", WIDTH),
                  ("activation_scale_format", "fp32"),
                  ("activation_scale_shape", scale_shape),
                  ("activation_scale_bytes", scale_bytes)))
    ops = [pack, dispatch, combine, unpack]
    if with_quant:
        ops.append(quant)
    ops.append(gate)
    return prefix, ops


def _mc2_lower(ops, prefix, **options):
    from mindie_sim.prediction.operators.mc2.mc2_graph import lower

    runtime = SimpleNamespace(
        kernels={"native_moe_protocol": {"kind": "arch35_fullmesh",
                                         **options.pop("protocol", {})}},
        metadata={}, moe_recipe="alltoall")
    return lower(ops, runtime=runtime, prefix=prefix, tokens=TOKENS,
                 top_k=TOPK, width=WIDTH, experts=32, ep=EP,
                 source_partitions=1, received_assignments=RECEIVED,
                 dtype="bf16", **options)


def test_mc2_reemits_a_receiver_quantizer_for_float_dispatch():
    prefix, ops = _mc2_graph()
    result = _mc2_lower(ops, prefix)
    ids = [op.op_id for op in result]
    assert prefix + ".pack.quantize" not in ids
    assert prefix + ".dispatch.received_quantize" in ids
    quantizer = next(op for op in result
                     if op.op_id == prefix + ".dispatch.received_quantize")
    gate = next(op for op in result if op.op_id == prefix + ".expert_up")
    assert gate.deps == (quantizer.op_id,)
    metadata = dict(quantizer.metadata)
    assert metadata["quantization_scope"] == "moe_received_expert_input"
    assert metadata["source_rows"] == RECEIVED
    scale = RECEIVED / (TOKENS * TOPK)
    assert quantizer.mte2_bytes == pytest.approx(800.0 * scale)
    assert quantizer.mte3_bytes == pytest.approx(200.0 * scale)
    assert quantizer.vector_ops == pytest.approx(4000.0 * scale)
    gate_meta = dict(gate.metadata)
    assert gate_meta["activation_payload_producer"] == quantizer.op_id
    assert gate_meta["activation_scale_producer"] == quantizer.op_id


def test_mc2_absorbs_the_pack_quantizer_of_a_quantized_dispatch():
    prefix, ops = _mc2_graph()
    result = _mc2_lower(ops, prefix,
                        protocol={"dispatch_quant": "per_token_int8"})
    ids = [op.op_id for op in result]
    assert prefix + ".pack.quantize" not in ids
    assert prefix + ".dispatch.received_quantize" not in ids
    gate = next(op for op in result if op.op_id == prefix + ".expert_up")
    assert dict(gate.metadata)["activation_producer"] == (
        "explicit_transport_quantization")
    assert gate.deps == (prefix + ".dispatch",)


def test_mc2_rejects_an_external_pack_quantizer_consumer():
    from mindie_sim.schema.common import ModelingError

    prefix, ops = _mc2_graph()
    consumer = _mc2_op("other.consumer", "elementwise", "scale_read",
                       deps=(prefix + ".pack.quantize",))
    with pytest.raises(ModelingError, match="mc2_cannot_absorb_external"):
        _mc2_lower([*ops, consumer], prefix)


def test_mc2_non_mc2_selection_falls_through_without_raising():
    from mindie_sim.prediction.operators.mc2.mc2_graph import select_common

    runtime = SimpleNamespace(
        kernels={}, metadata={"moe_comm_mode": "dynamic"},
        moe_recipe="alltoall")
    features = SimpleNamespace(tp=1, graph_buckets=(), metadata=(
        ("moe_comm_mode", "dynamic"), ("max_num_seqs", 1)))
    # world_size == 1 selects AllGather in the source; automatic selection
    # simply declines instead of raising a modeling error.
    assert select_common(runtime, features, tokens=1, top_k=4, experts=32,
                         ep=1) is None


# --------------------------------------------------------------------------- #
# #52.1: model-scope speculative analysis width.
# --------------------------------------------------------------------------- #
def _mtp_case():
    return parse_case(compose_case(
        "deepseek-v4.1-flash", hardware="a3", tp=1, ep=1, num_devices=1,
        layers=2, mtp_proposal_tokens=3, mtp_acceptance_length=2.5))


def test_analysis_view_expands_the_declared_verification_width():
    from mindie_sim.prediction.analysis_inputs import analysis_view

    case = _mtp_case()
    view = analysis_view(case, "r", 4, 128, "model", "decode")
    assert (view.q_len, view.draft_proposal_tokens) == (1, 3)
    control = analysis_view(case, "r", 1, 128, "model", "decode")
    assert (control.q_len, control.draft_proposal_tokens) == (1, 0)


@pytest.mark.parametrize("tokens", (2, 3, 5, 8))
def test_analysis_view_keeps_explicit_probe_rows(tokens):
    from mindie_sim.prediction.analysis_inputs import analysis_view

    case = _mtp_case()
    view = analysis_view(case, "r", tokens, 128, "model", "decode")
    assert (view.q_len, view.draft_proposal_tokens) == (tokens, 0)


def test_model_scope_routes_verification_rows_without_double_expansion():
    from mindie_sim.prediction.operator_analysis import target_graphs

    case = _mtp_case()
    control = target_graphs(case, phase="decode", tokens=1, context=128,
                            sequences=1, scope="model")
    verify = target_graphs(case, phase="decode", tokens=4, context=128,
                           sequences=1, scope="model")
    probe = target_graphs(case, phase="decode", tokens=8, context=128,
                          sequences=1, scope="model")
    assert control[0][0].local_shape[0] == 1
    assert verify[0][0].local_shape[0] == 4
    assert probe[0][0].local_shape[0] == 8


# --------------------------------------------------------------------------- #
# #52.2: pure SWA MQSMLA geometry.
# --------------------------------------------------------------------------- #
def test_mqsmla_pure_swa_has_zero_compressed_rows():
    from mindie_sim.prediction.operators.paged_mla.family_mqsmla import query_rows, query_shape
    from mindie_sim.schema.workload import RequestView

    case = parse_case(compose_case("deepseek-v4.1-flash", hardware="a5",
                                   tp=8, ep=8, num_devices=8, layers=4))
    views = (RequestView("r", 128, 131583, True, "decode"),)
    pure = next(plan for plan in case.model.layer_plans
                if plan.attention == "mla" and plan.compress_ratio == 0
                and plan.index_topk)
    shape = query_shape(views, pure, topk_present=True)
    assert shape["ratio"] == 0
    assert set(query_rows(shape)) == {(128, 0)}
    sparse = next(plan for plan in case.model.layer_plans
                  if plan.attention == "mla" and plan.compress_ratio > 0)
    sparse_shape = query_shape(views, sparse, topk_present=True)
    assert set(query_rows(sparse_shape)) == {(128, 512)}
    # A compressed cache declaration without an index TopK has no compressed
    # branch at all.
    no_topk = query_shape(views, replace(pure, compress_ratio=4),
                          topk_present=False)
    assert no_topk["ratio"] == 0 and no_topk["topk"] == 0
    assert set(query_rows(no_topk)) == {(128, 0)}


def test_mqsmla_rejects_padded_views_and_keeps_padded_tokens():
    from mindie_sim.prediction.operators.paged_mla.family_mqsmla import unsupported_reason
    from mindie_sim.schema.workload import RequestView

    raw = compose_case("deepseek-v4.1-flash", hardware="a5", tp=8, ep=8,
                       num_devices=8, layers=2)
    case = parse_case(raw)
    plan = next(plan for plan in case.model.layer_plans
                if plan.attention == "mla" and plan.compress_ratio == 0)
    parallel = case.runtime.parallel
    padded = (RequestView("r", 4, 128, False, "decode", padding=True),)
    assert unsupported_reason(
        views=padded, plan=plan, parallel=parallel, model=case.model,
        padded_tokens=4) == "mqsmla_binding_requires_contiguous_unpadded_decode_or_draft_rows"
    split = (RequestView("r", 4, 128, False, "decode",
                         query_ranges=((0, 2), (2, 4))),)
    assert unsupported_reason(
        views=split, plan=plan, parallel=parallel, model=case.model,
        padded_tokens=4) == "mqsmla_binding_requires_contiguous_unpadded_decode_or_draft_rows"
    # A wider physical capture bucket is admitted; a narrower one is not.
    contiguous = (RequestView("r", 4, 128, True, "decode"),)
    assert unsupported_reason(
        views=contiguous, plan=plan, parallel=parallel, model=case.model,
        padded_tokens=8) != "mqsmla_binding_requires_positive_padded_rows"
    assert unsupported_reason(
        views=contiguous, plan=plan, parallel=parallel, model=case.model,
        padded_tokens=2) == "mqsmla_binding_requires_positive_padded_rows"


# --------------------------------------------------------------------------- #
# #52.3: explicit HBM service modes follow the derived bandwidth.
# --------------------------------------------------------------------------- #
def test_hbm_service_modes_scale_with_the_derived_bandwidth():
    from mindie_sim.hardware_derivation import derive_hardware_sku
    from mindie_sim.schema.hardware import Hardware

    base = Hardware.from_mapping(_hardware_mapping())
    peak = float(dict(base.dies[0].rates)["hbm"])
    mode_rate = float(dict(base.dies[0].service_modes)["hbm_read"][1])
    cases = (
        ({"hbm_peak_bytes_per_s": peak * 0.5}, 0.5),
        ({"hbm_efficiency": 0.7}, 0.7),
        ({"hbm_peak_bytes_per_s": peak * 0.5, "hbm_efficiency": 0.6}, 0.3),
        ({"aic_count": base.dies[0].compute_units}, 1.0),
    )
    for overrides, factor in cases:
        derived = derive_hardware_sku(base, sku="derived-hbm", **overrides)
        die = derived.dies[0]
        assert float(dict(die.rates)["hbm"]) == pytest.approx(peak * factor)
        assert float(dict(die.service_modes)["hbm_read"][1]) == pytest.approx(
            mode_rate * factor)
        rederived = derive_hardware_sku(derived, sku="derived-hbm")
        assert float(dict(
            rederived.dies[0].service_modes)["hbm_read"][1]) == pytest.approx(
                mode_rate * factor)


def test_dram_bound_service_modes_scale_with_each_derived_hbm_rate():
    """ACLNN DRAM streams follow the declared HBM bandwidth, not a fixed rate.

    ``mte2_aic_dram`` / ``mte2_aiv_dram`` declare an HBM-serviced read stream
    while retaining their MTE2 resource name.  Re-deriving from the scaled
    profile must not apply the ratio a second time.
    """
    from mindie_sim.hardware_derivation import derive_hardware_sku
    from mindie_sim.schema.hardware import Hardware

    base = Hardware.from_mapping({"backend": "ascend", "sku": "Ascend950DT_9582"})
    original_hbm = float(dict(base.dies[0].rates)["hbm"])
    original = {name: float(dict(base.dies[0].service_modes)[name][1])
                for name in ("mte2_aic_dram", "mte2_aiv_dram")}
    for peak in (1.4e12, 1.2e12):
        factor = peak / original_hbm
        derived = derive_hardware_sku(
            base, sku="derived-dram", hbm_peak_bytes_per_s=peak)
        die = derived.dies[0]
        assert float(dict(die.rates)["hbm"]) == pytest.approx(peak)
        for name, rate in original.items():
            assert float(dict(die.service_modes)[name][1]) == pytest.approx(
                rate * factor)
        rederived = derive_hardware_sku(derived, sku="derived-dram")
        for name, rate in original.items():
            assert float(dict(rederived.dies[0].service_modes)[name][1]
                         ) == pytest.approx(rate * factor)


def _hardware_mapping():
    machine = compose_case("qwen3-32b", hardware="a3", tp=1, ep=1,
                          num_devices=1, layers=1)["machine"]
    mapping = json.loads(json.dumps(machine))
    devices = mapping.get("devices") or [mapping["device"]]
    for device in devices:
        device["service_modes"] = {
            "hbm_read": {"resource": "hbm",
                         "rate": float(device["rates"]["hbm"])}}
    return mapping


# --------------------------------------------------------------------------- #
# #52.4: uniform replica cloning and request rebinding.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dp", (32, 64))
def test_model_scope_uniform_replicas_match_unrolled_lowering(dp, monkeypatch):
    from mindie_sim.prediction import operator_analysis as oa

    raw = compose_case("qwen3-30b-a3b", hardware="a3", tp=1, ep=dp, dp=dp,
                       num_devices=dp, layers=1, prompt_tokens=8,
                       output_tokens=2, concurrency=1, max_num_seqs=1)
    case = parse_case(raw)
    real_ops = oa._target_ops
    calls = []

    def recording(case_, views_, scope_, **kwargs):
        calls.append((case_, views_, scope_, dict(kwargs)))
        return real_ops(case_, views_, scope_, **kwargs)

    monkeypatch.setattr(oa, "_target_ops", recording)
    from mindie_sim.prediction import local_projection_tp

    monkeypatch.setattr(oa, "bind_cohort_traffic", lambda _case, graphs: graphs)
    monkeypatch.setattr(local_projection_tp, "lower_local_projection_tp_cohort",
                        lambda graphs, **_kwargs: graphs)
    graphs = oa.target_graphs(case, phase="decode", tokens=1, context=128,
                              sequences=1, scope="model")
    assert len(calls) == 1
    first_case, _first_views, scope, kwargs = calls[0]
    assert scope == "model"
    assert len(graphs.programs) == dp
    for replica in sorted(graphs.programs)[1:]:
        target_views = tuple(view for view in kwargs["global_views"]
                             if view.request_id.startswith(
                                 f"analysis{replica}:"))
        unrolled_kwargs = dict(kwargs)
        unrolled_kwargs["replica"] = replica
        unrolled_kwargs["physical_rank"] = oa._physical_rank_for(case, replica)
        unrolled_kwargs["model_programs"] = {}
        unrolled = real_ops(first_case, target_views, "model",
                            **unrolled_kwargs)
        assert [op.to_dict() for op in unrolled] == [
            op.to_dict() for op in graphs[replica]]
        assert unrolled_kwargs["model_programs"][replica].to_dict(
            include_ops=True) == graphs.programs[replica].to_dict(
                include_ops=True)


# --------------------------------------------------------------------------- #
# #49: host PLE staging and zero duplicate device work.
# --------------------------------------------------------------------------- #
def _engram_case():
    return parse_case(compose_case(
        "qwen3.8-flash-next", hardware="a5", tp=2, ep=1, num_devices=2,
        layers=2, engram_cpu_offload=True, prompt_tokens=8, output_tokens=2,
        concurrency=1))


def test_ple_host_lookup_has_zero_duplicate_device_work():
    from mindie_sim.prediction.operator_analysis import target_operators

    ops = target_operators(_engram_case(), phase="prefill", tokens=8,
                           context=0, sequences=1, scope="body")
    lookups = [op for op in ops if op.recipe == "ple_host_uva_lookup"]
    assert lookups
    for op in lookups:
        assert (op.vector_ops, op.mte2_bytes, op.mte3_bytes, op.hbm_bytes) == (
            0.0, 0.0, 0.0, 0.0)
        work = dict(op.metadata)["resource_work"]
        assert set(work) == {"host_memory", "host_vector", "h2d"}
        assert work["host_memory"] == work["h2d"] > 0.0
        assert work["host_vector"] > 0.0


def test_host_staging_requires_qualified_bound_keys():
    from mindie_sim.prediction.program import HostStaging
    from mindie_sim.schema.common import ModelingError

    def context(work, names=None):
        return SimpleNamespace(
            resource_names=names if names is not None else {
                "host_memory": "host_memory@r0:s0",
                "host_vector": "host_vector@r0:s0",
                "h2d": "h2d@r0:s0"},
            work=work, minimum_ms=0.0)

    qualified = context({"host_memory@r0:s0": 81920.0,
                         "host_vector@r0:s0": 512.0, "h2d@r0:s0": 81920.0})
    program = HostStaging().lower(qualified)
    stages = {stage.name: dict(stage.work) for stage in program.stages}
    assert stages["lookup"] == {"host_memory@r0:s0": 81920.0}
    assert stages["decode"] == {"host_vector@r0:s0": 512.0}
    assert stages["transfer"] == {"h2d@r0:s0": 81920.0}
    with pytest.raises(ModelingError, match="host_staging_requires_bound"):
        HostStaging().lower(context({"host_memory@r0:s0": 1.0}))
    with pytest.raises(ModelingError, match="host_staging_requires_host_memory"):
        HostStaging().lower(context({}, names={}))


# --------------------------------------------------------------------------- #
# #50: bounded two-stage streaming sweep.
# --------------------------------------------------------------------------- #
def _sweep_case():
    return compose_case("qwen3-32b", hardware="a3", tp=1, ep=1, pp=1,
                        num_devices=1, layers=2, prompt_tokens=8,
                        output_tokens=2, concurrency=1, max_num_seqs=1)


def _sweep_kwargs(**extra):
    return {"context": 8, "output_budget": 2, "max_sequences_per_dp": 1, **extra}


def test_b1_screen_never_runs_the_graph(monkeypatch):
    from mindie_sim.prediction import efficiency
    from mindie_sim.prediction.evaluation import EvaluationContext
    from mindie_sim import sweep_engine
    from mindie_sim.sweep_resize import materialize_machine_for_devices

    def forbidden(*_args, **_kwargs):
        raise AssertionError("b1_screen executed a full graph")

    monkeypatch.setattr(efficiency, "_lower_and_run_graph", forbidden)
    parsed = parse_case(_sweep_case())
    parallel = ParallelSpec.from_mapping({"dp": 1, "tp": 1, "ep": 1, "pp": 1})
    machine = materialize_machine_for_devices(parsed, parsed, 1)
    candidate = sweep_engine.build_candidate_case(parsed, machine, parallel, ())
    static = sweep_engine.validate_candidate_statically(
        parsed, parallel, cand_devices=1, cand_machine=machine, part=(),
        phase="decode", context=8, q_tokens=1, emit_tokens=1.0,
        output_budget=2, max_sequences_per_dp=1)
    row = sweep_engine.b1_screen(
        parsed, candidate, parallel,
        common={"candidate_id": "c", "parallel": {"dp": 1}, "num_devices": 1,
                "ranks_per_node": 1},
        ctx=EvaluationContext.create(candidate), phase="decode", scope="body",
        context=8, q_tokens=1, emit_tokens=1.0, sla_ms=1e-9,
        b_vram_max=int(static["b_vram_max_joint"]),
        max_sequences_per_dp=1, output_budget=2, static=static)
    assert row["evaluation_stage"] == "b1_screen"
    assert row["evaluation_boundary"] == "b1_isolated_operator_sum"
    assert row["is_screening_estimate"] is True
    assert row["b1_sla_pass"] is True
    assert row["operator_sum_screen_ms"] == pytest.approx(row["step_ms_at_b1"])
    assert row["operator_sum_screen_ms"] > 0.0
    assert row["tps_screen_b1"] == pytest.approx(row["tps_at_b1"])
    assert row["graph_ttft_ms"] is None
    assert row["search"]["evaluated_candidates"] == [SCREEN_BATCH]


def test_finalist_top_k_ranks_the_global_screen():
    from mindie_sim.sweep import sweep_topologies_iter

    rows = list(sweep_topologies_iter(
        _sweep_case(), two_stage=True, finalist_top_k=1,
        num_devices_grid=(1, 2), **_sweep_kwargs()))
    screened = [row for row in rows if row["evaluation_stage"] == "b1_screen"]
    full = [row for row in rows if row["evaluation_stage"] == "full_sla"]
    assert len(screened) >= 2
    assert len(full) == 1
    best = min(screened, key=lambda row: (row["operator_sum_screen_ms"],
                                          row["candidate_id"]))
    assert full[0]["candidate_id"] == best["candidate_id"]
    assert full[0]["graph_step_ms"] == pytest.approx(full[0]["step_ms_at_b1"])


def test_streaming_checkpoint_repairs_a_torn_tail(tmp_path):
    from mindie_sim.sweep import reconstruct_sweep_summary, sweep_topologies_iter

    path = tmp_path / "sweep.jsonl"
    first = list(sweep_topologies_iter(
        _sweep_case(), output_jsonl=path, num_devices_grid=(1, 2),
        **_sweep_kwargs()))
    with path.open("ab") as stream:
        stream.write(b'{"candidate_id": "truncated')
    resumed = list(sweep_topologies_iter(
        _sweep_case(), checkpoint_jsonl=path, resume=True,
        num_devices_grid=(1, 2), **_sweep_kwargs()))
    assert len(resumed) == len(first)
    assert all(row.get("resumed_from_checkpoint") is True for row in resumed)
    assert [row["candidate_id"] for row in resumed] == [
        row["candidate_id"] for row in first]
    actual = reconstruct_sweep_summary(resumed)["candidates"]
    for row in actual:
        row.pop("resumed_from_checkpoint", None)
    assert actual == reconstruct_sweep_summary(first)["candidates"]
    assert path.read_bytes().endswith(b"\n")


def test_checkpoint_alias_conflict_is_rejected(tmp_path):
    from mindie_sim.sweep import sweep_topologies_iter

    with pytest.raises(ValueError, match="conflicting_checkpoint_paths"):
        list(sweep_topologies_iter(
            _sweep_case(), output_jsonl=tmp_path / "a.jsonl",
            checkpoint_jsonl=tmp_path / "b.jsonl", **_sweep_kwargs()))
    with pytest.raises(ValueError, match="finalist_top_k_must_be_positive"):
        list(sweep_topologies_iter(
            _sweep_case(), two_stage=True, finalist_top_k=0,
            **_sweep_kwargs()))


def test_resume_upgrades_screening_rows_and_skips_committed_stages(tmp_path):
    from mindie_sim.sweep import sweep_topologies_iter

    path = tmp_path / "resume.jsonl"
    rows = list(sweep_topologies_iter(
        _sweep_case(), two_stage=True, output_jsonl=path, num_devices=1,
        **_sweep_kwargs()))
    assert rows
    path.write_bytes(b"".join(
        (json.dumps({**row, "evaluation_stage": "b1_screen"},
                    sort_keys=True) + "\n").encode() for row in rows))
    resumed = list(sweep_topologies_iter(
        _sweep_case(), two_stage=True, output_jsonl=path, resume=True,
        num_devices=1, **_sweep_kwargs()))
    assert resumed
    assert not any(row.get("resumed_from_checkpoint") for row in resumed)
    assert any(row["evaluation_stage"] == "full_sla" for row in resumed)


def test_isolated_worker_runs_and_times_out():
    from mindie_sim.sweep import sweep_topologies_iter

    rows = list(sweep_topologies_iter(
        _sweep_case(), two_stage=True, finalist_top_k=1, isolate_workers=True,
        candidate_timeout_s=120.0, num_devices=1, **_sweep_kwargs()))
    assert any(row["evaluation_stage"] == "b1_screen" for row in rows)
    assert any(row["evaluation_stage"] == "full_sla" for row in rows)
    timed_out = list(sweep_topologies_iter(
        _sweep_case(), isolate_workers=True, candidate_timeout_s=1e-9,
        num_devices=1, **_sweep_kwargs()))
    assert [row["status"] for row in timed_out] == ["timeout"]
    assert timed_out[0]["feasibility_status"] == "candidate_timeout"


def test_sweep_cli_exposes_the_bounded_flags(tmp_path):
    from mindie_sim.__main__ import parser

    args = parser().parse_args([
        "sweep", "case.json", "--two-stage", "--finalist-top-k", "3",
        "--checkpoint-jsonl", str(tmp_path / "c.jsonl"), "--isolate-workers",
        "--resume"])
    assert args.finalist_top_k == 3
    assert args.checkpoint_jsonl == str(tmp_path / "c.jsonl")
    assert args.isolate_workers is True
    assert args.resume is True
    assert args.output_jsonl is None
