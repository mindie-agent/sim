"""Wave 3A: CANN 9.2.1 MLA / compressor / KeyPool / prolog alignments.

[Role] Prove the Six declared contracts conserve arithmetic, bytes and deps.
[Invariants] Every assertion derives from declared geometry or an operator
  graph; no captured timing, model name or measurement selects a branch.
[Timeouts & Locks] Tier 0-1, in-memory graphs only; no device or locks.
[Side-Effect Policy] Read-only over catalog cases and builder-local records.
"""
from dataclasses import replace
import json
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim import target_graphs
from mindie_sim.catalog.cache_format import (
    CacheFormat, block_quantized_mla_format, block_quantized_row_bytes,
)
from mindie_sim.catalog.draft_geometry import native_draft_layer_plans
from mindie_sim.catalog.quant import projection_quant
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction import compression, draft_context, family_mla_prolog
from mindie_sim.prediction.compression import RECIPE as COMPRESSOR_RECIPE
from mindie_sim.prediction.compression import compressor
from mindie_sim.prediction.family_mqsmla import (
    LEGACY_RECIPE as MQSMLA_LEGACY_RECIPE,
)
from mindie_sim.prediction.family_mqsmla import RECIPE as MQSMLA_RECIPE
from mindie_sim.prediction.implementations import IMPLEMENTATIONS
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.mla_native import index_layer_norm
from mindie_sim.prediction.mla_prolog_packed import (
    LAYOUT, MXFP8_QUERY_FORMAT, NATIVE_KIND_V3, NATIVE_KIND_V4, view as cache_view,
)
from mindie_sim.prediction.mqsmla import (
    LEGACY_RECIPE as KERNEL_LEGACY_RECIPE,
)
from mindie_sim.prediction.mqsmla import (
    RECIPE as KERNEL_RECIPE, build as build_mqsmla, query_rows,
    selective_fd_excluded,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.native_fusion_traffic import apply_traffic_profile
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec


ROOT = Path(__file__).resolve().parents[1]
GLM_CASE = (ROOT / "examples/cases/high-concurrency-20261030/matrix-extension/"
            "a3_glm53flash_tp16_aclgraph_c32.json")
KEYPOOL_IMPLEMENTATION = "vllm_ascend_cann_keypool_caller"
BLOCK = 128
HIDDEN = 4096
HEAD_WIDTH = 576
INDEX_WIDTH = 128
INDEX_HEADS = 32
KV_RANK = 512
ROPE_WIDTH = 64
PROLOG_QUERY_LORA_RANK = 1536


def _mla_plan(**overrides):
    fields = dict(
        index=0, attention="mla", architecture="mla", ffn="dense",
        intermediate_size=1024, num_heads=8, num_kv_heads=1, head_dim=HEAD_WIDTH,
        q_lora_rank=KV_RANK, kv_lora_rank=KV_RANK, qk_nope_head_dim=KV_RANK,
        qk_rope_head_dim=ROPE_WIDTH, v_head_dim=128, mla_variant="latent_mla",
        cache_kind="window_compressed", compress_ratio=4, compressor_width=1,
        compressor_state_rows=8, compressor_kind="gated_ring", owns_long_kv=True,
        kv_owner_layer_id=0, window=4096,
    )
    fields.update(overrides)
    return LayerPlan(**fields)


def _model(plan, *, quant=None, metadata=()):
    return ModelSpec(
        name="wave3a", layers=1, hidden_size=HIDDEN, num_heads=8,
        head_dim=HEAD_WIDTH, intermediate_size=1024, vocab_size=1024,
        attention="mla", ffn="dense", layer_plans=(plan,), metadata=metadata,
        quant=quant or QuantSpec("bf16"))


def _builder(plan, model):
    return Builder("s0.l0", plan, 0, (), model, "decode")


def _by_recipe(ops):
    return {op.recipe: op for op in ops}


# --- M1: declared NoPE geometry ------------------------------------------
def test_mla_use_nope_round_trips_and_selects_zero_rotary():
    from mindie_sim.schema.model import MlaSpec
    spec = MlaSpec.from_mapping({
        "q_lora_rank": KV_RANK, "kv_lora_rank": KV_RANK,
        "qk_nope_head_dim": KV_RANK, "qk_rope_head_dim": ROPE_WIDTH,
        "v_head_dim": 128, "mla_use_nope": True})
    assert spec.mla_use_nope is True
    assert spec.to_dict()["mla_use_nope"] is True
    plan = _mla_plan(mla_use_nope=True)
    assert plan.mla_do_rope is False
    assert plan.to_dict()["mla_use_nope"] is True
    assert _mla_plan().mla_do_rope is True
    assert _mla_plan(qk_rope_head_dim=0).mla_do_rope is False
    # The learned tail stays in the cached row width even without rotary work.
    assert _mla_plan(mla_use_nope=True).cache_row_elements == KV_RANK + ROPE_WIDTH


def test_draft_plans_inherit_native_fusion_and_router_fields():
    body = _mla_plan(
        packed_input_projection=True, mla_use_nope=True,
        router_output_dtype="fp32", router_dtype="bf16",
        router_precast_weight_copy=False, ffn="moe", num_experts=8, top_k=2,
        moe_intermediate_size=256, router_mode="noaux_tc", routed_scaling_factor=2.5,
        indexer=True, indexer_type="full", index_heads=INDEX_HEADS,
        index_dim=INDEX_WIDTH, index_topk=512, index_kpool=4,
        index_kpool_compress=True, index_input="hidden")
    model = _model(body)
    model = replace(model, layer_plans=(body,))
    draft = SimpleNamespace(
        layers=2, attention_kind="mla", mla_variant="latent_mla",
        hidden_size=HIDDEN, intermediate_size=256, num_heads=8, num_kv_heads=1,
        head_dim=HEAD_WIDTH, q_lora_rank=KV_RANK, kv_lora_rank=KV_RANK,
        qk_nope_head_dim=KV_RANK, qk_rope_head_dim=ROPE_WIDTH, v_head_dim=128,
        num_experts=8, top_k=2, expert_intermediate_size=256,
        num_shared_experts=0, o_lora_rank=0, o_groups=0,
        index_share_for_mtp_iteration=True, draft_compress_ratios=(),
        target_layer_ids=(0,), mhc_mult=0, mhc_lowrank=0,
        mhc_sinkhorn_iters=0, kv_cache_elements=0, qk_norm=False,
        q_output_gate=False, sliding_window=0, noncausal_query_block=False,
        source={"kind": "glm_mtp"},
    )
    plans = native_draft_layer_plans(model, draft)
    assert len(plans) == 2
    for plan in plans:
        assert plan.packed_input_projection is True
        assert plan.mla_use_nope is True
        assert plan.router_output_dtype == "fp32"
        assert plan.router_dtype == "bf16"
        assert plan.router_precast_weight_copy is False
        # Every proposal step updates its own KPool row; only the shared TopK
        # selection is a per-step execution decision.
        assert plan.owns_index_k is True
        assert plan.index_k_owner_layer_id == plan.index
        assert plan.indexer_type == "shared"
        assert plan.owns_topk is True


# --- M4: compressor ratio<=1 and single fused operator -------------------
def test_compressor_c1_layer_has_no_gate_or_pool():
    plan = _mla_plan(compress_ratio=1)
    model = _model(plan)
    terms = {term.name for term in layer_weight_terms(plan, model)}
    assert "compressor_wkv" in terms and "compressor_norm" in terms
    assert not {"compressor_gate", "compressor_ape"} & terms
    builder = _builder(plan, model)
    ready = compressor(builder, name="compress", source="hidden", tokens=8,
                       padded_tokens=8, valid_rows=2, width=HEAD_WIDTH,
                       plan=plan, model=model)
    recipes = {op.recipe for op in builder.ops}
    assert recipes == {"compress_values", "compress_norm", "compress_rotary"}
    assert ready == "s0.l0.compress_rotary"
    values = _by_recipe(builder.ops)["compress_values"]
    assert values.flops == 2 * 8 * HEAD_WIDTH * HIDDEN


def test_fused_compressor_conserves_member_work_and_reads_hidden_once():
    plan = _mla_plan()
    model = _model(plan)
    builder = _builder(plan, model)
    ready = compressor(builder, name="compress", source="hidden", tokens=8,
                       padded_tokens=8, valid_rows=2, width=HEAD_WIDTH,
                       plan=plan, model=model, return_latent=True, fused=True)
    assert len(builder.ops) == 1
    op = builder.ops[0]
    assert op.recipe == COMPRESSOR_RECIPE
    assert op.op_id == ready[0] == ready[1]
    meta = dict(op.metadata)
    assert meta["native_kind"] == "aclnnCompressor"
    assert meta["hidden_read_once"] is True
    pool_rows = 8 * HEAD_WIDTH
    assert op.flops == 2 * 2 * pool_rows * HIDDEN
    assert op.vector_ops == (6 * pool_rows + 5 * 2 * HEAD_WIDTH
                             + 6 * 2 * ROPE_WIDTH)
    weight_bytes = builder.quant.matrix_bytes(HEAD_WIDTH, HIDDEN,
                                              compression.COMPRESSION_RECIPE)
    # The rolling state is read and written once per bounded state row, not
    # once per projected token; the APE table is one ratio-wide FP32 plane.
    state_rows = min(8, plan.compressor_state_rows)
    state = state_rows * HEAD_WIDTH * int(dtype_bytes("fp32"))
    ape = plan.compress_ratio * HEAD_WIDTH * int(dtype_bytes("fp32"))
    norm_weight = HEAD_WIDTH * dtype_bytes(plan.compressor_norm_dtype)
    rope_table = 2 * ROPE_WIDTH * int(dtype_bytes("fp32")) * 2
    assert op.mte2_bytes == (8 * HIDDEN * dtype_bytes("bf16")
                             + 2 * weight_bytes + state + ape
                             + norm_weight + rope_table)
    # Rotated cache row plus the materialized pre-RoPE latent and the rolling
    # state write; hidden is read once and no intermediate round trip exists.
    assert op.mte3_bytes == (2 * 2 * HEAD_WIDTH * dtype_bytes("bf16") + state)
    assert op.returns == ("latent",)
    assert dict(op.metadata)["latent_shape"] == (2, HEAD_WIDTH)


# --- M1: native indexer LayerNorm and head weights ----------------------
def test_native_index_layer_norm_reads_parameters_and_emits_no_cast():
    plan = _mla_plan()
    model = _model(plan)
    builder = _builder(plan, model)
    ready = index_layer_norm(builder, source="hidden", rows=8, width=INDEX_WIDTH,
                             native=True)
    assert len(builder.ops) == 1
    op = builder.ops[0]
    assert ready == op.op_id
    assert op.family == "layernorm" and op.dtype == "bf16"
    assert op.vector_ops == 8 * 8 * INDEX_WIDTH
    assert op.mte2_bytes == 8 * INDEX_WIDTH * 2 + 2 * INDEX_WIDTH * 4
    assert op.mte3_bytes == 8 * INDEX_WIDTH * 2
    meta = dict(op.metadata)
    assert meta["native_kind"] == "aclnnLayerNorm"
    assert meta["parameter_read_bytes"] == {"gamma": INDEX_WIDTH * 4,
                                            "beta": INDEX_WIDTH * 4}
    # The explicit raw-eager decomposition keeps the upstream Cast pair.
    eager = _builder(plan, model)
    index_layer_norm(eager, source="hidden", rows=8, width=INDEX_WIDTH,
                     native=False)
    assert {item.family for item in eager.ops} == {"cast", "layernorm"}


def test_indexer_head_weights_feed_the_native_contract_without_casts():
    from mindie_sim.prediction.mla_cann_epilogs import index_head_weights
    plan = _mla_plan(indexer=True, indexer_type="full", index_heads=INDEX_HEADS,
                     index_dim=INDEX_WIDTH, index_topk=512, index_kpool=4,
                     index_kpool_compress=True, index_input="hidden")
    model = _model(plan)
    builder = _builder(plan, model)
    native = index_head_weights(builder, plan=plan, model=model,
                                quant=model.quant, deps="hidden",
                                pool_weights=None, tokens=8, native=True)
    ops = _by_recipe(builder.ops)
    assert set(ops) == {"index_weights"}
    assert ops["index_weights"].dtype == "bf16"
    assert native == ops["index_weights"].op_id
    # The retained analytic chain keeps its declared explicit casts.
    legacy = _builder(plan, model)
    index_head_weights(legacy, plan=plan, model=model, quant=model.quant,
                       deps="hidden", pool_weights=None, tokens=8, native=False)
    assert "index_weights_fp16" in {op.recipe for op in legacy.ops}


# --- M4: fused indexer quantize + cache scatter epilog ------------------
def _index_cache_ops():
    from mindie_sim.prediction.mla_cann_epilogs import index_row_quantization
    plan = _mla_plan(indexer=True, indexer_type="full", index_heads=INDEX_HEADS,
                     index_dim=INDEX_WIDTH, index_topk=512)
    model = _model(plan)
    builder = _builder(plan, model)
    quantized = builder.vector(
        "index_key_quantize", ("index_norm",), 8 * INDEX_WIDTH, 4,
        family="quantize", shape=(8, INDEX_WIDTH),
        read_bytes_override=8 * INDEX_WIDTH * 2,
        write_bytes_override=8 * (INDEX_WIDTH + 4),
        returns=("index_scales",),
        metadata=(("output_format", "int8"), ("scale_format", "fp16"),
                  ("scale_shape", (8, 1)), ("scale_count", 1)))
    stored = builder.store("index_store", (quantized,), 8, INDEX_WIDTH + 4,
                           "int8")
    fused = index_row_quantization(builder, stored, quantized,
                                   width=INDEX_WIDTH, native=True)
    return builder, fused


def test_indexer_quant_cache_replaces_producer_and_store_with_one_epilog():
    builder, fused = _index_cache_ops()
    assert len(builder.ops) == 1
    op = builder.ops[0]
    assert op.op_id == fused
    meta = dict(op.metadata)
    assert meta["native_kind"] == "aclnnIndexerQuantCache"
    assert meta["fused_members"] == ("s0.l0.index_key_quantize",
                                     "s0.l0.index_store")
    # BF16 input read once, one cache-row write; the intermediate payload
    # write plus cache read are gone.
    assert op.mte2_bytes == 8 * INDEX_WIDTH * 2
    assert op.mte3_bytes == 8 * (INDEX_WIDTH + 4)
    assert op.hbm_bytes == op.mte2_bytes + op.mte3_bytes
    assert op.vector_ops == 8 * INDEX_WIDTH * 4
    assert op.deps == ("index_norm",)


# --- M2: NoPE C8 520 B/row ---------------------------------------------
def test_nope_c8_row_is_five_hundred_twenty_bytes():
    fmt = block_quantized_mla_format()
    assert fmt == CacheFormat("fp8_e4m3", BLOCK, "fp16")
    assert block_quantized_row_bytes(KV_RANK) == 520
    # Kimi keeps the 64-element learned tail: 512 B INT8 + 128 B BF16 + 16 B
    # FP32 scale.
    packed = CacheFormat("int8", BLOCK, "fp32", ROPE_WIDTH, "bf16")
    assert packed.row_bytes(KV_RANK + ROPE_WIDTH) == 656
    assert cache_view(KV_RANK, ROPE_WIDTH)["row_bytes"] == 656


# --- M5: prolog V4/DSpark V3 admission ---------------------------------
def _kimi_quant():
    return QuantSpec(
        name="W8A8_MXFP8", weight_format="fp8_e4m3",
        activation_format="fp8_e4m3", expert_weight_format="fp8_e4m3",
        weight_block=(1, 32), expert_block=32, scale_format="fp8_e8m0",
        scale_bytes=1.0, activation_group_size=32,
        activation_scale_format="fp8_e8m0", activation_scale_pack=2,
        kv_dtype="int8", index_dtype="bf16", index_scale_dtype="none",
        category_overrides={
            "indexer_fp8": {"weight_format": "fp8_e4m3",
                            "activation_format": "fp8_e4m3"},
            "indexer": {"weight_format": "fp8_e4m3",
                        "activation_format": "fp8_e4m3"},
        })


def _prolog_case():
    plan = _mla_plan(indexer=True, indexer_type="full", index_heads=INDEX_HEADS,
                     index_dim=INDEX_WIDTH, index_topk=512, owns_index_k=True,
                     owns_topk=True, cache_kind="dense", compress_ratio=0,
                     q_lora_rank=PROLOG_QUERY_LORA_RANK)
    model = _model(plan, quant=_kimi_quant())
    return plan, model


def test_prolog_v4_admits_dcp8_mxfp8_and_c8_cache():
    plan, model = _prolog_case()
    family_mla_prolog.validate(
        plan=plan, parallel=ParallelSpec(tp=8, cp=8, pcp=1), model=model,
        quant=model.quant, lowering_passes=("activation_quantization",),
        cache_layout=LAYOUT)
    ops = [
        _op("s0.l0.q_a", "gemm", (8, PROLOG_QUERY_LORA_RANK, HIDDEN)),
        _op("s0.l0.q_norm", "rmsnorm", (8, PROLOG_QUERY_LORA_RANK)),
        _op("s0.l0.q_b", "gemm", (8, 8 * HEAD_WIDTH, KV_RANK)),
        _op("s0.l0.kv_a", "gemm", (8, HEAD_WIDTH, HIDDEN)),
        _op("s0.l0.kv_norm", "rmsnorm", (8, KV_RANK)),
        _op("s0.l0.q_absorb", "gemm", (64, KV_RANK, KV_RANK)),
    ]
    family_mla_prolog.declare(
        ops, prefix="s0.l0", plan=plan, model=model,
        parallel=ParallelSpec(tp=8, cp=8, pcp=1), quant=model.quant, tokens=8,
        members=tuple(op.op_id for op in ops), q_a="s0.l0.q_a",
        q_b="s0.l0.q_b", kv_a="s0.l0.kv_a", q_absorb="s0.l0.q_absorb",
        cache_layout=LAYOUT)
    declaration = _declaration(ops, "s0.l0.q_absorb")
    assert declaration["implementation"]["native_kind"] == NATIVE_KIND_V4
    conditions = declaration["implementation"]["conditions"]
    assert conditions["mxfp8_query"] is True and conditions["e8m0_group"] == 32
    assert conditions["data_parallel_context"] == 8
    assert declaration["tensor_view"]["do_rope"] is True


def test_prolog_v3_keeps_bf16_dspark_without_a_rotary_member():
    plan, model = _prolog_case()
    plan = replace(plan, mla_use_nope=True)
    model = replace(model, quant=QuantSpec("bf16", kv_dtype="bf16"))
    family_mla_prolog.validate(
        plan=plan, parallel=ParallelSpec(tp=8, cp=2, pcp=1), model=model,
        quant=model.quant, lowering_passes=())
    ops = [
        _op("s0.l0.q_a", "gemm", (8, PROLOG_QUERY_LORA_RANK, HIDDEN)),
        _op("s0.l0.q_norm", "rmsnorm", (8, PROLOG_QUERY_LORA_RANK)),
        _op("s0.l0.q_b", "gemm", (8, 8 * HEAD_WIDTH, KV_RANK)),
        _op("s0.l0.kv_a", "gemm", (8, HEAD_WIDTH, HIDDEN)),
        _op("s0.l0.kv_norm", "rmsnorm", (8, KV_RANK)),
        _op("s0.l0.kv_store", "cache_store", (8, HEAD_WIDTH)),
        _op("s0.l0.q_absorb", "gemm", (64, KV_RANK, KV_RANK)),
    ]
    members = tuple(op.op_id for op in ops)
    family_mla_prolog.declare(
        ops, prefix="s0.l0", plan=plan, model=model,
        parallel=ParallelSpec(tp=8, cp=2, pcp=1), quant=model.quant, tokens=8,
        members=members, q_a="s0.l0.q_a", q_b="s0.l0.q_b",
        kv_a="s0.l0.kv_a", q_absorb="s0.l0.q_absorb")
    declaration = _declaration(ops, "s0.l0.q_absorb")
    assert declaration["implementation"]["native_kind"] == NATIVE_KIND_V3
    assert declaration["tensor_view"]["do_rope"] is False
    assert declaration["implementation"]["conditions"]["query_rope_member"] is False
    # The traffic profile must not require a rotary phase for a NoPE prolog.
    members = tuple(op for op in ops if op.op_id != "s0.l0.q_absorb")
    _, ledger = apply_traffic_profile(
        list(members), declaration, "preserve_workspaces")
    assert ledger["profile"] == "preserve_workspaces"


def _declaration(ops, anchor):
    node = next(op for op in ops if op.op_id == anchor)
    return dict(dict(node.metadata)["structure_subgraphs"][-1])


def _op(op_id, family, shape):
    return OpSpec(
        op_id=op_id, family=family, phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=shape, local_shape=shape,
        padded_shape=shape, launch_shape=shape, dtype="bf16",
        metadata=(("activation_format", "bf16"), ("weight_format", "bf16")))


# --- M4: official MQSMLA registration and legacy guard ------------------
def test_official_mqsmla_recipe_replaces_the_cannbot_prototype():
    assert MQSMLA_RECIPE == "aclnn_mixed_quant_sparse_flash_mla"
    assert MQSMLA_LEGACY_RECIPE == "cannbot_mixed_quant_sparse_flash_mla"
    assert MQSMLA_RECIPE in IMPLEMENTATIONS and MQSMLA_LEGACY_RECIPE in IMPLEMENTATIONS
    from mindie_sim.prediction.family_mqsmla import SOURCE_RECIPE, recipes
    assert SOURCE_RECIPE == "mixed_quant_sparse_flash_mla"
    assert set(recipes()) == {MQSMLA_RECIPE, SOURCE_RECIPE,
                              MQSMLA_LEGACY_RECIPE}
    rows = tuple(query_rows({"sequences": ((1, 2048),) * 72, "window": 128,
                             "ratio": 4, "topk": 2048}))
    assert rows == ((128, 512),) * 72
    assert selective_fd_excluded(32, rows) is True
    plan = _mla_plan(mla_variant="latent_mqa", num_kv_heads=1, head_dim=512,
                     qk_rope_head_dim=64, qk_nope_head_dim=448,
                     cache_kind="window_compressed", compress_ratio=4,
                     index_topk=2048, window=512)
    metadata = (("mqsmla_geometry", {
        "sequences": ((1, 2048),) * 72, "window": 128, "ratio": 4,
        "topk": 2048, "heads": 64, "width": 512, "metadata_support_fd": False,
        "addressing": "scalar", "implementation": KERNEL_LEGACY_RECIPE}),)
    op = _op("s0.mqsmla", "attention", (72, 64, 512))
    op = replace(op, recipe=KERNEL_LEGACY_RECIPE, metadata=metadata,
                 source_boundary="kernel")
    with pytest.raises(ModelingError, match="selective_72_row_fd_branch"):
        build_mqsmla(op, _geometry(op, cores=32))
    official = replace(op, recipe=KERNEL_RECIPE, metadata=(
        ("mqsmla_geometry", {**dict(metadata)["mqsmla_geometry"],
                             "implementation": KERNEL_RECIPE}),))
    assert build_mqsmla(official, _geometry(official, cores=32)) is not None


def test_mqsmla_admits_draft_and_gates_the_legacy_72_row_geometry():
    from mindie_sim.prediction.family_mqsmla import (
        FORMATS, unsupported_reason,
    )
    from mindie_sim.schema.workload import RequestView
    plan = _mla_plan(
        mla_variant="latent_mqa", num_heads=64, num_kv_heads=1, head_dim=512,
        qk_nope_head_dim=448, qk_rope_head_dim=64, o_groups=0,
        cache_kind="window_compressed", compress_ratio=4, window=128,
        index_topk=2048)
    model = _model(plan, metadata=(("cache_representations", {
        region: dict(declaration) for region, declaration in FORMATS.items()}),))
    parallel = ParallelSpec(tp=8)
    draft = (RequestView("r", 1, 128, True, "draft"),)
    assert unsupported_reason(views=draft, plan=plan, parallel=parallel,
                              model=model, padded_tokens=1) is None
    rows = tuple(RequestView(f"r{index}", 1, 2048, True, "decode")
                 for index in range(72))
    official = unsupported_reason(
        views=rows, plan=plan, parallel=parallel, model=model,
        padded_tokens=72, cores=32, implementation=KERNEL_RECIPE)
    legacy = unsupported_reason(
        views=rows, plan=plan, parallel=parallel, model=model,
        padded_tokens=72, cores=32, implementation=KERNEL_LEGACY_RECIPE)
    assert official != "mqsmla_selective_72_row_fd_branch_not_modeled"
    assert legacy == "mqsmla_selective_72_row_fd_branch_not_modeled"


def _geometry(op, *, cores):
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction import mqsmla
    return Geometry(
        cores, {"vector": 2 * cores, "mte2_aiv": 2 * cores, "mte3": 2 * cores,
                "mte1": cores, "mte2_aic": cores, "fixpipe": cores},
        dict(mqsmla.BUFFERS),
        {"ub": 2 * cores, "l1": cores, "l0a": cores, "l0b": cores, "l0c": cores})


# --- M2/M4 graph-level: zero rotary, fused cache epilogs, MTP reuse -----
def _glm_raw(*, proposals=None, keypool=False, quant=None, cache=None,
             lowering_passes=None):
    raw = json.loads(GLM_CASE.read_text())
    raw["model"]["layers"] = 4
    if cache is not None:
        raw["model"].setdefault("metadata", {})["cache_representations"] = cache
    if quant is not None:
        raw["model"]["quant"] = quant
    if lowering_passes is not None:
        raw["runtime"]["lowering_passes"] = list(lowering_passes)
    if proposals is not None:
        raw["runtime"].setdefault("metadata", {})["draft_proposal_tokens"] = proposals
    if keypool:
        raw["runtime"]["kernels"].setdefault("lowerings", {})["pool_key_indexer"] = {
            "implementation": KEYPOOL_IMPLEMENTATION,
            "options": {"tail_capacity": 9, "block_table_columns": 8195}}
    return raw


def test_nope_body_graph_has_zero_rotary_and_a_block_quantized_cache_epilog():
    raw = _glm_raw(cache={"mla_kv": {"dtype": "fp8_e4m3", "group_size": BLOCK,
                                     "scale_dtype": "fp16"}})
    ops = target_graphs(raw, scope="body", phase="decode", tokens=6,
                        context=0, sequences=1, apply_body_defaults=False)[0]
    assert not [op for op in ops if op.recipe in
                {"rotary", "q_rotary", "kv_rotary"}]
    store = next(op for op in ops if op.recipe == "kv_store")
    meta = dict(store.metadata)
    assert meta["native_kind"] == "aclnnDynamicBlockQuant"
    assert meta["cache_row_bytes"] == 520
    assert meta["cache_epilog"]["scatter"] == "aclnnScatterPaKvCache"
    assert store.mte3_bytes == 6 * 520


def test_mtp_index_share_runs_five_producers_and_one_selection():
    raw = _glm_raw(proposals=5, keypool=True)
    graphs = target_graphs(raw, scope="model", phase="decode", tokens=6,
                          context=0, sequences=2)
    ops = graphs[0]
    producers = [op for op in ops if op.recipe == "cann_key_pool"]
    writers = [op for op in ops if op.recipe == "cann_keypool_cache_write"]
    selections = [op for op in ops if op.recipe == "cann_pool_key_indexer"]
    topk = [op for op in ops if op.recipe == "index_topk"]
    assert len(producers) == len(writers) == 6      # 5 draft steps + body
    assert len(selections) == len(topk) == 2        # draft0 + body
    draft0_topk = next(op for op in topk if op.op_id.startswith("draft0."))
    for step in range(5):
        attention = [op for op in ops
                     if op.op_id.startswith(f"draft{step}.") and op.family == "attention"]
        assert attention and all(draft0_topk.op_id in op.deps for op in attention)


def _packed_index_cache():
    return {"index_k": {"dtype": "fp8_e4m3", "group_size": 32,
                        "scale_dtype": "fp16"}}


def test_keypool_fp8_cache_write_is_one_hundred_thirty_two_bytes_per_pool():
    raw = _glm_raw(keypool=True, cache=_packed_index_cache(),
                   quant={"name": "bf16_fp8_index", "weight_format": "bf16",
                          "activation_format": "bf16", "native_format": "bf16",
                          "expert_weight_format": "bf16", "kv_dtype": "int8",
                          "index_dtype": "fp8_e4m3", "index_scale_dtype": "fp16"})
    ops = _by_recipe(target_graphs(raw, scope="body", phase="decode", tokens=6,
                                   context=131201, sequences=4)[0])
    write = ops["cann_keypool_cache_write"]
    meta = dict(write.metadata)
    assert meta["pooled_row_bytes"] == 132
    assert meta["pooled_scale_dtype"] == "fp32"
    assert meta["cache_dtype"] == "fp8_e4m3"
    indexer = ops["cann_pool_key_indexer"]
    indexer_meta = dict(indexer.metadata)
    assert indexer.dtype == "fp8_e4m3"
    assert indexer.flops == 2 * indexer_meta["safe_reference_scanned_pools"] * 32 * INDEX_WIDTH
    assert indexer_meta["pool_row_bytes"] == 132
    assert indexer_meta["cache_reuse"] == "batch_shared_prefix_l2_deduplicated"


# --- M5: external draft context prolog ---------------------------------
def test_draft_context_emits_one_prolog_and_one_dcp_row_shard():
    draft = SimpleNamespace(
        layers=2, hidden_size=HIDDEN, q_lora_rank=KV_RANK,
        kv_lora_rank=KV_RANK, qk_rope_head_dim=ROPE_WIDTH,
        activation_dtype="bf16")
    model = _model(_mla_plan(), quant=QuantSpec("bf16"))
    projection = _op("d:projection", "gemm", (8, HIDDEN, HIDDEN))
    ops = []
    ready = draft_context.prepare(
        ops, projection=projection, model=model, draft=draft, rows=8,
        query_rows=3, prefix="d:", stage=0, requests=(), parallel=ParallelSpec(cp=2))
    recipes = [op.recipe for op in ops]
    assert recipes == ["draft_context_norm"] + [
        item for _ in range(2) for item in
        (draft_context.PROLOG_RECIPE, draft_context.SCATTER_RECIPE)] + [
        "draft_query_embedding", "draft_context_query_ready"]
    prolog = next(op for op in ops if op.recipe == draft_context.PROLOG_RECIPE)
    scatter = next(op for op in ops if op.recipe == draft_context.SCATTER_RECIPE)
    row_bytes = (KV_RANK + ROPE_WIDTH) * dtype_bytes("bf16")
    assert row_bytes == 1152
    assert scatter.local_shape[0] == ceil(8 / 2)
    assert dict(scatter.metadata)["cache_epilog"]["kv_row_bytes"] == row_bytes
    assert dict(prolog.metadata)["native_kind"] == "aclnnMlaPrologV3WeightNz"
    assert dict(prolog.metadata)["do_rope"] is True
    assert ready == "d:context_query_ready"


def test_glm_case_never_claims_rotary_for_a_zero_width_layer():
    raw = _glm_raw()
    plan = parse_case(raw).model.layer_plans[0]
    assert plan.qk_rope_head_dim == 0 and plan.mla_do_rope is False


def test_projection_quant_keeps_bf16_head_weights_for_the_native_caller():
    plan = _mla_plan(indexer=True, index_heads=INDEX_HEADS,
                     index_dim=INDEX_WIDTH, index_kpool=4)
    quant = QuantSpec("W8A8_DYNAMIC", weight_format="int8",
                      activation_format="int8", native_format="bf16",
                      index_dtype="bf16")
    resolved = projection_quant(quant, plan)
    from mindie_sim.catalog.ops import compute_dtype_for
    assert compute_dtype_for(resolved, "indexer.weights") == ("bf16", "bf16")
    assert compute_dtype_for(resolved, "indexer.pool_projection") == ("fp32", "fp32")


def test_mxfp8_query_format_is_admitted_by_the_packed_view():
    from mindie_sim.prediction.mla_prolog_packed import validate_view
    declaration = {
        "latent_rank": KV_RANK, "rope_width": ROPE_WIDTH,
        "packed_cache": cache_view(KV_RANK, ROPE_WIDTH),
        "token_dtype": MXFP8_QUERY_FORMAT,
        "projection_weight_dtypes": (MXFP8_QUERY_FORMAT,) * 3,
        "absorbed_weight_dtype": "bf16", "norm_dtype": "bf16",
        "cache_dtype": "int8", "query_norm_dtype": MXFP8_QUERY_FORMAT,
    }
    validate_view(declaration)
    bad = {**declaration, "query_norm_dtype": "int8"}
    with pytest.raises(ModelingError, match="declared_token_projections"):
        validate_view(bad)
