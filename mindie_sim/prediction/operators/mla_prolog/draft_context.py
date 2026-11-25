"""Independent context-KV preparation for external block drafters.

[Role] Separate target-feature context rows from proposal query rows.
[Invariants] Each draft layer computes and stores its own BF16 context KV.
  Context projection is not reused as the query embedding.  Fused PrologV3
  executes all four MLA projections (``q_a``, ``kv_a``, ``q_b``, ``w_uk``)
  for local heads while discarding query outputs.  Component-major dense
  strides write directly to cache with zero standalone scatter; token-fused
  interleaved FlashMLA caches stage through a dense workspace and emit one
  ``aclnnScatterPaKvCache`` writing only block-owned DCP rows.
[Timeouts & Locks] O(draft layers); no locks or external work.
[Side-Effect Policy] Appends only to the caller-owned operator list.
"""
from collections.abc import Sequence
from dataclasses import replace
from math import ceil
from typing import Literal

from mindie_sim.catalog.ops import cache_store_op, compute_dtype_for, gemm_op, vector_op, barrier_op
from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.physical_recipe_base import Phase, io_weight, tiled_phased_program

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCE_REF = SourceRef(
    repository="vllm-project/vllm-ascend",
    revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
    relative_path="vllm_ascend/attention/mla_v1.py",
    source_sha256="f7f672e001651a748d2e88ef0f36797ef78db63321ac657f1c7a944c0f561c22",
)
NORM_OPS = 5
ROPE_OPS = 6
PROLOG_RECIPE = "draft_context_mla_prolog_v3"
SCATTER_RECIPE = "draft_context_kv_scatter"
PROLOG_NATIVE_KIND = "aclnnMlaPrologV3WeightNz"
PROLOG_V4_NATIVE_KIND = "aclnnMlaPrologV4WeightNz"
SCATTER_NATIVE_KIND = "aclnnScatterPaKvCache"
DYNAMIC_QUANT_RECIPE = "dynamic_quantize"
DYNAMIC_MX_QUANT_OP = "torch_npu.npu_dynamic_mx_quant"
DYNAMIC_INT8_QUANT_OP = "torch_npu.npu_dynamic_quant"
KV_ROW_BYTES_PER_ELEMENT = int(dtype_bytes("bf16"))
SLOT_INDEX_BYTES = 8
ROPE_TRIG_TABLES = 2
MXFP8_QUANT_GROUP_SIZE = 32
MXFP8_SCALE_BYTES = 1
INT8_SCALE_BYTES = int(dtype_bytes("fp32"))
QUANT_VEC_OPS_PER_ELEM = 4
QUANT_VEC_OPS_PER_GROUP = 3
PROLOG_CUBE_SHARE = 1.0
PROLOG_VECTOR_SHARE = 1.0
PROLOG_MEMORY_SHARE = 1.0
UNIT_CONTEXT_PARALLEL = 1
TWO_CACHE_COMPONENTS = 2
ASCEND_PROLOG_ARCHS = frozenset({"", "A2", "A3", "A5"})
FLASHMLA_STAGED_ARCHS = frozenset({"A5"})
FLASHMLA_SUPPORTED_Q_HEADS = frozenset({8, 12, 64, 96})
FLASHMLA_KV_LORA_RANK = 512
FLASHMLA_QK_ROPE_HEAD_DIM = 64
FLASHMLA_SUPPORTED_KV_DTYPES = frozenset({"bf16", "bfloat16", "fp16", "float16"})
CONTIGUOUS_DRAFT_LAYOUTS = frozenset({
    "split_contiguous", "contiguous_components", "contiguous_tuple",
    "component", "two_contiguous", "contiguous",
})
COMPONENT_MAJOR_DRAFT_LAYOUTS = frozenset({
    "component_major", "strided_component_major",
})
STAGED_INTERLEAVED_LAYOUTS = frozenset({
    "token_fused", "token_interleaved", "combined",
})
STRIDED_DRAFT_LAYOUTS = frozenset({
    "strided", "token_fused", "token_interleaved", "combined", "paired_sfa",
    "component_major", "strided_component_major",
})


def _resolve_arch(machine=None, runtime=None) -> str:
    direct = str(getattr(machine, "arch_family", "") or "").strip().upper()
    if direct:
        return direct
    dies = getattr(machine, "dies", ()) or ()
    if dies:
        die_arch = str(getattr(dies[0], "arch_family", "") or "").strip().upper()
        if die_arch:
            return die_arch
    rt_meta = dict(getattr(runtime, "metadata", ()) or ())
    return str(rt_meta.get("arch_family") or "").strip().upper()


def draft_cache_requires_prolog_v3(model, runtime=None, machine=None) -> bool:
    """Reference ``_cache_requires_prolog_v3_writer`` on the draft MLA cache."""
    model_meta = dict(getattr(model, "metadata", ()) or ())
    rt_meta = dict(getattr(runtime, "metadata", ()) or ())
    arch = _resolve_arch(machine=machine, runtime=runtime)
    if arch not in ASCEND_PROLOG_ARCHS:
        return False
    layout = str(rt_meta.get("draft_cache_layout")
                 or model_meta.get("draft_cache_layout") or "").strip().lower()
    if (layout in CONTIGUOUS_DRAFT_LAYOUTS
            or bool(rt_meta.get("draft_cache_contiguous")
                    or model_meta.get("draft_cache_contiguous"))):
        return False
    if layout in STRIDED_DRAFT_LAYOUTS:
        return True
    for key in ("draft_cache_representations", "cache_representations"):
        reps = rt_meta.get(key) if key in rt_meta else model_meta.get(key)
        if not isinstance(reps, dict):
            continue
        if (({"mla_nope", "mla_rope"} <= set(reps)
                or {"nope", "rope"} <= set(reps)
                or (key == "draft_cache_representations"
                    and len(reps) == TWO_CACHE_COMPONENTS
                    and "mla_kv" not in reps))
                and all(not isinstance(v, dict) or v.get("contiguous", True)
                        for v in reps.values())):
            return False
        if key == "draft_cache_representations" and isinstance(reps.get("mla_kv"), dict):
            if bool(reps["mla_kv"].get("contiguous_components", False)):
                return False
    allocator = str(model_meta.get("declared_cache_allocator")
                    or model_meta.get("cache_allocator") or "").strip().lower()
    return allocator == "paired_sfa"


def _supports_flashmla_staging(model, runtime=None, draft=None, parallel=None) -> bool:
    """Reference ``_can_use_flashmla`` (``mla_v1.py:1098-1107``) eligibility."""
    eff_parallel = parallel if parallel is not None else getattr(runtime, "parallel", None)
    pcp = max(UNIT_CONTEXT_PARALLEL,
              int(getattr(eff_parallel, "pcp", UNIT_CONTEXT_PARALLEL) or UNIT_CONTEXT_PARALLEL))
    if pcp > UNIT_CONTEXT_PARALLEL:
        return False
    spec = draft
    if spec is None:
        aux = getattr(model, "auxiliary", None)
        spec = getattr(aux, "dspark", None) or getattr(aux, "mtp", None) or model
    kv_lora = int(getattr(spec, "kv_lora_rank", 0) or 0)
    rope_dim = int(getattr(spec, "qk_rope_head_dim", 0) or 0)
    if kv_lora and kv_lora != FLASHMLA_KV_LORA_RANK:
        return False
    if rope_dim and rope_dim != FLASHMLA_QK_ROPE_HEAD_DIM:
        return False
    kv_dtype = str(getattr(spec, "kv_dtype", "") or "").strip().lower()
    if kv_dtype and kv_dtype not in FLASHMLA_SUPPORTED_KV_DTYPES:
        return False
    raw_heads = int(getattr(spec, "num_heads", 0) or 0)
    if raw_heads > 0:
        tp = max(UNIT_CONTEXT_PARALLEL,
                 int(getattr(eff_parallel, "tp", UNIT_CONTEXT_PARALLEL) or UNIT_CONTEXT_PARALLEL))
        dcp = max(UNIT_CONTEXT_PARALLEL,
                  int(getattr(eff_parallel, "cp", UNIT_CONTEXT_PARALLEL) or UNIT_CONTEXT_PARALLEL))
        local_heads = max(1, int(ceil(raw_heads / tp)))
        if local_heads not in FLASHMLA_SUPPORTED_Q_HEADS:
            return False
        if local_heads * dcp not in FLASHMLA_SUPPORTED_Q_HEADS:
            return False
    return True


def draft_cache_requires_prolog_v3_scatter(
    model, runtime=None, machine=None, *, draft=None, parallel=None,
) -> bool:
    """Whether ``_prolog_v3_writer_workspace`` must stage and scatter into history.

    Upstream ``mla_v1.py:1098-1107,1395-1405,2246-2289,2403-2415`` writes
    directly into component-major A2/A3 (and non-FlashMLA A5) caches with zero
    standalone scatter, and stages via a dense workspace plus
    ``reshape_and_cache`` only for token-interleaved A5 FlashMLA views where
    ``stride(1) != shape(2) * shape(3)``.
    """
    if not draft_cache_requires_prolog_v3(model, runtime=runtime, machine=machine):
        return False
    model_meta = dict(getattr(model, "metadata", ()) or ())
    rt_meta = dict(getattr(runtime, "metadata", ()) or ())
    layout = str(rt_meta.get("draft_cache_layout")
                 or model_meta.get("draft_cache_layout") or "").strip().lower()
    if layout in STAGED_INTERLEAVED_LAYOUTS:
        return True
    if layout in COMPONENT_MAJOR_DRAFT_LAYOUTS:
        return False
    for key in ("draft_cache_representations", "cache_representations"):
        reps = rt_meta.get(key) if key in rt_meta else model_meta.get(key)
        if isinstance(reps, dict) and isinstance(reps.get("mla_kv"), dict):
            mla_kv = reps["mla_kv"]
            for flag in ("token_interleaved", "token_fused", "interleaved"):
                if flag in mla_kv:
                    return bool(mla_kv[flag])
    arch = _resolve_arch(machine=machine, runtime=runtime)
    allocator = str(model_meta.get("declared_cache_allocator")
                    or model_meta.get("cache_allocator") or "").strip().lower()
    if not (arch in FLASHMLA_STAGED_ARCHS and (allocator == "paired_sfa" or layout == "paired_sfa")):
        return False
    return _supports_flashmla_staging(model, runtime=runtime, draft=draft, parallel=parallel)


SOURCE = {
    "repository": "vllm-ascend",
    "base_revision": "nightly-ci-main-a5-openeuler-cann9.2.1-20261111",
    "path": "vllm_ascend/models/kimi_k3/kimi_k3_dspark.py",
    "symbol": "exec_context_kv_prolog_v3",
    "lines": "211-240",
    "companion": "vllm_ascend/attention/mla_v1.py:2295-2356,2556-2594",
    "scope": "independent_source_semantics; not_capture_binary_identity",
}


def recipes():
    from mindie_sim.prediction.mechanism import PlannedRecipe
    return {PROLOG_RECIPE: PlannedRecipe(build_prolog),
            SCATTER_RECIPE: PlannedRecipe(build_scatter)}


def _prolog_work(op, geometry=None):
    matrix = (
        {dtype: float(amount) for dtype, amount in op.matrix_flops if amount}
        if op.matrix_flops
        else ({op.dtype: op.flops} if op.flops else {})
    )
    resources = {
        "hbm": op.hbm_bytes, "mte2": op.mte2_bytes,
        "mte3": op.mte3_bytes, "vector": op.vector_ops,
    }
    fixpipe_bytes = float(dict(getattr(op, "metadata", ()) or ()).get("query_fixpipe_bytes", 0.0) or 0.0)
    if fixpipe_bytes > 0.0:
        units = getattr(geometry, "resource_units", None)
        if units is None or "fixpipe" in units:
            resources["fixpipe"] = fixpipe_bytes
    return Work(matrix, resources)


def _prolog_program(context):
    op = context.op
    meta = dict(getattr(op, "metadata", ()) or ())
    hbm_total = float(getattr(op, "hbm_bytes", 0.0) or 0.0)
    if hbm_total > 0.0:
        read_hbm = min(1.0, max(0.0, float(getattr(op, "mte2_bytes", 0.0) or 0.0) / hbm_total))
        mte3_hbm = min(1.0, max(0.0, float(getattr(op, "mte3_bytes", 0.0) or 0.0) / hbm_total))
        fixpipe_hbm = max(0.0, 1.0 - read_hbm - mte3_hbm)
    else:
        read_hbm, fixpipe_hbm, mte3_hbm = 1.0, 0.0, 0.0
    qa_kva_share = min(1.0, max(0.0, float(meta.get("qa_kva_cube_share", PROLOG_CUBE_SHARE))))
    qb_uk_share = max(0.0, PROLOG_CUBE_SHARE - qa_kva_share)
    pre_qb_vec_share = min(1.0, max(0.0, float(meta.get("pre_qb_vector_share", PROLOG_VECTOR_SHARE))))
    q_rope_vec_share = max(0.0, PROLOG_VECTOR_SHARE - pre_qb_vec_share)
    has_fixpipe = any(
        role == "fixpipe" or str(name).startswith("fixpipe")
        for role, name in dict(context.resource_names).items()
    ) and any(str(k).startswith("fixpipe") and float(v) > 0.0 for k, v in dict(context.work).items())
    phases = [
        Phase("context_load", {"hbm": read_hbm, "mte2": PROLOG_MEMORY_SHARE}),
        Phase("qkv_a", {"cube": qa_kva_share}, after=("context_load",)),
        Phase("cq_kv_norm_rope", {"vector": pre_qb_vec_share}, after=("qkv_a",)),
    ]
    last_phase = "cq_kv_norm_rope"
    if qb_uk_share > 0.0 or fixpipe_hbm > 0.0 or has_fixpipe:
        qb_weights = {}
        if qb_uk_share > 0.0:
            qb_weights["cube"] = qb_uk_share
        if fixpipe_hbm > 0.0:
            qb_weights["hbm"] = fixpipe_hbm
        if has_fixpipe:
            qb_weights["fixpipe"] = PROLOG_MEMORY_SHARE
        phases.append(Phase("qb_wuk", qb_weights, after=(last_phase,)))
        last_phase = "qb_wuk"
    if q_rope_vec_share > 0.0:
        phases.append(Phase("q_rope", {"vector": q_rope_vec_share}, after=(last_phase,)))
        last_phase = "q_rope"
    phases.append(Phase("publish", {"hbm": mte3_hbm, "mte3": PROLOG_MEMORY_SHARE}, after=(last_phase,)))
    return tiled_phased_program(context, tuple(phases))


def _scatter_program(context):
    read_share = io_weight(context, "hbm_read")
    write_share = io_weight(context, "hbm_write")
    return tiled_phased_program(context, (
        Phase("gather_rank_rows", {"hbm": read_share, "mte2": PROLOG_MEMORY_SHARE}),
        Phase("paged_scatter", {"hbm": write_share, "mte3": PROLOG_MEMORY_SHARE,
                                "vector": PROLOG_VECTOR_SHARE},
              after=("gather_rank_rows",)),
    ))


def build_prolog(op, geometry):
    from mindie_sim.prediction.mechanism import OperatorPlan
    return OperatorPlan(_prolog_work(op, geometry), _prolog_program,
                        metadata={"physical_class": "DraftContextPrologV3",
                                  "stage_basis": "declared_phases_partition_member_work"})


def build_scatter(op, geometry):
    from mindie_sim.prediction.mechanism import OperatorPlan
    del geometry
    return OperatorPlan(_prolog_work(op), _scatter_program,
                        metadata={"physical_class": "DraftContextKvScatter",
                                  "stage_basis": "declared_phases_partition_member_work"})


def owned_rows(rows, parallel):
    """Rank-owned DCP row shard: ``ceil(rows / max(1, cp))``."""
    cp = max(UNIT_CONTEXT_PARALLEL,
             int(getattr(parallel, "cp", UNIT_CONTEXT_PARALLEL) or UNIT_CONTEXT_PARALLEL))
    return int(ceil(int(rows) / cp))


def owned_scatter_rows(rows, rows_by_view, views, contexts, parallel, block):
    """Worst-rank rows one draft layer stores under block ownership.

    A position block is owned atomically by one context rank, so the writer
    charges every row in that block to its owner.  ``ceil(rows / cp)`` would
    split one block across ranks and undercount the owner's write.
    """
    if rows_by_view and views and contexts:
        from mindie_sim.prediction.operators.mla.generic_latent_attention import owned_store_rows
        return int(owned_store_rows(rows_by_view, views, parallel, block,
                                    contexts=contexts))
    return owned_rows(rows, parallel)


def _masked_rows_by_view(rows_by_view, valid_rows: int) -> tuple[int, ...]:
    remaining = max(0, int(valid_rows))
    masked = []
    for count in rows_by_view:
        take = min(remaining, max(0, int(count)))
        masked.append(take)
        remaining -= take
    return tuple(masked)


def _owned_masked_rows(valid_flags, rows_by_view, views, contexts, parallel, block) -> int:
    """Intersect per-row validity flags with DCP block ownership across views."""
    from mindie_sim.catalog.attention_common import active_dcp_degree
    degrees = [active_dcp_degree(parallel, view.phase) for view in views]
    max_degree = max(degrees, default=UNIT_CONTEXT_PARALLEL)
    interleave = max(UNIT_CONTEXT_PARALLEL, int(block))
    per_rank = [0] * max_degree
    offset = 0
    for start, count, degree in zip(contexts, rows_by_view, degrees):
        span = max(0, int(count))
        view_flags = valid_flags[offset:offset + span]
        offset += span
        if degree <= UNIT_CONTEXT_PARALLEL:
            valid_in_view = sum(1 for ok in view_flags if ok)
            for rank in range(max_degree):
                per_rank[rank] += valid_in_view
            continue
        base_pos = int(start)
        for idx, ok in enumerate(view_flags):
            if ok:
                owner = ((base_pos + idx) // interleave) % degree
                per_rank[owner] += 1
    return max(per_rank, default=0)


def _layer_owned_rows(layer_valid, *, rows, rank_rows, rows_by_view, views, contexts, parallel, block) -> int:
    if layer_valid is None:
        return rank_rows
    if isinstance(layer_valid, Sequence) and not isinstance(layer_valid, (str, bytes)):
        flags = tuple(bool(x) if isinstance(x, bool) else int(x) >= 0 for x in layer_valid[:int(rows)])
        if not any(flags):
            return 0
        if len(flags) >= int(rows) and all(flags):
            return rank_rows
        if rows_by_view and views and contexts:
            return min(rank_rows, _owned_masked_rows(flags, rows_by_view, views, contexts, parallel, block))
        return min(rank_rows, owned_rows(sum(1 for ok in flags if ok), parallel))
    v_rows = max(0, int(layer_valid))
    if v_rows <= 0:
        return 0
    if v_rows >= int(rows):
        return rank_rows
    if rows_by_view and views and contexts:
        masked = _masked_rows_by_view(rows_by_view, v_rows)
        return min(rank_rows, owned_scatter_rows(v_rows, masked, views, contexts, parallel, block))
    return min(rank_rows, owned_rows(v_rows, parallel))


def _resolve_prolog_heads_and_dims(draft, parallel) -> tuple[int, int]:
    tp = max(UNIT_CONTEXT_PARALLEL,
             int(getattr(parallel, "tp", UNIT_CONTEXT_PARALLEL) or UNIT_CONTEXT_PARALLEL))
    raw_heads = int(getattr(draft, "num_heads", 0) or 0)
    local_heads = max(1, int(ceil(raw_heads / tp))) if raw_heads > 0 else 0
    raw_nope = int(getattr(draft, "qk_nope_head_dim", 0) or 0)
    if raw_nope > 0:
        return local_heads, raw_nope
    head_dim = int(getattr(draft, "head_dim", 0) or 0)
    rope_dim = int(getattr(draft, "qk_rope_head_dim", 0) or 0)
    fallback_nope = max(0, head_dim - rope_dim) if local_heads > 0 and head_dim > 0 else 0
    return local_heads, fallback_nope


def _prepared_prolog_heads(local_heads: int, prolog_dtype: str) -> int:
    """Reference ``mla_v1.py:1082,2098`` BF16 next-power-of-2 head padding."""
    if local_heads > 0 and prolog_dtype == "bf16":
        return 1 << (local_heads - 1).bit_length()
    return local_heads


def _resolve_prolog_dtypes(model, draft) -> tuple[str, bool]:
    qa_fmt = model.quant.format_for("mla.q_a")
    kva_fmt = model.quant.format_for("mla.kv_a")
    qb_fmt = model.quant.format_for("mla.q_b")
    mla_format = qa_fmt if qa_fmt == kva_fmt == qb_fmt else model.quant.format_for("mla")
    is_mxfp8 = mla_format == "fp8_e4m3"
    qa_dtype = compute_dtype_for(model.quant, "mla.q_a")[0]
    kva_dtype = compute_dtype_for(model.quant, "mla.kv_a")[0]
    qb_dtype = compute_dtype_for(model.quant, "mla.q_b")[0]
    prolog_dtype = (
        qa_dtype if qa_dtype == kva_dtype == qb_dtype
        else compute_dtype_for(model.quant, "mla")[0]
    ) or draft.activation_dtype
    return prolog_dtype, is_mxfp8


def _build_prolog_quant_op(
    *, name, norm_id, draft, rows, prolog_dtype, activation_bytes, layer_common,
) -> tuple[OpSpec | None, int]:
    """Emit dynamic activation quantizer before quantized PrologV3 (``mla_v1.py:2335-2351``)."""
    hidden = int(draft.hidden_size)
    if rows <= 0 or prolog_dtype not in {"fp8_e4m3", "int8"}:
        return None, rows * hidden * activation_bytes
    if prolog_dtype == "fp8_e4m3":
        group_size = MXFP8_QUANT_GROUP_SIZE
        scale_fmt = "fp8_e8m0"
        scale_per_group = MXFP8_SCALE_BYTES
        native_op = DYNAMIC_MX_QUANT_OP
    else:
        group_size = max(1, hidden)
        scale_fmt = "fp32"
        scale_per_group = INT8_SCALE_BYTES
        native_op = DYNAMIC_INT8_QUANT_OP
    groups_per_row = int(ceil(hidden / group_size))
    groups = rows * groups_per_row
    scale_bytes = groups * scale_per_group
    quant_out_bytes = rows * hidden * int(dtype_bytes(prolog_dtype)) + scale_bytes
    read_bytes = rows * hidden * activation_bytes
    vec_ops = QUANT_VEC_OPS_PER_ELEM * rows * hidden - groups + QUANT_VEC_OPS_PER_GROUP * groups
    quant_op = vector_op(
        op_id=name + ".mla_prolog_v3.quantize", deps=(norm_id,),
        elements=rows * hidden, ops_per_element=0, vector_ops_override=vec_ops,
        dtype=draft.activation_dtype, family="quantize", recipe=DYNAMIC_QUANT_RECIPE,
        shape=(rows, hidden), read_bytes_override=read_bytes,
        write_bytes_override=quant_out_bytes, returns=("activation_scales",),
        metadata=(
            ("output_format", prolog_dtype),
            ("primary_output", "payload"),
            ("activation_group_size", group_size),
            ("activation_scale_shape", (rows, groups_per_row)),
            ("activation_scale_format", scale_fmt),
            ("activation_scale_bytes", scale_bytes),
            ("native_op", native_op),
        ),
        **layer_common,
    )
    return quant_op, quant_out_bytes


def _build_prolog_v3_op(
    *, name, input_id, token_input_bytes, prolog_dtype, is_mxfp8, model, draft,
    rows, query_rows, parallel, kv_width, row_bytes, layer_rank_rows,
    staging_scatter, layer_common,
) -> OpSpec:
    do_rope = bool(draft.qk_rope_head_dim > 0)
    # CANN op-plugin npu_mla_prolog_v3 dispatches aclnnMlaPrologV3WeightNz
    # for BF16/INT8/MXFP8 doRope=True; V4WeightNz is the 9-input doRope=False ABI.
    native_kind = PROLOG_V4_NATIVE_KIND if (is_mxfp8 and not do_rope) else PROLOG_NATIVE_KIND
    qkv_width = draft.q_lora_rank + kv_width
    local_heads, qk_nope_head_dim = _resolve_prolog_heads_and_dims(draft, parallel)
    prepared_heads = _prepared_prolog_heads(local_heads, prolog_dtype)
    qk_head_dim = qk_nope_head_dim + draft.qk_rope_head_dim
    qa_kva_flops = TWO_CACHE_COMPONENTS * rows * qkv_width * draft.hidden_size
    qb_flops = TWO_CACHE_COMPONENTS * rows * prepared_heads * draft.q_lora_rank * qk_head_dim
    uk_flops = TWO_CACHE_COMPONENTS * rows * prepared_heads * qk_nope_head_dim * draft.kv_lora_rank
    total_flops = qa_kva_flops + qb_flops + uk_flops
    if prolog_dtype == "bf16":
        matrix_flops = (("bf16", total_flops),) if total_flops else ()
    else:
        matrix_flops = tuple(
            (dt, fl) for dt, fl in ((prolog_dtype, qa_kva_flops + qb_flops), ("bf16", uk_flops))
            if fl
        )
    if rows > 0:
        weight_bytes = (
            model.quant.matrix_bytes(draft.q_lora_rank, draft.hidden_size, "mla.q_a")
            + model.quant.matrix_bytes(kv_width, draft.hidden_size, "mla.kv_a")
            + (
                model.quant.matrix_bytes(prepared_heads * qk_head_dim, draft.q_lora_rank, "mla.q_b")
                if prepared_heads and qk_head_dim else 0
            )
            + prepared_heads * qk_nope_head_dim * draft.kv_lora_rank * KV_ROW_BYTES_PER_ELEMENT
        )
        gamma_bytes = (draft.q_lora_rank + draft.kv_lora_rank) * KV_ROW_BYTES_PER_ELEMENT
        rope_table_bytes = (
            ROPE_TRIG_TABLES * rows * draft.qk_rope_head_dim * KV_ROW_BYTES_PER_ELEMENT
            if do_rope else 0
        )
        slot_index_bytes = rows * SLOT_INDEX_BYTES
    else:
        weight_bytes = gamma_bytes = rope_table_bytes = slot_index_bytes = 0
    kv_write_rows = rows if staging_scatter else layer_rank_rows
    kv_write_bytes = kv_write_rows * row_bytes
    # MM4 w_uk publishes BF16 query_out via Cube fixpipe (not AIV mte3).
    q_nope_out_bytes = rows * prepared_heads * draft.kv_lora_rank * KV_ROW_BYTES_PER_ELEMENT
    q_rope_out_bytes = (
        rows * prepared_heads * (draft.qk_rope_head_dim if do_rope else 0)
        * KV_ROW_BYTES_PER_ELEMENT
    )
    mte2_bytes = token_input_bytes + weight_bytes + gamma_bytes + rope_table_bytes + slot_index_bytes
    mte3_bytes = kv_write_bytes + q_rope_out_bytes
    hbm_bytes = mte2_bytes + q_nope_out_bytes + mte3_bytes
    pre_qb_vec_ops = rows * (
        NORM_OPS * (draft.q_lora_rank + draft.kv_lora_rank)
        + (ROPE_OPS * draft.qk_rope_head_dim if do_rope else 0)
    )
    q_rope_vec_ops = rows * (ROPE_OPS * prepared_heads * draft.qk_rope_head_dim if do_rope else 0)
    vector_ops = pre_qb_vec_ops + q_rope_vec_ops
    qa_kva_cube_share = (qa_kva_flops / total_flops) if total_flops > 0 else PROLOG_CUBE_SHARE
    pre_qb_vec_share = (pre_qb_vec_ops / vector_ops) if vector_ops > 0 else PROLOG_VECTOR_SHARE
    if prepared_heads > 0:
        fused_members = (
            ("qkv_a", "q_b", "w_uk", "kv_norm", "kv_rope", "q_rope")
            if do_rope else ("qkv_a", "q_b", "w_uk", "kv_norm")
        )
    else:
        fused_members = ("qkv_a", "kv_norm", "kv_rope") if do_rope else ("qkv_a", "kv_norm")
    cp = int(getattr(parallel, "cp", UNIT_CONTEXT_PARALLEL) or UNIT_CONTEXT_PARALLEL)
    return OpSpec(
        op_id=name + ".mla_prolog_v3", deps=(input_id,),
        recipe=PROLOG_RECIPE, family="mla",
        semantic_shape=(rows, qkv_width), local_shape=(rows, qkv_width),
        padded_shape=(rows, qkv_width), launch_shape=(rows, qkv_width),
        flops=total_flops, matrix_flops=matrix_flops, vector_ops=vector_ops,
        hbm_bytes=hbm_bytes, mte2_bytes=mte2_bytes, mte3_bytes=mte3_bytes,
        dtype=prolog_dtype,
        metadata=(("native_kind", native_kind),
                  ("native_source", SOURCE),
                  ("do_rope", do_rope),
                  ("fused_members", fused_members),
                  ("local_heads", local_heads),
                  ("prepared_heads", prepared_heads),
                  ("query_fixpipe_bytes", q_nope_out_bytes),
                  ("qa_kva_cube_share", qa_kva_cube_share),
                  ("pre_qb_vector_share", pre_qb_vec_share),
                  ("query_outputs_discarded", True),
                  ("direct_cache_write", not staging_scatter),
                  ("kv_write_bytes", kv_write_bytes),
                  ("context_rows", rows), ("query_rows", query_rows),
                  ("data_parallel_context", cp),
                  ("dcp_degree", cp),
                  ("dcp_row_shard", layer_rank_rows),
                  ("rank_owned_rows", layer_rank_rows),
                  ("kv_row_bytes", row_bytes),
                  ("hidden_read_once", True),
                  ("timing_boundary", "expanded_semantic_operator")),
        **layer_common,
    )


def _emit_context_query_ready(ops, *, deps, rows, query_rows, prefix, draft, common):
    """Emit query embedding only when query_rows > 0, then return the ready barrier op_id."""
    if int(query_rows) > 0:
        embedding = vector_op(
            op_id=prefix + "query_embedding", deps=(), elements=query_rows * draft.hidden_size,
            ops_per_element=0, dtype="bf16", family="embedding",
            recipe="draft_query_embedding", **common)
        ready = barrier_op(
            op_id=prefix + "context_query_ready", deps=(*deps, embedding.op_id),
            recipe="draft_context_query_ready", shape=(rows, query_rows), **common)
        ops.extend((embedding, ready))
        return ready.op_id
    ready = barrier_op(
        op_id=prefix + "context_query_ready", deps=tuple(deps),
        recipe="draft_context_query_ready", shape=(rows, 0), **common)
    ops.append(ready)
    return ready.op_id


def prepare(ops, *, projection, model, draft, rows, query_rows,
            prefix, stage, requests, parallel=None, rows_by_view=None,
            views=None, contexts=None, block=UNIT_CONTEXT_PARALLEL,
            use_prolog_v3=True, layer_slot_valid_rows=None,
            requires_staging_scatter: bool | None = None,
            runtime=None, machine=None):
    proj_mode = str(getattr(draft, "context_projection_mode", "") or "none").strip()
    if proj_mode == "shared_compressed_kv":
        common = dict(stage=stage, layer=-1, requests=requests, phase="draft")
        return _emit_context_query_ready(
            ops, deps=(projection.op_id,), rows=rows, query_rows=query_rows,
            prefix=prefix, draft=draft, common=common)
    if getattr(draft, "attention_kind", "mla") == "gqa" or proj_mode in {"fused_kv_bank", "per_layer_qkv"}:
        from .draft_context_gqa import prepare_gqa_context
        return prepare_gqa_context(
            ops,
            projection=projection,
            model=model,
            draft=draft,
            rows=rows,
            query_rows=query_rows,
            prefix=prefix,
            stage=stage,
            requests=requests,
            parallel=parallel,
            owned_scatter_rows_fn=owned_scatter_rows,
            rows_by_view=rows_by_view,
            views=views,
            contexts=contexts,
            block=block,
            layer_slot_valid_rows=layer_slot_valid_rows,
        )
    common = dict(stage=stage, layer=-1, requests=requests, phase="draft")
    norm = vector_op(
        op_id=prefix + "context_norm", deps=(projection.op_id,),
        elements=rows * draft.hidden_size, ops_per_element=NORM_OPS,
        dtype="bf16", family="rmsnorm", recipe="draft_context_norm", **common)
    ops.append(norm)
    kv_width = draft.kv_lora_rank + draft.qk_rope_head_dim
    row_bytes = kv_width * KV_ROW_BYTES_PER_ELEMENT
    rank_rows = owned_scatter_rows(
        rows, rows_by_view, views, contexts, parallel, block)
    activation_bytes = dtype_bytes(draft.activation_dtype)
    staging_scatter = (
        bool(requires_staging_scatter)
        if requires_staging_scatter is not None
        else draft_cache_requires_prolog_v3_scatter(
            model, runtime=runtime, machine=machine, draft=draft, parallel=parallel,
        )
    )
    prolog_dtype, is_mxfp8 = _resolve_prolog_dtypes(model, draft)
    cp = int(getattr(parallel, "cp", UNIT_CONTEXT_PARALLEL) or UNIT_CONTEXT_PARALLEL)
    stores = []
    for layer in range(draft.layers):
        layer_valid = None
        if layer_slot_valid_rows is not None:
            layer_valid = (
                layer_slot_valid_rows[layer]
                if layer < len(layer_slot_valid_rows)
                else None
            )
            if layer_valid is None:
                continue
        layer_rank_rows = _layer_owned_rows(
            layer_valid, rows=rows, rank_rows=rank_rows,
            rows_by_view=rows_by_view, views=views, contexts=contexts,
            parallel=parallel, block=block,
        )
        name = prefix + f"context_kv{layer}"
        layer_common = {**common, "layer": layer}
        qkv_width = (draft.q_lora_rank + draft.kv_lora_rank
                     + draft.qk_rope_head_dim)
        if use_prolog_v3:
            quant_op, token_input_bytes = _build_prolog_quant_op(
                name=name, norm_id=norm.op_id, draft=draft, rows=rows,
                prolog_dtype=prolog_dtype, activation_bytes=activation_bytes,
                layer_common=layer_common,
            )
            if quant_op is not None:
                ops.append(quant_op)
            prolog = _build_prolog_v3_op(
                name=name,
                input_id=quant_op.op_id if quant_op is not None else norm.op_id,
                token_input_bytes=token_input_bytes,
                prolog_dtype=prolog_dtype, is_mxfp8=is_mxfp8,
                model=model, draft=draft,
                rows=rows, query_rows=query_rows, parallel=parallel,
                kv_width=kv_width, row_bytes=row_bytes,
                layer_rank_rows=layer_rank_rows,
                staging_scatter=staging_scatter,
                layer_common=layer_common,
            )
            ops.append(prolog)
            if not staging_scatter:
                stores.append(prolog.op_id)
                continue
            kv_ready = prolog.op_id
        else:
            qkv_proj = gemm_op(
                op_id=name + ".qkv_a", deps=(norm.op_id,),
                m=rows, n=qkv_width, k=draft.hidden_size, quant=model.quant,
                category="mla", family="gemm", recipe="draft_context_qkv_a",
                **layer_common)
            kv_norm = vector_op(
                op_id=name + ".kv_norm", deps=(qkv_proj.op_id,),
                elements=rows * draft.kv_lora_rank, ops_per_element=NORM_OPS,
                dtype=draft.activation_dtype, family="rmsnorm",
                recipe="draft_context_kv_norm", **layer_common)
            ops.extend((qkv_proj, kv_norm))
            kv_ready = kv_norm.op_id
            if draft.qk_rope_head_dim:
                kv_rope = vector_op(
                    op_id=name + ".kv_rope", deps=(kv_norm.op_id,),
                    elements=rows * draft.qk_rope_head_dim,
                    ops_per_element=ROPE_OPS, dtype=draft.activation_dtype,
                    family="rotary", recipe="draft_context_kv_rope",
                    **layer_common)
                ops.append(kv_rope)
                kv_ready = kv_rope.op_id
        stored = cache_store_op(
            op_id=name + ".kv_scatter", deps=(kv_ready,), rows=layer_rank_rows,
            row_bytes=row_bytes, dtype="bf16", kind="draft_context",
            recipe=SCATTER_RECIPE, **layer_common)
        stored = replace(stored, metadata=stored.metadata + (
            ("native_kind", SCATTER_NATIVE_KIND),
            ("cache_epilog", {
                "kind": SCATTER_NATIVE_KIND,
                "context_rows": rows,
                "rank_owned_rows": layer_rank_rows,
                "kv_row_bytes": row_bytes,
                "row_elements": kv_width,
            }),
            ("dcp_row_shard", layer_rank_rows),
            ("dcp_degree", cp)))
        ops.append(stored)
        stores.append(stored.op_id)
    return _emit_context_query_ready(
        ops, deps=tuple(stores) or (norm.op_id,), rows=rows, query_rows=query_rows,
        prefix=prefix, draft=draft, common=common)
