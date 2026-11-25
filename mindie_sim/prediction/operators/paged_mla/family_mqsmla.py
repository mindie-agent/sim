"""Explicit mixed-quant sparse MLA source binding for ordinary decode graphs.

[Role] Replace one batched attention boundary, preserving cache producers.
[Invariants] Packed cache formats are declared by the model and shared with
  capacity accounting. Source selection does not prove a captured binary ABI.
  One predicate decides whether an automatic binding is admissible, so a
  default can never install a geometry the planner rejects.
[Timeouts & Locks] O(active query rows), no file or device access.
[Side-Effect Policy] Returns immutable operators and a shared Engine recipe.
"""
from dataclasses import replace
from typing import Literal

from mindie_sim.catalog.cache_format import CacheFormat, cache_format
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.mechanism import PlannedRecipe
from .mqsmla import (
    COMPRESSED_CACHE_ROW_BYTES,
    DSL_NATIVE_KIND,
    DSL_SOURCE,
    KV_QUANT_BF16,
    KV_QUANT_FP8,
    KV_QUANT_MXFP8,
    KV_QUANT_ROW_BYTES,
    MQSMLA_HEAD_WIDTH,
    MQSMLA_ROPE_WIDTH,
    NATIVE_KIND,
    ORIGINAL_CACHE_ROW_BYTES,
    RECIPE,
    SOURCE,
    SOURCE_REFS as _MQSMLA_SOURCE_REFS,
    SUPPORTED_ADDRESSING,
    SUPPORTED_PHASES,
    is_abi_admitted,
    supported_heads,
    verify_mqsmla_abi,
    build,
)

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCE_REFS: tuple[SourceRef, ...] = _MQSMLA_SOURCE_REFS
#: The CANN 9.2.1 source-symbol spelling retained as an explicit alias.
SOURCE_RECIPE = "mixed_quant_sparse_flash_mla"
FP8_WINDOW_GROUP_SIZE = 32
FP4_COMPRESSED_GROUP_SIZE = 16
SINGLE_KV_HEAD = 1
UNIT_CONTEXT_PARALLEL = 1
MIN_ACTIVE_ROWS = 1
QUERY_ORIGIN = 0
NO_COMPRESSED_ROWS = 0
DEFAULT_PAGE_SIZE = 128
VALID_ADDRESSING = SUPPORTED_ADDRESSING
FORMATS = {
    "window": {"dtype": "fp8_e4m3", "group_size": FP8_WINDOW_GROUP_SIZE,
               "scale_dtype": "bf16"},
    "compressed_kv": {"dtype": "fp4_e2m1", "group_size": FP4_COMPRESSED_GROUP_SIZE,
                      "scale_dtype": "bf16"},
}


def quant_mode_of(model, head_dim):
    """Declared packed-KV mode of the arch35 mixed-quant kernel, or ``None``.

    Packed FP8 rows use seven native scale groups over the 448 NoPE lanes.
    The mixed FP8/FP4 declaration selects the retained DSL layout.
    """
    del head_dim
    return _declared_mode(cache_format(model, "window"),
                          cache_format(model, "compressed_kv"))


def _declared_mode(window, compressed):
    from mindie_sim.catalog.mqsmla_layout import packed_scale_type
    kind = packed_scale_type(window, compressed)
    if kind is not None:
        return KV_QUANT_FP8 if kind == "bf16" else KV_QUANT_MXFP8
    if window == compressed == CacheFormat("bf16"):
        return KV_QUANT_BF16
    return None


def _region_format(raw):
    if not isinstance(raw, dict):
        return None
    allowed = {"dtype", "group_size", "scale_dtype", "suffix_elements",
               "suffix_dtype", "packed_elements"}
    if set(raw) - allowed:
        return None
    return CacheFormat(**raw)


def declared_quant_mode(op):
    """Read one analytic attention op's declared cache representation pair."""
    view = dict(getattr(op, "metadata", ()) or ()).get("attention_tensor_view") or {}
    regions = view.get("cache_representations") or {}
    window, compressed = (_region_format(regions.get("window")),
                          _region_format(regions.get("compressed_kv")))
    if window is None or compressed is None:
        return None
    mode = _declared_mode(window, compressed)
    if mode is None:
        return None
    return mode


def recipes():
    """The official CANN recipe and its source-name alias."""
    return {RECIPE: PlannedRecipe(build), SOURCE_RECIPE: PlannedRecipe(build)}


def query_shape(views, plan, *, topk_present=False):
    """The exact per-row geometry the builder will consume."""
    effective_ratio = (plan.compress_ratio if topk_present or plan.owns_long_kv
                       else NO_COMPRESSED_ROWS)
    return {
        "sequences": tuple((int(v.q_len), int(v.context)) for v in views),
        "window": plan.window,
        "ratio": effective_ratio,
        "topk": plan.index_topk if topk_present else NO_COMPRESSED_ROWS,
    }


def _prefix_fields(views, runtime):
    from .indexed_attention import _prefix_declaration
    return _prefix_declaration(views, runtime)


def unsupported_reason(*, views, plan, parallel, model, padded_tokens,
                       addressing="scalar", metadata_support_fd=False,
                       return_softmax_lse=False, cores=None, topk_present=False,
                       implementation=RECIPE):
    """Return the first failed source-domain predicate, or ``None``.

    ``cores`` is the declared AIC owner count; when supplied, the retained
    SELECTIVE_FD_72 wrapper geometry is rejected through the same predicate the
    builder uses.
    """
    if (metadata_support_fd or addressing not in VALID_ADDRESSING
            or return_softmax_lse):
        return "mqsmla_supported_branch_is_whole_row_address_no_lse"
    tp = max(int(getattr(parallel, "tp", SINGLE_KV_HEAD) or SINGLE_KV_HEAD), SINGLE_KV_HEAD)
    cp = int(getattr(parallel, "cp", UNIT_CONTEXT_PARALLEL) or UNIT_CONTEXT_PARALLEL)
    if cp != UNIT_CONTEXT_PARALLEL:
        return "mqsmla_requires_context_parallel_cp1"
    if not supported_heads(plan.num_heads // tp):
        return "mqsmla_local_heads_outside_source_tp_tiling"
    groups = int(getattr(plan, "o_groups", 0) or 0)
    if groups and groups % tp:
        return "mqsmla_output_groups_not_divisible_by_tp"
    if (plan.num_kv_heads != SINGLE_KV_HEAD or plan.head_dim != MQSMLA_HEAD_WIDTH
            or plan.qk_rope_head_dim != MQSMLA_ROPE_WIDTH):
        return "mqsmla_requires_d512_rope64_single_kv_head"
    if model.activation_dtype != "bf16":
        return "mqsmla_requires_bf16_activation"
    valid_rows = sum(int(v.q_len) for v in views)
    if valid_rows < MIN_ACTIVE_ROWS or padded_tokens < valid_rows:
        return "mqsmla_binding_requires_positive_padded_rows"
    if any(v.phase not in SUPPORTED_PHASES or v.padding
           or (v.query_ranges and v.query_ranges != ((QUERY_ORIGIN, v.q_len),))
           for v in views):
        return "mqsmla_binding_requires_contiguous_unpadded_decode_or_draft_rows"
    if quant_mode_of(model, plan.head_dim) in {KV_QUANT_FP8, KV_QUANT_MXFP8}:
        # A declared packed-quantized representation is itself the contract.  A
        # BF16 cache keeps the plain attention path: the packed native kernel is
        # a quantized-cache binding, never a relabel of a BF16 declaration.
        return None
    for region, declaration in FORMATS.items():
        if cache_format(model, region) != CacheFormat(**declaration):
            return "mqsmla_cache_representation_contract_required:" + region
    return None


def validate(*, views, plan, parallel, model, padded_tokens,
             metadata_support_fd=False, addressing="scalar", return_softmax_lse=False,
             cores=None, topk_present=False, implementation=RECIPE):
    reason = unsupported_reason(
        views=views, plan=plan, parallel=parallel, model=model,
        padded_tokens=padded_tokens, addressing=addressing,
        metadata_support_fd=metadata_support_fd,
        return_softmax_lse=return_softmax_lse, cores=cores,
        topk_present=topk_present, implementation=implementation,
    )
    if reason:
        raise ModelingError(reason)


def _matches_dsl_formats(op) -> bool:
    view = dict(getattr(op, "metadata", ()) or ()).get("attention_tensor_view") or {}
    regions = view.get("cache_representations")
    if not isinstance(regions, dict) or not regions:
        return False
    return all(
        _region_format(regions.get(region)) == CacheFormat(**declaration)
        for region, declaration in FORMATS.items()
    )


def _layout_contract(quant_mode: int | None) -> str:
    if quant_mode is None:
        return "mixed_fp8_fp4_544_320"
    if quant_mode == KV_QUANT_FP8:
        return "aclnn_packed_kv_608"
    if quant_mode == KV_QUANT_MXFP8:
        return "aclnn_packed_kv_584"
    return "aclnn_dense_bf16"


def _abi_options_from_runtime(runtime) -> tuple[str, str | None, str, object | None, tuple]:
    rt_meta = dict(getattr(runtime, "metadata", ()) or ()) if runtime is not None else {}
    soc_arch = str(rt_meta.get("soc_arch") or getattr(runtime, "arch_family", "") or "A5").strip().upper()
    sku_id = str(rt_meta.get("sku_id") or "").strip().lower() or None
    track_spec = rt_meta.get("abi_track_spec")
    pr_track = str(getattr(track_spec, "track", "") or rt_meta.get("abi_pr_track") or "shipping")
    bindings = tuple(rt_meta.get("abi_manifest_bindings") or ())
    return soc_arch, sku_id, pr_track, track_spec, bindings


def lower_attention(ops, outputs, *, selection, views, plan, topk_present, prefix,
                    parallel=None, runtime=None):
    ids = set(outputs)
    members = [op for op in ops if op.op_id in ids]
    tp_degree = max(int(getattr(parallel, "tp", SINGLE_KV_HEAD) or SINGLE_KV_HEAD),
                    SINGLE_KV_HEAD)
    heads = plan.num_heads // tp_degree
    rows = tuple((v.q_len, v.context) for v in views)
    active_q = sum(q for q, _ in rows)
    effective_ratio = (plan.compress_ratio if topk_present or plan.owns_long_kv
                       else NO_COMPRESSED_ROWS)
    implementation = str(getattr(selection, "name", RECIPE))
    addressing = str(dict(getattr(selection, "options", {}) or {}).get("addressing", "scalar"))
    if addressing not in VALID_ADDRESSING:
        raise ModelingError("mqsmla_addressing_mode_unsupported")
    quant_mode = declared_quant_mode(members[0])
    if quant_mode is None and not _matches_dsl_formats(members[0]):
        raise ModelingError("mqsmla_cache_representation_contract_required")
    orig_view = dict(getattr(members[0], "metadata", ()) or ()).get("attention_tensor_view") or {}
    cache_reps = orig_view.get("cache_representations")
    shape = {"sequences": rows, "window": plan.window, "ratio": effective_ratio,
             "topk": plan.index_topk if topk_present else 0, "heads": heads,
             "width": MQSMLA_HEAD_WIDTH, "metadata_support_fd": False,
             "implementation": implementation,
             "addressing": addressing}
    if quant_mode is not None:
        shape["kv_quant_mode"] = quant_mode
        shape["packed_row_bytes"] = KV_QUANT_ROW_BYTES[quant_mode]
        shape.update(_prefix_fields(views, runtime))
    is_dsl = quant_mode is None
    native_kind = DSL_NATIVE_KIND if is_dsl else NATIVE_KIND
    source_rec = DSL_SOURCE if is_dsl else SOURCE
    api_namespace = "third_party_wheel" if is_dsl else "aclnn"
    api_entry = DSL_NATIVE_KIND if is_dsl else "cann_ops_transformer.ops.mixed_quant_sparse_flash_mla"
    kernel_name = "MixedQuantSparseFlashMlaDsl" if is_dsl else "MixedQuantSparseFlashMla"
    soc_arch, sku_id, pr_track, track_spec, bindings = _abi_options_from_runtime(runtime)
    page_size = int(getattr(runtime, "kv_block_size", 0) or DEFAULT_PAGE_SIZE)
    abi_verdict = verify_mqsmla_abi(
        soc_arch=soc_arch,
        sku_id=sku_id,
        quant_mode=quant_mode,
        active_q=active_q,
        heads=heads,
        batch=len(rows),
        compressed_cache=effective_ratio > 0,
        topk_present=bool(topk_present),
        topk=plan.index_topk if topk_present else 0,
        page_size=page_size,
        pr_track=pr_track,
        track_spec=track_spec,
        manifest_bindings=bindings,
    )
    query = (active_q, heads, MQSMLA_HEAD_WIDTH)
    tensor_view = {
        "kind": "mixed_quant_shared_kv", "query_shape": query,
        "output_shape": query,
        "query_dtype": "bf16", "output_dtype": "bf16",
        "original_cache_row_bytes": KV_QUANT_ROW_BYTES[quant_mode]
        if quant_mode is not None else ORIGINAL_CACHE_ROW_BYTES,
        "compressed_cache_row_bytes": KV_QUANT_ROW_BYTES[quant_mode]
        if quant_mode is not None else COMPRESSED_CACHE_ROW_BYTES,
        "kv_quant_mode": quant_mode,
        "compressed_cache": effective_ratio > 0,
    }
    if isinstance(cache_reps, dict) and cache_reps:
        tensor_view["cache_representations"] = dict(cache_reps)
    api_payload = {"namespace": api_namespace, "entry": api_entry}
    if not is_dsl:
        api_payload["aclnn_symbol"] = native_kind
    meta = {**dict(members[0].metadata), "mqsmla_geometry": shape,
            "logical_pairs": sum(dict(op.metadata).get("logical_pairs", 0) for op in members),
            "kv_sweeps": sum(dict(op.metadata).get("kv_sweeps", 0) for op in members),
            "abi_pr_track": pr_track,
            **({"abi_track_spec": track_spec} if track_spec is not None else {}),
            **({"abi_manifest_bindings": bindings} if bindings else {}),
            **({"abi_sku_id": sku_id} if sku_id else {}),
            "implementation_binding": {
                "selector": "mqsmla_attention", "implementation": implementation,
                "capture_binary_identity": "not_established",
                "abi_registry_verified": is_abi_admitted(abi_verdict),
                "abi_verdict_status": abi_verdict.status,
                "abi_entry_id": abi_verdict.matched_entry_id,
                "abi_symbol": abi_verdict.symbol_or_kernel or native_kind,
                "abi_rejection_reason": abi_verdict.rejection_reason,
                "layout_contract": _layout_contract(quant_mode),
                "source": source_rec,
                "coverage": "conditional_source_program",
                "caller_unmodeled": ["separate_aicpu_metadata_generation",
                                      "original_window_index_generation"]},
            "attention_tensor_view": tensor_view,
            "native_kind": native_kind,
            "aclnn_symbol": native_kind,
            "native_call": {
                "api": api_payload,
                "source": source_rec,
                "kernels": [{
                    "name": kernel_name,
                    "role": "mix",
                    "source_path": source_rec["path"],
                    "source_sha256": source_rec.get("source_sha256"),
                    "evidence": "source_declared",
                }],
                "execution_core": "mix",
            }}
    result = replace(members[0], op_id=prefix + ".mqsmla",
                     recipe=RECIPE,
                     deps=tuple(dict.fromkeys(dep for op in members for dep in op.deps)),
                     semantic_shape=query,
                     local_shape=query,
                     padded_shape=query,
                     launch_shape=query,
                     flops=sum(op.flops for op in members),
                     vector_ops=sum(op.vector_ops for op in members),
                     hbm_bytes=sum(op.hbm_bytes for op in members),
                     mte2_bytes=sum(op.mte2_bytes for op in members),
                     mte3_bytes=sum(op.mte3_bytes for op in members),
                     source_boundary="kernel", metadata=tuple(meta.items()))
    ops[:] = [op for op in ops if op.op_id not in ids]
    ops.append(result)
    return [result.op_id]
