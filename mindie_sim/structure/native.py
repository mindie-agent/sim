"""Canonical matrix and native operator structure contracts.

[Role] Project matrix and native operator execution metadata into canonical
  structure contract dictionaries consumed by structure exports and profiler SSOT.
[Invariants] Pure deterministic projections preserving exact contract keys.
[Timeouts & Locks] In-memory metadata inspection; no I/O or locks.
[Side-Effect Policy] Pure read-only projections."""
from __future__ import annotations

import math

def mxfp4_reference_coverage():
    """Packed payload accounting does not supply native MXFP4 execution."""
    return {
        "kind": "partial_program",
        "reason": "mxfp4_native_execution_unmodeled",
        "missing_work": ["native_fp8_fp4_matrix_service_or_explicit_operand_conversion",
                         "microscale_application"],
        "basis": (
            "The declared program uses BF16 matrix service after an assumed fused "
            "dequantization; packed weight bytes and scale traffic are retained, "
            "but conversion and native MX execution have no physical program."),
    }


def matrix_contract(op):
    meta, shape = op["metadata"], tuple(op["local_shape"])
    if len(shape) != 3:
        return None
    precision = (meta["activation_format"], meta["weight_format"], meta["output_format"])
    scales = (tuple(meta.get("activation_scale_shape", ())),
              meta.get("activation_scale_format", ""))
    wrapper = meta.get("matrix_mapping_domain") == "single_group_dense_wrapper"
    grouped = (op["family"] == "grouped_gemm" or wrapper
               or meta.get("matrix_mapping_domain") in {"aggregate_experts", "equal_expert_groups"})
    if grouped:
        bank = meta.get("matrix_weight_bank_groups")
        if not isinstance(bank, int) or bank < 1 or meta.get("bias"):
            return None
        if wrapper:
            declaration = meta.get("matrix_wrapper", {})
            if (bank != 1 or declaration.get("input_rank") != 2
                    or declaration.get("graph_mode") != "eager"
                    or tuple(declaration.get("group_list_values", ())) != (shape[0],)):
                return None
        if "expected_rows" in meta and meta["expected_rows"] != shape[0]:
            return None
        if "mc2_unfused_storage" in meta:
            storage = meta["mc2_unfused_storage"]
            capacity = storage["capacity_rows"]
            if (storage["valid_rows"] != shape[0] or capacity < shape[0]
                    or tuple(meta.get("input_capacity_shape", ())) != (capacity, shape[2])
                    or tuple(meta.get("output_capacity_shape", ())) != (capacity, shape[1])
                    or tuple(meta.get("activation_scale_capacity_shape", ())) != (capacity,)):
                return None
            shape, scales = (capacity, *shape[1:]), ((capacity,), scales[1])
        return "grouped_matrix", (bank, *shape), *precision, *scales
    if meta.get("matrix_groups", 1) != 1:
        # The generic shape stores aggregate M; it does not encode batch axes.
        # Native TransposeBatchMatMul has its own explicit typed projection.
        return None
    if meta.get("bias"):
        if not meta.get("bias_shape") or not meta.get("bias_dtype") or scales[0]:
            return None
        return ("matrix_bias", shape, *precision,
                tuple(meta["bias_shape"]), meta["bias_dtype"])
    return "matrix", shape, *precision, *scales


def _has_w4a8_down_epilogue(op):
    meta = op["metadata"]
    phases = ("pre", "matrix", "post_dequant", "output")
    return (
        op["recipe"] == "ascend_w4a8_msd_grouped_down"
        and op.get("source_boundary") == "kernel"
        and meta.get("compute_format") == "int4"
        and bool(meta.get("w4a8_msd_down"))
        and tuple(meta.get("w4a8_msd_down_phases", ())) == phases
        and all(phase in meta.get("w4a8_msd_down_ledger", {}) for phase in phases)
    )


def matrix_comparison_scope(op):
    meta = op["metadata"]
    grouped = op["family"] == "grouped_gemm" or meta.get("matrix_mapping_domain")
    scope = {
        "kind": "grouped_matrix_active_geometry" if grouped else "matrix_active_geometry",
        "compared": ["logical_M_N_K", "activation_weight_output_precision",
                     "activation_scale_shape_dtype", *(["expert_bank_count"] if grouped else [])],
        "not_compared": ["weight_scale_storage_representation", "scale_values",
                         "native_dispatch", "tiling",
                         *(["per_expert_row_histogram", "active_weight_reads", "group_list_values"]
                           if grouped else [])],
        "basis": "independently_declared_tensor_geometry; no_occurrence_or_workload_realization_proof",
    }
    if grouped:
        scope["routing_basis"] = meta.get("routing_basis", "undeclared")
        scope["work_assumption"] = (
            "Total rows constrain GEMM FLOPs, but the expert histogram and active weight "
            "reads remain those declared by the case.")
    if "mc2_unfused_storage" in meta:
        scope["kind"] = "grouped_matrix_allocated_tensor_geometry"
        scope["compared"][0] = "allocated_M_and_matrix_N_K"
        scope["not_compared"].append("valid_grouped_rows_not_proven_by_tensor_capacity")
        scope["work_assumption"] = (
            "MC2 determines allocated tensor rows. Declared integer group-list values "
            "alone determine matrix FLOPs and active weight reads; allocation is not valid work.")
    if meta.get("matrix_mapping_domain") == "single_group_dense_wrapper":
        declaration = meta["matrix_wrapper"]
        scope["kind"] = "source_selected_single_bank_matrix_geometry"
        scope["routing_basis"] = "source_wrapper_all_rows_in_one_group"
        scope["declared_group_list"] = list(declaration["group_list_values"])
        scope["work_assumption"] = (
            "One independently declared dense projection supplies every row to a "
            "single weight bank. The existing matrix program prices that projection; "
            "the wrapper declaration does not identify the captured occurrence.")
        scope["source_wrapper"] = declaration
    if meta.get("weight_format") in {"int4", "w4a8"} and meta.get("activation_format") == "int8":
        epilogue = ["packed_int4_output_rescale", "quantization_offset_correction"]
        if _has_w4a8_down_epilogue(op):
            scope["implemented_quantized_epilogue"] = {
                "program": op["recipe"], "work": epilogue,
                "basis": "source_pre_matrix_post_dequant_output_ledger",
            }
            scope["source_program_limitations"] = list(meta.get("w4a8_msd_down_limitations", ()))
        else:
            # Packed bytes alone do not implement MSD assist correction and
            # output rescaling. Only the selected complete source program does.
            scope["prediction_coverage"] = {
                "kind": "partial_program", "reason": "quantized_epilogue_unmodeled",
                "missing_work": epilogue,
                "basis": "declared_int4_int8_program_has_no_supported_correction_epilogue",
            }
    if meta.get("activation_scale_format") == "fp8_e8m0":
        scope["mx_execution_assumption"] = (
            "Declared matrix compute-format work and operand-scale traffic; MX scale "
            "application and native instruction scheduling are not separately resolved. "
            "This does not infer an additional output epilogue from tensor descriptors.")
    if (meta.get("weight_format") in {"mxfp4", "fp4", "fp4_e2m1"}
            and meta.get("dequant_before_compute")
            and meta.get("compute_format") in {"bf16", "fp8_e4m3", "fp8_e5m2"}):
        scope["prediction_coverage"] = mxfp4_reference_coverage()
    return scope


#: CANN 9.2.1 official mixed-quant sparse MLA vs folded 544/320 DSL ABI.
MQSMLA_RECIPE = "aclnn_mixed_quant_sparse_flash_mla"
MQSMLA_SOURCE_RECIPE = "mixed_quant_sparse_flash_mla"
MQSMLA_NATIVE_KIND = "aclnnMixedQuantSparseFlashMla"
DSL_MQSMLA_NATIVE_KIND = "tilelang_dsl.mixed_quant_sparse_flash_mla"
MQSMLA_RECIPES = frozenset({MQSMLA_RECIPE, MQSMLA_SOURCE_RECIPE})
MQSMLA_NATIVE_KINDS = frozenset({MQSMLA_NATIVE_KIND, DSL_MQSMLA_NATIVE_KIND})
MIXED_QUANT_NATIVE_KIND = "MixedQuantSparseFlashMla"
SPARSE_FLASH_ATTENTION_NATIVE = "SparseFlashAttention"
SPARSE_MLA_NATIVE_KIND = "SparseFlashMla"
SPARSE_SHARED_KV_NATIVE_KIND = "SparseAttnSharedkv"
GATED_RING_COMPRESSOR = "gated_ring"
A5_ARCH_FAMILY = "A5"
FLOATING_CACHE_DTYPES = frozenset({"bf16", "fp16", "bfloat16", "float16"})
# Packed window (FP8 values + BF16 scale per 32) and compressed
# (two FP4 per byte + BF16 scale per 16) rows of one 512-wide cache entry.
MQSMLA_WINDOW_ROW_BYTES = 544
MQSMLA_COMPRESSED_ROW_BYTES = 320
MQSMLA_FP8_ROW_BYTES = 608
MQSMLA_MXFP8_ROW_BYTES = 584
MQSMLA_CANN_PACKED_WIDTHS = frozenset({MQSMLA_FP8_ROW_BYTES, MQSMLA_MXFP8_ROW_BYTES})
MQSMLA_ORIGINAL_ROW_WIDTHS = frozenset({
    MQSMLA_WINDOW_ROW_BYTES, MQSMLA_FP8_ROW_BYTES, MQSMLA_MXFP8_ROW_BYTES,
})
MQSMLA_DSL_ROW_WIDTH_PAIR = (MQSMLA_WINDOW_ROW_BYTES, MQSMLA_COMPRESSED_ROW_BYTES)
MQSMLA_CANN_ROW_WIDTH_PAIRS = frozenset({
    (MQSMLA_FP8_ROW_BYTES, MQSMLA_FP8_ROW_BYTES),
    (MQSMLA_MXFP8_ROW_BYTES, MQSMLA_MXFP8_ROW_BYTES),
})
MQSMLA_ROW_WIDTH_PAIRS = frozenset({MQSMLA_DSL_ROW_WIDTH_PAIR}) | MQSMLA_CANN_ROW_WIDTH_PAIRS
MQSMLA_KV_QUANT_MODE_WIDTHS = {1: MQSMLA_FP8_ROW_BYTES, 2: MQSMLA_MXFP8_ROW_BYTES}
MQSMLA_HEAD_WIDTH = 512
MQSMLA_QUERY_RANK = 3
MQSMLA_CANN_CACHE_DTYPE = "fp8_e4m3"
MQSMLA_DSL_CACHE_DTYPE = "uint8"
MQSMLA_CACHE_DTYPE = MQSMLA_DSL_CACHE_DTYPE


def shared_kv_native_name(view):
    """Generation and hardware select the native sparse operator family.

    ``gated_ring`` (V4.1) is always the sparse-flash MLA kernel. On A5 the V4
    ``weighted_pool`` family also uses sparse flash MLA, taking the mixed-quant
    variant whenever the cache is not a plain floating representation. The
    legacy A2/A3 shared-KV operator stays restricted to ``weighted_pool``.
    """
    if view.get("compressor_kind") == GATED_RING_COMPRESSOR:
        return SPARSE_MLA_NATIVE_KIND
    if view.get("arch_family") == A5_ARCH_FAMILY:
        return (SPARSE_MLA_NATIVE_KIND
                if view.get("cache_dtype") in FLOATING_CACHE_DTYPES
                else MIXED_QUANT_NATIVE_KIND)
    return SPARSE_SHARED_KV_NATIVE_KIND


def grouped_situ_contract(view):
    """Verified MX A8W4 channel domain; routed row values remain independent."""
    experts, width, intermediate = (view[key] for key in (
        "local_experts", "input_width", "intermediate_width"))
    activation = view["activation"]
    if (experts < 1 or width <= 0 or width % 64 or intermediate <= 0 or intermediate % 64
            or (view["input_dtype"], view["weight_dtype"], view["output_dtype"])
            != ("fp8_e4m3", "mxfp4", "fp8_e4m3")
            or view.get("intermediate_dtype") != "bf16"
            or view.get("weight_scale_group_size") != 32
            or view.get("input_scale_dtype") != "fp8_e8m0"
            or view.get("output_scale_dtype") != "fp8_e8m0"
            or view.get("input_scale_group_size") != 32
            or view.get("output_scale_group_size") != 32
            or tuple(view.get("input_scale_tail", ())) != (width // 64, 2)
            or tuple(view.get("output_scale_tail", ())) != (intermediate // 64, 2)
            or activation.get("activation_kind") != "situ"
            or activation.get("activation_compute_dtype") != "fp32"
            or activation.get("activation_beta", 0) <= 0
            or not activation.get("activation_linear_beta")
            or activation["activation_linear_beta"] <= 0):
        return None
    return (
        "grouped_mlp_parameters_active", "GroupedMatmulSituQuant",
        (experts, width, intermediate), ("fp8_e4m3", "mxfp4", "fp8_e4m3"),
    )


def add_rmsnorm_bias_contract(op):
    meta, shape = op["metadata"], tuple(op["local_shape"])
    if (meta.get("native_norm_kind") != "AddRmsNormBias" or len(shape) != 2
            or not meta.get("norm_has_weight") or meta.get("norm_weight_offset")):
        return None
    x = (shape, op["dtype"], "ND")
    affine = ((shape[-1],), op["dtype"], "ND")
    beta = affine if meta.get("norm_has_bias") else ((), None, None)
    return ("native_tensor_abi", "AddRmsNormBias", (x, x, affine, beta),
            (x, ((shape[0], 1), "fp32", "ND"), x))


def layer_norm_contract(op):
    """Mean/variance normalization with explicitly declared affine parameters."""
    meta, shape = op["metadata"], tuple(op["local_shape"])
    parameter_dtype = meta.get("norm_parameter_dtype")
    if (op["family"] != "layernorm" or len(shape) != 2
            or not meta.get("norm_has_weight") or not meta.get("norm_has_bias")
            or not parameter_dtype):
        return None
    x = (shape, op["dtype"], "ND")
    affine = ((shape[-1],), parameter_dtype, "ND")
    return "layernorm_active", "LayerNormV3", x, affine, affine, x


def transpose_batchmatmul_contract(op):
    """Compare batch matrix geometry separately from its physical permutation."""
    meta = op["metadata"]
    view = meta.get("native_tensor_view", {})
    if view.get("kind") != "transpose_batchmatmul":
        return None
    batch, rows, n, k = (view[key] for key in ("batch", "rows", "n", "k"))
    if (tuple(op["local_shape"]) != (batch * rows, n, k)
            or meta.get("matrix_groups") != batch
            or meta.get("bias") or meta.get("activation_scale_shape")):
        return None
    dtypes = tuple(meta.get(key) for key in
                   ("activation_format", "weight_format", "output_format"))
    if any(dtype not in {"bf16", "fp16", "fp32"} for dtype in dtypes):
        return None
    return ("transpose_batchmatmul_active", "TransposeBatchMatMul",
            (batch, rows, n, k), *dtypes)


def gated_norm_contract(op):
    """RMS gated output with weight, no bias and one group per declared row."""
    meta = op["metadata"]
    shape = tuple(op["local_shape"])
    if (op["recipe"] != "linear_output_norm_gate" or len(shape) != 2
            or meta.get("norm_kind") != "rmsnorm"
            or meta.get("norm_has_weight") is not True
            or meta.get("norm_has_bias") is not False
            or meta.get("gate_present") is not True
            or meta.get("norm_group_width") != shape[-1]):
        return None
    rows, width = shape
    dtype = op["dtype"]
    tensor = (shape, dtype, "ND")
    if meta.get("recurrent_attention_kind") == "kda":
        return ("gated_norm_active_rows", "layer_norm_gated_fwd_kernel",
                tensor, ((width,), dtype, "ND"))
    return (
        "native_tensor_abi", "_layer_norm_fwd_1pass_kernel_npu",
        (tensor, ((width,), dtype, "ND"), tensor),
        (tensor, ((rows,), "fp32", "ND")),
    )


def gemma_norm_contract(op):
    """The declared weight offset distinguishes Gemma RMS from ordinary RMS."""
    meta, shape = op["metadata"], tuple(op["local_shape"])
    if (op["family"] != "rmsnorm" or len(shape) != 2
            or meta.get("norm_weight_offset") != 1.0
            or meta.get("norm_multiply_before_output_cast") is not True
            or meta.get("norm_has_weight") is not True):
        return None
    tensor = (shape, op["dtype"], "ND")
    return ("native_tensor_abi", "GemmaRmsNorm",
            (tensor, ((shape[-1],), op["dtype"], "ND")),
            (tensor, ((shape[0], 1), "fp32", "ND")))


def _mqsmla_attention_contract(op, view):
    """One packed-cache batch kernel per layer, not a captured binary ABI."""
    if not isinstance(view, dict):
        return None
    meta = op["metadata"]
    binding = meta.get("implementation_binding", {})
    native_kind = meta.get("native_kind")
    if (native_kind not in MQSMLA_NATIVE_KINDS
            or binding.get("implementation") not in MQSMLA_RECIPES
            or binding.get("capture_binary_identity") != "not_established"
            or op.get("source_boundary") != "kernel"):
        return None
    shape = tuple(view.get("query_shape", ()))
    orig_bytes = view.get("original_cache_row_bytes")
    if (len(shape) != MQSMLA_QUERY_RANK
            or any(type(dim) is not int or dim <= 0 for dim in shape)
            or shape[-1] != MQSMLA_HEAD_WIDTH
            or tuple(view.get("output_shape", ())) != shape
            or view.get("query_dtype") != "bf16"
            or view.get("output_dtype") != "bf16"
            or type(orig_bytes) is not int
            or orig_bytes not in MQSMLA_ORIGINAL_ROW_WIDTHS
            or type(view.get("compressed_cache")) is not bool):
        return None
    kv_quant_mode = view.get("kv_quant_mode")
    if orig_bytes in MQSMLA_CANN_PACKED_WIDTHS:
        if native_kind != MQSMLA_NATIVE_KIND:
            return None
        if (kv_quant_mode is not None
                and MQSMLA_KV_QUANT_MODE_WIDTHS.get(kv_quant_mode) != orig_bytes):
            return None
        contract_kind = MQSMLA_NATIVE_KIND
        cache_dtype = MQSMLA_CANN_CACHE_DTYPE
        allowed_pairs = MQSMLA_CANN_ROW_WIDTH_PAIRS
    else:
        if kv_quant_mode is not None:
            return None
        contract_kind = DSL_MQSMLA_NATIVE_KIND
        cache_dtype = MQSMLA_DSL_CACHE_DTYPE
        allowed_pairs = frozenset({MQSMLA_DSL_ROW_WIDTH_PAIR})
    compressed = ((), None, None)
    if view["compressed_cache"]:
        comp_bytes = view.get("compressed_cache_row_bytes")
        if (type(comp_bytes) is not int
                or (orig_bytes, comp_bytes) not in allowed_pairs):
            return None
        compressed = ((1, comp_bytes), cache_dtype, "ND")
    query = (shape, "bf16", "ND")
    return (
        "sparse_attention_active", contract_kind, (query,),
        (((1, orig_bytes), cache_dtype, "ND"), compressed),
        query,
    )


def attention_contract(op):
    """Source-declared dense or paged tensors, not the entire native ABI."""
    view = op["metadata"].get("attention_tensor_view")
    if op.get("recipe") in MQSMLA_RECIPES:
        return _mqsmla_attention_contract(op, view)
    if op.get("recipe") == "vllm_ascend_sharedkv_scfa":
        return _scfa_attention_contract(op, view)
    if op.get("recipe") == "ascend_sparse_flash_attention_arch22":
        return _indexed_attention_contract(op, view)
    if op.get("recipe") in {"ascend_paged_mla_decode_arch35", "ascend_paged_mla_decode_arch22"}:
        return _paged_mla_attention_contract(op, view)
    if not view or op.get("source_boundary"):
        return None
    if view.get("kind") in {"indexed_latent", "shared_kv"}:
        return _sparse_attention_contract(view)
    if view.get("kind") == "mixed_quant_shared_kv":
        # A packed ABI selects the mixed-quant producer; a plain floating
        # override keeps the generation's own sparse-MLA family.
        floating = view.get("cache_dtype") in FLOATING_CACHE_DTYPES
        return _sparse_attention_contract(
            view, native_name=None if floating else MIXED_QUANT_NATIVE_KIND)
    if view.get("kind") == "paged_gqa":
        return _paged_attention_contract(view)
    if (view.get("logical_layout") != "TND"
            or view.get("kv_storage") != "contiguous"):
        return None
    slots = {
        name: (tuple(view[name + "_shape"]), view[name + "_dtype"], "ND")
        for name in ("query", "key", "value", "output")
    }
    return ("attention_active_qkv", "FusedInferAttentionScore",
            tuple(slots[name] for name in ("query", "key", "value")), (slots["output"],))


def _paged_mla_attention_contract(op, view):
    """Native BNSD query/NBSD output; cache capacity and index values excluded."""
    geometry = op["metadata"].get("paged_mla_geometry", {})
    active = len(geometry.get("contexts", ()))
    batch = geometry.get("native_query_capacity", active)
    heads, page = geometry.get("heads", 0), geometry.get("page_size", 0)
    key_tile = 512 if op.get("recipe") == "ascend_paged_mla_decode_arch22" else 128
    if (not view or view.get("kind") != "paged_latent"
            or op.get("source_boundary") != "kernel"
            or op["metadata"].get("implementation_binding", {}).get("implementation")
            != op.get("recipe")
            or active < 1 or batch < active or not 1 <= heads <= 64 or page < 1 or key_tile % page
            or tuple(op["local_shape"]) != (active, heads, 1, 512)
            or tuple(op["padded_shape"]) != (batch, heads, 1, 512)
            or tuple(view.get("query_shape", ())) != (batch, heads, 1, 512)
            or tuple(view.get("query_rope_shape", ())) != (batch, heads, 1, 64)
            or tuple(view.get("output_shape", ())) != (heads, batch, 1, 512)
            or tuple(view.get("cache_head_shape", ())) != (1, 512)
            or tuple(view.get("rope_cache_head_shape", ())) != (1, 64)
            or any(view.get(name + "_dtype") != "bf16"
                   for name in ("query", "cache", "output"))):
        return None
    query = ((batch, heads, 1, 512), "bf16", "ND")
    rope = ((batch, heads, 1, 64), "bf16", "ND")
    cache = ((1, page, 512), "bf16", "ND")
    rope_cache = ((1, page, 64), "bf16", "ND")
    output = ((heads, batch, 1, 512), "bf16", "ND")
    return ("attention_paged_mla_qk_rope", "FusedInferAttentionScore",
            (query, rope), (cache, cache, rope_cache), output, batch)


def _indexed_attention_contract(op, view):
    """A source binding retains the same active tensors, not binary identity."""
    meta = op["metadata"]
    shape = meta.get("indexed_attention_geometry", {})
    if (not view or op.get("source_boundary") != "kernel"
            or view.get("kind") != "indexed_latent"
            or meta.get("implementation_binding", {}).get("implementation")
            != "ascend_sparse_flash_attention_arch22"
            or shape.get("window") != 0 or shape.get("compression_ratio") != 1
            or shape.get("cp") != 1 or shape.get("sparse_block_size", 1) != 1
            or (shape.get("latent"), shape.get("rope")) != (512, 64)
            or not 1 <= shape.get("heads", 0) <= 16):
        return None
    query = (shape.get("query"), shape.get("heads"), 512)
    if (tuple(view.get("query_shape", ())) != query
            or tuple(view.get("output_shape", ())) != query
            or tuple(view.get("query_rope_shape", ())) != (*query[:2], 64)
            or tuple(view.get("cache_head_shape", ())) != (1, 512)
            or tuple(view.get("rope_cache_head_shape", ())) != (1, 64)
            or any(view.get(key + "_dtype") != "bf16"
                   for key in ("query", "cache", "output"))
            or any(shape.get(key + "_dtype") != "bf16" for key in ("query", "cache"))):
        return None
    return _sparse_attention_contract(view)


def _scfa_attention_contract(op, view):
    """Keep the kernel boundary while exposing only its active tensor contract."""
    meta = op["metadata"]
    source = meta.get("sharedkv_attention_geometry", {})
    if (not view or op.get("source_boundary") != "kernel"
            or view.get("kind") != "shared_kv" or not view.get("compressed_cache")
            or source.get("branch") != "SCFA"
            or meta.get("implementation_binding", {}).get("implementation")
            != "vllm_ascend_sharedkv_scfa"):
        return None
    shape = (source.get("query"), source.get("heads"), source.get("width"))
    if (tuple(view.get("query_shape", ())) != shape
            or tuple(view.get("output_shape", ())) != shape
            or tuple(view.get("cache_head_shape", ())) != (1, source.get("width"))
            or any(view.get(name + "_dtype") != source.get(name + "_dtype")
                   for name in ("query", "cache", "output"))):
        return None
    return _sparse_attention_contract(view)


def _paged_attention_contract(view):
    required = {"query_shape", "output_shape", "key_tail_shape", "value_tail_shape",
                "query_dtype", "key_dtype", "value_dtype", "output_dtype", "batch_count"}
    if (not required <= view.keys() or view.get("logical_layout") != "TND"
            or view.get("kv_storage") != "paged"):
        return None
    query, output = tuple(view["query_shape"]), tuple(view["output_shape"])
    key, value = tuple(view["key_tail_shape"]), tuple(view["value_tail_shape"])
    dtypes = tuple(view[name + "_dtype"] for name in ("query", "key", "value", "output"))
    if (len(query) != 3 or output != query or len(key) != 2 or len(value) != 2
            or key[0] != value[0] or any(dim <= 0 for dim in (*query, *key, *value))
            or view["batch_count"] < 1 or len(set(dtypes)) != 1
            or dtypes[0] not in {"bf16", "fp16"}):
        return None
    # The frontend supplies block size and packed KV row width independently.
    # No observed cache capacity or factorization of the packed width is used.
    return (
        "attention_paged_active_qkv", "FusedInferAttentionScore",
        (query, dtypes[0], "ND"),
        ((key, dtypes[1], "ND"), (value, dtypes[2], "ND")),
        (output, dtypes[3], "ND"), view["batch_count"],
    )


def _sparse_attention_contract(view, native_name=None):
    query = (tuple(view["query_shape"]), view["query_dtype"], "ND")
    output = (tuple(view["output_shape"]), view["output_dtype"], "ND")
    cache = (tuple(view["cache_head_shape"]), view["cache_dtype"], "ND")
    if view["kind"] == "indexed_latent":
        rope = (tuple(view["query_rope_shape"]), view["query_dtype"], "ND")
        rope_cache = (tuple(view["rope_cache_head_shape"]), view["cache_dtype"], "ND")
        return ("sparse_attention_active",
                native_name or SPARSE_FLASH_ATTENTION_NATIVE,
                (query, rope), (cache, cache, rope_cache), output)
    compressed = cache if view["compressed_cache"] else ((), None, None)
    return ("sparse_attention_active", native_name or shared_kv_native_name(view),
            (query,), (cache, compressed), output)


def cache_contract(op):
    """Active input element counts and slot count, without native storage claims."""
    view = op["metadata"].get("cache_tensor_view")
    if not view or op.get("source_boundary") or op["metadata"].get("cache_kind") != "gqa":
        return None
    return (
        "kv_cache_store_active_elements", "ScatterPaKvCache",
        tuple((math.prod(view[name + "_shape"]), view[name + "_dtype"])
              for name in ("key", "value")),
        tuple(view["slot_shape"]),
    )
