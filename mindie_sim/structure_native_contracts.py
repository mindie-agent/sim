"""Native tensor boundaries derived from independently declared operator work.

[Role] Project frontend semantics onto captured attention/cache/fused interfaces.
[Invariants] No capture input, device selection or duration affects a declaration.
  A source call is not a kernel; excluded scalar/auxiliary values stay explicit.
[Timeouts & Locks] Pure bounded metadata arithmetic; no I/O or locks.
[Side-Effect Policy] Returns immutable contracts without importing the profiler.
"""
import math

#: CANN 9.2.1 official mixed-quant sparse MLA plus its retained prototype alias.
MQSMLA_RECIPE = "aclnn_mixed_quant_sparse_flash_mla"
MQSMLA_NATIVE_KIND = "aclnnMixedQuantSparseFlashMla"
LEGACY_MQSMLA_RECIPE = "cannbot_mixed_quant_sparse_flash_mla"
LEGACY_MQSMLA_NATIVE_KIND = "MqsmlaKernel___call__"
MQSMLA_RECIPES = frozenset({MQSMLA_RECIPE, LEGACY_MQSMLA_RECIPE})
MQSMLA_NATIVE_KINDS = frozenset({MQSMLA_NATIVE_KIND, LEGACY_MQSMLA_NATIVE_KIND})
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
MQSMLA_HEAD_WIDTH = 512
MQSMLA_QUERY_RANK = 3
MQSMLA_CACHE_DTYPE = "uint8"


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


_ADD_RMSNORM_ABI_TENSORS = {
    "x": ("input", "bf16"),
    "residual": ("input", "bf16"),
    "gamma": ("weight", "bf16"),
    "output": ("output", "bf16"),
    "rstd": ("output", "fp32"),
    "residual_out": ("output", "bf16"),
}
_ABI_DTYPE_ALIASES = {
    "bfloat16": "bf16", "float32": "fp32", "float16": "fp16",
    "int8": "int8", "int32": "int32", "int64": "int64",
}


def _abi_field(raw, name, default=None):
    if isinstance(raw, dict):
        return raw.get(name, default)
    return getattr(raw, name, default)


def _abi_dtype(value):
    text = str(value or "").strip().lower()
    return _ABI_DTYPE_ALIASES.get(text, text)


def _abi_positive_shape(raw):
    shape = _abi_field(raw, "shape", None)
    if not isinstance(shape, (list, tuple)) or not shape:
        return None
    values = []
    for item in shape:
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            return None
        values.append(item)
    return tuple(values)


def _add_rmsnorm_abi(tensors, attrs, *, sequence_rows="", sequence_ranks=1):
    """Validate the BF16 AddRmsNorm tensor/attr ABI without pricing it."""
    reasons = []
    if not isinstance(tensors, dict):
        return None, ("tensors_requires_mapping",)
    if sequence_rows not in {"", "shard", "selected"}:
        reasons.append("sequence_rows_invalid")
    if (isinstance(sequence_ranks, bool) or not isinstance(sequence_ranks, int)
            or sequence_ranks < 1):
        reasons.append("sequence_ranks_invalid")
    values = {}
    for name, (role, dtype) in _ADD_RMSNORM_ABI_TENSORS.items():
        raw = tensors.get(name)
        if raw is None or _abi_field(raw, "present", True) is False:
            reasons.append(f"tensor_missing:{name}")
            continue
        shape = _abi_positive_shape(raw)
        if shape is None:
            reasons.append(f"tensor_shape_invalid:{name}")
            continue
        actual_role = str(_abi_field(raw, "role", role) or role)
        actual_dtype = _abi_dtype(_abi_field(raw, "dtype", dtype))
        layout = str(_abi_field(raw, "layout", "ND") or "ND").upper()
        if actual_role != role:
            reasons.append(f"tensor_role_mismatch:{name}")
        if actual_dtype != dtype:
            reasons.append(f"tensor_dtype_mismatch:{name}")
        if layout != "ND":
            reasons.append(f"tensor_layout_mismatch:{name}")
        values[name] = shape
    epsilon = None
    if isinstance(attrs, dict) and attrs.get("epsilon") is not None:
        raw_eps = attrs["epsilon"]
        if (isinstance(raw_eps, bool) or not isinstance(raw_eps, (int, float))
                or not math.isfinite(float(raw_eps)) or float(raw_eps) <= 0):
            reasons.append("epsilon_invalid")
        else:
            epsilon = float(raw_eps)
    if reasons:
        return None, tuple(dict.fromkeys(reasons))
    if len(values["x"]) != 2:
        return None, ("tensor_rank_mismatch:x",)
    rows, hidden = values["x"]
    if values["residual"] != (rows, hidden):
        return None, ("tensor_shape_mismatch:residual",)
    if len(values["gamma"]) != 1 or values["gamma"][0] != hidden:
        return None, ("tensor_shape_mismatch:gamma",)
    if values["output"] != (rows, hidden):
        return None, ("tensor_shape_mismatch:output",)
    if len(values["residual_out"]) != 2 or values["residual_out"] != (rows, hidden):
        return None, ("tensor_shape_mismatch:residual_out",)
    rstd = values["rstd"]
    if not (rstd == (rows, 1) or rstd == (rows,)):
        return None, ("tensor_shape_mismatch:rstd",)
    binding_shapes = {
        "x": [hidden], "residual": [hidden], "gamma": [hidden],
        "output": [hidden], "rstd": [1], "residual_out": [hidden],
    }
    binding_attrs = {
        "hidden": hidden,
        "sequence_rows": sequence_rows,
        "sequence_ranks": sequence_ranks,
        "input_layout": "ND",
        "rstd_dtype": "fp32",
        "output_dtype": "bf16",
        "residual_dtype": "bf16",
        "call_name": "aclnnAddRmsNorm",
        "kernel_name": "AddRmsNorm",
    }
    occurrence_shapes = {
        "x": [rows, hidden], "residual": [rows, hidden], "gamma": [hidden],
        "output": [rows, hidden], "rstd": [rows, 1],
        "residual_out": [rows, hidden],
    }
    return {
        "kind": "add_rmsnorm",
        "family": "elementwise",
        "recipe": "norm",
        "rows": rows,
        "hidden": hidden,
        "act_bytes": 2.0,
        "dtype": "bf16",
        "layout": "ND",
        "members": ("residual_add", "rmsnorm"),
        "inputs": ("x", "residual", "gamma"),
        "outputs": ("output", "rstd", "residual_out"),
        "binding_shapes": binding_shapes,
        "binding_attrs": binding_attrs,
        "occurrence_shapes": occurrence_shapes,
        "epsilon": epsilon,
    }, ()


def add_rmsnorm_abi_region(tensors, attrs=None, *, sequence_rows="",
                           sequence_ranks=1):
    """Inverse tensor contract: a concrete AddRmsNorm ABI -> semantic region.

    Returns ``None`` when the ABI is not this fused residual+RMSNorm call.
    The result carries no duration, work rate or target observation.
    """
    region, _reasons = _add_rmsnorm_abi(
        tensors, attrs, sequence_rows=sequence_rows,
        sequence_ranks=sequence_ranks)
    return region


def add_rmsnorm_abi_reasons(tensors, attrs=None, *, sequence_rows="",
                            sequence_ranks=1):
    """Typed rejection reasons for the inverse AddRmsNorm ABI contract."""
    _region, reasons = _add_rmsnorm_abi(
        tensors, attrs, sequence_rows=sequence_rows,
        sequence_ranks=sequence_ranks)
    return reasons


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
    if (len(shape) != MQSMLA_QUERY_RANK
            or any(type(dim) is not int or dim <= 0 for dim in shape)
            or shape[-1] != MQSMLA_HEAD_WIDTH
            or tuple(view.get("output_shape", ())) != shape
            or view.get("query_dtype") != "bf16"
            or view.get("output_dtype") != "bf16"
            or view.get("original_cache_row_bytes") != MQSMLA_WINDOW_ROW_BYTES
            or type(view.get("compressed_cache")) is not bool):
        return None
    compressed = ((), None, None)
    if view["compressed_cache"]:
        if view.get("compressed_cache_row_bytes") != MQSMLA_COMPRESSED_ROW_BYTES:
            return None
        compressed = (((1, MQSMLA_COMPRESSED_ROW_BYTES), MQSMLA_CACHE_DTYPE, "ND"))
    query = (shape, "bf16", "ND")
    return (
        "sparse_attention_active", native_kind, (query,),
        (((1, MQSMLA_WINDOW_ROW_BYTES), MQSMLA_CACHE_DTYPE, "ND"), compressed),
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
