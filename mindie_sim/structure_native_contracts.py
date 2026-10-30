"""Native tensor boundaries derived from independently declared operator work.

[Role] Project frontend semantics onto captured attention/cache/fused interfaces.
[Invariants] No capture input, device selection or duration affects a declaration.
  A source call is not a kernel; excluded scalar/auxiliary values stay explicit.
[Timeouts & Locks] Pure bounded metadata arithmetic; no I/O or locks.
[Side-Effect Policy] Returns immutable contracts without importing the profiler.
"""
import math


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


def attention_contract(op):
    """Source-declared dense or paged tensors, not the entire native ABI."""
    view = op["metadata"].get("attention_tensor_view")
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


def _sparse_attention_contract(view):
    query = (tuple(view["query_shape"]), view["query_dtype"], "ND")
    output = (tuple(view["output_shape"]), view["output_dtype"], "ND")
    cache = (tuple(view["cache_head_shape"]), view["cache_dtype"], "ND")
    if view["kind"] == "indexed_latent":
        rope = (tuple(view["query_rope_shape"]), view["query_dtype"], "ND")
        rope_cache = (tuple(view["rope_cache_head_shape"]), view["cache_dtype"], "ND")
        return ("sparse_attention_active", "SparseFlashAttention",
                (query, rope), (cache, cache, rope_cache), output)
    compressed = cache if view["compressed_cache"] else ((), None, None)
    return ("sparse_attention_active", "SparseAttnSharedkv",
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
