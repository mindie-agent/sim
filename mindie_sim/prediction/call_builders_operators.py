"""Argument validation and operator construction for standalone ACLNN calls.

[Role] Lower matrix, vector and attention arguments to their declared
  operator geometry, arithmetic, traffic and metadata.
[Invariants] Tensor shapes and dtypes determine the work of a single
  invocation. Validation uses the same argument rules for each call kind.
[Timeouts & Locks] Finite geometry construction. No I/O or locks.
[Side-Effect Policy] Returns operator records without running device work.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes

ACLNN_ABI_SCHEMA = "mindie-sim.aclnn-abi/1"
ACLNN_DEFAULT_DTYPE = "bf16"
ACLNN_GROUPED_KIND = "grouped"
ACLNN_ATTENTION_KIND = "attention"
ACLNN_VECTOR_KIND = "vector"

ACLNN_PHASES = frozenset({"prefill", "decode", "mixed", "primitive"})
ACLNN_MATRIX_DTYPES = frozenset({"bf16", "bfloat16", "fp16", "float16",
                                 "fp32", "float32", "int8", "w8a8",
                                 "fp8", "fp8_e4m3", "fp8_e5m2",
                                 "int4", "w4a8", "mxfp4"})
ACLNN_VECTOR_DTYPES = frozenset({"bf16", "bfloat16", "fp16", "float16",
                                 "fp32", "float32"})
ACLNN_ATTENTION_DTYPES = frozenset({"bf16", "bfloat16", "fp16", "float16"})
ACLNN_ATTENTION_QUERY_DTYPES = ACLNN_VECTOR_DTYPES | frozenset({
    "fp8", "fp8_e4m3", "fp8_e5m2", "int8",
})
ACLNN_ATTENTION_KV_DTYPES = ACLNN_ATTENTION_QUERY_DTYPES | frozenset({
    "int4", "mxfp4",
})
MATRIX_MIN_RANK = 2
MATMUL_SHAPE_DIMS = 3
ATTN_UNBATCHED_RANK = 3
ATTN_BATCHED_RANK = 4
FLOPS_PER_FMA = 2
ATTENTION_SOFTMAX_OPS_PER_PAIR = 5
RMSNORM_OPS_PER_ELEM = 5
ADD_RMSNORM_OPS_PER_ELEM = 6
DYNAMIC_QUANT_OPS_PER_ELEM = 4


def _mapping(value, reason):
    if not isinstance(value, Mapping):
        raise ModelingError(reason)
    return dict(value)


def _text(value, reason):
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(reason)
    return value.strip()


def _positive_integer(value, reason):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(reason)
    return value


def _nonnegative_integer(value, reason):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingError(reason)
    return value


def _dtype(value, allowed, reason):
    dtype = _text(value or ACLNN_DEFAULT_DTYPE, reason).lower()
    if dtype not in allowed:
        raise ModelingError(f"{reason}:{dtype}")
    dtype_bytes(dtype)
    return dtype


def _boolean(value, reason):
    if not isinstance(value, bool):
        raise ModelingError(reason)
    return value


def _phase(value):
    phase = _text(value or "primitive", "aclnn_call_phase_invalid")
    if phase not in ACLNN_PHASES:
        raise ModelingError(f"aclnn_call_phase_unsupported:{phase}")
    return phase


def _shape(value, name):
    if isinstance(value, Mapping):
        value = value.get("shape")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ModelingError(f"aclnn_call_tensor_shape_invalid:{name}")
    shape = []
    for item in value:
        shape.append(_positive_integer(item, f"aclnn_call_tensor_shape_invalid:{name}"))
    return tuple(shape)


def _tensor(arguments, *names):
    tensors = arguments.get("tensors")
    if not isinstance(tensors, Mapping):
        return None, None
    for name in names:
        raw = tensors.get(name)
        if raw is None:
            continue
        return name, _shape(raw, name)
    return None, None


def _leading_product(shape):
    return math.prod(shape[:-2]) if len(shape) > 2 else 1


def _expert_rows(arguments):
    rows = arguments.get("expert_rows")
    if rows is None:
        rows = arguments.get("group_list")
    if rows is None:
        return None
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ModelingError("aclnn_grouped_matmul_expert_rows_invalid")
    values = tuple(_nonnegative_integer(
        value, "aclnn_grouped_matmul_expert_rows_invalid") for value in rows)
    if not any(values):
        raise ModelingError("aclnn_grouped_matmul_requires_active_expert")
    return values


def _matrix_geometry(arguments, kind):
    """Return ``(m, n, k, groups)`` for one standalone matrix ABI."""
    _, x_shape = _tensor(arguments, "x", "input", "activation", "a")
    _, w_shape = _tensor(arguments, "weight", "w", "b")
    if x_shape is None and arguments.get("x_shape") is not None:
        x_shape = _shape(arguments["x_shape"], "x_shape")
    if w_shape is None and arguments.get("weight_shape") is not None:
        w_shape = _shape(arguments["weight_shape"], "weight_shape")
    transpose_b = bool(arguments.get("transpose_b", False))
    batch, m, k = 1, None, None
    shape = arguments.get("shape")
    if x_shape is None and shape is not None:
        shape = _shape(shape, "shape")
        if len(shape) != MATMUL_SHAPE_DIMS:
            raise ModelingError("aclnn_matmul_shape_requires_m_n_k")
        m, n, k = shape
        shape = None
    else:
        n = None
    if x_shape is not None:
        if len(x_shape) < MATRIX_MIN_RANK:
            raise ModelingError("aclnn_matmul_requires_rank_at_least_two_activation")
        batch = _leading_product(x_shape)
        m, k = x_shape[-2:]
    weight_count = 1
    if w_shape is not None:
        if len(w_shape) < MATRIX_MIN_RANK:
            raise ModelingError("aclnn_matmul_requires_rank_at_least_two_weight")
        weight_count = _leading_product(w_shape)
        first, second = w_shape[-2:]
        n = first if transpose_b else second
        weight_k = second if transpose_b else first
        if k is not None and weight_k != k:
            raise ModelingError("aclnn_matmul_weight_k_dimension_mismatch")
        k = weight_k if k is None else k
    rows = _expert_rows(arguments) if kind == ACLNN_GROUPED_KIND else None
    if rows is not None:
        m, groups = sum(rows), sum(1 for value in rows if value)
    else:
        if kind == ACLNN_GROUPED_KIND:
            groups = _positive_integer(
                arguments.get("groups", weight_count),
                "aclnn_grouped_matmul_requires_positive_groups")
        else:
            groups = max(1, weight_count)
        if kind == ACLNN_GROUPED_KIND and weight_count not in (1, groups):
            raise ModelingError("aclnn_grouped_matmul_weight_group_mismatch")
    m = _positive_integer(arguments.get("m", m),
                          "aclnn_matmul_requires_positive_m") * (
                              batch if rows is None else 1)
    n = _positive_integer(arguments.get("n", n),
                          "aclnn_matmul_requires_positive_n")
    k = _positive_integer(arguments.get("k", k),
                          "aclnn_matmul_requires_positive_k")
    if kind != ACLNN_GROUPED_KIND and weight_count not in (1, batch):
        raise ModelingError("aclnn_matmul_weight_batch_mismatch")
    return m, n, k, groups


def _quant_spec(symbol, kind, arguments):
    from mindie_sim.catalog.types import QuantSpec
    quantized = kind == ACLNN_GROUPED_KIND or "QuantBatchMatmul" in symbol
    activation = _dtype(
        arguments.get("activation_dtype") or arguments.get("dtype")
        or _optional_tensor_dtype(
            arguments, ("x", "input", "a"), ACLNN_MATRIX_DTYPES,
            "aclnn_call_activation_dtype_unsupported")
        or ("int8" if quantized else ACLNN_DEFAULT_DTYPE),
        ACLNN_MATRIX_DTYPES, "aclnn_call_activation_dtype_unsupported")
    default_weight = ("int8" if quantized and activation == "int8"
                      else activation)
    weight = _dtype(
        arguments.get("weight_dtype")
        or _optional_tensor_dtype(
            arguments, ("weight", "w", "b"), ACLNN_MATRIX_DTYPES,
            "aclnn_call_weight_dtype_unsupported")
        or default_weight,
        ACLNN_MATRIX_DTYPES, "aclnn_call_weight_dtype_unsupported")
    output = _dtype(
        arguments.get("output_dtype")
        or _optional_tensor_dtype(
            arguments, ("output", "out", "y"), ACLNN_VECTOR_DTYPES,
            "aclnn_call_output_dtype_unsupported")
        or "bf16",
        ACLNN_VECTOR_DTYPES, "aclnn_call_output_dtype_unsupported")
    return QuantSpec(
        name="aclnn_grouped_matmul" if kind == ACLNN_GROUPED_KIND
        else "aclnn_quant_batch_matmul_v3" if quantized else "aclnn_matmul",
        weight_format=weight, activation_format=activation,
        output_format=output), activation


def _matrix_metadata(symbol, kind, arguments, m, n, k, groups):
    metadata = [("api", symbol), ("aclnn_symbol", symbol),
                ("aclnn_abi_schema", ACLNN_ABI_SCHEMA),
                ("aclnn_call_kind", kind),
                ("aclnn_geometry", {"m": m, "n": n, "k": k,
                                    "matrix_groups": groups}),
                ("program_kind", "matmul")]
    if kind == ACLNN_GROUPED_KIND:
        metadata.extend((
            ("matrix_mapping_domain", "aggregate_experts"),
            ("routing_basis", "probability_moments"),
            ("expected_active_experts", float(groups)),
            ("expected_rows", float(m)),
            ("matrix_groups", groups),
        ))
        rows = _expert_rows(arguments)
        if rows is not None:
            # ExpectedGroupedMatmul consumes probability moments, not a
            # realized histogram.  Retain the declared direct counts as ABI
            # evidence without reinterpreting them as an integer dispatch.
            metadata.extend((("aclnn_group_list", rows),
                             ("group_list_semantics",
                              "direct_counts_not_cumulative_offsets")))
    return tuple(metadata)


def _build_matrix(arguments, symbol, kind, recipe, family):
    from mindie_sim.catalog.ops import gemm_op
    m, n, k, groups = _matrix_geometry(arguments, kind)
    quant, activation = _quant_spec(symbol, kind, arguments)
    metadata = _matrix_metadata(symbol, kind, arguments, m, n, k, groups)
    op = gemm_op(
        op_id="aclnn_call", deps=(), stage=0, layer=0, requests=(),
        m=m, n=n, k=k, quant=quant, category="linear",
        phase=_phase(arguments.get("phase")), recipe=recipe, family=family,
        weight_count=groups, activation_dtype=activation, metadata=metadata)
    return op, groups


def _optional_tensor_dtype(arguments, names, allowed, reason):
    tensors = arguments.get("tensors")
    if isinstance(tensors, Mapping):
        for name in names:
            raw = tensors.get(name)
            if isinstance(raw, Mapping) and raw.get("dtype"):
                return _dtype(raw["dtype"], allowed, reason)
    return None


def _tensor_dtype(arguments, names, fallback):
    tensors = arguments.get("tensors")
    if isinstance(tensors, Mapping):
        for name in names:
            raw = tensors.get(name)
            if isinstance(raw, Mapping) and raw.get("dtype"):
                return _dtype(raw["dtype"], ACLNN_VECTOR_DTYPES,
                              "aclnn_call_tensor_dtype_unsupported")
    return _dtype(arguments.get("dtype", fallback), ACLNN_VECTOR_DTYPES,
                  "aclnn_call_dtype_unsupported")


def _vector_geometry(arguments):
    _, shape = _tensor(arguments, "x", "input", "self", "src", "dst",
                       "output")
    if shape is None:
        for key in ("x_shape", "shape"):
            if arguments.get(key) is not None:
                shape = _shape(arguments[key], key)
                break
    if shape is not None:
        rows = arguments.get("rows", arguments.get(
            "tokens", shape[-2] if len(shape) > 1 else 1))
        columns = arguments.get(
            "columns", arguments.get(
                "cols", arguments.get(
                    "hidden", arguments.get(
                        "width", shape[-1] if len(shape) > 1
                        else shape[0]))))
        return _positive_integer(rows, "aclnn_vector_requires_positive_rows"), \
            _positive_integer(columns, "aclnn_vector_requires_positive_columns")
    if "elements" in arguments:
        elements = _positive_integer(arguments["elements"],
                                     "aclnn_vector_requires_positive_elements")
        return 1, elements
    return (_positive_integer(arguments.get("rows", arguments.get("tokens")),
                              "aclnn_vector_requires_positive_rows"),
            _positive_integer(arguments.get(
                "columns", arguments.get("cols", arguments.get(
                    "hidden", arguments.get("width")))),
                "aclnn_vector_requires_positive_columns"))


def _vector_work(kind, arguments):
    rows, columns = _vector_geometry(arguments)
    elements = rows * columns
    if kind == "rmsnorm":
        dtype = _tensor_dtype(arguments, ("x", "input", "gamma"), ACLNN_DEFAULT_DTYPE)
        act_bytes = dtype_bytes(dtype)
        gamma_bytes = columns * act_bytes
        read = elements * act_bytes + gamma_bytes
        write = elements * act_bytes + rows * dtype_bytes("fp32")
        return dtype, elements, RMSNORM_OPS_PER_ELEM, read, write, (rows, columns), {
            "reduction_width": columns, "norm_has_weight": True}
    if kind == "add_rmsnorm":
        dtype = _tensor_dtype(arguments, ("x", "residual", "gamma"),
                              ACLNN_DEFAULT_DTYPE)
        act_bytes = dtype_bytes(dtype)
        read = 2 * elements * act_bytes + columns * act_bytes
        write = 2 * elements * act_bytes + rows * dtype_bytes("fp32")
        return dtype, elements, ADD_RMSNORM_OPS_PER_ELEM, read, write, (rows, columns), {
            "reduction_width": columns, "norm_has_weight": True,
            "norm_has_residual": True}
    if kind == "swiglu":
        dtype = _tensor_dtype(arguments, ("x", "input", "gate"), ACLNN_DEFAULT_DTYPE)
        act_bytes = dtype_bytes(dtype)
        read = 2 * elements * act_bytes
        write = elements * act_bytes
        return dtype, elements, RMSNORM_OPS_PER_ELEM, read, write, (rows, columns), {
            "activation_kind": "swiglu", "gate_layout": "gate_first"}
    if kind == "quantize":
        dtype = _tensor_dtype(arguments, ("x", "input"), ACLNN_DEFAULT_DTYPE)
        output = _dtype(arguments.get("output_dtype", "int8"),
                        frozenset({"int8", "fp8", "fp8_e4m3", "fp8_e5m2"}),
                        "aclnn_dynamic_quant_output_dtype_unsupported")
        read = elements * dtype_bytes(dtype)
        write = elements * dtype_bytes(output) + rows * dtype_bytes("fp32")
        return dtype, elements, DYNAMIC_QUANT_OPS_PER_ELEM, read, write, (rows, columns), {
            "input_format": dtype, "output_format": output,
            "activation_scale_format": "fp32", "activation_scale_shape": (rows,)}
    if kind == "tensor_copy":
        dtype = _tensor_dtype(arguments, ("src", "self", "x", "input"),
                              ACLNN_DEFAULT_DTYPE)
        target = _dtype(arguments.get("output_dtype", dtype), ACLNN_VECTOR_DTYPES,
                        "aclnn_inplace_copy_output_dtype_unsupported")
        read = elements * dtype_bytes(dtype)
        write = elements * dtype_bytes(target)
        return dtype, elements, 0.0, read, write, (rows, columns), {
            "input_format": dtype, "output_format": target,
            "copy_semantics": "inplace_copy_into_destination"}
    raise ModelingError(f"aclnn_vector_kind_unsupported:{kind}")


def _build_vector(arguments, symbol, vector_kind, recipe, family):
    from mindie_sim.catalog.ops import vector_op
    dtype, elements, ops_per_element, read, write, shape, extra = _vector_work(
        vector_kind, arguments)
    metadata = (
        ("api", symbol), ("aclnn_symbol", symbol),
        ("aclnn_abi_schema", ACLNN_ABI_SCHEMA),
        ("aclnn_call_kind", ACLNN_VECTOR_KIND),
        *tuple(sorted(extra.items())),
    )
    op = vector_op(
        op_id="aclnn_call", deps=(), stage=0, layer=0, requests=(),
        elements=elements, ops_per_element=ops_per_element, dtype=dtype,
        family=family, phase=_phase(arguments.get("phase")), recipe=recipe,
        shape=shape, read_bytes_override=float(read),
        write_bytes_override=float(write), metadata=metadata)
    return op


def _first_integer(arguments, *names, default=None):
    for name in names:
        if arguments.get(name) is not None:
            return arguments[name]
    return default


def _attention_geometry(arguments):
    tensors = arguments.get("tensors")
    query = None
    if isinstance(tensors, Mapping):
        for name in ("query", "q", "query_shape", "q_shape"):
            raw = tensors.get(name)
            if raw is not None:
                query = _shape(raw, name)
                break
    if query is None:
        for name in ("query_shape", "q_shape", "shape"):
            if arguments.get(name) is not None:
                query = _shape(arguments[name], name)
                break
    if query is None:
        batch = _positive_integer(arguments.get("batch", 1),
                                  "aclnn_attention_requires_positive_batch")
        heads = _positive_integer(
            _first_integer(arguments, "heads", "num_heads"),
            "aclnn_attention_requires_positive_heads")
        query_tokens = _positive_integer(
            _first_integer(arguments, "query_tokens", "q_len", "seq_len",
                           "query_len"),
            "aclnn_attention_requires_positive_query_tokens")
        key_tokens = _positive_integer(
            _first_integer(arguments, "key_tokens", "kv_len", "kv_seq_len",
                           default=query_tokens),
            "aclnn_attention_requires_positive_key_tokens")
        head_dim = _positive_integer(
            _first_integer(arguments, "head_dim", "d"),
            "aclnn_attention_requires_positive_head_dim")
        v_head_dim = _positive_integer(
            _first_integer(arguments, "v_head_dim", "value_head_dim",
                           default=head_dim),
            "aclnn_attention_requires_positive_v_head_dim")
        kv_heads = _positive_integer(
            _first_integer(arguments, "kv_heads", default=heads),
            "aclnn_attention_requires_positive_kv_heads")
        return batch, heads, query_tokens, key_tokens, head_dim, v_head_dim, kv_heads
    if len(query) == ATTN_BATCHED_RANK:
        batch, heads, query_tokens, head_dim = query
    elif len(query) == ATTN_UNBATCHED_RANK:
        batch, heads, query_tokens, head_dim = 1, *query
    else:
        raise ModelingError("aclnn_attention_query_requires_rank_three_or_four")
    key_tokens = _first_integer(arguments, "key_tokens", "kv_len",
                                "kv_seq_len", default=None)
    kv_heads = _first_integer(arguments, "kv_heads", default=None)
    key_shape = None
    if isinstance(tensors, Mapping):
        for name in ("key", "k", "key_shape", "kv_shape"):
            raw = tensors.get(name)
            if raw is not None:
                key_shape = _shape(raw, name)
                break
    if key_shape is None:
        for name in ("key_shape", "kv_shape"):
            if arguments.get(name) is not None:
                key_shape = _shape(arguments[name], name)
                break
    if key_shape is not None:
        if len(key_shape) == ATTN_BATCHED_RANK:
            key_batch, shape_heads, shape_tokens, shape_dim = key_shape
        elif len(key_shape) == ATTN_UNBATCHED_RANK:
            key_batch, shape_heads, shape_tokens, shape_dim = 1, *key_shape
        else:
            raise ModelingError("aclnn_attention_key_requires_rank_three_or_four")
        if key_batch != batch or shape_dim != head_dim:
            raise ModelingError("aclnn_attention_key_geometry_mismatch")
        key_tokens = shape_tokens if key_tokens is None else key_tokens
        kv_heads = shape_heads if kv_heads is None else kv_heads
    value_shape = None
    if isinstance(tensors, Mapping):
        for name in ("value", "v", "value_shape", "v_shape"):
            raw = tensors.get(name)
            if raw is not None:
                value_shape = _shape(raw, name)
                break
    if value_shape is None:
        for name in ("value_shape", "v_shape"):
            if arguments.get(name) is not None:
                value_shape = _shape(arguments[name], name)
                break
    v_head_dim = _first_integer(arguments, "v_head_dim", "value_head_dim",
                               default=None)
    if value_shape is not None:
        if len(value_shape) == ATTN_BATCHED_RANK:
            value_batch, value_heads, value_tokens, shape_v_dim = value_shape
        elif len(value_shape) == ATTN_UNBATCHED_RANK:
            value_batch, value_heads, value_tokens, shape_v_dim = 1, *value_shape
        else:
            raise ModelingError("aclnn_attention_value_requires_rank_three_or_four")
        if value_batch != batch:
            raise ModelingError("aclnn_attention_value_geometry_mismatch")
        key_tokens = value_tokens if key_tokens is None else key_tokens
        if value_tokens != key_tokens:
            raise ModelingError("aclnn_attention_value_geometry_mismatch")
        kv_heads = value_heads if kv_heads is None else kv_heads
        if value_heads != kv_heads:
            raise ModelingError("aclnn_attention_value_geometry_mismatch")
        v_head_dim = shape_v_dim
    key_tokens = _positive_integer(
        query_tokens if key_tokens is None else key_tokens,
        "aclnn_attention_requires_positive_key_tokens")
    kv_heads = _positive_integer(
        heads if kv_heads is None else kv_heads,
        "aclnn_attention_requires_positive_kv_heads")
    v_head_dim = _positive_integer(
        head_dim if v_head_dim is None else v_head_dim,
        "aclnn_attention_requires_positive_v_head_dim")
    if batch < 1 or heads < 1 or query_tokens < 1 or head_dim < 1:
        raise ModelingError("aclnn_attention_requires_positive_geometry")
    return batch, heads, query_tokens, key_tokens, head_dim, v_head_dim, kv_heads


def _attention_op(arguments, symbol, recipe):
    from mindie_sim.schema.operators import OpSpec
    batch, heads, query_tokens, key_tokens, head_dim, v_head_dim, kv_heads = \
        _attention_geometry(arguments)
    attrs = arguments.get("attrs")
    attrs = (_mapping(attrs, "aclnn_attention_attrs_requires_mapping")
             if attrs is not None else {})
    causal = _boolean(attrs.get("causal", arguments.get("causal", True)),
                      "aclnn_attention_causal_requires_boolean")
    block_size = arguments.get("block_size", attrs.get("block_size"))
    paged = block_size is not None
    block_size = (_positive_integer(block_size,
                                    "aclnn_attention_block_size_invalid")
                  if paged else 0)
    sparse_mode = attrs.get("sparse_mode", arguments.get("sparse_mode"))
    sparse_mode = (_nonnegative_integer(sparse_mode,
                                        "aclnn_attention_sparse_mode_invalid")
                   if sparse_mode is not None else None)
    top_left_causal = sparse_mode in (0, 2)
    if causal:
        visible = min(query_tokens, key_tokens)
        if top_left_causal:
            pairs = (visible * (visible + 1) // 2
                     + (query_tokens - visible) * key_tokens)
        else:
            pairs = visible * key_tokens - visible * (visible - 1) // 2
    else:
        pairs = query_tokens * key_tokens
    query_dtype = _dtype(
        arguments.get("query_dtype")
        or _optional_tensor_dtype(
            arguments, ("query", "q"), ACLNN_ATTENTION_QUERY_DTYPES,
            "aclnn_attention_query_dtype_unsupported")
        or ACLNN_DEFAULT_DTYPE,
        ACLNN_ATTENTION_QUERY_DTYPES, "aclnn_attention_query_dtype_unsupported")
    key_dtype = _dtype(
        arguments.get("key_dtype") or arguments.get("kv_dtype")
        or _optional_tensor_dtype(
            arguments, ("key", "k"), ACLNN_ATTENTION_KV_DTYPES,
            "aclnn_attention_kv_dtype_unsupported")
        or query_dtype,
        ACLNN_ATTENTION_KV_DTYPES, "aclnn_attention_kv_dtype_unsupported")
    value_dtype = _dtype(
        arguments.get("value_dtype") or arguments.get("kv_dtype")
        or _optional_tensor_dtype(
            arguments, ("value", "v"), ACLNN_ATTENTION_KV_DTYPES,
            "aclnn_attention_kv_dtype_unsupported")
        or key_dtype,
        ACLNN_ATTENTION_KV_DTYPES, "aclnn_attention_kv_dtype_unsupported")
    output_dtype = _dtype(
        arguments.get("output_dtype")
        or _optional_tensor_dtype(
            arguments, ("output", "out"), ACLNN_ATTENTION_QUERY_DTYPES,
            "aclnn_attention_output_dtype_unsupported")
        or query_dtype,
        ACLNN_ATTENTION_QUERY_DTYPES, "aclnn_attention_output_dtype_unsupported")
    query_bytes = batch * heads * query_tokens * head_dim * dtype_bytes(query_dtype)
    key_bytes = batch * kv_heads * key_tokens * head_dim * dtype_bytes(key_dtype)
    value_bytes = batch * kv_heads * key_tokens * v_head_dim * dtype_bytes(value_dtype)
    kv_bytes = key_bytes + value_bytes
    output_bytes = batch * heads * query_tokens * v_head_dim * dtype_bytes(output_dtype)
    flops = float(FLOPS_PER_FMA * batch * heads * pairs * (head_dim + v_head_dim))
    vector_ops = float(ATTENTION_SOFTMAX_OPS_PER_PAIR * batch * heads * pairs)
    view = {
        "kind": "paged_gqa" if paged else "dense_gqa",
        "logical_layout": "TND",
        "kv_storage": "paged" if paged else "contiguous",
        "query_shape": [batch, heads, query_tokens, head_dim],
        "key_shape": [batch, kv_heads, key_tokens, head_dim],
        "value_shape": [batch, kv_heads, key_tokens, v_head_dim],
        "output_shape": [batch, heads, query_tokens, v_head_dim],
        "query_dtype": query_dtype, "key_dtype": key_dtype,
        "value_dtype": value_dtype, "output_dtype": output_dtype,
        "key_tail_shape": [block_size or key_tokens, kv_heads * head_dim],
        "value_tail_shape": [block_size or key_tokens, kv_heads * v_head_dim],
        "batch_count": batch, "causal": causal,
    }
    if paged:
        view["page_size"] = block_size
    metadata = (
        ("api", symbol), ("aclnn_symbol", symbol),
        ("aclnn_abi_schema", ACLNN_ABI_SCHEMA),
        ("aclnn_call_kind", ACLNN_ATTENTION_KIND),
        ("attention_tensor_view", view),
        ("attention_geometry", {
            "batch": batch, "heads": heads, "kv_heads": kv_heads,
            "query_tokens": query_tokens, "key_tokens": key_tokens,
            "head_dim": head_dim, "v_head_dim": v_head_dim,
            "causal": causal, "paged": paged, "block_size": block_size or None,
            "sparse_mode": sparse_mode, "top_left_causal": top_left_causal,
            "query_dtype": query_dtype, "key_dtype": key_dtype,
            "value_dtype": value_dtype, "output_dtype": output_dtype}),
    )
    return OpSpec(
        op_id="aclnn_call", family="attention", phase=_phase(arguments.get("phase")),
        stage=0, layer=0, requests=(),
        semantic_shape=(batch, heads, query_tokens, head_dim),
        local_shape=(batch, heads, query_tokens, head_dim),
        padded_shape=(batch, heads, query_tokens, head_dim),
        launch_shape=(batch, heads, query_tokens, head_dim),
        flops=flops, vector_ops=vector_ops,
        hbm_bytes=query_bytes + kv_bytes + output_bytes,
        mte2_bytes=query_bytes + kv_bytes, mte3_bytes=output_bytes,
        dtype=query_dtype, recipe=recipe, metadata=metadata)
