"""Source-backed native views and precision boundaries for latent attention.

[Role] Keep batched matrices and the indexer's FP32 LayerNorm explicit.
[Invariants] Transpose views add no copy; casts carry real input/output traffic.
  Native correspondence does not select a runtime kernel or prove an occurrence.
[Timeouts & Locks] Pure finite operator construction; no I/O or locks.
[Side-Effect Policy] Appends operators only to the caller-owned builder.
"""
from mindie_sim.schema.common import dtype_bytes


LAYER_NORM_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/ops/layernorm.py",
    "source_sha256": "5d23daa6eba43ddcefcb7ef79a1a2894e3d38a90d9c02ddfe9120f88e8ef745d",
    "symbol": "LayerNorm.forward",
    "scope": "independent_source_semantics; not_capture_binary_identity",
}
BATCHED_MATRIX_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/attention/sfa_v1.py",
    "source_sha256": "058aeb723a4ef3cdb0e07a5fc0180ec53f8227f62d98710144ecffb9cc66678f",
    "symbol": "_q_proj_and_k_up_proj/_v_up_proj",
    "scope": "independent_source_semantics; not_capture_binary_identity",
}
GROUPED_OUTPUT_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/attention/dsa_v41.py",
    "source_sha256": "1da24c133d0fd473e367aea28419dabd7f42d5e90e4a9215c35e787c182cc195",
    "symbol": "npu_transpose_batchmatmul",
    "scope": "independent_source_semantics; not_capture_binary_identity",
}


def batched_matrix_metadata(*, tokens, heads, input_width, output_width, role):
    """Project head-wise matrix work without materializing transpose views."""
    return (
        ("native_tensor_view", {
            "kind": "transpose_batchmatmul",
            "batch": heads, "rows": tokens, "n": output_width, "k": input_width,
            "input_shape": (tokens, heads, input_width),
            "weight_shape": (heads, input_width, output_width),
            "output_shape": (tokens, heads, output_width),
            "input_permutation": (1, 0, 2), "output_permutation": (1, 0, 2),
            "permutation_scope": "logical_view; physical_native_layout_not_selected",
            "role": role,
        }),
        ("native_implementation", {
            "native_kind": "TransposeBatchMatMul", "selection": None,
            "conditions": {
                "scope": "independent_per_head_matrix_with_transposed_views",
                "permutation_does_not_materialize_copy": True,
                "native_dispatch_and_physical_input_order_not_selected": True,
            },
        }),
        ("native_source", GROUPED_OUTPUT_SOURCE if role == "grouped_output"
         else BATCHED_MATRIX_SOURCE),
    )


NATIVE_LAYER_NORM_KIND = "aclnnLayerNorm"
NATIVE_LAYER_NORM_OPS = 8
FP32_BYTES = int(dtype_bytes("fp32"))


def index_layer_norm(builder, *, source, rows, width, input_dtype=None,
                     output_dtype=None, native=True):
    """Normalize the indexer key with one declared producer.

    ``aclnnLayerNorm`` (``LayerNormV3``) consumes the activation dtype and
    writes it back while accumulating mean/variance in FP32 in UB; gamma and
    beta stay FP32 parameters.  No external cast operator exists on that path.
    The explicit ``raw_eager`` decomposition keeps the upstream PyTorch FP32
    wrapper as its own two casts.
    """
    dtype = input_dtype or builder.model.activation_dtype
    output_dtype = output_dtype or dtype
    elements = rows * width
    parameters = {"gamma": width * FP32_BYTES, "beta": width * FP32_BYTES}
    if native:
        return builder.vector(
            "index_norm", (source,), elements, NATIVE_LAYER_NORM_OPS,
            family="layernorm", dtype=dtype, shape=(rows, width),
            read_bytes_override=elements * dtype_bytes(dtype)
                                + sum(parameters.values()),
            write_bytes_override=elements * dtype_bytes(output_dtype),
            metadata=(("reduction_width", width), ("norm_has_weight", True),
                      ("norm_has_bias", True), ("norm_eps", 1e-6),
                      ("norm_parameter_dtype", "fp32"),
                      ("parameter_read_bytes", parameters),
                      ("norm_gamma_runtime_bytes", parameters["gamma"]),
                      ("output_format", output_dtype),
                      ("native_kind", NATIVE_LAYER_NORM_KIND),
                      ("zero_external_casts", True),
                      ("native_source", LAYER_NORM_SOURCE)))
    ready = source
    if dtype != "fp32":
        ready = builder.vector(
            "index_norm_input_fp32", (source,), elements, 1, family="cast",
            dtype="fp32", shape=(rows, width),
            read_bytes_override=elements * dtype_bytes(dtype),
            write_bytes_override=elements * FP32_BYTES,
            metadata=(("input_format", dtype), ("output_format", "fp32"),
                      ("native_source", LAYER_NORM_SOURCE)))
    ready = builder.vector(
        "index_norm", (ready,), elements, NATIVE_LAYER_NORM_OPS, family="layernorm",
        dtype="fp32", shape=(rows, width),
        read_bytes_override=(rows + 2) * width * FP32_BYTES,
        write_bytes_override=elements * FP32_BYTES,
        metadata=(("reduction_width", width), ("norm_has_weight", True),
                  ("norm_has_bias", True), ("norm_eps", 1e-6),
                  ("norm_parameter_dtype", "fp32"), ("native_source", LAYER_NORM_SOURCE)))
    if output_dtype != "fp32":
        ready = builder.vector(
            "index_norm_output_cast", (ready,), elements, 1, family="cast",
            dtype=output_dtype, shape=(rows, width),
            read_bytes_override=elements * FP32_BYTES,
            write_bytes_override=elements * dtype_bytes(output_dtype),
            metadata=(("input_format", "fp32"), ("output_format", output_dtype),
                      ("native_source", LAYER_NORM_SOURCE)))
    return ready
