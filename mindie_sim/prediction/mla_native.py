"""Source-backed native views and precision boundaries for latent attention.

[Role] Keep batched matrices and the indexer's FP32 LayerNorm explicit.
[Invariants] Transpose views add no copy; casts carry real input/output traffic.
  Native correspondence does not select a runtime kernel or prove an occurrence.
[Timeouts & Locks] Pure finite operator construction; no I/O or locks.
[Side-Effect Policy] Appends operators only to the caller-owned builder.
"""
from mindie_sim.schema.common import dtype_bytes


LAYER_NORM_SOURCE = {
    "repository": "vllm",
    "base_revision": "58d3918e3ea0a544ffedadad2ba84559e9c51d8f",
    "source_state": "clean_at_revision",
    "path": "vllm/model_executor/layers/layernorm.py",
    "sha256": "3460eba01758849027bb59b8592626406e33ab25b23d2def729bb0ec42afce67",
    "symbol": "LayerNorm.forward", "lines": [310, 325],
    "scope": "independent_source_semantics; not_capture_binary_identity",
}
BATCHED_MATRIX_SOURCE = {
    "repository": "vllm-ascend",
    "base_revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "source_state": "clean_at_revision",
    "path": "vllm_ascend/attention/sfa_v1.py",
    "sha256": "c4ea3f33ba6c71d9a9163dc397c12d92dba67aa26c292ab1ce52605b3c91ad42",
    "symbol": "_q_proj_and_k_up_proj/_v_up_proj", "lines": [917, 948],
    "scope": "independent_source_semantics; not_capture_binary_identity",
}
GROUPED_OUTPUT_SOURCE = {
    "repository": "vllm-ascend",
    "base_revision": "ae923905948f360f4b7ab06df52d0789a0c9316e",
    "source_state": "clean_at_revision",
    "path": "vllm_ascend/attention/dsa_v1.py",
    "sha256": "3fb4e94672e6adb931cb2e2a164201000822fe89250fd3bfd3585ba397acf52e",
    "symbol": "npu_transpose_batchmatmul", "lines": [1530, 1572],
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


def index_layer_norm(builder, *, source, rows, width):
    """LayerNorm.forward normalizes FP32 and casts back to its input dtype."""
    dtype = builder.model.activation_dtype
    elements = rows * width
    ready = source
    if dtype != "fp32":
        ready = builder.vector(
            "index_norm_input_fp32", (source,), elements, 1, family="cast",
            dtype="fp32", shape=(rows, width),
            read_bytes_override=elements * dtype_bytes(dtype),
            write_bytes_override=elements * 4,
            metadata=(("input_format", dtype), ("output_format", "fp32"),
                      ("native_source", LAYER_NORM_SOURCE)))
    ready = builder.vector(
        "index_norm", (ready,), elements, 8, family="layernorm",
        dtype="fp32", shape=(rows, width),
        read_bytes_override=(rows + 2) * width * 4,
        write_bytes_override=elements * 4,
        metadata=(("reduction_width", width), ("norm_has_weight", True),
                  ("norm_has_bias", True), ("norm_eps", 1e-6),
                  ("norm_parameter_dtype", "fp32"), ("native_source", LAYER_NORM_SOURCE)))
    if dtype != "fp32":
        ready = builder.vector(
            "index_norm_output_cast", (ready,), elements, 1, family="cast",
            dtype=dtype, shape=(rows, width),
            read_bytes_override=elements * 4,
            write_bytes_override=elements * dtype_bytes(dtype),
            metadata=(("input_format", "fp32"), ("output_format", dtype),
                      ("native_source", LAYER_NORM_SOURCE)))
    return ready
