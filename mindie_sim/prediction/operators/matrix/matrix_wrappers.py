"""Explicit dense-to-single-bank MXFP8 wrapper declarations.

[Role] Preserve a source caller's one-group tensor view on an existing matrix,
  its eager 1-element INT64 device group-list fill producer, and post-processed
  paired E8M0 weight-scale storage.
[Invariants] The wrapper does not turn routed experts into dense GEMMs, infer
  graph dispatch or change arithmetic or recipe selection. A selected projection
  is bias-free, two-dimensional and explicitly eager. Checkpoint weight-scale
  extent remains in provenance separately from paired runtime scale storage.
[Timeouts & Locks] One bounded graph traversal; no I/O or locks.
[Side-Effect Policy] Returns immutable operators and metadata replacements only.
"""
from dataclasses import replace
from typing import Literal

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
_VLLM_ASCEND_REV = "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb"
_OP_PLUGIN_REV = "48189acf95882e23fae56ea96ceeb89358cb42da"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=_VLLM_ASCEND_REV,
        relative_path="vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py",
        source_sha256="f9d03f16ae93fb678da5ff0135848226b585960dc1abe45252719ce7eb29e65a",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=_VLLM_ASCEND_REV,
        relative_path="vllm_ascend/models/kimi_k3/kimi_k3.py",
        source_sha256="aa666ec743d3c8cd10d58196be9e05d0606aa4ca89d14a94f327a75c9fd44613",
    ),
    SourceRef(
        repository="Ascend/op-plugin",
        revision=_OP_PLUGIN_REV,
        relative_path="op_plugin/ops/opapi/GroupedMatmulKernelNpuOpApi.cpp",
        source_sha256="a58dd51addfa3511735fcf1f46355a4597f62920cc5d24edc335b59610128fdc",
    ),
    SourceRef(
        repository="Ascend/op-plugin",
        revision=_OP_PLUGIN_REV,
        relative_path="op_plugin/ops/opapi/FillKernelNpuOpApi.cpp",
        source_sha256="a514f43d59b0394ced4dae6af31c7b80b50e2171da2792aa500e6ad5f15683ba",
    ),
)

MX_SCALE_GROUP = 32
MX_SCALE_PAIR_LANES = 2
MX_SCALE_GROUP_PAIR = MX_SCALE_PAIR_LANES * MX_SCALE_GROUP
MX_SCALE_PAIR_ROUNDUP = MX_SCALE_GROUP_PAIR - 1
MX_SCALE_BYTES = int(dtype_bytes("fp8_e8m0"))
GROUP_LIST_DTYPE = "int64"
GROUP_LIST_BYTES = float(dtype_bytes(GROUP_LIST_DTYPE))
GROUP_LIST_FILL_RECIPE = "group_list_fill"
_FILL_PASSTHROUGH_META = (
    "graph_captured", "graph_segment", "structure_block", "structure_role",
)


IMPLEMENTATION = "vllm_ascend_mxfp8_single_group_eager"
MAPPING_DOMAIN = "single_group_dense_wrapper"
SOURCE = {
    "image_id": "sha256:883670fcc91cbd75a7070b00501be0ecd81a266b81366a21bf02139d8e6a33e1",
    "wrapper": {
        "path": "vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py",
        "sha256": "51ece928e77420a7f6600e2d87c46e9397a5140b6acee324aa75fdb97266d009",
        "lines": [167, 221],
    },
    "caller": {
        "path": "vllm_ascend/models/kimi_k3.py",
        "sha256": "be9adc792774f919f443caf483eb2e04e4453617465381f47a482dd4736ec63f",
        "lines": [120, 147],
    },
    "basis": "declared_source_wrapper_not_capture_dispatch_proof",
}


def recipes():
    """A tensor-view wrapper leaves the selected matrix program intact."""
    return {}


def _domain_issue(op, meta):
    if (op.family == "grouped_gemm" or meta.get("program_kind") != "matmul"
            or meta.get("matrix_groups", 1) != 1 or meta.get("matrix_mapping_domain")):
        return "requires_single_dense_matrix"
    if len(op.local_shape) != 3 or min(op.local_shape) <= 0:
        return "requires_positive_M_N_K"
    rows, _, width = op.local_shape
    if meta.get("bias") or meta.get("bias_shape"):
        return "requires_bias_free_projection"
    if (meta.get("activation_format"), meta.get("weight_format")) != ("fp8_e4m3", "fp8_e4m3"):
        return "requires_mxfp8_operands"
    if meta.get("output_format") not in {"bf16", "fp16"}:
        return "requires_float_output"
    if (meta.get("activation_group_size") != MX_SCALE_GROUP
            or meta.get("activation_scale_format") != "fp8_e8m0"
            or tuple(meta.get("activation_scale_shape", ())) != (
                rows, (width + MX_SCALE_PAIR_ROUNDUP) // MX_SCALE_GROUP_PAIR, MX_SCALE_PAIR_LANES)
            or meta.get("weight_representation") != "fp8_block1x32_fp8_e8m0"):
        return "requires_group32_e8m0_scales"
    if meta.get("expected_rows", rows) != rows:
        return "requires_all_recorded_rows_active"
    if meta.get("graph_captured"):
        return "requires_eager_dispatch"
    return None


def _group_list_fill_op(op: OpSpec, rows: int) -> OpSpec:
    """Emit the eager 1-element INT64 device ``torch.full`` group-list producer."""
    passthrough = tuple(
        (key, value) for key, value in op.metadata
        if key in _FILL_PASSTHROUGH_META
    )
    return OpSpec(
        op_id=f"{op.op_id}.group_list_fill",
        family="tensor_fill",
        phase=op.phase,
        stage=op.stage,
        layer=op.layer,
        requests=op.requests,
        deps=op.deps,
        semantic_shape=(1,),
        local_shape=(1,),
        padded_shape=(1,),
        launch_shape=(1,),
        dtype=GROUP_LIST_DTYPE,
        layout="flat",
        recipe=GROUP_LIST_FILL_RECIPE,
        hbm_bytes=GROUP_LIST_BYTES,
        mte2_bytes=0.0,
        mte3_bytes=GROUP_LIST_BYTES,
        vector_ops=1.0,
        stream=op.stream,
        metadata=(
            ("native_kind", "Fill"),
            ("aclnn_symbol", "aclnnInplaceFillScalar"),
            ("fill_value", int(rows)),
            ("output_format", GROUP_LIST_DTYPE),
            ("group_list_consumer", op.op_id),
            ("source", SOURCE),
            *passthrough,
        ),
    )


def _paired_weight_scale_storage(op: OpSpec, meta: dict):
    """Post-processed paired E8M0 weight-scale storage per ``w8a8_mxfp8.py:269-276``."""
    _, cols, width = op.local_shape
    scale_pairs = (width + MX_SCALE_PAIR_ROUNDUP) // MX_SCALE_GROUP_PAIR
    unpadded_groups = (width + MX_SCALE_GROUP - 1) // MX_SCALE_GROUP
    paired_shape = (1, scale_pairs, cols, MX_SCALE_PAIR_LANES)
    checkpoint_shape = tuple(meta.get("weight_scale_shape") or (cols, unpadded_groups))
    payload_bytes = float(meta.get("weight_payload_bytes", cols * width))
    checkpoint_packed = float(meta.get("packed_weight_bytes", meta.get("weight_bytes", payload_bytes)))
    checkpoint_scale_bytes = max(0.0, checkpoint_packed - payload_bytes)
    paired_scale_bytes = float(cols * scale_pairs * MX_SCALE_PAIR_LANES * MX_SCALE_BYTES)
    runtime_packed = payload_bytes + paired_scale_bytes
    delta = runtime_packed - checkpoint_packed
    return (
        payload_bytes,
        checkpoint_packed,
        checkpoint_scale_bytes,
        paired_scale_bytes,
        runtime_packed,
        delta,
        paired_shape,
        checkpoint_shape,
    )


def apply_matrix_wrapper(ops, selection, runtime, *, local_tokens=None):
    """Bind explicitly named projection recipes after representation lowering.

    ``projection_recipes`` declares the caller domain, not an inferred model
    name. Graph-policy annotations are added later in the ordinary frontend;
    therefore absence of ``graph_captured`` cannot establish eager execution.
    """
    if selection is None:
        return tuple(ops)
    if selection.name != IMPLEMENTATION:
        raise ModelingError("matrix_wrapper_implementation_unsupported:" + selection.name)
    selected = selection.options.get("projection_recipes", ())
    if isinstance(selected, str) or not selected:
        raise ModelingError("matrix_wrapper_requires_projection_recipes")
    selected = set(selected)
    # These builders represent the two routed transforms whose source caller
    # uses this wrapper. The same image uses one fused KDA QKV projection;
    # selecting the current separate Q/K/V recipes would invent three calls.
    if selected - {"latent_projection"}:
        raise ModelingError("matrix_wrapper_source_caller_unsupported:"
                            + ",".join(sorted(selected - {"latent_projection"})))
    if runtime.graph_mode.lower() not in {"none", "off", "eager", "disabled"}:
        raise ModelingError("matrix_wrapper_requires_explicit_eager_dispatch")
    result, seen = [], set()
    for op in ops:
        if op.recipe not in selected:
            result.append(op)
            continue
        seen.add(op.recipe)
        meta = dict(op.metadata)
        # Existing source programs own their native call/kernel representation.
        if op.plan_required or "implementation_binding" in meta or "matrix_wrapper" in meta:
            result.append(op)
            continue
        issue = _domain_issue(op, meta)
        if issue:
            raise ModelingError(f"matrix_wrapper_domain_miss:{op.op_id}:{issue}")
        rows = op.local_shape[0]
        fill_op = _group_list_fill_op(op, rows)
        (
            payload_bytes,
            checkpoint_packed,
            checkpoint_scale_bytes,
            paired_scale_bytes,
            runtime_packed,
            delta,
            paired_shape,
            checkpoint_shape,
        ) = _paired_weight_scale_storage(op, meta)
        # The 1-element INT64 group_list write is accounted on fill_op; do not fold
        # its scalar 8 B read into matrix mte2_bytes, which matrix_mapping splits
        # into tiled activation A bytes (L1->L0A MTE1).
        new_mte2 = float(op.mte2_bytes) + delta
        new_hbm = float(op.hbm_bytes) + delta
        if new_mte2 < 0.0 or new_hbm < 0.0:
            raise ModelingError(f"matrix_wrapper_negative_weight_scale_traffic:{op.op_id}")
        meta.update({
            "matrix_mapping_domain": MAPPING_DOMAIN,
            "matrix_weight_bank_groups": 1,
            "group_list_producer": fill_op.op_id,
            "weight_payload_bytes": payload_bytes,
            "packed_weight_bytes": runtime_packed,
            "weight_bytes": runtime_packed,
            "weight_scale_shape": paired_shape,
            "runtime_weight_scale_bytes": paired_scale_bytes,
            "checkpoint_weight_scale_shape": checkpoint_shape,
            "checkpoint_weight_scale_bytes": checkpoint_scale_bytes,
            "checkpoint_packed_weight_bytes": checkpoint_packed,
            "matrix_wrapper": {
                "implementation": IMPLEMENTATION,
                "native_kind": "GroupedMatmul",
                "input_rank": 2,
                "graph_mode": "eager",
                "group_list_shape": (1,),
                "group_list_dtype": GROUP_LIST_DTYPE,
                "group_list_values": (rows,),
                "group_list_producer": fill_op.op_id,
                "group_type": 0,
                "split_item": 2,
                "weight_scale_shape": paired_shape,
                "runtime_weight_scale_bytes": paired_scale_bytes,
                "checkpoint_weight_scale_shape": checkpoint_shape,
                "checkpoint_weight_scale_bytes": checkpoint_scale_bytes,
                "source_selection": "explicit_matrix_wrapper",
                "source_capability": "AscendW8A8MXFP8Dynamic",
                "source_role": "single_group_dense_wrapper",
                "source_basis": SOURCE["basis"],
                "source_identity": {
                    "wrapper_image_id": SOURCE["image_id"],
                    "current_executable_source": False,
                    "basis": SOURCE["basis"],
                },
                "matched_predicate": (
                    "explicit_matrix_wrapper + no_bias + two_dimensional + "
                    "fp8_e4m3_activation/weight + bf16_or_fp16_output + "
                    "group32_e8m0 + none/off/eager_graph + all_recorded_rows_active"),
                "proven_predicates": {
                    "implementation": IMPLEMENTATION,
                    "ascend_dynamic_mxfp8": True,
                    "two_dimensional": True,
                    "bias_free": True,
                    "active_forward_rows": rows,
                    "graph_mode": str(runtime.graph_mode),
                    "eager_dispatch": True,
                    "dtypes": ("fp8_e4m3", "fp8_e4m3", meta["output_format"]),
                },
                "source": SOURCE,
                "prediction_boundary": "existing_matrix_program",
            },
        })
        wrapped_op = replace(
            op,
            deps=tuple(dict.fromkeys((*op.deps, fill_op.op_id))),
            mte2_bytes=new_mte2,
            hbm_bytes=new_hbm,
            metadata=tuple(meta.items()),
        )
        result.extend((fill_op, wrapped_op))
    missing = selected - seen
    # An idle DP participant still computes received experts, but its caller
    # has no local dense projections. Only the caller can establish this.
    if missing and local_tokens != 0:
        raise ModelingError("matrix_wrapper_projection_recipes_unmatched:" + ",".join(sorted(missing)))
    return tuple(result)
