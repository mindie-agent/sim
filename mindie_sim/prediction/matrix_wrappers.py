"""Explicit dense-to-single-bank MXFP8 wrapper declarations.

[Role] Preserve a source caller's one-group tensor view on an existing matrix.
[Invariants] The wrapper does not turn routed experts into dense GEMMs, infer
  graph dispatch or change arithmetic, traffic, dependencies or recipe selection.
  A selected projection is bias-free, two-dimensional and explicitly eager.
[Timeouts & Locks] One bounded graph traversal; no I/O or locks.
[Side-Effect Policy] Returns immutable metadata replacements only.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError


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
    if (meta.get("activation_group_size") != 32
            or meta.get("activation_scale_format") != "fp8_e8m0"
            or tuple(meta.get("activation_scale_shape", ())) != (rows, (width + 63) // 64, 2)
            or meta.get("weight_representation") != "fp8_block1x32_fp8_e8m0"):
        return "requires_group32_e8m0_scales"
    if meta.get("expected_rows", rows) != rows:
        return "requires_all_recorded_rows_active"
    if meta.get("graph_captured"):
        return "requires_eager_dispatch"
    return None


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
        meta.update({
            "matrix_mapping_domain": MAPPING_DOMAIN,
            "matrix_weight_bank_groups": 1,
            "matrix_wrapper": {
                "implementation": IMPLEMENTATION,
                "native_kind": "GroupedMatmul",
                "input_rank": 2,
                "graph_mode": "eager",
                "group_list_shape": (1,),
                "group_list_dtype": "int64",
                "group_list_values": (rows,),
                "group_type": 0,
                "split_item": 2,
                "source": SOURCE,
                "prediction_boundary": "existing_matrix_program",
            },
        })
        result.append(replace(op, metadata=tuple(meta.items())))
    missing = selected - seen
    # An idle DP participant still computes received experts, but its caller
    # has no local dense projections. Only the caller can establish this.
    if missing and local_tokens != 0:
        raise ModelingError("matrix_wrapper_projection_recipes_unmatched:" + ",".join(sorted(missing)))
    return tuple(result)
