"""Explicit complete native MXFP8 gate/up boundary from ordinary MoE groups.

[Role] Bind existing matrix/activation/quantizer producer contracts to one plan.
[Invariants] The expert down consumer remains outside; no intermediate output
  may escape a fusion that keeps it in UB. User selection is explicit.
[Timeouts & Locks] Bounded member/metadata inspection; no locks or I/O.
[Side-Effect Policy] Immutable declarations only; no source or capture reads.
"""
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from .mxfp8_grouped import MX_SCALE_GROUP, MX_SCALE_GROUP_PAIR, MX_SCALE_PAIR_BYTES
from .mxfp8_swiglu import L1_K_GRANULE, OUTPUT_N_ALIGNMENT, PROFILE, RECIPE, ROLE, SOURCE

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="gmm/grouped_matmul_swiglu_quant_v2/op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_basic_tiling.cpp",
        source_sha256="85b86162c25c80dc626b643182a6bad3f11dab5d6ddc3ab6538ad30f813d36a0",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="gmm/common/cgmct/kernel/kernel_gmm_swiglu_mxquant.h",
        source_sha256="799f7caf064cb129177529d70210bc8cbc94e2d314bc08cb6b13924b2f13c54e",
    ),
)

CAPABILITIES = frozenset({
    "A5", "decode", "mxfp8", "mx32", "e4m3", "grouped_gate_up",
    "fused_swiglu_quant", "cgmct_source_program", "complete_region_only",
})


def capabilities():
    """Machine-readable fused-region boundary and typed attributes."""
    return {
        "selector": "native_fusion",
        "program_profile": PROFILE,
        "recipe": RECIPE,
        "role": ROLE,
        "tensor_view_dtypes": {
            "input": "fp8_e4m3", "weight": "fp8_e4m3",
            "output": "fp8_e4m3", "input_scale": "fp8_e8m0",
            "output_scale": "fp8_e8m0",
        },
        "scale_group_size": MX_SCALE_GROUP,
        "members": ("matmul", "swiglu", "quantize"),
        "options": ("l1_k", "compact"),
        "requires": (
            "exact_role", "complete_three_member_integer_grouped_chain",
            "no_external_intermediate", "explicit_l1_k_multiple_of_128",
            "expert_weight_scale_group32", "no_bias_or_epilogue",
        ),
        "source": SOURCE,
    }


def bind(declaration, members, outputs, selection, options):
    view = declaration.get('tensor_view')
    if not isinstance(view, dict):
        raise ModelingError('mxfp8_swiglu_requires_declared_tensor_view')
    for key in ('intermediate_width', 'input_width', 'local_experts'):
        if type(view.get(key)) is not int or view[key] < 1:
            raise ModelingError('mxfp8_swiglu_requires_positive_' + key)

    if (declaration['role'] != ROLE or
            any(view.get(key) != 'fp8_e4m3' for key in ('input_dtype','weight_dtype','output_dtype'))
            or any(view.get(key) != 'fp8_e8m0' for key in ('input_scale_dtype','output_scale_dtype'))
            or any(view.get(key) != MX_SCALE_GROUP for key in (
                'weight_scale_group_size', 'input_scale_group_size', 'output_scale_group_size'))):
        raise ModelingError('mxfp8_swiglu_requires_E4M3_MX32_input_weight_output')
    matrices=[op for op in members if dict(op.metadata).get('program_kind') == 'matmul']
    activations=[op for op in members if op.family == 'swiglu']
    quantizers=[op for op in members if op.family == 'quantize']
    if len(members) != 3 or len(matrices) != 1 or len(activations) != 1 or len(quantizers) != 1:
        raise ModelingError('mxfp8_swiglu_requires_gate_up_SwiGLU_quantizer')
    matrix,activation,quant=matrices[0],activations[0],quantizers[0]
    md=dict(matrix.metadata)
    rows=tuple(md.get('matrix_group_rows',()))
    width,k=view['intermediate_width'],view['input_width']
    if width % OUTPUT_N_ALIGNMENT or k % MX_SCALE_GROUP_PAIR:
        raise ModelingError('mxfp8_swiglu_requires_N64_K64_geometry')
    expected_in_tail = (k // MX_SCALE_GROUP_PAIR, MX_SCALE_PAIR_BYTES)
    expected_out_tail = (width // MX_SCALE_GROUP_PAIR, MX_SCALE_PAIR_BYTES)
    if (tuple(view.get('input_scale_tail', ())) != expected_in_tail
            or tuple(view.get('output_scale_tail', ())) != expected_out_tail):
        raise ModelingError('mxfp8_swiglu_requires_paired_e8m0_scales')
    if (not rows or sum(rows) != matrix.local_shape[-3] or len(rows) != view['local_experts']
            or matrix.local_shape[-2:] != (2*width,k)
            or md.get('weight_representation') != 'fp8_block1x32_fp8_e8m0'
            or md.get('dequant_before_compute') or md.get('matrix_epilogue') or md.get('bias')
            or matrix.op_id not in activation.deps or activation.op_id not in quant.deps
            or set(outputs) != {quant.op_id}):
        raise ModelingError('mxfp8_swiglu_requires_complete_integer_grouped_chain_no_external_intermediate')
    in_scale_shape = md.get('activation_scale_shape')
    if in_scale_shape is not None and tuple(in_scale_shape) != (sum(rows), *expected_in_tail):
        raise ModelingError('mxfp8_swiglu_requires_paired_e8m0_scales')
    w_scale_shape = md.get('weight_scale_shape')
    expected_w_scale_shape = (view['local_experts'], k // MX_SCALE_GROUP_PAIR, 2 * width, MX_SCALE_PAIR_BYTES)
    if w_scale_shape is not None and tuple(w_scale_shape) != expected_w_scale_shape:
        raise ModelingError('mxfp8_swiglu_requires_paired_e8m0_weight_scales')
    out_scale_shape = dict(quant.metadata).get('activation_scale_shape')
    if out_scale_shape is not None and tuple(out_scale_shape) != (sum(rows), *expected_out_tail):
        raise ModelingError('mxfp8_swiglu_requires_paired_e8m0_scales')
    if set(options)-{'l1_k','compact'} or type(options.get('l1_k')) is not int:
        raise ModelingError('mxfp8_swiglu_requires_explicit_l1_k')
    if options['l1_k'] < L1_K_GRANULE or options['l1_k'] % L1_K_GRANULE:
        raise ModelingError('mxfp8_swiglu_requires_explicit_l1_k_multiple_of_128')
    if 'compact' in options and type(options['compact']) is not bool:
        raise ModelingError('mxfp8_swiglu_compact_must_be_boolean')
    if (activation.dtype != 'bf16' or quant.dtype != 'bf16'
            or activation.local_shape != (sum(rows), width)
            or quant.local_shape != (sum(rows), width)):
        raise ModelingError('mxfp8_swiglu_requires_bf16_swiglu_input_and_mx_output_rows')
    config={'rows':rows,'width':width,'k':k,**options}
    return {
        'fusion_group_key':declaration['key'],
        'fusion_semantic_ids':[op.op_id for op in members],
        'fusion_semantic_recipes':[{'op_id':op.op_id,'family':op.family,'recipe':op.recipe,
                                   'local_shape':op.local_shape} for op in members],
        'fusion_external_outputs':outputs,'fusion_role':ROLE,'fusion_tensor_view':view,
        'fusion_program_profile':PROFILE,'mxfp8_swiglu_config':config,
        'fusion_boundary':{'implementation':selection.name,'native':declaration['implementation'],
            'source':declaration.get('source'),'program_profile':PROFILE,
            'submitted_kernels':1,'semantic_phase_count':len(members),
            'evidence':{'program_source':SOURCE,'capture_binary_identity':'not_established',
                        'routing_basis':'explicit_integer_rows'},
            'timing_basis':'source_AIC_two_products_to_FP32_UB_then_AIV_SwiGLU_MX',
            'traffic':{'profile':None,'limitations':['member_work_replaced_by_complete_source_plan']},
            'fidelity':{'kind':'source_phase_tile_program','native_dispatch_reconstructed':False,
                        'l1_k':'explicit_input','local_edges':'source_AIC_AIV_flag_dependencies'}},
        **{key:value for key,value in members[-1].metadata
           if key in {'structure_block','structure_role','graph_captured','graph_segment'}},
    }


__all__ = ["CAPABILITIES", "bind", "capabilities"]
