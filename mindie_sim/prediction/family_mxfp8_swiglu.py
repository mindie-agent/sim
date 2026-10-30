"""Explicit complete native MXFP8 gate/up boundary from ordinary MoE groups.

[Role] Bind existing matrix/activation/quantizer producer contracts to one plan.
[Invariants] The expert down consumer remains outside; no intermediate output
  may escape a fusion that keeps it in UB. User selection is explicit.
[Timeouts & Locks] Bounded member/metadata inspection; no locks or I/O.
[Side-Effect Policy] Immutable declarations only; no source or capture reads.
"""
from mindie_sim.schema.common import ModelingError
from .mxfp8_swiglu import PROFILE, RECIPE, ROLE, SOURCE


def bind(declaration, members, outputs, selection, options):
    view=declaration['tensor_view']
    if (declaration['role'] != ROLE or
            any(view.get(key) != 'fp8_e4m3' for key in ('input_dtype','weight_dtype','output_dtype'))
            or any(view.get(key) != 'fp8_e8m0' for key in ('input_scale_dtype','output_scale_dtype'))
            or any(view.get(key) != 32 for key in ('weight_scale_group_size','input_scale_group_size','output_scale_group_size'))):
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
    if (not rows or sum(rows) != matrix.local_shape[-3] or len(rows) != view['local_experts']
            or matrix.local_shape[-2:] != (2*width,k)
            or md.get('weight_representation') != 'fp8_block1x32_fp8_e8m0'
            or md.get('dequant_before_compute') or md.get('matrix_epilogue') or md.get('bias')
            or matrix.op_id not in activation.deps or activation.op_id not in quant.deps
            or set(outputs) != {quant.op_id}):
        raise ModelingError('mxfp8_swiglu_requires_complete_integer_grouped_chain_no_external_intermediate')
    if set(options)-{'l1_k','compact'} or type(options.get('l1_k')) is not int:
        raise ModelingError('mxfp8_swiglu_requires_explicit_l1_k')
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
