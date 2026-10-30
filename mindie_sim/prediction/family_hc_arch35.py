"""Explicit arch35 HC source profiles, separate from the legacy A3 candidates.

[Role] Bind the ordinary HC semantic groups to source-owned physical programs.
[Invariants] Profiles replace all member work exactly once. Source ABI has
  three HCPre outputs; the captured extra [B,4] input/output is not established.
  No timing parameter or measured residual participates in the planner.
[Timeouts & Locks] Four immutable geometries per source template cache; no I/O.
[Side-Effect Policy] Returns immutable source plans without editing a case.
"""
from dataclasses import replace
from functools import lru_cache
from types import MappingProxyType

from mindie_sim.schema.common import ModelingError
from .mechanism import OperatorPlan, Work
from .program import KernelProgram, ParallelProgram, Stage
from .hc_arch35_program import (
    allocations, phase_program, post_geometry, post_owner, pre_geometry,
    pre_owner, pre_program, pre_second_owner, total_work,
)

PRE_PROFILE = 'hc_pre_arch35_mk_175ef4d4'
COMPACT_PRE_PROFILE = 'hc_pre_arch35_mk_compact_175ef4d4'
POST_PROFILE = 'hc_post_arch35_bf16_175ef4d4'
SOURCE = {
    'repository': 'vllm-ascend',
    'revision': '175ef4d49c2072f9233619636f0309f4f88b4e2c',
    'pre_entry': {'path': 'csrc/moe/hc_pre/op_kernel/hc_pre.cpp',
        'sha256': '284443c5d0fdd7e9b1abad9ec581a4cb2eafd060c490866955a70f646dccd32c',
        'lines': '49-74'},
    'pre_host': {'path': 'csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h',
        'sha256': 'fd624ad5d2d1daa96a87ab57b040bceb45c2b0d176cc7276a62baba150e424ea',
        'lines': '295-429'},
    'pre_cube': {'path': 'csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute_arch35.h',
        'sha256': 'cf197ddead3807f57c4791e173eee8de4c1116c19bea6a17ddae12b40d9a2b3b',
        'lines': '26-32,65-190,216-233'},
    'post_entry': {'path': 'csrc/moe/hc_post/op_kernel/hc_post.cpp',
        'sha256': 'c735e1e4534bc2d701fbd06636e42082567c81ea3f0ed24e4e10b9a71867eda4',
        'lines': '39-59'},
    'pre_core': {'path': 'csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core_arch35.h',
        'sha256': '2077956c339dce8090ee39869baba03476848295e7659d9abade63f26b638481', 'lines': '36-65,80-231,276-389'},
    'pre_vector': {'path': 'csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h',
        'sha256': '48d81b71d8db2b5bbc789a9c7005b8566384143fa0d5c4810cd2c6873c587f79', 'lines': '134-203,237-317,431-553,591-703,879-1075'},
    'post_host': {'path': 'csrc/moe/hc_post/op_host/hc_post_tiling_arch35.h',
        'sha256': '659f6f0e4d3126a88580202de3fc40e17d6ae6662a4783cd67247e6a3c41ff01', 'lines': '34,184-203'},
    'post_bf16': {'path': 'csrc/moe/hc_post/op_kernel/hc_post_bfloat16.h',
        'sha256': '72559bcd3201153c00b4493818ddef751104e964cce8691bffb52368b62b5a50', 'lines': '96-119,210-362'},
}
LIMITATIONS = (
    'source_capture_binary_identity_not_established',
    'source_three_output_HCPre_does_not_cover_extra_B4_input_output',
    'vector_elementary_arithmetic_and_transpose_envelope_not_instruction_issue_model',
    'scalar_GetValue_payload_only_no_scalar_latency_declaration',
    'MTE1_base_pipe_bytes_no_directional_instruction_rate_assumption',
    'source_control_flags_zero_latency_readiness_dependencies',
    'pre_stage2_row_batches_conservative_serial_completion',
)


def _frozen_work(work):
    return Work(MappingProxyType(dict(work.matrix)), MappingProxyType(dict(work.resources)))


def _frozen_groups(groups):
    return tuple(tuple(replace(phase, work=_frozen_work(phase.work)) for phase in phases)
                 for phases in groups)


@lru_cache(maxsize=4)
def _pre_template(g, iterations):
    # Future layers share source IR, never bound programs, rates or Engine events.
    # Capacity checks still run before this cache lookup on each physical device.
    first = _frozen_groups(pre_owner(g, owner) for owner in range(g.k_parts))
    second = _frozen_groups(pre_second_owner(g, owner, iterations)
                           for owner in range((g.rows + g.row_former - 1) // g.row_former))
    return first, second, _frozen_work(total_work((*first, *second)))


@lru_cache(maxsize=4)
def _post_template(rows, hidden, units):
    used = min(rows, units)
    former, rem = divmod(rows, used)
    groups = _frozen_groups(post_owner({'hidden': hidden}, owner,
                                     former + (owner < rem), units)
                            for owner in range(used))
    return groups, _frozen_work(total_work(groups))


def selected_plan(op, geometry):
    meta = dict(op.metadata)
    profile = meta.get('fusion_boundary', {}).get('traffic_profile')
    if profile not in {PRE_PROFILE, COMPACT_PRE_PROFILE, POST_PROFILE}:
        return None
    view = meta['fusion_tensor_view']
    roles = ('vector', 'mte2_aiv', 'mte3')
    if profile in {PRE_PROFILE, COMPACT_PRE_PROFILE}:
        roles += ('mte1', 'mte2_aic', 'fixpipe')
    for role in roles:
        if role not in geometry.resource_units:
            raise ModelingError(f'hc_arch35_requires_declared_pipe:{role}')
    # Leave mixed Pre unclassified so the existing hardware policy selects
    # max(Cube, Vector) once from real work, rather than inventing a class name.
    diagnostic = {'source': SOURCE, 'profile': profile,
                  'source_execution_domain': 'mixed' if profile != POST_PROFILE else 'vector',
                  'limitations': LIMITATIONS}
    if profile in {PRE_PROFILE, COMPACT_PRE_PROFILE}:
        exposed = set(meta['fusion_external_outputs'])
        for name, phase in zip(meta['fusion_semantic_ids'], meta['fusion_phases']):
            if name in exposed and phase['recipe'] not in {'hc_sinkhorn', 'hc_pre_mix'}:
                raise ModelingError('hc_arch35_cannot_export_internal_intermediate')
        g = pre_geometry(view, geometry)
        first, second, work = _pre_template(g, view['sinkhorn_iters'])
        from .hc_arch35_compact import eligible
        compact=profile==COMPACT_PRE_PROFILE and eligible(g)
        diagnostic.update(geometry=vars(g), logical_projection_flops=2 * g.rows * g.width * 24,
                          physical_projection_flops=2 * g.m_l1 * g.width * 32,
                          source_workspace_bytes=g.rows * g.k_parts * 25 * 4,
                          compact_pipeline={
                              'requested':profile==COMPACT_PRE_PROFILE,
                              'applied':compact,
                              'domain':'one_L1_panel_per_Cube_owner',
                              'outside_domain':'original_detailed_DAG',
                              'contention':'coupled_fluid_interior; original_prologue_RMS_projection_outputs',
                          })
        return OperatorPlan(work,
            lambda context: pre_program(g, first, second, context,compact=compact), diagnostic)
    size = post_geometry(view, geometry)
    units = geometry.resource_units['vector']
    used = min(view['tokens'], units)
    groups, work = _post_template(view['tokens'], view['hidden'], units)
    def program(context):
        buffers = allocations(context, 'post', {'ub': size}, range(used), units)
        return KernelProgram((Stage('post_body', acquire=buffers, free=buffers,
            program=ParallelProgram(tuple(phase_program(p, context) for p in groups))),))
    diagnostic.update(active_owners=used, ub_per_owner=size, d_tile=4096)
    return OperatorPlan(work, program, diagnostic)
