"""Installed CLI checks for typed caller representations and source programs.

[Role] Exercise selected public frontend paths in the wheel verifier's venv.
[Invariants] Complete selected layers retain every TP/DP/EP participant. Checks
  establish executable caller contracts, not measured performance accuracy.
[Timeouts & Locks] Each CLI call uses the parent's independent 60 second budget.
[Side-Effect Policy] Write temporary inputs only inside the verifier directory.
"""
import json


def verify_source_callers(run, root, work):
    folder = root / 'examples/cases/high-concurrency-20261030'

    def graph(path, label, tokens=1, sequences=1, layers=1):
        raw = json.loads(path.read_text())
        raw['model']['layers'] = layers
        if 'pp_layer_partition' in raw['runtime']:
            raw['runtime']['pp_layer_partition'] = [layers]
        selected = work / (label + '.json')
        selected.write_text(json.dumps(raw) + '\n')
        result = run(['graph', selected, '--scope', 'body', '--phase', 'decode',
                      '--tokens', tokens, '--sequences', sequences, '--context', 2048,
                      '--input-ready-ms', 7], label)
        assert result['duration_ms'] > 0
        assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
        assert not any(result['execution']['live_allocated'].values())
        assert all(row['ready_ms'] >= 7 for row in result['timeline'])
        return [row['operator'] for row in result['isolated']['operators']]

    ops = graph(folder / 'qwen-grouped-int32-conditional/'
                'qwen35-35b-a5-tp4-c4-grouped-int32.json',
                'graph-source-grouped-int32-caller', tokens=4, sequences=4)
    gate = next(o for o in ops if o['recipe'] == 'balanced_grouped_gate_up')
    activation = next(o for o in ops if o['recipe'] == 'grouped_int32_dequant_swiglu')
    assert gate['metadata']['output_format'] == 'int32'
    assert 'matrix_epilogue' not in gate['metadata']
    assert activation['source_boundary'] == 'kernel'
    assert activation['metadata']['input_row_scale_producer']
    assert activation['metadata']['activation_scale_format'] == 'fp32'

    ops = graph(folder / 'v41-mc2-mx32-conditional/'
                'a5_v41_fp8_tp1_dp8_ep8_tpot5ms_c1-mx32-fullmesh-source.json',
                'graph-source-mc2-mx32-producer')
    gate = next(o for o in ops if o['recipe'] == 'balanced_grouped_gate_up')
    assert gate['metadata']['activation_producer'] == 'explicit_transport_quantization'
    assert gate['metadata']['activation_scale_producer'] == gate['metadata']['activation_payload_producer']
    assert not any(o['op_id'] == gate['op_id'] + '.quantize' for o in ops)
    phases = [o['metadata'].get('mc2_phase', {}) for o in ops]
    assert any(p.get('dispatch_quant') == 'mx' and p.get('payload_dtype') == 'fp8_e4m3'
               for p in phases)

    ops = graph(folder / 'v41-mx32-conditional/'
                'a5_v41_fp8_tp1_dp8_ep8_tpot5ms_c1-prepared-mx32-source.json',
                'graph-source-prepared-mx32-matrix')
    matrices = [o for o in ops if o['recipe'] == 'prepared_mx32_aic_matrix']
    assert len(matrices) == 24  # Three projections on every DP8 participant.
    assert all(o['layout'] == 'fp8_block1x32_fp8_e8m0' for o in matrices)
    assert all(o['mte3_bytes'] == 0 and o['metadata']['fixpipe_bytes'] > 0 for o in matrices)

    ops = graph(root / 'examples/source-families/v41-hc-arch35-source-control.json',
                'graph-source-hc-arch35-mixed-and-vector')
    hc = [o for o in ops if o['metadata'].get('fusion_role') in {'hc_pre', 'hc_post'}]
    assert {o['metadata']['fusion_role'] for o in hc} == {'hc_pre', 'hc_post'}
    for op in hc:
        program = op['metadata']['program_execution']
        assert program['profile'] in {'hc_pre_arch35_mk_175ef4d4', 'hc_post_arch35_bf16_175ef4d4'}
        assert 'source_three_output_HCPre_does_not_cover_extra_B4_input_output' in program['limitations']


    for tokens, sequences in ((1, 1), (4, 4)):
        ops = graph(root / 'examples/source-families/glm-routing-state-control.json',
                    f'graph-source-glm-state-routing-q{tokens}-b{sequences}',
                    tokens=tokens, sequences=sequences, layers=4)
        copies = [op for op in ops if op['recipe'] == 'triton_noncontiguous_conv_state_copy']
        assert {op['metadata']['direction'] for op in copies} == {'gather', 'scatter'}
        assert all(op['local_shape'][:2] == [sequences, 6] for op in copies)
        routing = next(op for op in ops if op['recipe'] == 'ascend_moe_init_routing_v3_mxfp8')
        gate = next(op for op in ops if op['recipe'] == 'balanced_grouped_gate_up')
        assert routing['source_boundary'] == 'kernel' and routing['hbm_bytes'] > 0
        assert all(op['source_boundary'] == 'kernel' and op['hbm_bytes'] > 0 for op in copies)
        assert gate['metadata']['activation_producer'] == 'explicit_routing_quantization'
        assert gate['metadata']['activation_payload_producer'] == routing['op_id']
        assert gate['metadata']['activation_scale_producer'] == routing['op_id']
        assert not any(op['op_id'] == gate['op_id'] + '.quantize' for op in ops)

    _verify_hccl_callers(run, work)
    _verify_mc2_tp_int8_caller(run, root, work)


def _hccl_case(collective, expansion, template):
    """Small ordinary transformer; all resource/dispatch values are declarations."""
    model = {'name': 'hccl-installed-control', 'layers': 1, 'hidden_size': 1680,
             'num_heads': 4, 'num_kv_heads': 2, 'head_dim': 8,
             'intermediate_size': 64, 'vocab_size': 128}
    binding = {'expansion': expansion, 'template': template}
    kernels = {'hccl': {collective: binding}}
    if expansion == 'AIV':
        binding['block_limit'] = 64
    if expansion.startswith('CCU_'):
        # This one-shot source domain is one <=4096-byte reduction fragment.
        model['hidden_size'] = 256
    if collective in {'allgather', 'reduce_scatter'}:
        # The ordinary SP caller emits both AG and RS. Select only the target
        # HCCL family; the other retains its existing analytical program.
        model.update(ffn='moe', num_experts=4, top_k=2, moe_intermediate_size=32,
                     layer_plans=[{'index': 0, 'attention': 'gqa', 'ffn': 'dense',
                         'num_heads': 4, 'num_kv_heads': 2, 'head_dim': 8,
                         'intermediate_size': 64, 'residual_block_size': 2}])
        kernels['sequence_parallel'] = {
            'implementation': 'vllm_ascend_kimi_sequence_parallel',
            'enable_expert_parallel': True, 'padding': 'none',
            'attention_output': 'separate_reduce_scatter',
            'shard_materialization': 'view', 'shared_expert_layout': 'tp_sharded'}
    return {
        'machine': {'backend': 'ascend', 'sku': 'Ascend950DT_9582', 'num_devices': 2,
                    'device': {'launch_us': 1, 'rates': {'hccl_copy': 1e12,
                        'hccl_reduce': 1e12, 'hccl_control': 1e9,
                        **({'ccu_copy': 1e12, 'ccu_buffer_bytes': 1e12,
                            'ccu_reduce_elements': 1e12, 'ccu_control': 1e9}
                           if expansion.startswith('CCU_') else {})},
                        'resource_units': {'hccl_copy': 1, 'hccl_reduce': 1, 'hccl_control': 1,
                            **({role: 1 for role in ('ccu_copy', 'ccu_buffer_bytes',
                                'ccu_reduce_elements', 'ccu_control')}
                               if expansion.startswith('CCU_') else {})}}},
        'model': model,
        'runtime': {'tp': 2, 'dp': 1, 'ep': 2 if 'sequence_parallel' in kernels else 1,
                    'pp': 1, 'graph_mode': 'none', 'kernels': kernels},
        'workload': {'requests': [{'id': 'control', 'prompt_tokens': 1, 'output_tokens': 1}]},
    }


def _verify_hccl_callers(run, work):
    for collective, expansion, template in (
        ('allreduce', 'AICPU_TS', 'mesh_oneshot'),
        ('allgather', 'AICPU_TS', 'mesh'),
        ('reduce_scatter', 'AICPU_TS', 'mesh'),
        ('allreduce', 'AIV', 'mesh_oneshot'),
        ('allreduce', 'AIV', 'mesh_twoshot'),
        ('allgather', 'AIV', 'mesh'),
        ('allreduce', 'AICPU_TS', 'nhr'),
        ('allgather', 'AICPU_TS', 'nhr'),
        ('reduce_scatter', 'AICPU_TS', 'nhr'),
        ('allreduce', 'CCU_MS', 'mesh_oneshot_v1'),
        ('allreduce', 'CCU_MS', 'mesh_oneshot_v2'),
        ('allgather', 'CCU_SCHED', 'mesh_mem2mem_v1'),
        ('allgather', 'CCU_SCHED', 'mesh_mem2mem_v2'),
    ):
        label = f'graph-source-hccl-{expansion.lower()}-{collective}-{template}'
        selected = work / (label + '.json')
        selected.write_text(json.dumps(_hccl_case(collective, expansion, template)) + '\n')
        report = run(['graph', selected, '--scope', 'body', '--phase', 'decode',
                      '--tokens', 1, '--sequences', 2, '--context', 16,
                      '--input-ready-ms', 7, '--include-submission', '--replay-scope', 'none',
                      '--event-limit', 5000], label)
        assert report['duration_ms'] > 0
        execution = report['execution']
        assert execution['live_tasks'] == execution['live_programs'] == 0
        assert not any(execution['live_allocated'].values())
        assert all(row['ready_ms'] >= 7 for row in report['timeline'])
        assert all(row['start_ms'] >= 7 for row in report['events'] if row['demand'])
        operators = [row['operator'] for row in report['isolated']['operators']]
        local = [op for op in operators if op['family'] == 'hccl_local']
        assert local and all(op['metadata']['hccl_selection']['collective'] == collective
                             and op['metadata']['hccl_selection']['expansion'] == expansion
                             for op in local)
        assert all(op['metadata']['program_execution']['hccl_selection']['template'] == template
                   for op in local)
        assert sum(op['hbm_bytes'] for op in local) > 0
        if expansion == 'AICPU_TS' or expansion.startswith('CCU_'):
            assert all(not op['metadata']['program_execution']['unpriced_service_work'] for op in local)
        entries = [op['metadata']['device_kernel_group'] for op in operators
                   if op['metadata'].get('device_kernel_group', {}).get('entry')]
        exits = [op['metadata']['device_kernel_group'] for op in operators
                 if op['metadata'].get('device_kernel_group', {}).get('exit')]
        assert entries and {entry['id'] for entry in entries} == {exit['id'] for exit in exits}
        retired = [event for event in report['events']
                   if ':kernel:' in event['task'] and event['task'].endswith(':retire')]
        assert len(retired) == len(entries)
        assert execution['submission']['eager_invocations'] >= len(entries)
        assert any(value > 0 for name, value in execution['serviced'].items()
                   if name.startswith('link:'))
        if expansion == 'AIV':
            assert all(entry['buffers'][0]['id'] == 'hccl_aiv_copy_queue' for entry in entries)
            assert execution['peak_allocated']['ub@r0:s0'] >= 190*1024+2560


def _verify_mc2_tp_int8_caller(run, root, work):
    """One actual Qwen layer retains GDN, W8A8 and the INT32 grouped caller."""
    path = root / ('examples/cases/high-concurrency-20261030/qwen35-hccl-conditional/'
                   'qwen35-35b-a5-tp4-c4-hccl-source.json')
    raw = json.loads(path.read_text())
    raw['model']['layers'] = 1
    if 'pp_layer_partition' in raw['runtime']:
        raw['runtime']['pp_layer_partition'] = [1]
    raw['runtime']['kernels'].update(
        native_moe_protocol={'kind': 'arch35_fullmesh', 'max_inflight_per_rank': 4,
                             'dispatch_quant': 'per_token_int8'},
        moe_token_partition={'kind': 'vllm_ascend_mc2_tp_slice'})
    selected = work / 'mc2-tp-int8-grouped-caller.json'
    selected.write_text(json.dumps(raw) + '\n')
    result = run(['graph', selected, '--scope', 'body', '--phase', 'decode',
                  '--tokens', 4, '--sequences', 4, '--context', 128,
                  '--input-ready-ms', 7], 'graph-source-mc2-tp-int8-grouped-caller')
    assert result['duration_ms'] > 0
    execution = result['execution']
    assert execution['live_tasks'] == execution['live_programs'] == 0
    assert not any(execution['live_allocated'].values())
    assert all(row['ready_ms'] >= 7 for row in result['timeline'])
    ops = {row['operator']['op_id']: row['operator'] for row in result['isolated']['operators']}
    assert ops['s0.l0.router']['local_shape'][0] == 16
    assert ops['s0.l0.topk']['local_shape'][0] == 4
    dispatch, gate, activation, down = (ops['s0.l0.' + name]
        for name in ('dispatch', 'expert_up', 'expert_act', 'expert_down'))
    assert dispatch['metadata']['mc2_phase']['source_tokens'] == 4
    assert dispatch['metadata']['mc2_phase']['received_assignments'] == 32
    assert 'activation_scales' in dispatch['returns']
    assert dispatch['metadata']['activation_scale_shape'] == [32]
    assert dispatch['metadata']['activation_scale_bytes'] == 128
    assert dispatch['metadata']['activation_scale_format'] == 'fp32'
    assert gate['metadata']['activation_payload_producer'] == dispatch['op_id']
    assert gate['metadata']['output_format'] == 'int32'
    assert gate['op_id'] + '.quantize' not in ops
    assert activation['recipe'] == 'grouped_int32_dequant_swiglu'
    assert activation['metadata']['input_row_scale_producer'] == dispatch['op_id']
    assert activation['metadata']['output_format'] == 'int8'
    assert down['metadata']['activation_scale_producer'] == activation['op_id']
    assert down['metadata']['output_format'] == 'bf16'
    gather = ops['s0.l0.mc2_tp_gather']
    assert gather['metadata']['hccl_selection']['payload_bytes'] == 16384
    assert gather['metadata']['hccl_selection']['collective'] == 'allgather'
    timeline = {row['op_id']: row for row in result['timeline']}
    assert timeline['s0.l0.expert_up']['ready_ms'] >= timeline['s0.l0.dispatch']['complete_ms']
    assert timeline['s0.l0.expert_act']['ready_ms'] >= max(
        timeline['s0.l0.dispatch']['complete_ms'], timeline['s0.l0.expert_up']['complete_ms'])
    assert timeline['s0.l0.mc2_tp_gather.hccl.entry']['ready_ms'] >= timeline['s0.l0.unpack']['complete_ms']
    serving = run(['predict', selected, '--scope', 'body', '--phase', 'decode',
                   '--tokens', 4, '--sequences', 4, '--context', 128],
                  'predict-body-source-mc2-tp-int8-grouped-caller')
    assert serving['status'] == 'modeled' and serving['body_forward_ms'] > 0
    assert serving['execution']['live_tasks'] == serving['execution']['live_programs'] == 0
    assert not any(serving['execution']['live_allocated'].values())
    assert any(row['op_id'] == 's0.l0.mc2_tp_gather' for row in serving['timeline'])
