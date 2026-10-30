"""Installed CLI checks for owner services and explicit network protocols.

[Role] Exercise newly exported source services outside the checkout.
[Invariants] Controls use synthetic rates, never measured model residuals.
[Timeouts & Locks] Each CLI process uses the wheel verifier's 60 s budget.
[Side-Effect Policy] Writes only verifier temporary inputs.
"""
import json


def verify_whitebox_services(run, root, work):
    path = root / 'examples/communication/ccu-two-die-allgather-tp8.json'
    result = run(['graph', path, '--scope', 'model', '--phase', 'decode',
                  '--tokens', 1, '--context', 16], 'graph-ccu-two-endpoint-allgather')
    serviced = result['execution']['serviced']
    assert serviced['ccu_copy@r0:s0'] > 0
    assert any(':tx:port' in key and value for key, value in serviced.items())
    assert any(':rx:port' in key and value for key, value in serviced.items())
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    result = run(['predict', path], 'predict-ccu-two-endpoint-allgather')
    assert result['status'] == 'modeled'
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0

    raw = json.loads((work / 'mc2-tp-int8-dsq-arch35.json').read_text())
    raw['machine'].setdefault('device', {})['compute_arch'] = {'instruction_services': {
        'scalar_gm_access': {'unit': 'calls', 'owner_axis': 'vector', 'issue_ns': 1000,
                             'provenance': {'kind': 'synthetic_owner_supply'}}}}
    path = work / 'scalar-owner-service.json'
    path.write_text(json.dumps(raw) + '\n')
    result = run(['graph', path, '--scope', 'body', '--phase', 'decode', '--tokens', 4,
                  '--sequences', 4, '--context', 128], 'graph-owner-source-scalar-service')
    assert result['execution']['serviced']['scalar_gm_access@r0:s0'] > 0
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0

    raw = json.loads((root / 'examples/source-families/hccl-two-level-control.json').read_text())
    for link in raw['machine']['links']:
        link.update(semantics='per_port', switch_hops=2, switch_hop_latency_us=.5,
                    alignment_bytes=64, transaction_bytes=256, protocol_overhead_bytes=16)
    path = work / 'hccl-two-level-switched.json'
    path.write_text(json.dumps(raw) + '\n')
    result = run(['graph', path, '--scope', 'body', '--phase', 'decode', '--tokens', 2,
                  '--context', 16], 'graph-hccl-two-level-switch-and-signals')
    resources = result['execution']['serviced']
    assert any(':tx:port' in key and value for key, value in resources.items())
    assert any(':rx:port' in key and value for key, value in resources.items())
    assert any(row['recipe'] == 'hccl_remote_notify' for row in result['timeline'])

    raw = json.loads((root / 'examples/disaggregated.json').read_text())
    connector = raw['connector']
    raw['connector'] = {'link': {
        'name': 'installed-handoff', 'semantics': 'per_port',
        'capacity_bytes_per_ms': connector['bytes_per_s'] / 1000,
        'latency_us': connector.get('latency_us', 0),
        'switch_hops': 2, 'switch_hop_latency_us': .5,
        'transaction_bytes': 256, 'alignment_bytes': 256, 'protocol_overhead_bytes': 32}}
    path = work / 'pd-physical-connector.json'
    path.write_text(json.dumps(raw) + '\n')
    result = run(['predict', path], 'predict-pd-physical-framing-and-switch')
    execution = result['execution']
    wire = execution['connector']['wire_bytes']
    assert wire > execution['handoff_bytes'] > 0
    assert execution['serviced']['pd:link:installed-handoff:tx:port0'] == wire
    assert execution['serviced']['pd:link:installed-handoff:rx:port1'] == wire
    assert execution['connector']['transactions'] > 0
    assert execution['live_tasks'] == execution['live_programs'] == 0

    path = root / 'examples/source-families/arch22-mc2-caller-control.json'
    for command in ('graph', 'predict'):
        result = run([command, path, '--scope', 'body', '--phase', 'decode', '--tokens', 1,
                      '--context', 128], command + '-arch22-native-mc2-caller')
        assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0

    raw = {
        'model': {'catalog_key': 'glm-5.3', 'layers': 1,
                  'layer_overrides': {'0': {'qk_rope_head_dim': 0}}},
        'machine': {'backend': 'ascend', 'sku': 'Ascend950DT_9582', 'num_devices': 8,
                    'device': {'rates': {'aicpu_metadata_call': 10000},
                               'resource_units': {'aicpu_metadata_call': 1}}},
        'runtime': {'tp': 8, 'kernels': {'lowerings': {'sparse_mla_metadata': {
            'implementation': 'vllm_ascend_sparse_mla_metadata',
            'options': {'layer_groups': [[0]]}}}}},
        'workload': {'requests': [{'id': 'r', 'prompt_tokens': 1, 'output_tokens': 1}]}}
    path = work / 'sparse-metadata-controller.json'
    path.write_text(json.dumps(raw) + '\n')
    result = run(['predict', path, '--scope', 'body', '--tokens', 4,
                  '--sequences', 2, '--context', 128], 'body-sparse-metadata-controller')
    assert sum(row['native_kind'] == 'SparseFlashMlaMetadata'
               for row in result['timeline'] if 'native_kind' in row) == 1
    assert result['execution']['serviced']['aicpu_metadata_call@r0:s0'] == 1
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0

    device = raw['machine']['device']
    device['rates']['joint_metadata'] = 1e9
    device['work_units'] = {'joint_metadata': 'busy_ns'}
    device['resource_units']['joint_metadata'] = 1
    raw['runtime']['kernels']['service_regimes'] = [{
        'id': 'installed-joint-native-control', 'match': {'kernel': 'SparseFlashMlaMetadata'},
        'response': {'fixed_us': 50}, 'execution': {
            'kind': 'joint_native', 'resource': 'joint_metadata', 'reference_rate': 1e9,
            'native_kernel': 'SparseFlashMlaMetadata'},
        'provenance': {'kind': 'synthetic_installed_control'}}]
    path = work / 'joint-native-service.json'
    path.write_text(json.dumps(raw) + '\n')
    result = run(['predict', path, '--scope', 'body', '--tokens', 4,
                  '--sequences', 2, '--context', 128], 'body-joint-native-service')
    assert result['execution']['serviced']['joint_metadata@r0:s0'] == 50000
    assert not result['execution']['serviced'].get('aicpu_metadata_call@r0:s0')
    assert result['parameters']['joint_native_services'][0]['analytical_reference_work'][
        'aicpu_metadata_call'] == 1
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0

    path = root / 'examples/communication/analytic-three-level-control.json'
    result = run(['graph', path, '--scope', 'body', '--phase', 'decode', '--tokens', 2,
                  '--context', 16], 'graph-analytic-three-level-hierarchy')
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    assert any(row['recipe'] == 'hccl_remote_notify' for row in result['timeline'])
