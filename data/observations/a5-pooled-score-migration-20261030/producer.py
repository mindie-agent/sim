"""Transport frozen GLM Triton score responses to the corrected pool boundary.

[Role] Preserve the existing operator function while renaming its physical axes.
[Invariants] No refit, no Body residual, and no A3 CANN parameters. The old
  260-byte uncompressed-pair feature was a geometry proxy, not native traffic.
[Timeouts & Locks] Stream capture CSVs; finite Body probes; no device contexts.
[Side-Effect Policy] Local explicit cases and migration evidence only.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mindie_sim import predict, target_graphs, operator_costs

HERE = Path(__file__).resolve().parent
PROFILER = ROOT.parent/'profiler/examples/structure'
OUT = ROOT/'examples/cases/pooled-score-migration-20261030'
NAMES = ('a5_glm53flash_tp8_c4_p8k', 'a5_glm53flash_heldout_c12_p4k',
         'a5_glm53flash_heldout_c24_p12k')
NATIVE = '_glm5_next_lightning_indexer_score_kernel'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def identity(path):
    return {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest(),
            'size_bytes': path.stat().st_size}


def migrate(rule):
    previous = deepcopy(rule)
    match = rule['match']
    assert match['recipe'] == 'index_scores' and match['dtype'] == 'bf16'
    match.update({'recipe': 'index_pool_scores', 'dtype': 'fp32', 'local_shape.1': 128,
                  'metadata.pool_size': 4, 'metadata.score_mode': 'linear_weighted_query',
                  'metadata.native_source.revision': '8c30e4f491f73cba737a1dacc522894e437cdf3d'})
    del match['local_shape.2']
    # Preserve exactly f(260*causal_pairs). The coefficient is not interpreted
    # as an HBM rate after the source decomposition corrects pooled traffic.
    for name, coordinate in rule['coordinates'].items():
        assert coordinate['path'] == 'work.hbm'
        coordinate.update({'path': 'metadata.causal_token_pairs',
                           'scale': 260 * coordinate.get('scale', 1)})
    rule['provenance']['coordinate_migration'] = {
        'basis': 'algebraic_transport_of_existing_operator_response; zero_refit',
        'old_feature': 'uncompressed_causal_pairs * (128 BF16 bytes + 4 score bytes)',
        'new_feature': 'causal_token_pairs * 260; geometry_proxy_not_native_HBM_traffic',
        'original_rule': previous,
        'source': {'revision': '8c30e4f491f73cba737a1dacc522894e437cdf3d',
                   'path': 'vllm_ascend/ops/triton/glm5_next_lightning_indexer.py',
                   'lines': '37-114,174-212'},
        'boundary': 'FP32 qbar plus BF16 pooled-cache matvec; topk and query weighting remain separate',
        'native_binary_identity': 'not_established; existing_native_name_and_tensor_ABI_binding_retained'}
    return rule


def native_evidence(entry, rule):
    """Use retained complete Body populations; A5 Triton CSV shapes are N/A."""
    if 'heldout' in entry['case_id']:
        path = Path('/Volumes/data/mindie-sim-data/acceptance-archives-20261030/heldout-prepared/native-populations') / (
            'body-operator-' + entry['case_id'] + '-heldout-prepared.actual.json')
        actual = json.loads(path.read_text())
        families = [f for f in actual['families'] if f['native_operator_type'] == NATIVE]
        count = sum(f['observed_us']['count'] for f in families)
        mean = sum(f['observed_us']['mean'] * f['observed_us']['count'] for f in families)/count
        evidence = identity(path)
    else:
        pin = rule['provenance']['evidence_identity']
        path = Path(pin['path'])
        assert identity(path)['sha256'] == pin['sha256']
        proof = json.loads(path.read_text())
        for part in rule['provenance']['json_pointer'].strip('/').split('/'):
            proof = proof[int(part)] if isinstance(proof, list) else proof[part]
        count, mean = proof['count'], proof['mean_us']
        evidence = {**identity(path), 'json_pointer': rule['provenance']['json_pointer']}
    return {'count': count, 'mean_us': mean, 'source': evidence,
            'native_tensor_ABI': 'unavailable_in_A5_Triton_rows',
            'boundary': 'retained_symbol_and_source_semantics; exact_installed_binary_not_proven',
            'scope': 'retained_complete_native_call_population_within_selected_Body_geometry'}


def timeout(signum, frame):
    raise TimeoutError('pooled_score_migration_probe_exceeded_90_seconds')


def main():
    registry = json.loads((PROFILER/'cases.json').read_text())
    report = {'scope': 'existing_A5_Triton_score_service_coordinate_transport',
              'new_fitted_parameters': 0, 'rows': []}
    signal.signal(signal.SIGALRM, timeout)
    for entry in registry['cases']:
        if entry['phase'] != 'decode' or entry['case_id'] not in NAMES:
            continue
        source = PROFILER/entry['model_case']['path']
        original = json.loads(source.read_text())
        raw = deepcopy(original)
        rules = raw['runtime']['kernels']['service_regimes']
        replacements = [migrate(rule) for rule in rules if rule['match'].get('recipe') == 'index_scores']
        assert len(replacements) == 1
        raw['runtime']['metadata']['pooled_score_coordinate_migration'] = {
            'original_case': identity(source), 'new_fitted_parameters': 0,
            'report': str((HERE/'validation.json').relative_to(ROOT))}
        output = OUT/(entry['case_id']+'.json')
        save(output, raw)
        geometry = entry['geometry']
        kwargs = {'scope': 'body', 'phase': 'decode', 'tokens': geometry['tokens_per_sequence'],
                  'context': geometry['context_tokens'], 'sequences': geometry['sequences_per_replica']}
        signal.alarm(90)
        before = predict(original, **kwargs, event_limit=0)
        signal.alarm(90)
        after = predict(raw, **kwargs, event_limit=0)
        signal.alarm(0)
        tiny = deepcopy(raw)
        tiny['model']['layers'] = 4
        ops = [op for op in target_graphs(tiny, **kwargs)[0] if op.recipe == 'index_pool_scores']
        assert len(ops) == 1
        cost = operator_costs(tiny, ops)['operators'][0]
        meta = dict(ops[0].metadata)
        native = native_evidence(entry, replacements[0])
        expected_feature = (geometry['sequences_per_replica'] * (geometry['tokens_per_sequence'] *
            geometry['context_tokens'] + geometry['tokens_per_sequence'] * (geometry['tokens_per_sequence']+1)//2))*260
        assert meta['causal_token_pairs']*260 == expected_feature
        check = {'case_id': entry['case_id'], 'source': identity(source), 'output_case': identity(output),
            'geometry': geometry, 'body_before_ms': before['body_forward_ms'],
            'body_after_ms': after['body_forward_ms'], 'native_operator': native,
            'score_predicted_us': cost['duration_ms']*1000,
            'transported_feature': expected_feature, 'actual_logical_hbm_bytes': ops[0].hbm_bytes,
            'unpooled_causal_pairs': meta['causal_token_pairs'], 'scanned_pools': meta['scanned_pools'],
            'score_shape': ops[0].local_shape, 'score_dtype': ops[0].dtype,
            'live_tasks': after['execution']['live_tasks'], 'live_programs': after['execution']['live_programs']}
        if native['mean_us']:
            check['native_operator_relative_error'] = check['score_predicted_us']/native['mean_us']-1
        report['rows'].append(check)
        save(HERE/'validation.json', report)
        print(json.dumps(check), flush=True)


if __name__ == '__main__':
    main()
