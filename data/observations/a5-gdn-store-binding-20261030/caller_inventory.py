"""Inventory existing frozen local responses and native norm caller boundaries.

[Role] Expose exact reusable domains and structural fusion candidates.
[Invariants] Read-only graph construction; no response fit or Body correction.
  Missing capture shapes remain conditional, and old stride guards stay intact.
[Timeouts & Locks] Bounded local JSON and four finite graph constructions.
[Side-Effect Policy] Write one local inventory only; no cases or runtime change.
"""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('gdn_store', Path(__file__).with_name('produce.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from mindie_sim import target_graphs
from mindie_sim.prediction.service_regimes import axis_equal, lookup

RECIPES = {'ascend_causal_conv1d_update', 'decode_q_l2norm',
           'decode_k_l2norm', 'linear_output_norm_gate'}


def main():
    rules = []
    for name in ('qwen36-a5-tp2.json', 'qwen35-a5-tp4-ep4.json'):
        path = base.ROOT / 'examples/cases/longctx-decode-20261030' / name
        for rule in base.load(path)['runtime']['kernels']['service_regimes']:
            if rule['match'].get('recipe') in RECIPES:
                rules.append({'source': base.identity(path), 'rule': rule})
    report = {'scope': 'existing_local_response_domains_and_norm_fusion_inventory',
        'new_parameters': 0, 'body_executions': 0, 'rules': rules, 'cases': [],
        'limitations': ['Q/K L2 and output norm capture shapes are N/A; source/adjacent-call shapes remain conditional',
            'Old convolution cache_index_row_stride4 is not new-capture producer evidence',
            'Fusion call count and ABI are structural evidence, not kernel-time calibration',
            'vllm_ascend_native_norm add_rmsnorm branch names AddRmsNormBias even without bias; new capture says AddRmsNorm']}
    native = base.load(base.HERE / 'validation.json')
    for source_record in native['cases']:
        model, tp = source_record['model'], source_record['tp']
        raw = base.load(Path(source_record['case']['path']))
        row = {'model': model, 'tp': tp, 'case': source_record['case'], 'geometries': []}
        for batch in (1, 4, 24 if tp == 2 else 8):
            ops = target_graphs(raw, scope='body', phase='decode', tokens=4,
                                context=126000, sequences=batch)[0]
            comparisons = []
            for item in rules:
                rule = item['rule']
                selected = [op for op in ops if op.recipe == rule['match']['recipe']]
                counts = Counter()
                for op in selected:
                    axes = {'hardware': 'Ascend950DT_9582', 'dtype': op.dtype,
                        'recipe': op.recipe, 'family': op.family, 'phase': op.phase,
                        'local_shape': op.local_shape, 'metadata': dict(op.metadata)}
                    missing = tuple(key for key, value in rule['match'].items()
                                    if not axis_equal(lookup(axes, key), value))
                    counts[missing] += 1
                comparisons.append({'rule_id': rule['id'], 'matching_calls': counts.get((), 0),
                    'comparisons': [{'mismatched_axes': list(k), 'calls': v} for k, v in counts.items()]})
            row['geometries'].append({'batch': batch, 'query_tokens': 4, 'query_rows': batch * 4,
                'operators': [{'recipe': op.recipe, 'local_shape': op.local_shape,
                    'cache_index_row_stride': dict(op.metadata).get('cache_index_row_stride')}
                    for op in ops if op.layer == 0 and op.recipe in RECIPES], 'rule_domains': comparisons})
        fused = deepcopy(raw)
        fused['runtime']['lowering_passes'] = ['activation_quantization', 'norm_quant', 'residual_norm']
        fused['runtime']['kernels']['lowerings']['add_rmsnorm_quant'] = {
            'implementation': 'vllm_ascend_native_norm',
            'options': {'external_gamma_shift': True, 'has_bias': False}}
        ops = target_graphs(fused, scope='body', phase='decode', tokens=4,
                            context=126000, sequences=4)[0]
        row['fusion_candidate'] = {
            'lowering_passes': fused['runtime']['lowering_passes'],
            'add_rmsnorm_quant_binding': fused['runtime']['kernels']['lowerings']['add_rmsnorm_quant'],
            'semantic_families': dict(Counter(op.family for op in ops if 'norm' in op.family)),
            'native_selected': dict(Counter(dict(op.metadata).get('native_norm_kind') for op in ops
                                           if dict(op.metadata).get('native_norm_kind'))),
            'external_gamma_shift_calls': sum(op.recipe == 'gemma_gamma_shift' for op in ops),
            'first_fused_call': next({'recipe': op.recipe, 'local_shape': op.local_shape,
                'metadata': {k: v for k, v in dict(op.metadata).items()
                    if any(s in k for s in ('norm', 'format', 'scale'))}}
                for op in ops if op.family == 'add_rmsnorm_quant')}
        row['native_norm_groups'] = [g for g in source_record['native']['groups']
            if g['native'] in {'RmsNorm', 'AddRmsNorm', 'AddRmsNormDynamicQuant', 'AddRmsNormDynamicMxQuant'}]
        report['cases'].append(row)
    base.save(base.HERE / 'caller-frozen-inventory.json', report)
    print(json.dumps([{'model': r['model'], 'tp': r['tp'], 'fused': r['fusion_candidate']['native_selected'],
        'domains': [{'batch': g['batch'], 'hits': {p['rule_id']: p['matching_calls']
            for p in g['rule_domains'] if p['matching_calls']}} for g in r['geometries']]} for r in report['cases']], indent=2))


if __name__ == '__main__':
    main()
