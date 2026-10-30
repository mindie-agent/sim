"""Bind existing normalization fusion callers to the four new B4 cases.

[Role] Reuse semantic fusion passes and the existing native INT8/MXFP8 ABI.
[Invariants] Preserve residual outputs and one external gamma producer per
  selected fused call. Frozen response domains and coefficients stay unchanged.
[Timeouts & Locks] Stream bounded native samples; optional Body budget is 90 s.
[Side-Effect Policy] Local cases, Profiler manifest and validation only.
"""
import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import signal
import subprocess

spec = importlib.util.spec_from_file_location('hccl_frozen', Path(__file__).with_name('hccl_frozen.py'))
hccl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hccl)
base = hccl.base
from mindie_sim import target_graphs
from mindie_sim.structure_contracts import predicted_contract
from mindie_profiler.tensor_contracts import observed_contract

PASSES = ['activation_quantization', 'norm_quant', 'residual_norm']
BINDING = {'implementation': 'vllm_ascend_native_norm',
           'options': {'has_bias': False, 'external_gamma_shift': True}}
CASES = base.ROOT / 'examples/cases/longctx-decode-20261030/a5-gdn-caller-fused'


def native_contracts(record, kind):
    groups = [g for g in record['native']['groups'] if g['native'] == kind]
    targets = {g['first_locator']['csv_line'] for g in groups}
    result = []
    for row in base.kernel_rows(Path(record['native']['source']['path']), max_rows=5_000_000, timeout_s=90):
        if row.line in targets:
            contract = observed_contract(row)
            assert contract is not None
            result.append({'locator': row.event.locator, 'contract': contract,
                           'input_shapes': row.input_shapes, 'input_dtypes': row.input_dtypes,
                           'output_shapes': row.output_shapes, 'output_dtypes': row.output_dtypes,
                           'input_formats': row.input_formats, 'output_formats': row.output_formats})
            targets.remove(row.line)
            if not targets:
                break
    assert not targets
    return result, sum(g['calls_per_forward'] for g in groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predict', action='store_true', help='Run four complete B4 Bodies after structural validation')
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, base.timeout)
    native = base.load(base.HERE / 'validation.json')
    prior = base.load(base.HERE / 'hccl-frozen-validation.json')
    original_manifest = base.load(hccl.MANIFEST)
    report = {'scope': 'existing_native_norm_caller_binding; no_new_latency_parameters',
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base.ROOT, text=True).strip(),
        'new_parameters': 0, 'response_domain_changes': 0, 'lowering_passes': PASSES,
        'binding': BINDING, 'geometry': {'tokens': 4, 'sequences': 4, 'context': 126000, 'scope': 'body'},
        'previous_hccl_comparison': base.identity(base.HERE / 'hccl-frozen-validation.json'),
        'cases': []}
    manifest = {'schema_version': 'mindie-cases/1', 'cases': [],
        'policy': {'scope': 'four_B4_caller_variants_against_same_complete_native_Bodies',
                   'new_target_timings_used_to_fit': False}}
    for record in native['cases']:
        model, tp = record['model'], record['tp']
        before = next(r for r in prior['cases'] if (r['model'], r['tp'], r['batch']) == (model, tp, 4))
        source = Path(before['case']['path'])
        assert base.identity(source) == before['case']
        raw = base.load(source)
        old = target_graphs(raw, scope='body', phase='decode', tokens=4, context=126000, sequences=4)[0]
        assert not any(op.recipe == 'gemma_gamma_shift' for op in old)
        raw['runtime']['lowering_passes'] = PASSES
        raw['runtime']['kernels']['lowerings']['add_rmsnorm_quant'] = BINDING
        raw['runtime']['metadata']['native_norm_caller_binding'] = {
            'source_case': before['case'], 'lowering_passes': PASSES,
            'selected_family': 'add_rmsnorm_quant', 'external_gamma_shift': True,
            'captured_binary_identity': 'unobserved', 'target_timing_used_to_fit': False,
            'unquantized_add_rmsnorm': 'semantic_only; native selector names AddRmsNormBias rather than captured AddRmsNorm',
            'local_point_rules_added': 0, 'existing_HCCL_rules_changed': False}
        raw['runtime']['metadata']['source_compatible_gdn']['frozen_or_point_service_rules'] = len(raw['runtime']['kernels']['service_regimes'])
        output = CASES / f'a5-{model}-tp{tp}-b4-caller-fused.json'
        base.save(output, raw)
        ops = target_graphs(raw, scope='body', phase='decode', tokens=4, context=126000, sequences=4)[0]
        selected = [op for op in ops if op.family == 'add_rmsnorm_quant']
        gamma = [op for op in ops if op.recipe == 'gemma_gamma_shift']
        assert len(selected) == len(gamma) == 80
        gamma_ids = {op.op_id for op in gamma}
        assert all(sum(dep in gamma_ids for dep in op.deps) == 1 for op in selected)
        assert all(dict(op.metadata)['norm_weight_offset'] == 0 for op in selected)
        assert all('residual' in op.returns for op in selected)
        assert Counter(op.recipe for op in ops)['residual_add'] == 1
        assert raw['runtime']['kernels']['service_regimes'] == base.load(source)['runtime']['kernels']['service_regimes']
        kind = 'AddRmsNormDynamicQuant' if model == 'qwen36' else 'AddRmsNormDynamicMxQuant'
        actual, count = native_contracts(record, kind)
        assert count == len(selected)
        contracts = {predicted_contract(op.to_dict()) for op in selected}
        assert len(contracts) == 1
        assert all(sample['contract'] in contracts for sample in actual)
        gamma_native = [g for g in record['native']['groups']
            if g['native'] == 'Add' and g['input_shapes'] == '"5120;"' and g['output_shapes'] == '"5120"']
        gamma_count = sum(g['calls_per_forward'] for g in gamma_native)
        assert gamma_count == 128
        result = {'model': model, 'tp': tp, 'case': base.identity(output),
            'baseline_case': before['case'], 'baseline_engine_revision': prior['revision'],
            'baseline_body': before['after'], 'observed_body_ms': before['observed_body_ms'],
            'native_kind': kind, 'predicted_contract': next(iter(contracts)),
            'native_samples': actual, 'native_fused_calls_per_body': count,
            'semantic_family_counts': dict(Counter(op.family for op in ops if 'norm' in op.family)),
            'external_gamma': {'selected_calls': 80, 'native_add_5120_calls': gamma_count,
                'one_dependency_per_selected_fused_call': True, 'preexisting_gamma_calls': 0,
                'remaining_48': 'unselected initial/plain residual norms retain semantic weight offset; no extra gamma appended'},
            'remaining_standalone_residual_calls': 1,
            'response_rules_unchanged': True, 'body_prediction_status': 'not_requested'}
        if args.predict:
            result['after'] = hccl.bounded_body(raw, 4, before['observed_body_ms']['mean'])
            result['body_prediction_status'] = result['after']['status']
        report['cases'].append(result)
        original = next(r for r in original_manifest['cases']
            if r['case_id'] == f'a5-{model}-tp{tp}-decode-c4-p126000')
        entry = deepcopy(original)
        entry['case_id'] += '-caller-fused'
        entry['id'] = 'decode:' + entry['case_id']
        entry['sim_case_path'] = str(output)
        entry['model_case'] = base.identity(output)
        entry['supported_scopes'] = ['body', 'operators']
        entry['calibration'] = {'origin': 'frozen_older_HCCL_response_plus_source_caller',
            'evaluation_relation': 'new_target_not_used_for_fitting; previously_inspected_not_blind',
            'target_native_timings_used_in_rules': False,
            'evidence_refs': [before['case'], base.identity(output)]}
        manifest['cases'].append(entry)
        base.save(base.HERE / 'caller-binding-validation.json', report)
        base.save(base.HERE / 'caller-cases.json', manifest)
        print(json.dumps({'model': model, 'tp': tp, 'native_fused_calls': count,
            'external_gamma': len(gamma), 'body_status': result['body_prediction_status'],
            'body_ms': result.get('after', {}).get('body_ms')}), flush=True)
    if args.predict and any(r['body_prediction_status'] != 'completed' for r in report['cases']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
