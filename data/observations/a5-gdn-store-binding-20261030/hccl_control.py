"""Preserve both unresolved AIV source candidates without latency selection.

[Role] Bind the retained HCCL AIV programs to complete new A5 Body graphs.
[Invariants] Native blocks and semantic payload are fixed by evidence. Unknown
  dispatch remains conditional; unsupported TwoShot/3 never becomes Block6.
[Timeouts & Locks] One bounded 90-second Body at a time; no external work.
[Side-Effect Policy] Local case copies and comparison evidence only.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import resource
import signal
import subprocess
import time

spec = importlib.util.spec_from_file_location('gdn_store', Path(__file__).with_name('produce.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from mindie_sim import predict
from mindie_sim.prediction.hccl_contracts import HcclSelection, legacy_mesh_candidate
from mindie_sim.prediction.hccl_aiv import blocks
from mindie_sim.schema.common import ModelingError


def main():
    signal.signal(signal.SIGALRM, base.timeout)
    start = time.monotonic()
    binding = base.load(base.HERE / 'validation.json')
    report = {'scope': 'conditional_AIV_source_candidates_with_fixed_native_block_counts',
        'baseline': base.identity(base.HERE / 'validation.json'),
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base.ROOT, text=True).strip(),
        'new_parameters': 0, 'cases': [],
        'selection_policy': 'preserve all supported candidates; never select by Body error',
        'dispatch_status': 'unknown; retained legacy threshold is not current cost-selector or override proof'}
    for record in binding['cases']:
        model, tp = record['model'], record['tp']
        raw = base.load(Path(record['case']['path']))
        retained = base.load(Path(record['retained_validation']['path']))
        native = [g for g in record['native']['groups'] if g['native'] == 'hcom_allReduce_']
        expected_blocks = 3 if tp == 2 else 60
        assert {g['block_num'] for g in native} == {expected_blocks}
        assert {g['core'] for g in native} == {'COMMUNICATION'}
        assert sum(g['calls_per_forward'] for g in native) == 128
        payload = 16 * 5120 * 2
        legacy = legacy_mesh_candidate(collective='allreduce', expansion='AIV', payload_bytes=payload, ranks=tp)
        row = {'model': model, 'tp': tp, 'base_case': record['case'], 'native_source': record['native']['source'],
            'payload_bytes': payload, 'payload_basis': 'source semantic M16 hidden5120 BF16 output; HCCL CSV args N/A',
            'native_block_num': expected_blocks, 'native_calls_per_body': 128,
            'native_observation': native, 'legacy_selector_candidate': legacy,
            'baseline_body_ms': record['after']['body_ms'],
            'baseline_body_error': record['after']['body_relative_error'], 'candidates': []}
        for template in ('mesh_oneshot', 'mesh_twoshot'):
            case = deepcopy(raw)
            declaration = {'expansion': 'AIV', 'template': template, 'block_limit': expected_blocks, 'axes': ['tp']}
            case['runtime']['kernels']['hccl'] = {'allreduce': declaration}
            case['runtime']['metadata']['hccl_source_candidate'] = {
                'declaration': declaration, 'native_block_num': expected_blocks,
                'legacy_selector_candidate': legacy, 'captured_template': 'unobserved',
                'source_dispatch': 'conditional_candidate; block count and AivKernel name are insufficient',
                'payload_bytes_at_B4_q4': payload, 'new_latency_parameters': 0}
            candidate = {'template': template, 'declaration': declaration,
                         'legacy_threshold_agrees': template == legacy}
            try:
                owners = blocks(HcclSelection('AIV', 'allreduce', template, tp, payload,
                                              block_limit=expected_blocks), 64)
            except ModelingError:
                owners = None
            candidate['modeled_launch_owners'] = owners
            candidate['native_block_count_matches_modeled_owners'] = owners == expected_blocks
            if tp == 2 and template == 'mesh_twoshot':
                tiny = deepcopy(case)
                tiny['model']['layers'] = 1
                try:
                    predict(tiny, scope='body', phase='decode', tokens=4, context=126000, sequences=4, event_limit=0)
                except ModelingError as error:
                    candidate.update({'status': 'unsupported_source_branch', 'error': str(error),
                                      'block_count_changed': False, 'full_body_executed': False})
                else:
                    raise AssertionError('TwoShot/3 now executes: remove obsolete unsupported expectation and evaluate')
            else:
                output = base.ROOT / 'examples/cases/longctx-decode-20261030/a5-gdn-hccl-source' / f'a5-{model}-tp{tp}-{template}.json'
                base.save(output, case)
                candidate.update({'status': 'completed', 'case': base.identity(output),
                                  'result': base.prediction(case, retained)})
                candidate['result']['body_passed_15percent'] = abs(candidate['result']['body_relative_error']) <= .15
                assert len(candidate['result']['hccl_calls']) == 128
            row['candidates'].append(candidate)
            print(json.dumps({'model': model, 'tp': tp, 'template': template, 'status': candidate['status'],
                **({'body_ms': candidate['result']['body_ms'], 'error': candidate['result']['body_relative_error'],
                    'first_hccl_us': candidate['result']['hccl_calls'][0]['duration_us']}
                   if 'result' in candidate else {'error': candidate['error']})}), flush=True)
        report['cases'].append(row)
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['elapsed_seconds'] = time.monotonic()-start
        base.save(base.HERE / 'hccl-validation.json', report)


if __name__ == '__main__':
    main()
