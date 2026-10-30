"""Transfer three existing A3 FFN responses after exact native ABI checks.

[Role] Bind count-prefix, fused gate/up and expert-down to independent q4 inputs.
[Invariants] Copy old responses and domains unchanged; no target latency fit.
  Shared BF16, TP join reduction and FAI declarations remain the pinned source.
[Timeouts & Locks] Serial Body predictions each have a 90-second budget.
[Side-Effect Policy] Local candidate case, manifest and evidence only; no device work.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import argparse
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from mindie_sim import predict, target_graphs

BASE_REVISION='1c5939cd'
BASE_PATH='examples/cases/longctx-decode-20261030/a3-fai-regular/a3-qwen35-tp4-long-context-source.json'
PARAMETERS=ROOT/'data/observations/parameters/a3-native-decode-20261030/parameters.json'
OLD_EVIDENCE=PARAMETERS.with_name('operator-evidence.json')
OUTPUT=ROOT/'examples/cases/longctx-decode-20261030/a3-qwen35-frozen-ffn/qwen35-a3-tp4-frozen-ffn.json'
RECIPES={'count_prefix':'expert_group_list_cumsum','gate_up':'declared_native_fusion','expert_down':'balanced_grouped_down'}
NATIVE={'count_prefix':'Cumsum','gate_up':'GroupedMatmulSwigluQuant','expert_down':'GroupedMatmul'}


def identity(path):
    payload=path.read_bytes()
    return {'path':str(path),'sha256':sha256(payload).hexdigest(),'size_bytes':len(payload)}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def timeout(*_):
    raise TimeoutError('qwen35_frozen_ffn_Body_exceeded_90s')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predict',action='store_true')
    args=parser.parse_args()
    signal.signal(signal.SIGALRM,timeout)
    original_bytes=subprocess.check_output(['git','show',f'{BASE_REVISION}:{BASE_PATH}'],cwd=ROOT,timeout=15)
    raw=json.loads(original_bytes)
    old_records=json.loads(OLD_EVIDENCE.read_text())['records']
    rules=[r for r in json.loads(PARAMETERS.read_text())['service_regimes']
           if r['match'].get('recipe') in RECIPES.values()]
    assert len(rules)==3
    assert raw['runtime']['moe_recipe']=='tp_join_reduce'
    quant=dict(raw['model']['quant']['category_overrides'])
    assert dict(quant['shared_expert'])=={'weight_format':'bf16','activation_format':'bf16'}
    kernels=raw['runtime']['kernels']
    assert not kernels.get('lowerings')
    kernels['lowerings']={'group_list_prefix':'vllm_ascend_group_list_prefix',
        'native_fusion':{'implementation':'declared_native_fusion',
                         'options':{'roles':['grouped_gate_up_activation_quant']}}}
    kernels['service_regimes'].extend(deepcopy(rules))
    raw['runtime']['metadata']['frozen_ffn_transfer']={
        'base_revision':BASE_REVISION,'base_case_path':BASE_PATH,
        'base_case_sha256':sha256(original_bytes).hexdigest(),
        'parameter_identity':identity(PARAMETERS),'operator_evidence_identity':identity(OLD_EVIDENCE),
        'selected_rule_ids':[r['id'] for r in rules],
        'parameter_changes':0,'domain_changes':0,'new_target_timings_used_to_fit':False,
        'evaluation_relation':'new_q4_128K_captures_not_used_to_assign_old_4K_responses; previously_inspected_not_blind',
        'count_prefix_contract':'native descriptor N/A; 256 INT64 entries inferred from fused consumer ABI and pinned count-to-prefix caller',
        'captured_binary_identity':'unobserved','routing_values':'not_exported; declared force-EPLB realization remains conditional'}
    raw['runtime']['metadata']['ffn_source_contract']['grouped_execution']='declared fused gate/up/activation/quant, explicit count-prefix producer and unchanged old exact-domain native responses; captured binary unobserved'
    save(OUTPUT,raw)
    report={'scope':'frozen_A3_FFN_response_transfer_to_new_q4_Bodies',
        'runtime_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=15).strip(),
        'base_revision':BASE_REVISION,'base_case_sha256':sha256(original_bytes).hexdigest(),
        'candidate_case':identity(OUTPUT),'parameters':identity(PARAMETERS),'old_operator_evidence':identity(OLD_EVIDENCE),
        'response_rules_copied_verbatim':rules,'new_coefficients':0,'domain_changes':0,
        'boundary':'whole target Body; native activity diagnostics are not a wall-span correction',
        'cases':[]}
    manifest={'schema_version':'mindie-cases/1','cases':[],
        'policy':{'selection':'only exact B1/B4 source hashes physically reviewed',
                  'new_target_timings_used_to_fit':False}}
    original_manifest=json.loads((ROOT.parent/'profiler/examples/structure/cases.json').read_text())
    for batch in (1,4):
        ledger_path=HERE/f'native-b{batch}.json';ledger=json.loads(ledger_path.read_text())
        rows=batch*4*8
        comparisons=[]
        for op,kind in NATIVE.items():
            observed,=[g for g in ledger['native_groups'] if g['abi']['Type']==kind]
            assert observed['calls_per_body']==40
            candidates=[r for r in old_records if r['op']==op and r['coordinate']==(256 if op=='count_prefix' else rows)]
            matched=[r for r in candidates if r['native_abi']==observed['abi']]
            assert matched,(batch,op)
            rule=next(r for r in rules if r['match']['recipe']==RECIPES[op])
            coordinate=256 if op=='count_prefix' else rows
            response=rule['response']['fixed_us']+sum(t['us_per_unit']*coordinate for t in rule['response']['terms'])
            comparisons.append({'op':op,'native_abi':observed['abi'],'observed_mean_us':observed['mean_us'],
                'frozen_response_us':response,'observed_calls_per_body':observed['calls_per_body'],
                'matching_old_records':len(matched),'representative_old_source':{k:matched[0][k] for k in ['capture','sha256','split','coordinate','first','consumer_abi']},
                'new_locators':observed['locators'],'all_physical_ABI_fields_equal':True})
        ops=target_graphs(raw,scope='body',phase='decode',tokens=4,context=126000,sequences=batch)[0]
        selected={recipe:[o for o in ops if o.recipe==recipe] for recipe in RECIPES.values()}
        assert all(len(v)==40 for v in selected.values())
        for fused in selected['declared_native_fusion']:
            assert fused.local_shape==(rows,128)
            prefix=next(o for o in selected['expert_group_list_cumsum'] if o.layer==fused.layer)
            assert prefix.op_id in fused.deps and prefix.local_shape==(256,) and prefix.dtype=='int64'
            down=next(o for o in selected['balanced_grouped_down'] if o.layer==fused.layer)
            assert down.deps==(fused.op_id,) and down.local_shape==(rows,2048,128)
        body={'status':'not_requested'}
        if args.predict:
            started=time.monotonic();signal.alarm(90)
            try:
                result=predict(raw,scope='body',phase='decode',tokens=4,context=126000,sequences=batch,event_limit=0)
            except TimeoutError as exc:
                body={'status':'timed_out','error':str(exc),'elapsed_seconds':time.monotonic()-started}
            else:
                ms=result['body_forward_ms'];observed_ms=ledger['body_observed_ms']['observed_ms']['mean']
                body={'status':'completed','predicted_ms':ms,'observed_ms':observed_ms,
                    'relative_error':ms/observed_ms-1,'passed_15percent':abs(ms/observed_ms-1)<=.15,
                    'elapsed_seconds':time.monotonic()-started,'execution':result['execution'],
                    'service_regimes':result['parameters'].get('service_regimes',[])}
                save(HERE/f'prediction-b{batch}.json',result)
            finally:
                signal.alarm(0)
        row={'batch':batch,'query_tokens':4,'context_tokens':126000,'case_id':ledger['case_id'],
            'source':ledger['source_identity'],'ledger':identity(ledger_path),'abi_comparisons':comparisons,
            'source_graph_count':{k:len(v) for k,v in selected.items()},
            'shared_precision_and_TP_join_unchanged':True,'prior_ee3_body':ledger['body_observed_ms'],
            'after':body}
        report['cases'].append(row)
        entry=deepcopy(next(r for r in original_manifest['cases'] if r['source']['sha256']==ledger['source_identity']['sha256']))
        entry['case_id']+='-frozen-ffn';entry['id']='decode:'+entry['case_id']
        entry['sim_case_path']=str(OUTPUT);entry['model_case']=identity(OUTPUT)
        entry['supported_scopes']=['body','operators']
        entry['calibration']={'origin':'frozen_older_operator_responses',
            'evaluation_relation':'new_q4_128K_target_not_used_for_fitting; previously_inspected_not_blind',
            'target_native_timings_used_in_rules':False,'evidence_refs':[identity(PARAMETERS),identity(ledger_path)]}
        manifest['cases'].append(entry)
        report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save(HERE/'validation.json',report);save(HERE/'cases.json',manifest)
        print(json.dumps({'batch':batch,'abi_matches':[r['op'] for r in comparisons],**{k:v for k,v in body.items() if k not in ['execution','service_regimes']}}),flush=True)
    if args.predict and any(r['after']['status']!='completed' for r in report['cases']):
        raise SystemExit(1)


if __name__=='__main__':
    main()
