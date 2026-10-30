"""[Role] Assign conditional small-M grouped native-service regimes.
[Invariants] Only c64 native operator means assign bank-service coefficients;
  no Body total, c128 response or allocated activation rows enter assignment.
[Timeouts & Locks] Compact local metadata only; no subprocess/network/locks.
[Side-Effect Policy] Writes a parameter manifest and separate condition cases.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PROF=ROOT.parent/'profiler/examples/structure/acceptance-20261030-v41-operators-86c1ee8f-source-symmetric'
QOBS=ROOT/'data/verification/qwen35-source-residual-pmu-20261030.json'
VNAME={64:'a5_v41_fp8_tp1_dp8_ep8_tpot30ms_c64',128:'a5_v41_fp8_tp1_dp8_ep8_c128_p2k'}


def identity(path):
    return {'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest(),'size_bytes':path.stat().st_size}


def observed():
    q=json.loads(QOBS.read_text());v={}
    for c,name in VNAME.items():
        p=PROF/name/'body-operator-family-scoreboard.json'
        report=json.loads(p.read_text());case=report['cases'][0]
        v[c]={'report':identity(p),'source':case['source_identity'],'geometry':case['geometry'],'families':{}}
        for family in case['families']:
            if family['native_operator_type']=='GroupedMatmul':
                reference=family['conditional_source_role_references'][0]
                correspondence=reference['source_correspondence']
                v[c]['families'][family['declared_role']]={'observed_us':family['observed_us'],
                  'correspondence':{k:correspondence[k] for k in ('observed_buffer_row_capacity','declared_valid_routed_rows','observed_valid_routed_rows','observed_group_list_values','validated_geometry')},
                  'original_role':'unassigned'}
    return {x['c']:x for x in q},v


def rule(identifier,mean,banks,n,k,fmt,output,source):
    mx=fmt=='mxfp4'; group_rows={'path':'metadata.matrix_group_rows'}
    match={'hardware':'Ascend950DT_9582','family':'grouped_gemm','dtype':'bf16' if mx else 'int8',
      'metadata.weight_format':fmt,'metadata.activation_format':'fp8_e4m3' if mx else 'int8',
      'metadata.output_format':output}
    if mx:
        match.update({'metadata.program_execution.implementation':'ascend_mxa8w4_nznk',
                      'metadata.program_execution.mxa8w4.producer_mode':'dynamic_buffer4'})
    else:match['metadata.matrix_output_contract']=('int32_quant_matmul_without_epilogue' if output=='int32' else 'int32_cast_row_scale_column_scale_output_cast')
    return {'id':identifier,'match':match,
      'coordinates':{'active_banks':{**group_rows,'reduce':'positive_count'},
        'declared_banks':{**group_rows,'reduce':'len'},'max_rows':{**group_rows,'reduce':'max'},
        'n':{'path':'local_shape.1'},'k':{'path':'local_shape.2'}},
      'domain':{'active_banks':{'min':banks if mx else 32,'max':banks},
        'declared_banks':{'min':banks,'max':banks},'max_rows':{'min':1,'max':2 if mx else 16},
        'n':{'min':n,'max':n},'k':{'min':k,'max':k}},
      'response':{'fixed_us':0,'terms':[{'coordinate':'active_banks','us_per_unit':mean/banks,
        'scale_with':{'resource':'mte2_aiv' if mx else 'mte2_aic','reference_rate':4224000000000}}]},
      'provenance':{'kind':'conditional_native_grouped_service_calibration','boundary':'inclusive native kernel completion',
        'assignment':{'native_mean_us':mean,'declared_active_banks':banks,'cohort_label':64,'source':source},
        'geometry':'declared balanced histogram; captured group_list values unknown',
        'scale_assumption':'inverse private weight-input pipe rate; engineered effective-regime hypothesis, not a hardware bandwidth measurement',
        'input_precision':fmt,'output_precision':output,
        'validation_role':'c128 not used in this assignment; previously viewed observations, not blinded historical holdout',
        'cross_model_validation':'same implementation/geometry observations pending',
        'excludes':['Body_fit','model_name_selector','batch_selector','PMU_activity_sum','capacity_as_valid_rows']}}


def main():
    q,v=observed(); qf={f['role']:f for f in q[64]['families']}; rules=[]
    for role,n,k,output in (('up',1024,2048,'int32'),('down',2048,512,'bf16')):
        rules.append(rule('a5-int8-grouped-'+output+'-smallrows-effective-bank-service-v1',qf['GroupedMatmul/'+role]['mean_us']['Duration(us)'],64,n,k,'int8',output,identity(QOBS)))
    for role,n,k in (('expert_up',4608,5120),('expert_down',5120,2304)):
        f=v[64]['families'][role]
        rules.append(rule('a5-mxa8w4-nznk-'+str(n)+'x'+str(k)+'-smallrows-effective-bank-service-v1',f['observed_us']['mean'],48,n,k,'mxfp4','bf16',v[64]['report']))
    manifest={'schema':'operator-service-calibration/1','producer':identity(Path(__file__)),
      'service_regimes':rules,'q35_source':identity(QOBS),'v41_compact_observations':v,
      'roles':{'64':'parameter_assignment','128':'not_used_in_parameter_assignment','4':'Q35 extra non-assignment geometry validation'},
      'raw_original_role':'unassigned','body_durations_used_for_assignment':False}
    (HERE/'grouped-parameters.json').write_text(json.dumps(manifest,indent=2)+'\n')
    dsq=json.loads((HERE/'parameters.json').read_text())['service_regimes']
    output=ROOT/'examples/cases/high-concurrency-20261030/grouped-effective-service-conditional';output.mkdir(exist_ok=True)
    for c in (4,64,128):
        p=ROOT/f'examples/cases/high-concurrency-20261030/qwen35-mc2-tp-dsq-source-conditional/qwen35-35b-a5-tp4-c{c}-mc2-tp-dsq-source.json'
        raw=json.loads(p.read_text());raw['name']+=' / conditional grouped and DSQ effective service'
        raw['runtime']['kernels']['service_regimes']=deepcopy(dsq+rules)
        raw['runtime']['metadata']['grouped_effective_service']={'parent':identity(p),'parameters':identity(HERE/'grouped-parameters.json'),'role':manifest['roles'][str(c)]}
        (output/f'qwen35-c{c}-grouped-dsq-effective-service.json').write_text(json.dumps(raw,indent=2)+'\n')
    for c,name in VNAME.items():
        p=ROOT/f'examples/cases/high-concurrency-20261030/v41-composed-arch35-conditional/{name}-composed-arch35-source.json'
        raw=json.loads(p.read_text());raw['name']+=' / conditional grouped effective service'
        raw['runtime']['kernels']['service_regimes']=deepcopy(rules)
        raw['runtime']['metadata']['grouped_effective_service']={'parent':identity(p),'parameters':identity(HERE/'grouped-parameters.json'),'role':manifest['roles'][str(c)],'cohort':'symmetric_declared; not captured group_list proof'}
        (output/f'v41-c{c}-grouped-effective-service.json').write_text(json.dumps(raw,indent=2)+'\n')
    print(json.dumps({r['id']:r['response']['terms'][0]['us_per_unit'] for r in rules}))

if __name__=='__main__':main()
