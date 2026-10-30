"""Assign two fixed-domain native grouped responses from 64K primitive calls.

[Role] Freeze continuous rows-per-declared-bank responses and one EP1 case.
[Invariants] Only two64K hold sources train; 126K and other64K sources validate.
  Native duration is joint service, never an inferred private-pipe bandwidth.
[Timeouts & Locks] Compact manifests and pure frontend/primitive runs only.
[Side-Effect Policy] Local parameter/case/validation JSON; no runtime mutation.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from mindie_sim import target_graphs, operator_costs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.services import service_identity

SOURCE = ROOT.parent / 'profiler/examples/structure/cases/longctx-a5-qwen35-tp4-source-model.json'
CASE = ROOT / 'examples/cases/longctx-decode-20261030/qwen35-a5-tp4-ep1-grouped-joint.json'
QUEUE = 'native_grouped_call'
ROLES = {'grouped_gate_up': (256,2048), 'grouped_down': (2048,128)}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')


def identity(path):
    return {'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest(),
            'size_bytes':path.stat().st_size}


def signature(group):
    """Normalize only the routed-M axis, preserving weights/scales/format."""
    abi = deepcopy(group['abi'])
    parts = abi['Input Shapes'].strip('"').split(';')
    dimensions = parts[0].split(','); dimensions[0]='M';parts[0]=','.join(dimensions)
    assert parts[-1]==str(group['routed_rows'])
    parts[-1]='M';abi['Input Shapes']=';'.join(parts)
    out=abi['Output Shapes'].strip('"').split(',');out[0]='M'
    abi['Output Shapes']=','.join(out)
    assert abi['Input Data Types']=='INT8;INT8;INT32;DT_BF16;FLOAT;FLOAT16;FLOAT16;INT64;FLOAT'
    assert abi['Input Formats']=='ND;FRACTAL_NZ;ND;ND;ND;ND;ND;ND;ND'
    assert abi['Output Data Types']=='DT_BF16' and abi['Block Num']=='32' and abi['Mix Block Num']=='64'
    assert parts[-2]=='256'
    return abi


def grouped(raw, batch, context=65536):
    graph = target_graphs(raw,scope='body',phase='decode',tokens=4,
                          sequences=batch,context=context)[0]
    return [o for o in graph if o.layer==0 and o.family=='grouped_gemm']


def make_case():
    raw=json.loads(SOURCE.read_text())
    assert raw['runtime']['ep']==raw['runtime']['dp']==1 and raw['runtime']['tp']==4
    raw['name']='A5 Qwen3.5 TP4 EP1 conditional grouped native response'
    raw['model']['quant']['category_overrides']['routed_expert']={
        'scale_format':'bf16','scale_bytes':2}
    # Pin the inspected physical domain. The additional queue is not a hardware
    # pipe: one complete native kernel occupies one declared device service.
    raw['machine']=parse_case(raw).machine.to_dict()
    for die in raw['machine']['devices']:
        die['rates'][QUEUE]=1e9
        die.setdefault('work_units',{})[QUEUE]='busy_ns'
        die['resource_units'][QUEUE]=1
    return raw


def assign(raw, native):
    selected=grouped(raw,1);parsed=parse_case(raw);device=parsed.machine.dies[0].to_dict()
    guards={'hardware':'Ascend950DT_9582','core_count':device['compute_units']}
    guards.update({'physical_rates.resources.'+k:v for k,v in device['rates'].items()})
    guards.update({'physical_rates.compute_peaks.'+k:v for k,v in device['compute_peaks'].items()})
    guards.update({'physical_rates.service_modes.'+k:[v['resource'],v['rate']]
                   for k,v in device['service_modes'].items()})
    guards.update({'resource_units.'+k:v for k,v in device['resource_units'].items()})
    rules=[]
    for op in selected:
        n,k=ROLES[op.recipe]
        train=[]
        for source in native['cases']:
            if source['assignment_role']!='parameter_assignment':continue
            row=next(g for g in source['groups'] if (g['n'],g['k'])==(n,k))
            assert row['calls_per_forward']['min']==row['calls_per_forward']['max']==40
            train.append({'case_id':source['case_id'],'source':source['source_identity'],
                'extraction':source['extraction_identity'],'routed_rows':row['routed_rows'],
                'declared_banks':256,'rows_per_bank':row['routed_rows']/256,
                'mean_us':row['mean_us'],'calls':row['count'],
                'first_csv_line':row['first_csv_line'],'last_csv_line':row['last_csv_line'],
                'native_signature':signature(row)})
        train.sort(key=lambda x:x['routed_rows'])
        assert [t['routed_rows'] for t in train]==[32,128]
        assert train[0]['native_signature']==train[1]['native_signature']
        x0,x1=(t['rows_per_bank'] for t in train);y0,y1=(t['mean_us'] for t in train)
        slope=(y1-y0)/(x1-x0);fixed=y0-slope*x0
        assert slope>=0 and fixed>=0
        meta=dict(op.metadata);service=service_identity(parsed,op)
        match={**guards,**{key:service[key] for key in
            ('runtime','revision','format','implementation','dtype','layout','replay_scope')},
            'family':'grouped_gemm','phase':'decode','recipe':op.recipe,
            'metadata.weight_format':'int8','metadata.activation_format':'int8',
            'metadata.output_format':'bf16','metadata.activation_scale_format':'fp32',
            'metadata.matrix_output_contract':'int32_cast_row_scale_column_scale_output_cast',
            'metadata.matrix_weight_bank_groups':256,
            'metadata.routing_basis':'probability_moments',
            'local_shape.1':n,'local_shape.2':k}
        rules.append({'id':'a5-ep1-int8-nz-bf16-'+op.recipe+'-joint-v1',
            'match':match,
            'coordinates':{'rows_per_declared_bank':{'path':'local_shape.0','scale':1/256}},
            'domain':{'rows_per_declared_bank':{'min':.125,'max':.5}},
            'response':{'fixed_us':fixed,'terms':[{'coordinate':'rows_per_declared_bank','us_per_unit':slope}]},
            'execution':{'kind':'joint_native','resource':QUEUE,'reference_rate':1e9,
                'native_kernel':train[0]['native_signature']['Name']},
            'provenance':{'kind':'conditional_native_operator_response',
                'boundary':'inclusive single GroupedMatmul kernel duration',
                'training':train,'original_role':'unassigned','assignment':'retrospective64K hold',
                'histogram':'group_list values unknown; rows/256 is an aggregate coordinate, not a balanced histogram',
                'native_weight_column_scale':'BF16 [256,N]; per-token scale FP32[M]',
                'scope':'Explicit EP1 caller case; column-scale format is retained in case declaration and native protocol, not exposed as an independent regime-match axis.',
                'limitations':['binary revision and actual route histogram unobserved',
                    'correlated calls within two captures, not independent hardware trials',
                    'no opaque duration converted into HBM/MTE/Cube supply',
                    'only unmodified physical-rate domain; changes revert to analytical program',
                    '126K sources and64K o32 sources never assign coefficients',
                    'no whole-Body totals, residuals or concurrency labels enter response']}})
    return rules


def validate(raw, native):
    before=deepcopy(raw);before['runtime']['kernels'].pop('service_regimes')
    rows=[]
    for source in native['cases']:
        for group in source['groups']:
            signature(group)
            geometry=group['forward_geometry_counts']
            assert len(geometry)==1
            batch=geometry[0]['sequences'];assert group['routed_rows']==batch*4*8
            op=next(o for o in grouped(raw,batch,source['geometry']['context_tokens'])
                    if (o.local_shape[1],o.local_shape[2])==(group['n'],group['k']))
            rule=next(r for r in raw['runtime']['kernels']['service_regimes']
                      if r['match']['recipe']==op.recipe)
            assert signature(group)==rule['provenance']['training'][0]['native_signature']
            baseline=operator_costs(before,[op],input_ready_ms=7)['operators'][0]
            result=operator_costs(raw,[op],input_ready_ms=7)['operators'][0]
            assert result['program']=='JointNativeService'
            assert result['analytical_reference_work']=={
                'cube':op.flops,'vector':op.vector_ops,'hbm':op.hbm_bytes,
                'mte2':op.mte2_bytes,'mte3':op.mte3_bytes}
            assert result['operator']==baseline['operator']
            assert not result['service_before_input'] and result['live_tasks']==0
            assert not result['service_delta']
            delta=result['duration_ms']*1000/group['mean_us']-1
            rows.append({'case_id':source['case_id'],'role':source['assignment_role'],
                'recipe':op.recipe,'routed_rows':group['routed_rows'],'sequences':batch,
                'native_mean_us':group['mean_us'],'predicted_us':result['duration_ms']*1000,
                'relative_error':delta,'within15pct':abs(delta)<=.15,
                'baseline_analytical_us':baseline['duration_ms']*1000,
                'analytical_reference_work':result['analytical_reference_work'],
                'serviced':result['serviced'],'declared_service':result['declared_service'],
                'service_before_input':result['service_before_input'],'live_tasks':result['live_tasks']})
    checks=[]
    for axis,key in [('rates','hbm'),('rates','mte2_aic'),('rates','vector'),('compute_peaks','int8')]:
        altered=deepcopy(raw)
        for die in altered['machine']['devices']:die[axis][key]*=.5
        plain=deepcopy(altered);plain['runtime']['kernels'].pop('service_regimes')
        for op in grouped(altered,1):
            a=operator_costs(altered,[op])['operators'][0];b=operator_costs(plain,[op])['operators'][0]
            assert a['program']=='ExpectedGroupedMatmul'
            assert a['duration_ms']==b['duration_ms'] and a['serviced']==b['serviced']
            checks.append({'axis':axis+'.'+key,'recipe':op.recipe,'program':a['program'],
                           'duration_ms':a['duration_ms'],'equal_plain_analytical':True})
    for op in grouped(raw,8):
        assert operator_costs(raw,[op])['operators'][0]['program']=='ExpectedGroupedMatmul'
    return {'boundary':'native primitive only; no complete Body prediction run',
        'split':'retrospective source separation; original role unassigned',
        'cases':rows,'physical_rate_miss_controls':checks,'outside_rows256':'analytical',
        '126k_pairs':sum(r['role']=='fixed_126k_validation' for r in rows),
        '126k_within15pct':sum(r['role']=='fixed_126k_validation' and r['within15pct'] for r in rows)}


if __name__=='__main__':
    native=json.loads((HERE/'native.json').read_text());raw=make_case();rules=assign(raw,native)
    raw['runtime']['kernels']={'service_regimes':rules}
    raw['runtime']['metadata']['native_grouped_response']={
        'source_case':identity(SOURCE),'parameters':'data/observations/a5-qwen35-ep1-grouped-20261030/parameters.json',
        'role':'explicit conditional grouped response; EP1 retained',
        'case_change':'routed_expert column scale BF16/2B from native ABI; all other quant categories unchanged'}
    raw['runtime']['metadata']['new_latency_parameter_fit']=True
    write(HERE/'parameters.json',{'schema':'operator-service-calibration/1',
        'producer':identity(Path(__file__)),'native_ledger':identity(HERE/'native.json'),
        'service_regimes':rules,'source_case':identity(SOURCE),'body_durations_used':False})
    write(CASE,raw)
    report=validate(raw,native);write(HERE/'validation.json',report)
    print(json.dumps({'case':str(CASE),'126k_pairs':report['126k_pairs'],
        '126k_within15pct':report['126k_within15pct'],'coefficients':[
            [r['id'],r['response']] for r in rules]},indent=2))
