"""[Role] Assign one explicit conditional A5 DSQ effective-service regime.
[Invariants] Only the c64 operator mean determines the coefficient. No Body
  duration or held-out response enters assignment. Group geometry is declared,
  not reconstructed from allocation capacity or unavailable group_list values.
[Timeouts & Locks] Local compact inputs, no subprocess/network/locks.
[Side-Effect Policy] Writes parameter/role manifests and separate example cases.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from mindie_sim.prediction.structure import parse_case

HERE=Path(__file__).resolve().parent
OBS=ROOT/'data/verification/qwen35-source-residual-pmu-20261030.json'
BASE=ROOT/'examples/cases/high-concurrency-20261030/qwen35-mc2-tp-dsq-source-conditional'
OUT=ROOT/'examples/cases/high-concurrency-20261030/qwen35-dsq-effective-service-conditional'


def identity(path):
    return {'path':str(path.relative_to(ROOT)),'sha256':sha256(path.read_bytes()).hexdigest(),'size_bytes':path.stat().st_size}


def main():
    observations=json.loads(OBS.read_text())
    calibration=next(r for r in observations if r['c']==64)
    family=next(f for f in calibration['families'] if f['role']=='DequantSwigluQuant')
    base_path=BASE/'qwen35-35b-a5-tp4-c64-mc2-tp-dsq-source.json'
    case=parse_case(json.loads(base_path.read_text()))
    reference=dict(case.machine.dies[0].rates)['vector']
    visits=64 # Explicit source normal-branch ownership, independently retained.
    coefficient=family['mean_us']['Duration(us)']/visits
    regime={
      'id':'a5-arch35-dsq-full-load-normal-smallrows-effective-service-v1',
      'match':{'hardware':'Ascend950DT_9582','recipe':'grouped_int32_dequant_swiglu','dtype':'int8',
        'metadata.program_execution.grouped_dequant_schedule.kind':'arch35_full_load',
        'metadata.program_execution.grouped_dequant_schedule.host_special_group_cut':False},
      'coordinates':{
        'visits':{'path':'metadata.program_execution.grouped_dequant_schedule.groups_per_owner','reduce':'max'},
        'width':{'path':'local_shape.1'},
        'max_group_rows':{'path':'metadata.grouped_dequant.expert_rows','reduce':'max'},
        'min_group_rows':{'path':'metadata.grouped_dequant.expert_rows','reduce':'min'},
        'rows':{'path':'local_shape.0'},
        'capacity':{'path':'metadata.program_execution.grouped_dequant_schedule.input_capacity_rows'},
        'aiv_units':{'path':'metadata.program_execution.grouped_dequant_schedule.physical_aiv'}},
      'domain':{'visits':{'min':64,'max':64},'width':{'min':512,'max':512},
        'min_group_rows':{'min':8,'max':16},'max_group_rows':{'min':8,'max':16},
        'rows':{'min':512,'max':1024},'capacity':{'min':2048,'max':4096},'aiv_units':{'min':64,'max':64}},
      'response':{'fixed_us':0,'terms':[{'coordinate':'visits','us_per_unit':coefficient,
        'scale_with':{'resource':'vector','reference_rate':reference}}]},
      'provenance':{'kind':'conditional_operator_service_calibration',
        'boundary':'inclusive whole native DSQ device-call completion, not incremental latency',
        'geometry':'declared forced-balanced histogram; actual group_list values unavailable',
        'calibration_observation':identity(OBS),'calibration_cohort':{'concurrency_label':64,'calls':family['calls'],
          'mean_us':family['mean_us']['Duration(us)'],'source_normal_owner_group_visits':visits},
        'assignment':'mean native duration / maximum positive group visits per source AIV owner',
        'response_semantics':'max(existing physical execution, effective inclusive service)',
        'scaling_assumption':'inverse declared total AIV vector throughput; engineered extrapolation, not measured cross-rate response',
        'excluded':['Body_total_fit','per_model_selector','batch_selector','PMU_sum','hidden_extra_polling','allocation_as_valid_rows'],
        'data_roles':'new parameter-assignment/validation usage only; raw historic split remains unassigned',
        'cross_model_validation':'pending same implementation/domain observations; not established by this receipt'}}
    manifest={'schema':'operator-service-calibration/1','producer':identity(Path(__file__)),
      'service_regimes':[regime],
      'observation_roles':[{'source':identity(OBS),'cohort_concurrency_label':64,'role':'parameter_assignment',
        'raw_original_role':'unassigned','native_operator':'DequantSwigluQuant'},
        {'source':identity(OBS),'cohort_concurrency_label':128,'role':'not_used_in_this_parameter_assignment',
         'raw_original_role':'unassigned','native_operator':'DequantSwigluQuant',
         'independence':'previously inspected observations; not a historically blinded holdout'},
        {'source':identity(OBS),'cohort_concurrency_label':4,'role':'out_of_domain_nonapplication_control',
         'raw_original_role':'unassigned','native_operator':'DequantSwigluQuant'}],
      'not_calibrated':{'grouped_matmul':'active bank distributions and exact output branch remain conditional; separate coefficients not assigned here',
        'v41':'captured group_list/peer-valid rows unavailable; do not derive bank service from capacity',
        'Body':'no Body durations read for coefficient assignment'}}
    (HERE/'parameters.json').write_text(json.dumps(manifest,indent=2)+'\n')
    OUT.mkdir(exist_ok=True)
    for c in (4,64,128):
        p=BASE/f'qwen35-35b-a5-tp4-c{c}-mc2-tp-dsq-source.json'
        raw=json.loads(p.read_text()); raw['name']+=' / conditional effective DSQ service'
        raw['runtime']['kernels']['service_regimes']=deepcopy(manifest['service_regimes'])
        raw['runtime']['metadata']['dsq_effective_service_condition']={
          'parent':identity(p),'parameters':identity(HERE/'parameters.json'),
          'scope':'only matched normal-branch DSQ; special branch remains physical source control',
          'geometry':'declared balanced valid rows; upstream allocation ABI remains incomplete'}
        (OUT/f'qwen35-35b-a5-tp4-c{c}-dsq-effective-service.json').write_text(json.dumps(raw,indent=2)+'\n')
    print(json.dumps({'coefficient_us_per_owner_group_visit':coefficient,'reference_vector_rate':reference,
      'calibration_calls':family['calls'],'new_cases':3}))

if __name__=='__main__':main()
