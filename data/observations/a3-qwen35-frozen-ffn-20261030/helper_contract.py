"""Verify the retained helper geometry without assigning a timing parameter.

[Role] Preserve source and native consumer evidence for q2-to-q4 helper reuse.
[Invariants] N/A tensor descriptors stay unknown. Recurrent history/checkpoint
  differences are excluded from the helper contract, never normalized away.
[Timeouts & Locks] Small local JSON/source reads and 15-second Git reads only.
[Side-Effect Policy] Read-only; returns a compact source-conditional contract.
"""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

SOURCE_ROOT=Path('/Volumes/data/code/vllm-ascend-pr16544-review-fixes')
SOURCE_REVISION='8c30e4f491f73cba737a1dacc522894e437cdf3d'


def shape(text):
    return [tuple(int(v) for v in x.split(',')) for x in text.strip('"').split(';')]


def _retained(path,lines):
    data=path.read_bytes()
    return {'path':str(path),'sha256':sha256(data).hexdigest(),'size_bytes':len(data),'lines':lines}


def _git_source(path,lines):
    data=subprocess.check_output(['git','show',f'{SOURCE_REVISION}:{path}'],cwd=SOURCE_ROOT,timeout=15)
    return {'repository':str(SOURCE_ROOT),'revision':SOURCE_REVISION,'path':path,
            'sha256':sha256(data).hexdigest(),'size_bytes':len(data),'lines':lines}


def helper_contract(root,here,old_records):
    source=root/'data/observations/sources/gdn-image-c3d00a196905'
    report={'scope':'source_geometry_homology_for_frozen_helper_responses',
        'sources':{
            'caller':_retained(source/'gdn.py.txt',[[317,340],[367,374],[532,536]]),
            'gating':_retained(source/'fused_gdn_gating.py.txt',[[14,56],[59,101]]),
            'norm_kernel':_git_source('vllm_ascend/ops/triton/layernorm_gated.py',[[16,103],[106,168]]),
            'norm_wrapper':_git_source('vllm_ascend/ops/layernorm.py',[[157,197]])},
        'geometry':{
            'gating':'a,b contiguous [T,8]; wrapper sets seq_len=1 and grid=(40,1), BLK_HEADS=8, BLK_BATCHES=64; no per-request q/context/history argument',
            'gating_dtype':'BF16 a,b and beta; FP32 g; native beta/g consumer slots corroborate output precision',
            'norm_gate':'caller reshapes BF16 recurrent output and z to [T*8,128]; one 128-wide group, BLOCK_M=64, grid=(ceil(T*8/64),1); no q/context/history argument',
            'expert_add':'two BF16 [T,2048] operands and BF16 output, ND, scalar FLOAT, AIV Block40/Mix0'},
        'limitations':['Source geometry and native exported fields agree; capture binary identity is unobserved.',
            'Gating/norm tensor descriptors are N/A; their full typed ABI is source-conditional, not directly exported.',
            'Old q2 and new q4 recurrent sequence metadata, state capacity, history and checkpoints differ; no recurrent/conv/route_pack response is transferred.'],
        'new_coefficients':0,'cases':[]}
    for batch in (1,4):
        t=batch*4
        ledger=json.loads((here/f'native-b{batch}.json').read_text())
        new,=[g for g in ledger['native_groups'] if g['abi']['Type']=='RecurrentGatedDeltaRule']
        old=next(r for r in old_records if r['op']=='recurrent' and r['width']==2048 and
                 r['coordinate']==t and r['split']=='operator_calibration')
        a,b=old['native_abi'],new['abi']
        sa,sb=shape(a['Input Shapes']),shape(b['Input Shapes'])
        da,db=a['Input Data Types'].split(';'),b['Input Data Types'].split(';')
        assert sa[3]==sb[3]==(t,8) and da[3]==db[3]=='DT_BF16'
        assert sa[7]==sb[7]==(t,8) and da[7]==db[7]=='FLOAT'
        assert shape(a['Output Shapes'])[0]==shape(b['Output Shapes'])[0]==(t,8,128)
        assert a['Output Data Types'].split(';')[0]==b['Output Data Types'].split(';')[0]=='DT_BF16'
        helper=[]
        for op,typ,blocks in [('gating','fused_gdn_gating_kernel',40),
                              ('norm_gate','_layer_norm_fwd_1pass_kernel_npu',(t*8+63)//64),
                              ('expert_add','ForeachAddList',40)]:
            group,=[g for g in ledger['native_groups'] if g['abi']['Type']==typ]
            matches=[r for r in old_records if r['op']==op and r['width']==2048 and
                     r['coordinate']==t and r['native_abi']==group['abi']]
            assert matches and group['abi']['Block Num']==str(blocks)
            helper.append({'op':op,'old_rows':len(matches),'native_abi':group['abi'],
                'expected_source_blocks':blocks,'old_source':{k:matches[0][k] for k in ('capture','sha256','first')},
                'new_locations':group['locators']})
        report['cases'].append({'batch':batch,'tokens':t,'old_q':2,'new_q':4,
            'same_flat_geometry':True,'gating_grid':[40,1],'norm_grid':[(t*8+63)//64,1],
            'old_recurrent_consumer':{k:old[k] for k in ('capture','sha256','native_abi','first')},
            'new_recurrent_consumer':{'source':ledger['source_identity'],'native_abi':new['abi'],'locations':new['locators']},
            'helper_checks':helper})
    return report
