"""Fit observed A5 local service responses, retaining unused geometry checks.

[Role] Bind direction/shape-specific copy, routing and collective responses.
[Invariants] Parameters use native call observations, never Body gaps. B16/B64
  assign affine terms; B32 is unused by this assignment, not historically blind.
[Timeouts & Locks] Bounded retained JSON inputs; no remote work or locks.
[Side-Effect Policy] Write explicit reusable parameters and complete case inputs.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mindie_sim.prediction.structure import parse_case


def load(path):
    assert path.stat().st_size < 50 * 1024**2
    return json.loads(path.read_text())


def identity(path):
    return {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()}


def affine(a, b):
    slope = (b[1] - a[1]) / (b[0] - a[0])
    intercept = a[1] - slope*a[0]
    assert slope >= 0 and intercept >= 0
    return intercept, slope


def main():
    directory = Path(__file__).resolve().parent
    source = ROOT/'data/verification/conv-state-memory-service-audit-20261030.json'
    copy = {r['batch']: r for r in load(source)['rows']}
    profiler = ROOT.parent/'profiler/examples/structure/acceptance-20261030-glm-source-complete-operators/families'
    family, sources = {}, [identity(source)]
    for batch in (16, 32, 64):
        path = profiler/f'a5_glm53flash_tp8_c{batch}_p4k/body-operator-family-scoreboard.json'
        family[batch] = load(path)['cases'][0]['families']
        sources.append(identity(path))
    parent = ROOT/'examples/cases/high-concurrency-20261030/glm-hccl-join-reduce-conditional'
    reference = load(parent/'a5_glm53flash_tp8_c64_p4k.json')
    hardware = parse_case(reference).machine
    rates = dict(hardware.dies[0].rates)
    rules, checks = [], []

    def add(name, match, coordinate, values, *, resource=None, domain=None, more_coordinates=None, more_domain=None):
        intercept, slope = affine((16, values[16]), (64, values[64]))
        prediction = intercept + 32*slope
        checks.append({'id': name, 'parameter_assignment': [16, 64],
                       'observed_us': values, 'unused_in_assignment': 32,
                       'validation_predicted_us': prediction,
                       'validation_relative_error': prediction/values[32]-1})
        term = {'coordinate': 'rows', 'us_per_unit': slope}
        if resource:
            term['scale_with'] = {'resource': resource, 'reference_rate': rates[resource]}
        rules.append({'id': name,
            'match': {'hardware': hardware.hardware_key or hardware.sku, **match},
            'coordinates': {'rows': coordinate, **(more_coordinates or {})},
            'domain': {'rows': domain or {'min': 16, 'max': 64}, **(more_domain or {})},
            'response': {'fixed_us': intercept, 'terms': [term]},
            'provenance': {'basis': 'observed_native_call_affine_service',
                'original_split': 'unassigned', 'assignment_batches': [16, 64],
                'validation_batch': 32, 'validation_history': 'previously_examined_not_pristine_holdout',
                'observed_mean_us': values, 'sources': sources,
                'rate_interpretation': ('effective_service_scaling_assumption_not_isolated_pipe_measurement'
                                        if resource else 'fixed_protocol_response_in_declared_hardware_regime')}})

    for direction in ('gather', 'scatter'):
        values = {b: copy[b]['directions'][direction]['pmu']['Duration(us)']['mean'] for b in copy}
        add('a5-strided-state-'+direction,
            {'recipe': 'triton_noncontiguous_conv_state_copy', 'dtype': 'bf16',
             'metadata.direction': direction, 'local_shape.1': 6, 'local_shape.2': 3072,
             'resource_units.vector': 64}, {'path': 'local_shape.0'}, values,
            resource='mte3' if direction == 'scatter' else 'mte2_aiv')

    def means(native):
        result = {}
        for b, rows in family.items():
            samples = [f['observed_us'] for f in rows if f['native_operator_type'] == native]
            result[b] = sum(s['count']*s['mean'] for s in samples)/sum(s['count'] for s in samples)
        return result

    add('a5-mxfp8-routing-count',
        {'recipe': 'ascend_moe_init_routing_v3_mxfp8', 'dtype': 'bf16',
         'metadata.moe_init_config.experts': 288, 'metadata.moe_init_config.topk': 8,
         'metadata.moe_init_config.width': 4096},
        {'path': 'metadata.moe_init_config.tokens', 'scale': .25}, means('MoeInitRoutingV3'),
        resource='vector')
    add('a5-aicpu-mesh-oneshot-payload',
        {'kernel': 'HCCL_AICPU_TS_allreduce_mesh_oneshot',
         'metadata.hccl_selection.ranks': 8, 'metadata.hccl_selection.dtype_bytes': 2},
        {'path': 'metadata.hccl_selection.payload_bytes', 'scale': 1/32768},
        means('hcom_allReduce_AicpuKernel'))
    add('a5-mxfp8-grouped-swiglu-small-banks',
        {'recipe': 'ascend_mxfp8_grouped_swiglu_quant',
         'metadata.mxfp8_swiglu_config.k': 4096, 'metadata.mxfp8_swiglu_config.width': 256},
        {'path': 'local_shape.0', 'scale': 1/32}, means('GroupedMatmulSwigluQuantV2'),
        resource='vector', more_coordinates={'groups': {'path': 'metadata.mxfp8_swiglu_config.rows', 'reduce': 'len'}},
        more_domain={'groups': {'min': 288, 'max': 288}})
    packet = {'schema': 'mindie-sim.observable-regimes/1', 'rules': rules,
              'validation': checks, 'scope': 'native_kernel_response_not_body_fit'}
    (directory/'parameters.json').write_text(json.dumps(packet, indent=2)+'\n')
    output = parent.parent/'glm-observable-service'
    output.mkdir(exist_ok=True)
    for batch in (16, 32, 64):
        raw = deepcopy(load(parent/f'a5_glm53flash_tp8_c{batch}_p4k.json'))
        raw['name'] = f'a5_glm53flash_tp8_c{batch}_p4k:observable-service'
        raw['runtime']['kernels']['service_regimes'] = rules
        raw['runtime'].setdefault('metadata', {})['observable_service_parameters'] = str((directory/'parameters.json').relative_to(ROOT))
        (output/f'a5_glm53flash_tp8_c{batch}_p4k.json').write_text(json.dumps(raw, indent=2)+'\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
