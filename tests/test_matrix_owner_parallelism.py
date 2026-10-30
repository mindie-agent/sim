"""Underfilled matrix grids expose physical owners without changing the GEMM."""
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.matrix_mapping import DEFAULT_CANDIDATES, UNDERFILLED_CANDIDATES, owner_tiles

ROOT = Path(__file__).resolve().parents[1]


def price(m=128, n=288, k=4096, *, groups=1, options=None, matrix_scale=1, dtype='fp32'):
    raw = json.loads((ROOT / 'examples/quickstart.json').read_text())
    raw['machine'] = {'backend': 'ascend', 'sku': 'Ascend950DT_9582', 'num_devices': 1,
                      'device': {'compute_peaks': {'fp32': 27.0336e12 * matrix_scale}}}
    raw['runtime']['kernels'] = {'programs': {'matmul': {'kind': 'steady_matmul', **(options or {})}}}
    op = gemm_op(op_id='matrix', deps=('input',), stage=0, layer=0, requests=(),
                 m=m, n=n, k=k, weight_count=groups,
                 quant=QuantSpec(dtype, weight_format=dtype, activation_format=dtype,
                                 weight_block=(128, 128)),
                 category='dense_mlp', phase='decode')
    return graph_costs(raw, (op,), external_inputs={'input': 7}, isolated=False)


def mapping(report):
    return report['parameters']['matrix_mappings'][0]


def test_small_rectangular_matrix_uses_more_owners_with_same_semantic_work():
    old = price(options={'underfilled_candidates': ()})
    new = price()
    chosen = mapping(new)['selected']
    assert len(owner_tiles(128, 288, chosen['tile_m'], chosen['tile_n'], 32)) >= 16
    assert len(owner_tiles(128, 288, 128, 128, 32)) == 3
    assert new['duration_ms'] - 7 < (old['duration_ms'] - 7) / 4
    before, after = old['execution']['serviced'], new['execution']['serviced']
    assert after['matrix@r0:s0'] == pytest.approx(before['matrix@r0:s0'])
    assert after['memory@r0:s0'] == pytest.approx(before['memory@r0:s0'])
    # Independent owners fetch more private operands; more parallelism is not
    # free reuse across MTE2/L1. Global HBM reuse remains the declared policy.
    assert after['mte2_aic@r0:s0'] > before['mte2_aic@r0:s0']
    assert all(row['ready_ms'] >= 7 for row in new['timeline'])
    assert new['execution']['live_tasks'] == new['execution']['live_programs'] == 0
    assert all(abs(v) < 1e-6 for v in new['execution']['live_allocated'].values())


def test_independent_group_axes_and_ragged_edges_keep_logical_totals():
    result = price(m=57, n=145, k=319, groups=3)
    service = result['execution']['serviced']
    assert service['matrix@r0:s0'] == pytest.approx(2 * 57 * 145 * 319 / 27.0336e12 * 1000)
    assert service['memory@r0:s0'] == pytest.approx((57 * 319 + 3 * 145 * 319 + 57 * 145) * 4)
    assert mapping(result)['groups'] == 3
    assert result['execution']['live_tasks'] == 0


def test_explicit_candidate_domain_and_already_full_grid_are_preserved():
    fixed = price(options={'candidates': DEFAULT_CANDIDATES})
    old = price(options={'underfilled_candidates': ()})
    assert fixed['duration_ms'] == old['duration_ms']
    large = price(m=1024, n=4096, k=128)
    control = price(m=1024, n=4096, k=128, options={'underfilled_candidates': ()})
    assert large['duration_ms'] == control['duration_ms']
    assert large['execution']['serviced'] == control['execution']['serviced']


def test_hardware_compute_sweep_changes_execution_without_changing_flops():
    reports = [price(matrix_scale=factor) for factor in (.5, 1, 2)]
    assert reports[0]['duration_ms'] > reports[1]['duration_ms'] > reports[2]['duration_ms']
    for factor, report in zip((.5, 1, 2), reports):
        assert report['execution']['serviced']['matrix@r0:s0'] * factor == pytest.approx(
            reports[1]['execution']['serviced']['matrix@r0:s0'])


@pytest.mark.parametrize('dtype', ['bf16', 'fp8_e4m3', 'int8'])
@pytest.mark.parametrize('n', [24, 288])
def test_low_precision_default_keeps_base_mapping_and_explicit_subtiles_remain_available(dtype, n):
    default = price(m=64, n=n, dtype=dtype)
    base = price(m=64, n=n, dtype=dtype,
                 options={'candidates': DEFAULT_CANDIDATES})
    explicit = price(m=64, n=n, dtype=dtype,
                     options={'underfilled_candidates': UNDERFILLED_CANDIDATES})
    assert default['duration_ms'] == base['duration_ms']
    assert default['execution']['serviced'] == base['execution']['serviced']
    assert mapping(default)['selected']['tile_m'] >= 128
    assert mapping(default)['selected']['tile_n'] >= 128
    assert explicit['duration_ms'] < default['duration_ms']
    assert explicit['execution']['serviced']['matrix@r0:s0'] == pytest.approx(
        default['execution']['serviced']['matrix@r0:s0'])
    assert explicit['execution']['serviced']['memory@r0:s0'] == pytest.approx(
        default['execution']['serviced']['memory@r0:s0'])
    assert all(row['ready_ms'] >= 7 for row in explicit['timeline'])
    assert explicit['execution']['live_tasks'] == explicit['execution']['live_programs'] == 0


def test_equal_hbm_bounds_do_not_add_another_serial_owner_wave():
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.matrix_steady import SteadyMatmul
    from mindie_sim.prediction.program import RecipeContext
    op = gemm_op(op_id='groups', deps=(), stage=0, layer=0, requests=(),
                 m=258, n=33, k=4096, weight_count=2,
                 quant=QuantSpec('bf16'), category='dense_mlp', phase='decode')
    pools = {'l1': 1024 * 1024, 'l0a': 128 * 1024, 'l0b': 128 * 1024, 'l0c': 512 * 1024}
    times = []
    for factor in (.5, 1, 2):
        rates = {'cube': 39e6, 'hbm': 34500, 'mte2_aic': 802000,
                 'mte2_aiv': 1, 'mte1': 192000 * factor, 'mte3': 102000, 'vector': 1e9}
        units = {key: 1 if key == 'hbm' else 32 for key in rates}
        context = RecipeContext(op, {'cube': op.flops}, {key: key for key in rates}, rates, 32,
                                resource_units=units, pool_names={key: key for key in pools},
                                pool_capacities=pools, pool_units={key: 32 for key in pools})
        engine = Engine([Resource(key, rate, units=units[key]) for key, rate in rates.items()],
                        pools=pools, pool_units={key: 32 for key in pools})
        done = SteadyMatmul(underfilled_candidates=UNDERFILLED_CANDIDATES).lower(context).start(
            engine, 'groups', engine.submit('input', release=7))
        engine.advance(until=(done,))
        times.append(done.time - 7)
        assert engine.serviced['hbm'] == pytest.approx((258 * 4096 + 2 * 33 * 4096 + 258 * 33) * 2)
        assert not engine._tasks
    # At the HBM bound, smaller residency used to select two serial output
    # waves and make doubled MTE1 slower. The same bound has a one-wave plan.
    assert times[0] > times[1] >= times[2] - 1e-9
