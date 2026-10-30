"""Physical labels may share compilation; resource aliases and ownership may not."""
from dataclasses import replace
from math import lcm

import pytest

from mindie_sim.prediction.mxfp4_panels import compiled_panels, panel_parts
from mindie_sim.prediction.mxfp4_geometry import panel_sizes
from mindie_sim.prediction.refill_steady import compact_refill
from test_mxfp4_grouped import source_case, gate, physical_op
from test_mxfp4_panel_ir_cache import source_ir, execute


def uncached(context, args):
    k, m, n = (args[key] for key in ('k', 'm', 'n'))
    kb, ka, _ = panel_sizes(m, n)
    panels = (k + kb - 1) // kb
    count, remainder = divmod(panels, lcm(ka, 4096) // kb)
    if not remainder and k % kb:
        count -= 1
    period = lcm(ka, 4096) // kb
    def parts(index):
        return panel_parts(context, index=index,
                           **{key: value for key, value in args.items() if key != 'shared_clients'})
    return compact_refill(context, tuple(parts(i) for i in range(period)) if count else (), count,
                          tuple(parts(i) for i in range(count * period, panels)),
                          shared_clients=args['shared_clients'])


def rename(context, suffix):
    names = {name: name.replace('@r0:', f'@r{suffix}:') for name in context.rates}
    return replace(context, resource_names={role: names[name] for role, name in context.resource_names.items()},
                   rates={names[name]: rate for name, rate in context.rates.items()})


def test_256_physical_owner_bindings_share_two_templates(source_case, gate):
    context, _, _ = source_ir(source_case, physical_op(gate, (1,), 160, 5120))
    compiled_panels.cache_clear()
    for rank in range(8):
        physical = rename(context, rank)
        for owner in range(32):
            for k in (5120, 256):
                args = dict(k=k, m=1, n=160, repeat=1, owner=owner, cores=32, aivs=64,
                            directional=False, shared_clients={physical.resource_names['hbm']:32})
                bound = compiled_panels(physical, **args)
                if rank in (0, 7) and owner in (0, 31):
                    original = uncached(physical, args)
                    assert bound == original
                    assert execute(bound, physical) == execute(original, physical)
    assert compiled_panels.cache_info().misses == 512
    assert compiled_panels.template_cache_info().misses == 2
    assert compiled_panels.template_cache_info().hits == 510


@pytest.mark.parametrize('pipe_units', (16, 32, 64, 128))
@pytest.mark.parametrize('owner', (0, 15, 31))
def test_divisible_pipe_ownership_rebinds_real_units(source_case, gate, pipe_units, owner):
    context, _, _ = source_ir(source_case, physical_op(gate, (1,), 160, 320))
    context = replace(context, resource_units={**context.resource_units,
                      'mte2_aiv': pipe_units, 'mte3': pipe_units,
                      'mte1': pipe_units, 'mte2_aic': pipe_units})
    args = dict(k=320, m=3, n=160, repeat=2, owner=owner, cores=32, aivs=64,
                directional=False, shared_clients={context.resource_names['hbm']:3})
    original = uncached(context, args)
    cached = compiled_panels(context, **args)
    assert cached == original
    assert execute(cached, context) == execute(original, context)


def test_alias_changes_and_directional_modes_keep_exact_numeric_contract(source_case, gate):
    context, _, _ = source_ir(source_case, physical_op(gate, (1,), 160, 4160))
    # Read and publish share a physical AIV pipe; all programmed phases must
    # retain that alias rather than independently receiving two capacities.
    alias = context.resource_names['mte2_aiv']
    context = replace(context, resource_names={**context.resource_names, 'mte3':alias},
                      service_modes={**context.service_modes,
                         'mte1_l0a':('mte1', 1e10), 'mte1_l0b':('mte1', 2e10)})
    for owner in (0, 31):
        for share in (1, 7):
            for factor in (.5, 1, 2):
                physical = replace(context, rates={**context.rates, alias:context.rates[alias]*factor})
                args = dict(k=4160, m=17, n=160, repeat=2, owner=owner, cores=32, aivs=64,
                            directional=True, shared_clients={physical.resource_names['hbm']:share})
                cached = compiled_panels(physical, **args)
                original = uncached(physical, args)
                assert cached == original
                assert execute(cached, physical) == execute(original, physical)


def test_resource_sort_order_is_part_of_normalized_alias_topology(source_case, gate):
    context, _, _ = source_ir(source_case, physical_op(gate, (1,), 160, 320))
    names = {name: f'z{index:03d}' for index, name in enumerate(reversed(sorted(context.rates)))}
    context = replace(context, resource_names={role:names[name] for role,name in context.resource_names.items()},
                      rates={names[name]:rate for name,rate in context.rates.items()})
    args = dict(k=320, m=3, n=160, repeat=1, owner=7, cores=32, aivs=64,
                directional=False, shared_clients={context.resource_names['hbm']:32})
    cached = compiled_panels(context, **args)
    original = uncached(context, args)
    assert cached == original
    assert execute(cached, context) == execute(original, context)


def test_aic_and_aiv_alias_keeps_overlapping_owner_sets(source_case, gate):
    context, _, _ = source_ir(source_case, physical_op(gate, (1,), 160, 320))
    alias = context.resource_names['mte2_aiv']
    context = replace(context, resource_names={**context.resource_names, 'mte2_aic':alias},
                      resource_units={**context.resource_units, 'mte2_aic':64})
    compiled_panels.cache_clear()
    for owner in (0, 7, 31):
        args = dict(k=320, m=3, n=160, repeat=1, owner=owner, cores=32, aivs=64,
                    directional=False, shared_clients={context.resource_names['hbm']:8})
        bound = compiled_panels(context, **args)
        original = uncached(context, args)
        assert bound == original
        assert execute(bound, context) == execute(original, context)
        load = bound.stages[0].program.stages
        assert load[0].placement[alias] == (owner*2, owner*2+1)
        assert load[1].placement[alias] == (owner*2,)
        assert load[7].placement[alias] == (owner*2+1,)
    assert compiled_panels.template_cache_info().misses == 1
