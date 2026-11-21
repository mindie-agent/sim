"""Source-compatible dense MLA binding for the retained SplitCore program.

[Role] Bind q1 BF16 latent MLA and the existing caller head materializations.
[Invariants] The shared caller ABI projects geometry-derived SplitCore
  ownership unless an explicit host declaration overrides it. Retained KV rows
  occupy tile storage independently of query capacity and active query work.
[Timeouts & Locks] O(batch + host owners), no locks or I/O.
[Side-Effect Policy] Appends immutable ordinary graph operators only.
"""
from mindie_sim.schema.common import ModelingError

from . import family_paged_mla, paged_mla_membase

IMPLEMENTATION = paged_mla_membase.IMPLEMENTATION
BRANCH = 'arch22_bf16_noquant_mla'
PROGRAM_RECIPE = 'mindie_sim.prediction.paged_mla_membase.build'


def recipes():
    return {IMPLEMENTATION: family_paged_mla.planned_recipe(
        paged_mla_membase.build)}


def _common_kwargs(kwargs):
    options = dict(kwargs)
    options.update(
        implementation=IMPLEMENTATION,
        branch=BRANCH,
        program_recipe=PROGRAM_RECIPE,
        source_host_dispatch=True)
    return options


def _validate_retained(retained_kv_rows):
    if (type(retained_kv_rows) is not int
            or not 0 <= retained_kv_rows <= paged_mla_membase.KEY_TILE):
        raise ModelingError('paged_mla_arch22_retained_rows_must_fit_source_s2_tile')


def native_contract(**kwargs):
    """Derive the source host partition when ``dispatch=None``."""
    _validate_retained(kwargs.get('retained_kv_rows', 0))
    return family_paged_mla.native_contract(**_common_kwargs(kwargs))


def native_plan(**kwargs):
    """Return the SplitCore recipe plan for the same declaration."""
    _validate_retained(kwargs.get('retained_kv_rows', 0))
    options = _common_kwargs(kwargs)
    options.setdefault('build', paged_mla_membase.build)
    return family_paged_mla.native_plan(**options)


def validate(*, dispatch=None, retained_kv_rows=0, native_query_capacity=None, **kwargs):
    family_paged_mla.validate(dispatch=dispatch,
                              native_query_capacity=native_query_capacity, **kwargs)
    _validate_retained(retained_kv_rows)


def lower_attention(b, *, padded_tokens=None, dispatch=None, retained_kv_rows=0,
                    native_query_capacity=None, **kwargs):
    _validate_retained(retained_kv_rows)
    return family_paged_mla.lower_attention(
        b, padded_tokens=padded_tokens, dispatch=dispatch,
        native_query_capacity=native_query_capacity,
        retained_kv_rows=retained_kv_rows,
        implementation=IMPLEMENTATION, branch=BRANCH,
        program_recipe=PROGRAM_RECIPE, source_host_dispatch=True, **kwargs)
