"""Source-compatible dense MLA binding for the retained SplitCore program.

[Role] Bind q1 BF16 latent MLA and the existing caller head materializations.
[Invariants] The shared caller ABI projects geometry-derived SplitCore
  ownership unless an explicit host declaration overrides it. Retained KV rows
  occupy tile storage independently of query capacity and active query work.
[Timeouts & Locks] O(batch + host owners), no locks or I/O.
[Side-Effect Policy] Appends immutable ordinary graph operators only.
"""
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef

from mindie_sim.prediction.operators.paged_mla import family_paged_mla, paged_mla_membase

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = paged_mla_membase.SOURCES
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


def _validate_declaration_geometry(geometry):
    paged_mla_membase.validate_domain(
        geometry['contexts'],
        geometry['physical_heads'],
        page_size=geometry['page_size'],
        retained_kv_rows=geometry['retained_kv_rows'],
    )


def native_contract(**kwargs):
    """Derive the source host partition when ``dispatch=None``."""
    _validate_retained(kwargs.get('retained_kv_rows', 0))
    geometry = family_paged_mla.native_geometry(
        views=kwargs['views'], plan=kwargs['plan'],
        parallel=kwargs['parallel'], runtime=kwargs['runtime'],
        padded_tokens=kwargs.get('padded_tokens'),
        kv_splits=kwargs.get('kv_splits', 1),
        native_query_capacity=kwargs.get('native_query_capacity'),
        dispatch=kwargs.get('dispatch'), source_host_dispatch=True,
        retained_kv_rows=kwargs.get('retained_kv_rows', 0))
    _validate_declaration_geometry(geometry)
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
    views = kwargs['views']
    plan = kwargs['plan']
    parallel = kwargs['parallel']
    logical = plan.num_heads // parallel.tp
    physical = 1 << (logical - 1).bit_length()
    paged_mla_membase.validate_domain(
        tuple(view.context for view in views), physical,
        retained_kv_rows=retained_kv_rows)


def lower_attention(b, *, padded_tokens=None, dispatch=None, retained_kv_rows=0,
                    native_query_capacity=None, **kwargs):
    _validate_retained(retained_kv_rows)
    geometry = family_paged_mla.native_geometry(
        views=kwargs['views'], plan=kwargs['plan'],
        parallel=kwargs['runtime'].parallel, runtime=kwargs['runtime'],
        padded_tokens=padded_tokens, kv_splits=kwargs.get('kv_splits', 1),
        native_query_capacity=native_query_capacity, dispatch=dispatch,
        source_host_dispatch=True, retained_kv_rows=retained_kv_rows)
    _validate_declaration_geometry(geometry)
    return family_paged_mla.lower_attention(
        b, padded_tokens=padded_tokens, dispatch=dispatch,
        native_query_capacity=native_query_capacity,
        retained_kv_rows=retained_kv_rows,
        implementation=IMPLEMENTATION, branch=BRANCH,
        program_recipe=PROGRAM_RECIPE, source_host_dispatch=True, **kwargs)
