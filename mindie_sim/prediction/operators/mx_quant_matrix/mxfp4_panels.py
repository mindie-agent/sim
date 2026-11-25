"""Immutable physical panel programs for repeated MXA8W4 source geometry.

[Role] Compile the existing source panel DAG once per physical binding/shape.
[Invariants] Cache keys contain no operator, source metadata, callback or event.
  Cache hits create the same tasks in the same order with the same service.
  Output/Fixpipe and allocation lifetimes stay with the owning matrix program.
[Timeouts & Locks] At most 128 physical templates and 1024 bound IR entries;
  each contains one producer
  prologue, one coupled interior and one final consumer, never K-panel history.
[Side-Effect Policy] Pure IR construction; executing it uses the caller Engine.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from math import lcm
from types import MappingProxyType
from typing import Literal

from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.mechanism import Work
from .mxfp4_geometry import (
    MX_SCALE_PUBLISH_K, Config, activation_scale_bytes,
    panel_sizes, panel_work,
)
from .mxfp4_registers import CLASSES, widen, transpose
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage
from mindie_sim.prediction.operators.matrix.refill_steady import compact_refill
from .mxfp4_pipeline import Slots

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCE_REFS: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_cube_compute.h",
        source_sha256="8515f2f55c6b7fa9140193f231334c0d9b57ca6d1c6d9bb9d3509e994eb23846",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vec_compute.h",
        source_sha256="7ee00904eade2145ce3a3bd12d67883733b549240dd562be316563f6282f3493",
    ),
)
REGISTER_ROLES = ('reg_data_writes', 'reg_ub_accesses', 'reg_ub_stores', 'reg_shuffle_lanes')
TEMPLATE_CACHE_SIZE = 128


@dataclass(frozen=True)
class _Binding:
    names: tuple
    rates: tuple
    units: tuple
    peaks: tuple
    modes: tuple
    core_count: int

    @classmethod
    def from_context(cls, context):
        return cls(tuple(sorted(context.resource_names.items())),
                   tuple(sorted(context.rates.items())),
                   tuple(sorted(context.resource_units.items())),
                   tuple(sorted(context.compute_peaks.items())),
                   tuple(sorted((k, tuple(v)) for k, v in context.service_modes.items())),
                   context.core_count)

    def context(self):
        return RecipeContext(op=None, work={}, resource_names=dict(self.names),
                             rates=dict(self.rates), core_count=self.core_count,
                             resource_units=dict(self.units), compute_peaks=dict(self.peaks),
                             service_modes=dict(self.modes))


def _l2_permitted(context, *, l2_die: int | None = None) -> bool:
    if context.resource_names.get('l2') is None:
        return False
    view = getattr(context, 'admission_memory_view', None)
    if view is not None:
        if l2_die is not None:
            return view.effective_l2_bytes(l2_die) > 0
        caps = getattr(view, '_l2_capacity', None)
        if isinstance(caps, dict) and len(caps) == 1:
            sole_die = next(iter(caps))
            return view.effective_l2_bytes(sole_die) > 0
    return True


def panel_parts(context, *, k, m, n, repeat, index, owner, cores, aivs, directional,
                ownership=None, activation_l2=0.0, weight_l2=0.0, l2_die=None):
    """The existing source producer and micro-consumer DAG for one K panel.

    This helper is also available to the uncompressed caller; it does not
    choose a timing model or retain invocation/allocation state.  ``activation_l2``
    is the declared share of this column pass's activation panel whose lines are
    already resident from the first pass over the same rows; that share is served
    by the die's L2 rather than the off-chip stream.  ``weight_l2`` is the share
    of a B column panel re-issued from the owner's on-chip panel after the first
    M tile.
    """
    config = Config(k, n, (m,))
    ac, *vec = ownership if ownership is not None else _ownership(context, owner, cores, aivs)
    offset = index * panel_sizes(m, n)[0]
    a, vectors, micros = panel_work(config, m, n, offset)
    payload = a - activation_scale_bytes(config, m, offset)
    share = float(activation_l2)
    wshare = float(weight_l2)
    if (share > 0 or wshare > 0) and not _l2_permitted(context, l2_die=l2_die):
        share = wshare = 0.0
    activation = {'hbm': (payload * (1.0 - share) + (a - payload)) * repeat,
                  'mte2_aic': a * repeat}
    if share > 0:
        activation['l2'] = payload * repeat * share
    loads = [Stage('activation', context.bind(Work(resources=activation)), placement=ac)]
    for sub, work in enumerate(vectors):
        def vstage(name, resources, after=()):
            return Stage(f'{name}{sub}', context.bind(Work(resources=resources).scaled(repeat)),
                         after=after, placement=vec[sub])
        packed = {'hbm': work['weight'] * (1.0 - wshare), 'mte2_aiv': work['weight']}
        if wshare > 0:
            packed['l2'] = work['weight'] * wshare
        loads.extend((
            vstage('scale_read', {'hbm': work['scale'], 'mte2_aiv': work['scale']}),
            vstage('scale_transpose', transpose(work['gather_b16_lanes']).capacity_work(CLASSES).resources,
                   (f'scale_read{sub}',)),
            vstage('scale_publish', {'mte3': work['scale_publish']}, (f'scale_transpose{sub}',)),
            vstage('packed_read', packed, (f'scale_read{sub}',)),
            vstage('widen', widen(work['expanded']).capacity_work(CLASSES).resources,
                   (f'packed_read{sub}', f'scale_transpose{sub}')),
            vstage('publish', {'mte3': work['expanded']}, (f'widen{sub}', f'scale_publish{sub}')),
        ))
    steps = tuple((
        Stage('operand', context.bind(Work(resources=(
            {'mte1_l0a': left * repeat, 'mte1_l0b': right * repeat} if directional else
            {'mte1': (left + right) * repeat}))), placement=ac),
        Stage('cube', context.bind(Work({'fp8_e4m3': flops * repeat})),
              after=('operand',), placement=ac),
    ) for left, right, flops in micros)
    return tuple(loads), steps


def _immutable(program):
    # Only these immutable source definitions are cached. Events, iterators,
    # callbacks and allocation ownership are created by Program.start later.
    return KernelProgram(tuple(replace(
        stage, work=MappingProxyType(dict(stage.work)),
        caps=MappingProxyType(dict(stage.caps)),
        placement=MappingProxyType({key: tuple(units) for key, units in stage.placement.items()}),
        program=_immutable(stage.program) if stage.program is not None else None,
    ) for stage in program.stages), minimum_ms=program.minimum_ms)


def _ownership(context, owner, cores, aivs):
    return (context.owned(owner, cores, ('mte2_aic', 'mte1', 'cube', 'fixpipe')),
            *(context.owned(owner * 2 + sub, aivs,
                           ('mte2_aiv', 'vector', 'mte3', *REGISTER_ROLES))
              for sub in (0, 1)))


def _normalize(binding, owner, cores, aivs, shared_clients):
    """Keep physical alias/partition relationships; remove only their labels."""
    context = binding.context()
    placements = _ownership(context, owner, cores, aivs)
    names = {name: f'r{i:03d}' for i, name in enumerate(sorted(set(context.resource_names.values())))}
    units = {key: tuple(sorted({unit for item in placements for unit in item.get(key, ())}))
             for key in names}
    relative = tuple(tuple((names[key], tuple(units[key].index(unit) for unit in item))
                           for key, item in placement.items()) for placement in placements)
    normalized = _Binding(
        tuple((role, names[name]) for role, name in binding.names),
        tuple((names[name], context.rates[name]) for name in names),
        binding.units, binding.peaks, binding.modes, binding.core_count,
    )
    clients = tuple((names[key], value) for key, value in shared_clients if key in names)
    return normalized, relative, clients, {value: key for key, value in names.items()}, {
        names[key]: value for key, value in units.items()}


def _rebind(program, names, units):
    """Clone immutable definitions only; preserve task order and numeric values."""
    return KernelProgram(tuple(replace(
        stage,
        work=MappingProxyType({names[key]: value for key, value in stage.work.items()}),
        caps=MappingProxyType({names[key]: value for key, value in stage.caps.items()}),
        placement=MappingProxyType({names[key]: tuple(units[key][unit] for unit in placed)
                                    for key, placed in stage.placement.items()}),
        program=_rebind(stage.program, names, units) if stage.program is not None else None,
    ) for stage in program.stages), minimum_ms=program.minimum_ms)


@lru_cache(maxsize=TEMPLATE_CACHE_SIZE)
def _template(binding, k, m, n, repeat, cores, aivs, directional, ownership, shared_clients,
              producer_mode, producer_slots, activation_l2, weight_l2):
    context = binding.context()
    kb, ka, _ = panel_sizes(m, n)
    panels = (k + kb - 1) // kb
    period = lcm(ka, MX_SCALE_PUBLISH_K) // kb
    count, remainder = divmod(panels, period)
    if not remainder and k % kb:
        count -= 1
    ownership = tuple(dict(item) for item in ownership)

    def parts(index):
        if producer_mode == "dynamic_buffer4":
            from .mxfp4_source_refill import source_panel
            factory = source_panel
        else:
            factory = panel_parts
        return factory(context, k=k, m=m, n=n, repeat=repeat, index=index,
                           owner=0, cores=cores, aivs=aivs, directional=directional,
                           ownership=ownership, activation_l2=activation_l2,
                           weight_l2=weight_l2)

    factory, options = compact_refill, {}
    if producer_mode == "dynamic_buffer4":
        from .mxfp4_source_refill import compact_source
        factory, options = compact_source, {"slots": producer_slots}
    program = factory(
        context, tuple(parts(i) for i in range(period)) if count else (), count,
        tuple(parts(i) for i in range(count * period, panels)),
        shared_clients=dict(shared_clients), **options,
    )
    return _immutable(program)


@lru_cache(maxsize=1024)
def _compile(binding, k, m, n, repeat, owner, cores, aivs, directional, shared_clients,
             producer_mode, producer_slots, activation_l2, weight_l2):
    normalized, ownership, clients, names, units = _normalize(
        binding, owner, cores, aivs, shared_clients)
    template = _template(normalized, k, m, n, repeat, cores, aivs, directional, ownership, clients,
                         producer_mode, producer_slots, activation_l2, weight_l2)
    return _rebind(template, names, units)


def _clear():
    _compile.cache_clear()
    _template.cache_clear()


def compiled_panels(context, *, k, m, n, repeat, owner, cores, aivs,
                    directional=False, shared_clients=None, producer_mode="serial",
                    producer_slots=Slots(), activation_l2=0.0, weight_l2=0.0, l2_die=None):
    """Return reusable, event-free IR; only scalar physical bindings are retained.

    The outer cache distinguishes physical ranks; its inner templates preserve
    resource aliases and relative ownership while removing labels. Kernel names,
    requests, invocation floors, buffer allocations and callbacks are not
    consumed by this subprogram and must not enter its cache key/value.
    """
    if producer_mode not in ("serial", "dynamic_buffer4"):
        raise ValueError("mxa8w4_unknown_producer_mode")
    act_l2 = float(activation_l2)
    wgt_l2 = float(weight_l2)
    if (act_l2 > 0.0 or wgt_l2 > 0.0) and not _l2_permitted(context, l2_die=l2_die):
        act_l2 = wgt_l2 = 0.0
    return _compile(_Binding.from_context(context), k, m, n, repeat, owner, cores,
                    aivs, directional, tuple(sorted((shared_clients or {}).items())),
                    producer_mode, producer_slots, act_l2, wgt_l2)


compiled_panels.cache_info = _compile.cache_info
compiled_panels.cache_clear = _clear
compiled_panels.template_cache_info = _template.cache_info
