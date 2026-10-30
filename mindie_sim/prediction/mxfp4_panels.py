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

from .mechanism import Work
from .mxfp4_geometry import Config, panel_sizes, panel_work
from .mxfp4_registers import CLASSES, widen, transpose
from .program import KernelProgram, RecipeContext, Stage
from .refill_steady import compact_refill
from .mxfp4_pipeline import Slots

REGISTER_ROLES = ('reg_data_writes', 'reg_ub_accesses', 'reg_ub_stores', 'reg_shuffle_lanes')


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


def panel_parts(context, *, k, m, n, repeat, index, owner, cores, aivs, directional,
                ownership=None):
    """The existing source producer and micro-consumer DAG for one K panel.

    This helper is also available to the uncompressed caller; it does not
    choose a timing model or retain invocation/allocation state.
    """
    config = Config(k, n, (m,))
    ac, *vec = ownership if ownership is not None else _ownership(context, owner, cores, aivs)
    a, vectors, micros = panel_work(config, m, n, index * panel_sizes(m, n)[0])
    loads = [Stage('activation', context.bind(Work(resources={
        'hbm': a * repeat, 'mte2_aic': a * repeat})), placement=ac)]
    for sub, work in enumerate(vectors):
        def vstage(name, resources, after=()):
            return Stage(f'{name}{sub}', context.bind(Work(resources=resources).scaled(repeat)),
                         after=after, placement=vec[sub])
        loads.extend((
            vstage('scale_read', {'hbm': work['scale'], 'mte2_aiv': work['scale']}),
            vstage('scale_transpose', transpose(work['gather_b16_lanes']).capacity_work(CLASSES).resources,
                   (f'scale_read{sub}',)),
            vstage('scale_publish', {'mte3': work['scale_publish']}, (f'scale_transpose{sub}',)),
            vstage('packed_read', {'hbm': work['weight'], 'mte2_aiv': work['weight']},
                   (f'scale_read{sub}',)),
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


@lru_cache(maxsize=128)
def _template(binding, k, m, n, repeat, cores, aivs, directional, ownership, shared_clients,
              producer_mode, producer_slots):
    context = binding.context()
    kb, ka, _ = panel_sizes(m, n)
    panels = (k + kb - 1) // kb
    period = lcm(ka, 4096) // kb
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
                           ownership=ownership)

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
             producer_mode, producer_slots):
    normalized, ownership, clients, names, units = _normalize(
        binding, owner, cores, aivs, shared_clients)
    template = _template(normalized, k, m, n, repeat, cores, aivs, directional, ownership, clients,
                         producer_mode, producer_slots)
    return _rebind(template, names, units)


def _clear():
    _compile.cache_clear()
    _template.cache_clear()


def compiled_panels(context, *, k, m, n, repeat, owner, cores, aivs,
                    directional=False, shared_clients=None, producer_mode="serial",
                    producer_slots=Slots()):
    """Return reusable, event-free IR; only scalar physical bindings are retained.

    The outer cache distinguishes physical ranks; its inner templates preserve
    resource aliases and relative ownership while removing labels. Kernel names,
    requests, invocation floors, buffer allocations and callbacks are not
    consumed by this subprogram and must not enter its cache key/value.
    """
    if producer_mode not in ("serial", "dynamic_buffer4"):
        raise ValueError("mxa8w4_unknown_producer_mode")
    return _compile(_Binding.from_context(context), k, m, n, repeat, owner, cores,
                    aivs, directional, tuple(sorted((shared_clients or {}).items())),
                    producer_mode, producer_slots)


compiled_panels.cache_info = _compile.cache_info
compiled_panels.cache_clear = _clear
compiled_panels.template_cache_info = _template.cache_info
