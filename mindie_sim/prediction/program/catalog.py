"""Pure compilation of public program declarations into causal catalogs.

[Role] Translate existing KernelProgram/PipelineTemplate declarations into the
  frozen ProgramCatalog/FiniteProgramSpec records and bind fresh BindingFrames.
[Invariants] Compilation is pure: no Engine, Event, allocation, cursor or
  elapsed time enters a catalog. Names are labels; local ids, finite absolute
  indices and allocation symbols are the structural identity. Every existing
  Stage work amount is service milliseconds and is preserved exactly.
  Repeated homogeneous producers compile a bounded class catalog; index_of is
  the only per-index lookup and performs no enumeration.
[Timeouts & Locks] None: pure bounded construction O(number of declared
  classes * stages). No all-panel materialization.
[Side-Effect Policy] None. Submission helpers only call the caller's Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from mindie_sim.prediction.runtime.contracts import (
    AllocationBinding,
    AllocationKey,
    AllocationSpec,
    BindingFrame,
    CapTerm,
    CatalogGeneration,
    EdgeSpec,
    FiniteProgramSpec,
    OccurrenceKey,
    PlacementSpec,
    PoolKey,
    PortBinding,
    PortSpec,
    ProgramCatalog,
    ReleaseSpec,
    StageClass,
    StageSpec,
    WorkTerm,
)
from mindie_sim.prediction.program import (
    KernelProgram,
    ParallelProgram,
    PipelineProgram,
    PipelineTemplate,
    PipelineTemplateProgram,
    Stage,
)


# ---------------------------------------------------------------------------
# Canonical structural identity


def _canonical(value: Any) -> Any:
    """Deterministic JSON-safe projection; never reads live runtime state."""
    if isinstance(value, Mapping):
        rows = [[_canonical(key), _canonical(item)]
                for key, item in value.items()]
        rows.sort(key=lambda row: json.dumps(row[0], sort_keys=True))
        return {"$map": rows}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        # repr distinguishes -0.0, NaN spelling and integral floats without
        # introducing host-dependent formatting.
        return {"$float": repr(value)}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if callable(value):
        return {"$callable": [
            str(getattr(value, "__module__", "")),
            str(getattr(value, "__qualname__", "")),
            id(value),
        ]}
    if is_dataclass(value):
        return {"$type": type(value).__qualname__, **{
            item.name: _canonical(getattr(value, item.name))
            for item in fields(value)}}
    return {"$type": type(value).__qualname__, "$repr": repr(value)}


def _digest(value: Any) -> str:
    payload = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _catalog_identity(catalog: ProgramCatalog) -> str:
    """Content identity of a catalog, excluding its derived display ids."""
    return _digest({
        "generation": catalog.generation,
        "stages": catalog.stages,
        "ports": catalog.ports,
        "external_edges": catalog.external_edges,
        "allocation_symbols": catalog.allocation_symbols,
        "state_schema": catalog.state_schema,
        "finite": None if catalog.finite is None else {
            "total": catalog.finite.total,
            "slots": catalog.finite.slots,
            "classes": catalog.finite.classes,
            "setup": catalog.finite.setup,
            "drain": catalog.finite.drain,
            "setup_ms": catalog.finite.setup_ms,
            "minimum_ms": catalog.finite.minimum_ms,
            "translation": catalog.finite.translation,
        },
        "host_spec": catalog.host_spec,
        "selected_call_count": catalog.selected_call_count,
        "semantic_sources": catalog.semantic_sources,
        "guard": catalog.guard,
        "translation": catalog.translation,
    })


# ---------------------------------------------------------------------------
# Stage / allocation translation



def _template_scopes(template: PipelineTemplate | None,
                     producer_bindings: Any) -> dict[str, str]:
    scopes: dict[str, str] = {}
    rows: tuple = ()
    if producer_bindings is not None:
        recorder = getattr(producer_bindings, "rows", None)
        if callable(recorder):
            rows = tuple(recorder(getattr(template, "owner", 0)))
    if template is not None:
        rows = (*rows, *template.bindings)
    if isinstance(producer_bindings, Mapping):
        rows = (*rows, producer_bindings)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = row.get("name")
        scope = row.get("scope")
        if isinstance(name, str) and scope in {
                "invocation", "output", "panel", "slot"}:
            previous = scopes.get(name)
            if previous is not None and previous != scope:
                raise ModelingError(f"program_allocation_scope_conflict:{name}")
            scopes[name] = str(scope)
    return scopes


def _compile_stages(stages: Sequence[Stage], *, class_id: int,
                    scopes: Mapping[str, str],
                    symbols: dict[str, int],
                    specifications: list[AllocationSpec],
                    streams: dict[str, int],
                    nested: Mapping[str, ProgramCatalog] | None = None
                    ) -> tuple[StageSpec, ...]:
    """Compile one relative stage group. ``after``/``previous`` are local."""
    stages = tuple(stages)
    if not stages:
        raise ModelingError("program_catalog_stage_group_empty")
    local: dict[str, int] = {}
    for index, stage in enumerate(stages):
        if not isinstance(stage, Stage):
            raise ModelingError(
                f"program_catalog_requires_stage:{type(stage).__name__}")
        if stage.name in local:
            raise ModelingError(f"program_catalog_duplicate_stage:{stage.name}")
        local[stage.name] = index
    specs: list[StageSpec] = []
    for index, stage in enumerate(stages):
        edges: list[EdgeSpec] = []
        for key in stage.after:
            if key not in local:
                raise ModelingError(
                    f"program_catalog_unknown_predecessor:{stage.name}:{key}")
            edges.append(EdgeSpec(source_node=local[key]))
        previous_seen: set[int] = set()
        for key in stage.previous:
            if key not in local:
                raise ModelingError(
                    f"program_catalog_unknown_previous:{stage.name}:{key}")
            source = local[key]
            if source not in previous_seen:
                edges.append(EdgeSpec(source_node=source, iteration_delta=-1))
                previous_seen.add(source)
        work = tuple(WorkTerm(
            physical_resource_role=str(role), issued=float(amount),
            useful=float(amount), unit="ms", service_conversion=1.0)
            for role, amount in stage.work.items() if float(amount) != 0.0)
        caps = tuple(CapTerm(physical_resource_role=str(role), rate=float(rate))
                     for role, rate in stage.caps.items())
        placement = tuple(PlacementSpec(
            physical_resource_role=str(role),
            units=tuple(int(unit) for unit in units))
            for role, units in stage.placement.items())
        acquire: list[AllocationSpec] = []
        release: list[ReleaseSpec] = []
        for allocation in stage.acquire:
            acquire.append(_allocation_spec(
                allocation, scopes, symbols, specifications))
        for allocation in stage.free:
            _allocation_spec(allocation, scopes, symbols, specifications)
            release.append(ReleaseSpec(
                symbol=symbols[allocation.name],
                scope_kind=scopes.get(allocation.name, "invocation")))
        nested_catalog = (nested or {}).get(stage.name)
        if stage.program is not None and nested_catalog is None:
            raise ModelingError(
                f"program_catalog_nested_program_uncompiled:{stage.name}")
        specs.append(StageSpec(
            local_id=index,
            label=stage.name,
            class_id=class_id,
            relative_after=tuple(edges),
            stream_symbol=streams.setdefault(stage.name, len(streams)),
            work=work,
            latency_expr=float(stage.latency),
            caps=caps,
            coupled=bool(stage.coupled),
            priority=int(stage.priority),
            cohort_size=int(stage.cohort_size),
            placement=placement,
            acquire=tuple(acquire),
            release=tuple(release),
            state_effects=(),
            output_ports=(),
            nested_catalog=nested_catalog,
            condition=None,
        ))
    return tuple(specs)


def _allocation_spec(allocation, scopes: Mapping[str, str],
                     symbols: dict[str, int],
                     specifications: list[AllocationSpec]) -> AllocationSpec:
    name = allocation.name
    scope = scopes.get(name, "invocation")
    amount = float(allocation.size)
    units = tuple(int(unit) for unit in allocation.units) or (0,)
    existing = next((spec for spec in specifications if spec.label == name), None)
    if existing is not None:
        if (existing.pool_role != allocation.pool
                or float(existing.amount) != amount
                or tuple(existing.units) != units
                or existing.scope_kind != scope):
            raise ModelingError(f"program_allocation_conflict:{name}")
        return existing
    symbol = len(specifications)
    symbols[name] = symbol
    spec = AllocationSpec(
        symbol=symbol,
        label=name,
        pool_role=str(allocation.pool),
        amount=amount,
        scope_kind=scope,
        units=units,
        external_port=None,
        condition=None,
        generation=0,
    )
    specifications.append(spec)
    return spec


def _freeze_catalog(*, implementation_id: str,
                    generation: CatalogGeneration,
                    stages: tuple[StageSpec, ...],
                    ports: tuple[PortSpec, ...],
                    external_edges: tuple[EdgeSpec, ...],
                    allocation_symbols: tuple[AllocationSpec, ...],
                    finite: FiniteProgramSpec | None,
                    selected_call_count: int,
                    semantic_sources: tuple[str, ...],
                    guard: Any = None) -> ProgramCatalog:
    skeleton = ProgramCatalog(
        catalog_id="catalog",
        implementation_id=implementation_id,
        generation=generation,
        stages=stages,
        ports=ports,
        external_edges=external_edges,
        allocation_symbols=allocation_symbols,
        finite=finite,
        selected_call_count=selected_call_count,
        semantic_sources=semantic_sources,
        guard=guard,
    )
    digest = _catalog_identity(skeleton)
    return ProgramCatalog(
        catalog_id="catalog:" + digest[:32],
        implementation_id=implementation_id + ":" + digest[:16],
        generation=generation,
        stages=stages,
        ports=ports,
        external_edges=external_edges,
        allocation_symbols=allocation_symbols,
        finite=finite,
        selected_call_count=selected_call_count,
        semantic_sources=semantic_sources,
        guard=guard,
    )


# ---------------------------------------------------------------------------
# Ordinary kernel DAG


def catalog_compilable(program) -> bool:
    """True when a declaration has no opaque producer boundary.

    Opaque nested callbacks remain executable through the same public Engine,
    but they cannot be folded into an immutable catalog or a finite class.
    """
    if isinstance(program, KernelProgram):
        if not program.stages:
            return False
        # The causal Engine's flat catalog path submits a nested catalog
        # without the parent stage's dependencies today, so nested producers
        # must use the public Stage.submit path until that is fixed. Never run
        # nested work before its declared ready boundary.
        if any(stage.program is not None for stage in program.stages):
            return False
        # Cross-stage release identity is owned by the repeating/Engine public
        # submit path today; a catalog release must bind the same live grant.
        acquired_at: dict[str, int] = {}
        for index, stage in enumerate(program.stages):
            for allocation in stage.acquire:
                acquired_at.setdefault(allocation.name, index)
        for index, stage in enumerate(program.stages):
            for allocation in stage.free:
                if acquired_at.get(allocation.name) != index:
                    return False
        return True
    if isinstance(program, PipelineTemplateProgram):
        # The causal Engine's finite node loop currently executes flat class
        # stages only, and a release must bind the same live grant across
        # stages/panels. Keep the repeating consumer's Stage.submit path for
        # nested programs and allocation lifecycles.
        return (not _template_has_nested(program.template)
                and not _template_has_allocations(program.template))
    if isinstance(program, PipelineProgram):
        # A callback producer may encode per-wave tail geometry; only its
        # explicit template/index catalog can prove a bounded class mapping.
        return False
    if isinstance(program, ParallelProgram):
        return bool(program.programs) and all(
            catalog_compilable(child) for child in program.programs)
    return False


def _template_has_allocations(template: PipelineTemplate) -> bool:
    return any(
        stage.acquire or stage.free
        for stage in (*template.setup, *template.drain,
                      *(stage for phase in template.phases
                        for stage in phase.stages)))


def _template_has_nested(template: PipelineTemplate) -> bool:
    return any(
        stage.program is not None
        for stage in (*template.setup, *template.drain,
                      *(stage for phase in template.phases
                        for stage in phase.stages)))


def _compile_nested(program, *, generation: CatalogGeneration) -> ProgramCatalog:
    if isinstance(program, KernelProgram):
        return _compile_kernel(program, generation=generation, port_specs=())
    if isinstance(program, PipelineTemplateProgram):
        return _compile_template(
            program.template, generation=generation, port_specs=())
    if isinstance(program, PipelineProgram):
        return _compile_template(
            program.template(), generation=generation, port_specs=())
    if isinstance(program, ParallelProgram):
        children = tuple(
            _compile_nested(child, generation=generation)
            for child in program.programs)
        stages = tuple(StageSpec(
            local_id=index,
            label=f"parallel:{index}",
            class_id=0,
            nested_catalog=child,
        ) for index, child in enumerate(children))
        return _freeze_catalog(
            implementation_id="parallel-group",
            generation=generation,
            stages=stages,
            ports=(),
            external_edges=(),
            allocation_symbols=(),
            finite=None,
            selected_call_count=sum(child.selected_call_count for child in children),
            semantic_sources=tuple(
                source for child in children for source in child.semantic_sources))
    raise ModelingError(
        f"program_catalog_unsupported_nested:{type(program).__name__}")


def _compile_kernel(program: KernelProgram, *,
                    generation: CatalogGeneration,
                    port_specs: Sequence[PortSpec]) -> ProgramCatalog:
    declarations = list(program.stages)
    if not declarations and not program.minimum_ms:
        raise ModelingError("program_catalog_kernel_requires_stages")
    names = [stage.name for stage in declarations]
    if len(set(names)) != len(names):
        raise ModelingError("program_catalog_duplicate_stage")
    if program.minimum_ms:
        declarations.append(Stage("__observation__", latency=float(program.minimum_ms)))
    if declarations:
        declarations.append(Stage(
            "__complete__", after=tuple(stage.name for stage in declarations)))
    symbols: dict[str, int] = {}
    specifications: list[AllocationSpec] = []
    streams: dict[str, int] = {}
    scopes: dict[str, str] = {}
    nested = {
        stage.name: _compile_nested(stage.program, generation=generation)
        for stage in declarations if stage.program is not None
    }
    specs = _compile_stages(
        declarations, class_id=0, scopes=scopes, symbols=symbols,
        specifications=specifications, streams=streams, nested=nested)
    body_calls = sum(
        1 for stage in specs
        if stage.work or stage.latency_expr or stage.acquire or stage.release
        or stage.caps or stage.placement)
    return _freeze_catalog(
        implementation_id="kernel-program",
        generation=generation,
        stages=specs,
        ports=tuple(port_specs),
        external_edges=(),
        allocation_symbols=tuple(specifications),
        finite=None,
        selected_call_count=body_calls,
        semantic_sources=tuple(stage.name for stage in program.stages),
    )


# ---------------------------------------------------------------------------
# Finite pipeline template


def _compile_template(template: PipelineTemplate, *,
                      generation: CatalogGeneration,
                      port_specs: Sequence[PortSpec]) -> ProgramCatalog:
    if not isinstance(template, PipelineTemplate):
        raise ModelingError("program_catalog_requires_pipeline_template")
    scopes = _template_scopes(template, None)
    symbols: dict[str, int] = {}
    specifications: list[AllocationSpec] = []
    streams: dict[str, int] = {}

    classes: list[StageClass] = []
    phase_classes: list[tuple[int, ...]] = []
    for phase_id, phase in enumerate(template.phases):
        groups = phase.residue_stages or (phase.stages,)
        ids: list[int] = []
        for group in groups:
            class_id = len(classes)
            nested = {
                stage.name: _compile_nested(stage.program, generation=generation)
                for stage in group if stage.program is not None
            }
            classes.append(StageClass(
                class_id=class_id,
                stages=_compile_stages(
                    group, class_id=class_id, scopes=scopes,
                    symbols=symbols, specifications=specifications,
                    streams=streams, nested=nested)))
            ids.append(class_id)
        phase_classes.append(tuple(ids))

    setup_nested = {
        stage.name: _compile_nested(stage.program, generation=generation)
        for stage in template.setup if stage.program is not None
    }
    drain_nested = {
        stage.name: _compile_nested(stage.program, generation=generation)
        for stage in template.drain if stage.program is not None
    }
    setup = _compile_stages(
        template.setup, class_id=-1, scopes=scopes, symbols=symbols,
        specifications=specifications, streams=streams,
        nested=setup_nested) if template.setup else ()
    drain = _compile_stages(
        template.drain, class_id=-2, scopes=scopes, symbols=symbols,
        specifications=specifications, streams=streams,
        nested=drain_nested) if template.drain else ()

    if template.indexed_panels is not None:
        total = int(template.indexed_panels)
        producer_index = template.index_of
        ends: tuple[int, ...] = ()

        def index_of(index, _producer=producer_index, _ids=phase_classes,
                     _total=total):
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("program_catalog_index_requires_integer")
            if index < 0 or index >= _total:
                raise IndexError("program_catalog_index_out_of_range")
            phase_id, end, output, panel, slot = _producer(index)
            if (isinstance(phase_id, bool) or not isinstance(phase_id, int)
                    or not 0 <= phase_id < len(_ids)):
                raise ModelingError("program_catalog_index_mapping_invalid")
            ids = _ids[phase_id]
            class_id = ids[index % len(ids)] if len(ids) > 1 else ids[0]
            return (class_id, int(end), int(output), int(panel), int(slot))

        body_stages = len(classes[0].stages) if classes else 0
        selected = total * body_stages
    else:
        counts = tuple(int(phase.count) for phase in template.phases)
        total = sum(counts)
        ends = tuple(sum(counts[:position + 1])
                     for position in range(len(counts)))

        def index_of(index, _counts=counts, _ends=ends, _ids=phase_classes,
                     _total=total):
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("program_catalog_index_requires_integer")
            if index < 0 or index >= _total:
                raise IndexError("program_catalog_index_out_of_range")
            phase_id = 0
            while phase_id < len(_ends) and index >= _ends[phase_id]:
                phase_id += 1
            if phase_id >= len(_ids):
                raise ModelingError("program_catalog_index_mapping_invalid")
            ids = _ids[phase_id]
            class_id = ids[index % len(ids)] if len(ids) > 1 else ids[0]
            return (class_id, _ends[phase_id], 0, index,
                    index % max(1, int(template.slots)))

        body_stages = len(classes[0].stages) if classes else 0
        selected = total * body_stages

    finite = FiniteProgramSpec(
        total=total,
        slots=int(template.slots),
        classes=tuple(classes),
        index_of=index_of,
        setup=setup,
        drain=drain,
        setup_ms=float(template.setup_ms),
        minimum_ms=float(template.minimum_ms),
        translation=None,
    )
    return _freeze_catalog(
        implementation_id="finite-pipeline",
        generation=generation,
        stages=(),
        ports=tuple(port_specs),
        external_edges=(),
        allocation_symbols=tuple(specifications),
        finite=finite,
        selected_call_count=selected,
        semantic_sources=tuple(phase.kind for phase in template.phases),
    )


# ---------------------------------------------------------------------------
# Public compiler / binder


def compile_program(program, *, generation: CatalogGeneration,
                    port_specs: Sequence[PortSpec] = ()) -> ProgramCatalog:
    """Compile one immutable public program declaration.

    ``KernelProgram`` yields a flat relative DAG. ``PipelineTemplateProgram``
    and ``PipelineProgram`` yield a bounded finite class catalog. No Event,
    grant, cursor or measured duration is captured.
    """
    if isinstance(program, KernelProgram):
        return _compile_kernel(
            program, generation=generation, port_specs=port_specs)
    if isinstance(program, PipelineTemplateProgram):
        return _compile_template(
            program.template, generation=generation, port_specs=port_specs)
    if isinstance(program, PipelineProgram):
        if not program.homogeneous:
            raise ModelingError(
                "program_catalog_opaque_pipeline_requires_finite_window")
        return _compile_template(
            program.template(), generation=generation, port_specs=port_specs)
    if isinstance(program, ParallelProgram):
        return _compile_nested(program, generation=generation)
    raise ModelingError(
        f"program_catalog_unsupported_program:{type(program).__name__}")


def compile_pipeline_template(template: PipelineTemplate, *,
                              generation: CatalogGeneration,
                              port_specs: Sequence[PortSpec] = ()) -> ProgramCatalog:
    return _compile_template(
        template, generation=generation, port_specs=port_specs)


def _derived_allocations(catalog: ProgramCatalog, occurrence: OccurrenceKey,
                         memory_domain: str | None,
                         coordinates: Mapping[str, int]) -> dict[int, AllocationBinding]:
    result: dict[int, AllocationBinding] = {}
    for spec in catalog.allocation_symbols:
        amount = spec.amount
        if not isinstance(amount, (int, float)) or isinstance(amount, bool):
            raise ModelingError(
                f"program_bind_allocation_scalar_requires_expression_eval:"
                f"{spec.label}")
        scope = str(spec.scope_kind)
        ordinal = 0
        if scope == "panel":
            ordinal = int(coordinates.get("panel_ordinal", 0))
        elif scope == "output":
            ordinal = int(coordinates.get("output_ordinal", 0))
        elif scope == "slot":
            ordinal = int(coordinates.get("slot_ordinal", 0))
        units = tuple(spec.units) or (0,)
        pool = PoolKey(
            physical_domain=str(memory_domain or "local"),
            rank=int(occurrence.rank),
            tier=str(spec.pool_role),
            name=str(spec.pool_role),
            unit=int(units[0]),
            quantity_unit="bytes",
        )
        key = AllocationKey(
            occurrence=occurrence,
            scope_kind=scope,
            scope_ordinal=ordinal,
            symbol=int(spec.symbol),
            generation=int(spec.generation),
        )
        result[int(spec.symbol)] = AllocationBinding(
            key=key,
            pool=pool,
            amount=float(amount),
            backing=None,
            byte_ranges=(),
            reservation_parent=None,
            owner="",
            storage=str(spec.label),
            access="readwrite",
            last_use=None,
            acquire=True,
        )
    return result


def bind_program(catalog: ProgramCatalog, *, occurrence: OccurrenceKey,
                 geometry: Mapping[str, Any],
                 values: Mapping[str, PortBinding],
                 resources: Mapping[str, str],
                 allocations: Sequence[AllocationBinding] = (),
                 state_reads: Mapping[str, Any] = (),
                 conditions: Mapping[str, bool] = (),
                 placement: Mapping[str, Sequence[int]] = (),
                 host_fifo: str | None = None,
                 stream_bindings: Mapping[int, str] = (),
                 memory_domain: str | None = None) -> BindingFrame:
    """Bind a fresh invocation frame. Never reuses another occurrence."""
    if not isinstance(occurrence, OccurrenceKey):
        raise ModelingError("program_bind_requires_occurrence_key")
    supplied: dict[int, AllocationBinding] = {}
    for binding in allocations:
        if not isinstance(binding, AllocationBinding):
            raise ModelingError("program_bind_requires_allocation_bindings")
        if binding.key.symbol in supplied:
            raise ModelingError(
                f"program_bind_duplicate_allocation_symbol:{binding.key.symbol}")
        supplied[binding.key.symbol] = binding
    derived = _derived_allocations(catalog, occurrence, memory_domain, {})
    for symbol, binding in derived.items():
        supplied.setdefault(symbol, binding)
    declared = {int(spec.symbol) for spec in catalog.allocation_symbols}
    unknown = set(supplied) - declared
    if unknown:
        raise ModelingError(
            "program_bind_unknown_allocation_symbol:" +
            ",".join(str(symbol) for symbol in sorted(unknown)))
    return BindingFrame(
        occurrence=occurrence,
        geometry=dict(geometry),
        values=dict(values),
        resources=dict(resources),
        allocations=supplied,
        state_reads=dict(state_reads or ()),
        conditions=dict(conditions or ()),
        placement={key: tuple(units)
                   for key, units in dict(placement or ()).items()},
        host_fifo=host_fifo,
        stream_bindings={int(key): str(value)
                         for key, value in dict(stream_bindings or ()).items()},
        memory_domain=memory_domain,
        coordinates={},
    )


def catalog_resource_roles(catalog: ProgramCatalog) -> tuple[str, ...]:
    """Every physical role the catalog can address, in stable order."""
    roles: set[str] = set()
    groups = [catalog.stages]
    if catalog.finite is not None:
        groups.extend((catalog.finite.setup, catalog.finite.drain,
                       *(item.stages for item in catalog.finite.classes)))
    for stages in groups:
        for stage in stages:
            roles.update(term.physical_resource_role for term in stage.work)
            roles.update(term.physical_resource_role for term in stage.caps)
            roles.update(item.physical_resource_role for item in stage.placement)
    return tuple(sorted(roles))


def _generation(engine) -> CatalogGeneration:
    generation = getattr(engine, "catalog_generation", None)
    if isinstance(generation, CatalogGeneration):
        return generation
    return CatalogGeneration(
        implementation_generation=str(getattr(engine, "uid", "") or ""),
    )


def _occurrence(engine) -> OccurrenceKey:
    creator = getattr(engine, "new_occurrence", None)
    if not callable(creator):
        raise ModelingError("program_catalog_engine_requires_new_occurrence")
    occurrence = creator()
    if not isinstance(occurrence, OccurrenceKey):
        raise ModelingError("program_catalog_engine_occurrence_invalid")
    return occurrence


def submit_program_catalog(engine, catalog: ProgramCatalog, *,
                           name: str, after: Sequence = (),
                           owner: str = ""):
    """Submit one catalog once with a fresh occurrence and binding frame."""
    submitter = getattr(engine, "submit_catalog", None)
    if not callable(submitter):
        raise ModelingError("program_catalog_engine_requires_submit_catalog")
    occurrence = _occurrence(engine)
    roles = catalog_resource_roles(catalog)
    binding = bind_program(
        catalog,
        occurrence=occurrence,
        geometry={},
        values={},
        resources={role: role for role in roles},
        memory_domain=None,
    )
    return submitter(catalog, binding, name=name, after=tuple(after),
                     owner=owner)


def submit_program(engine, program, *, name: str, ready,
                   owner: str = ""):
    catalog = compile_program(program, generation=_generation(engine))
    return submit_program_catalog(
        engine, catalog, name=name, after=(ready,), owner=owner)


def _has_cross_stage_previous(template: PipelineTemplate) -> bool:
    groups = [template.setup, template.drain]
    for phase in template.phases:
        groups.extend(phase.residue_stages or (phase.stages,))
    for group in groups:
        names = {stage.name for stage in group}
        for stage in group:
            if any(dep != stage.name or dep not in names for dep in stage.previous):
                return True
    return False


def submit_template(engine, template: PipelineTemplate, *, name: str, ready,
                    owner: str = ""):
    if _has_cross_stage_previous(template):
        from mindie_sim.prediction.runtime.contracts import Submission
        from mindie_sim.prediction.scheduling import start_pipeline_template
        done = start_pipeline_template(
            engine, template, name, ready, owner=owner)
        return Submission(
            occurrence=_occurrence(engine), start=ready, done=done,
            ports={}, allocations={}, selected_program_id="finite-pipeline")
    catalog = compile_pipeline_template(
        template, generation=_generation(engine))
    return submit_program_catalog(
        engine, catalog, name=name, after=(ready,), owner=owner)
