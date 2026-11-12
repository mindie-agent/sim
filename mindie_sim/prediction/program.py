"""Composable kernel recipes independent of model names.

[Role] Translate physical operator work into finite resource programs.
[Invariants] A recipe changes stages, work, supply and lifetimes explicitly.
  A measured kernel span is a completion constraint, never invented FLOPs.
  Repeated tiles are generated through a bounded window, including fill/drain.
[Timeouts & Locks] Pure virtual execution; live tiles are bounded by slots.
[Side-Effect Policy] Submits tasks only to the caller's Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import NATIVE_WORK_UNITS
from .engine import Allocation, Engine, Event


_EMPTY_MAPPING = MappingProxyType({})


def scope_allocations(name, items):
    """Rename immutable buffer identities without reflecting dataclass fields."""
    return tuple(Allocation(f"{name}:{a.name}", a.pool, a.size, a.units)
                 for a in items) if items else ()


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    work: Mapping[str, float] = field(default_factory=dict)
    after: tuple[str, ...] = ()
    latency: float = 0.0
    caps: Mapping[str, float] = field(default_factory=dict)
    acquire: tuple[Allocation, ...] = ()
    free: tuple[Allocation, ...] = ()
    program: Program | None = None
    placement: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    previous: tuple[str, ...] = ()
    coupled: bool = False

    def __post_init__(self):
        # Empty/default identity is immutable and shared. Nonempty
        # caller-owned dicts still receive a proper MappingProxyType copy;
        # an existing proxy is already immutable. This avoids three new
        # proxy+dict objects for the common empty caps/work/placement case.
        work = self.work
        if not work:
            work = _EMPTY_MAPPING
        else:
            # MappingProxyType is a read-only view, not frozen ownership. Copy
            # every nonempty caller mapping, including an ordinary proxy.
            work = MappingProxyType(dict(work))
        caps = self.caps
        if not caps:
            caps = _EMPTY_MAPPING
        else:
            caps = MappingProxyType(dict(caps))
        placement = self.placement
        if not placement:
            placement = _EMPTY_MAPPING
        else:
            placement = MappingProxyType(
                {key: tuple(units) for key, units in placement.items()})
        object.__setattr__(self, "work", work)
        object.__setattr__(self, "caps", caps)
        object.__setattr__(self, "placement", placement)
        for name in ("after", "acquire", "free", "previous"):
            # tuple(existing tuple) is the same object; a caller list or other
            # sequence becomes an independent snapshot, including empty lists.
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def prepare(self, engine, name, scoped):
        return engine._prepare(
            name, self.work, latency=self.latency, caps=self.caps,
            acquire=scoped(self.acquire), free=() if self.program else scoped(self.free),
            placement=self.placement, coupled=self.coupled,
        )

    def submit(self, engine, name, predecessors, scoped, owner, prepared=None):
        if prepared is None:
            prepared = self.prepare(engine, name, scoped)
        event = engine._submit_prepared(prepared, name, after=predecessors, owner=owner)
        if self.program is not None:
            event = self.program.start(engine, name + ":program", event, owner=owner)
            if self.free:
                event = engine.submit(name + ":free", after=(event,), free=scoped(self.free), owner=owner)
        return event


class Program(Protocol):
    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event: ...


@dataclass(frozen=True)
class ParallelProgram:
    programs: tuple[Program, ...]

    def start(self, engine, name, ready, *, owner=""):
        events = tuple(program.start(engine, f"{name}:group{index}", ready, owner=owner)
                       for index, program in enumerate(self.programs))
        return engine.submit(name + ":complete", after=events or (ready,), owner=owner)


@dataclass(frozen=True)
class KernelProgram:
    stages: tuple[Stage, ...]
    minimum_ms: float = 0.0

    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event:
        events: dict[str, Event] = {}
        scoped = lambda items: scope_allocations(name, items)
        for stage in self.stages:
            predecessors = tuple(events[key] for key in stage.after) or (ready,)
            events[stage.name] = stage.submit(
                engine, f"{name}:{stage.name}", predecessors, scoped, owner)
        if self.minimum_ms:
            events["observation"] = engine.submit(
                name + ":observation", after=(ready,), latency=self.minimum_ms, owner=owner,
            )
        return engine.submit(name + ":complete", after=tuple(events.values()) or (ready,), owner=owner)


@dataclass(frozen=True)
class PipelineProgram:
    """A repeated stage DAG with explicit stream ordering and buffer lifetime.

    ``tile(index)`` supplies real tail-tile work. Every stage name denotes a
    serial instruction stream; distinct names may overlap. Buffer slots limit
    how many tiles can be resident. No steady-state II replaces finite work.
    """

    count: int
    slots: int
    tile: Callable[[int], Sequence[Stage]]
    minimum_ms: float = 0.0
    setup_ms: float = 0.0

    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event:
        # One finite tile has no pipeline recurrence to protect.  Keep the
        # same individual Stage submissions, dependencies and physical times,
        # but elide the setup/span/outer-completion adapters that only forward
        # this single tile's completion.
        if self.count == 1 and self.slots == 1 and not self.setup_ms and not self.minimum_ms:
            complete = engine.future(name + ":complete")

            def compact_begin(_: Event) -> None:
                local: dict[str, Event] = {}
                scoped = lambda items: scope_allocations(name, items)
                for stage in tuple(self.tile(0)):
                    deps = tuple(local[key] for key in stage.after)
                    local[stage.name] = stage.submit(
                        engine, f"{name}:tile0:{stage.name}", deps or (ready,),
                        scoped, owner)
                engine.submit(
                    name + ":tile0:retire",
                    after=tuple(local.values()) or (ready,), owner=owner,
                    on_complete=lambda _event: engine.resolve(complete))

            engine.submit(name + ":begin", after=(ready,), owner=owner,
                          on_complete=compact_begin)
            return complete
        complete = engine.future(name + ":pipeline_complete")
        ready = engine.submit(name + ":setup", after=(ready,), latency=self.setup_ms, owner=owner)
        observation = engine.submit(name + ":span", after=(ready,), latency=self.minimum_ms, owner=owner)
        stage_tail: dict[str, Event] = {}
        prepared_slots = [None for _ in range(min(self.slots, self.count))]
        next_tile, finished = 0, 0
        scoped = lambda items: scope_allocations(name, items)

        def retire(_: Event) -> None:
            nonlocal finished
            finished += 1
            if finished == self.count:
                stage_tail.clear()
                prepared_slots.clear()
                engine.resolve(complete)
            else:
                issue()

        def issue() -> None:
            nonlocal next_tile
            if next_tile >= self.count:
                return
            index = next_tile
            next_tile += 1
            local: dict[str, Event] = {}
            previous = dict(stage_tail)
            slot = index % self.slots
            stages = tuple(self.tile(index))
            cached = prepared_slots[slot]
            if cached is None or (cached[0] is not stages and (
                    len(cached[0]) != len(stages)
                    or any(old is not new for old, new in zip(cached[0], stages)))):
                compiled = tuple((stage, stage.prepare(
                    engine, f"{name}:tile{index}:{stage.name}", scoped)) for stage in stages)
                cached = stages, compiled
                prepared_slots[slot] = cached
            for stage, prepared in cached[1]:
                deps = [local[key] for key in stage.after]
                deps.extend(previous[key] for key in stage.previous if key in previous)
                if stage.name in stage_tail:
                    deps.append(stage_tail[stage.name])
                task_name = f"{name}:tile{index}:{stage.name}"
                event = stage.submit(engine, task_name, tuple(deps) or (ready,),
                                     scoped, owner, prepared)
                local[stage.name] = stage_tail[stage.name] = event
            # At most one stage set per live slot: unique/generated tail stages
            # replace the previous set instead of growing an execution history.
            engine.submit(f"{name}:tile{index}:retire", after=tuple(local.values()),
                          on_complete=retire, owner=owner)

        def begin(_: Event) -> None:
            if not self.count:
                engine.resolve(complete)
                return
            for _ in range(min(self.slots, self.count)):
                issue()

        if self.count < 0 or self.slots < 1:
            raise ValueError("pipeline_requires_nonnegative_tiles_and_positive_slots")
        engine.submit(name + ":begin", after=(ready,), on_complete=begin, owner=owner)
        return engine.submit(name + ":complete", after=(complete, observation), owner=owner)


@dataclass(frozen=True)
class RecipeContext:
    op: object
    work: Mapping[str, float]
    resource_names: Mapping[str, str]
    rates: Mapping[str, float]
    core_count: int
    minimum_ms: float = 0.0
    setup_ms: float = 0.0
    pool_names: Mapping[str, str] = field(default_factory=dict)
    resource_units: Mapping[str, int] = field(default_factory=dict)
    compute_peaks: Mapping[str, float] = field(default_factory=dict)
    pool_units: Mapping[str, int] = field(default_factory=dict)
    service_modes: Mapping[str, tuple[str, float]] = field(default_factory=dict)
    native_work: Mapping[str, float] = field(default_factory=dict)
    service_identity: Mapping = field(default_factory=dict)
    record_service: Callable | None = None
    work_units: Mapping[str, str] = field(default_factory=lambda: dict(NATIVE_WORK_UNITS))
    service_boundary: str = "device_kernel"
    # Physical resource -> (payload granule, wire frame including protocol overhead).
    transaction_frames: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    collective_plan: Callable[[], object] | None = None
    pool_capacities: Mapping[str, float] = field(default_factory=dict)
    record_mapping: Callable | None = None
    invocation_timing: object = None
    store_queues: Mapping[str, Mapping] = field(default_factory=dict)
    # IDs reserved by an enclosing physical kernel invocation. Isolated phase
    # analysis leaves this empty and must retain its own buffer requirements.
    group_buffers: tuple[str, ...] = ()
    joint_native_service: Mapping | None = None
    # Selected observable regime, including continuity group, execution mode,
    # response basis and the distinct response/executed durations.
    service_regime: Mapping | None = None
    # Per-stage capacity caps and diagnostics for a measured MAC service.  The
    # caps constrain only the selected physical pipe; other stages keep their
    # ordinary work and overlap.
    mac_caps: Mapping[str, float] = field(default_factory=dict)
    mac_service: Mapping | None = None
    # Source-selected implementation for this matched invocation only.  It is
    # never installed into the case or reused by an off-domain invocation.
    program_recipe: object = None

    def for_phase(self, work, *, kernel, dtype=None, layout=None, axes=None):
        """Bind explicitly identified phase work without repeating kernel setup."""
        op = work.apply(self.op)
        identity = {**self.service_identity, "kernel": kernel}
        if dtype is not None:
            op = replace(op, dtype=dtype)
            identity["dtype"] = dtype
        if layout is not None:
            op = replace(op, layout=layout)
            identity["layout"] = layout
        if axes is not None:
            identity["axes"] = axes
        native = {**work.resources, "cube": sum(work.matrix.values())}
        identity = json.loads(json.dumps(identity, allow_nan=False))
        return replace(
            self, op=op, work=self.bind(work),
            native_work={role: amount for role, amount in native.items() if amount},
            setup_ms=0, minimum_ms=0, service_identity=identity,
            service_boundary="resource_phase", invocation_timing=None,
        )

    def physical_role(self, role):
        return self.service_modes[role][0] if role in self.service_modes else role

    def placement(self, role, units):
        return {self.resource_names[self.physical_role(role)]: tuple(units)}

    def allocate(self, name, pool, size, units):
        return Allocation(name, self.pool_names[pool], size, tuple(units))

    def owned(self, owner, owners, roles):
        from .ownership import unit_group
        roles = {self.physical_role(role) for role in roles}
        return {self.resource_names[role]: unit_group(
            owner, owners, self.resource_units.get(role, self.core_count))
            for role in roles if role in self.resource_names}

    def bind(self, work) -> dict[str, float]:
        """Resolve typed stage work onto shared physical resources, in ms units."""
        demand = {}
        for dtype, flops in work.matrix.items():
            if not flops:
                continue
            if dtype not in self.compute_peaks:
                raise ModelingError(f"compute_precision_rate_required:{dtype}")
            name = self.resource_names["cube"]
            demand[name] = demand.get(name, 0) + 1000 * flops / self.compute_peaks[dtype]
        for role, amount in work.resources.items():
            if not amount:
                continue
            if role in self.service_modes:
                resource = self.physical_role(role)
                rate = self.service_modes[role][1]
                name = self.resource_names[resource]
                # Modes retain physical byte counts. Only service is expressed
                # in the base pipe's units so its different instructions contend.
                amount *= self.rates[name] * 1000 / rate
                role = resource
            if role in {"mte2", "mte3"} and role not in self.resource_names:
                continue
            if role not in self.resource_names:
                raise ModelingError(f"physical_resource_rate_missing:{role}")
            name = self.resource_names[role]
            demand[name] = demand.get(name, 0) + amount
        return demand


class Recipe(Protocol):
    def lower(self, context: RecipeContext) -> Program: ...


class ResourceEnvelope:
    """Aggregate independent work where no instruction-stage recipe is declared."""

    def lower(self, context: RecipeContext) -> KernelProgram:
        return KernelProgram(
            (Stage("setup", latency=context.setup_ms),
             Stage("service", context.work, after=("setup",))),
            minimum_ms=context.minimum_ms,
        )


@dataclass(frozen=True)
class TransferRecipe:
    """Per-path transactions with slots held through post-service response.

    ``response_ms`` is a fixed return delay after transmission. It is not the
    inclusive response span used by the address-dependent scalar store model.
    """

    transaction_bytes: int | None = None
    outstanding: int | None = None
    response_ms: float = 0.0

    def lower(self, context: RecipeContext) -> KernelProgram:
        from .transfer import FiniteTransfer
        paths = []
        for resource, amount in context.work.items():
            frame = context.transaction_frames.get(resource)
            if frame is not None:
                payload, wire = frame
                if self.transaction_bytes is not None and self.transaction_bytes != payload:
                    raise ModelingError(f"transfer_granule_conflicts_with_link:{resource}")
                granule = wire
            else:
                granule = self.transaction_bytes if self.transaction_bytes is not None else 1
            paths.append(FiniteTransfer({resource: amount}, granule,
                                        self.outstanding, self.response_ms))
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("transactions", after=("setup",), program=ParallelProgram(tuple(paths))),
        ), minimum_ms=context.minimum_ms)


class RecipeBook:
    """Explicit implementation bindings; adding recipes does not edit the loop."""

    def __init__(self, recipes: Mapping[str, Recipe] | None = None,
                 *, families: Mapping[str, Recipe] | None = None, default: Recipe | None = None,
                 programs: Mapping[str, Recipe] | None = None):
        from .matrix import TiledMatmul
        from .matrix_mapping import MappedMatmul
        from .matrix_steady import SteadyMatmul
        from .state_program import RecurrentVector
        generic_grouped = (programs or {}).get("matmul", MappedMatmul())
        self._generic_grouped = generic_grouped
        balanced_matrix = (families or {}).get("grouped_gemm", generic_grouped)
        from .generic_latent_attention import build as _generic_latent_build
        from .mechanism import PlannedRecipe as _PlannedRecipe
        self._explicit_recipes = frozenset((recipes or {}).keys())
        # ``_explicit_recipes`` also receives internally registered source
        # recipes.  Only this set is the caller's own override.
        self._caller_recipes = frozenset((recipes or {}).keys())
        self._explicit_families = frozenset((families or {}).keys())
        self._family_kinds = {}
        self.recipes = {
            "balanced_grouped_gate_up": balanced_matrix,
            "balanced_grouped_down": balanced_matrix,
            **(recipes or {}),
        }
        self.families = {"recurrent_scan": RecurrentScan(), "recurrent_vector": RecurrentVector(),
                         "engram_rh2d": HostStaging(),
                         "grouped_gemm": (programs or {}).get("matmul", SteadyMatmul()),
                         **(families or {})}
        self.programs = {"matmul": TiledMatmul(128, 128),
                         "dense_latent_attention": _PlannedRecipe(_generic_latent_build),
                         **(programs or {})}
        self.default = default or ResourceEnvelope()

    def select(self, op: object) -> Recipe:
        from .mechanism import PlannedRecipe
        if op.plan_required:
            recipe = self.recipes.get(op.recipe)
            if not isinstance(recipe, PlannedRecipe):
                raise ModelingError(f"source_plan_binding_required:{op.recipe}")
            if op.source_boundary == "call" and recipe.kernels is None:
                raise ModelingError(f"source_kernel_expansion_required:{op.recipe}")
            return recipe
        view = dict(dict(op.metadata).get("attention_tensor_view", {}) or {})
        explicit = (str(getattr(op, "recipe", "")) in self._explicit_recipes
                    or str(getattr(op, "family", "")) in self._explicit_families)
        if (not explicit
                and dict(op.metadata).get("program_kind") == "dense_latent_attention_generic"
                and view.get("kind") == "dense_paged_latent"
                and "latent_attention_schedule" in dict(op.metadata)):
            recipe = self.programs.get("dense_latent_attention")
            if isinstance(recipe, PlannedRecipe):
                return recipe
        kind = dict(op.metadata).get("program_kind")
        recipe_name = str(getattr(op, "recipe", ""))
        # Priority: a caller-supplied recipe override, then an explicitly
        # selected grouped_gemm physical family for the generic balanced
        # names, then the ordinary recipe/family/default lookup.  Specific
        # alternate recipes (K3, negative-weight, explicit grouped variants)
        # keep their own binding and are never replaced by the NXNK family.
        generic_grouped = {"balanced_grouped_gate_up",
                           "balanced_grouped_down"}
        if recipe_name in self._caller_recipes:
            recipe = self.recipes[recipe_name]
        elif (str(getattr(op, "family", "")) in self._explicit_families
              and str(getattr(op, "family", "")) != "grouped_gemm"):
            # An explicit caller family is authoritative over any automatic
            # source leaf, including the shared W4A8 default.
            recipe = self.families[op.family]
        elif str(getattr(op, "family", "")) in {"gate_up_projection",
                                                "down_projection"}:
            from .shared_w4a8 import (config_from_op as _shared_config,
                                      matches_source_interface as _shared_match,
                                      recipe as _shared_recipe)
            if _shared_match(op):
                # A valid ordinary shared W4A8 interface with a bad local
                # shape/scale must surface its error, not fall back.
                _shared_config(op)
                recipe = _shared_recipe()
            else:
                recipe = self.recipes.get(recipe_name, self.families.get(
                    op.family, self.programs.get(kind, self.default)))
        elif (str(getattr(op, "family", "")) == "grouped_gemm"
              and "grouped_gemm" in self._explicit_families
              and recipe_name in generic_grouped):
            recipe = self.families["grouped_gemm"]
            # The shared MXA8W4 NZNK family is source-specific.  Do not apply
            # it to unsupported W8/other layouts: the existing typed
            # config_from_op is the gate; on rejection use the ordinary
            # recipe default (never a blanket all-grouped W4 binding).
            if self._family_kinds.get("grouped_gemm") == "ascend_mxa8w4_nznk":
                from .mxfp4_grouped import (config_from_op as _mxfp4_config,
                                            matches_source_interface)
                if not matches_source_interface(op):
                    # Unsupported W8/other layout/weight group: use the
                    # ordinary analytical grouped recipe that existed before
                    # the source family.
                    recipe = self._generic_grouped
                else:
                    # A valid source interface with malformed/inconsistent
                    # rows must surface its real error, not silently fall back.
                    _mxfp4_config(op)
        else:
            recipe = self.recipes.get(recipe_name, self.families.get(
                op.family, self.programs.get(kind, self.default)))
        if "matrix_group_rows" in dict(op.metadata):
            from .grouped_rows import adapt
            recipe = adapt(recipe, op)
        else:
            from .expected_grouped import adapt
            recipe = adapt(recipe, op)
        return recipe

    def prepare(self, op, geometry):
        from .mechanism import PlannedRecipe
        recipe = self.select(op)
        if op.source_boundary == "call":
            raise ModelingError(f"source_call_requires_kernel_expansion:{op.recipe}")
        return recipe.prepare(op, geometry) if isinstance(recipe, PlannedRecipe) else (op, recipe)

    def expand(self, ops):
        from .boundaries import expand_call
        from .mechanism import PlannedRecipe
        result = []
        for op in ops:
            recipe = self.select(op)
            result.extend(expand_call(op, recipe.kernels)
                          if (isinstance(recipe, PlannedRecipe) and recipe.kernels
                              and op.source_boundary != "kernel") else (op,))
        return tuple(result)

    @classmethod
    def from_mapping(cls, raw):
        from .implementations import recipes as implementation_recipes
        from .mc2_graph import recipes as native_moe_recipes
        from .hccl_graph import recipes as hccl_recipes
        from .sparse_kv_offload import recipes as sparse_kv_offload_recipes
        from .matrix import TiledMatmul, HierarchicalMatmul
        from .matrix_mapping import MappedMatmul
        from .matrix_phase import PhaseMatmul
        from .matrix_steady import SteadyMatmul
        from .indexed_attention import recipe as indexed_attention
        from .indexed_quant_attention import recipe as indexed_quant_attention
        from .attention_decode import StagedGQADecode
        from .attention_prefill import StagedGQAPrefill
        from .pointwise import TiledPointwise
        from .state_program import RecurrentVector
        from .services import ResourceService
        from .tensor_copy import recipe as aiv_tensor_copy
        from .mxfp4_grouped import recipe as mxa8w4_nznk
        from .bf16_grouped import recipe as bf16_grouped
        from .mxfp8_grouped import recipe as mxfp8_grouped
        from .family_kda_recurrent import recipes as kda_recurrent_recipes
        from .family_attnres import recipes as attn_res_recipes
        from .collectives import PairwiseCollective, RingCollective
        from .direct_peer import DirectPeerCollective
        factories = {"envelope": ResourceEnvelope, "tiled_matmul": TiledMatmul,
                     "hierarchical_matmul": HierarchicalMatmul, "mapped_matmul": MappedMatmul,
                     "phase_matmul": PhaseMatmul,
                     "steady_matmul": SteadyMatmul,
                     "indexed_attention_pipeline": indexed_attention,
                     "indexed_attention_packed_int8": indexed_quant_attention,
                     "staged_gqa_decode": StagedGQADecode,
                     "staged_gqa_prefill": StagedGQAPrefill,
                     "transfer": TransferRecipe, "recurrent_scan": RecurrentScan,
                     "recurrent_vector": RecurrentVector, "tiled_pointwise": TiledPointwise,
                     "aiv_tensor_copy": aiv_tensor_copy,
                     "ascend_mxa8w4_nznk": mxa8w4_nznk,
                     "ascend_bf16_grouped": bf16_grouped,
                     "ascend_mxfp8_grouped_nz": mxfp8_grouped,
                     "resource_service": ResourceService, "ring_collective": RingCollective,
                     "pairwise_collective": PairwiseCollective,
                     "direct_peer_collective": DirectPeerCollective}
        def build(values):
            return {name: factories[value["kind"]](**{k: v for k, v in value.items() if k != "kind"})
                    for name, value in values.items()}
        explicit = {**native_moe_recipes(raw), **hccl_recipes(raw),
                    **sparse_kv_offload_recipes(),
                    **implementation_recipes(raw.get("lowerings", {})),
                    **kda_recurrent_recipes(),
                    **attn_res_recipes(),
                    **build(raw.get("recipes", {}))}
        book = cls(explicit, families=build(raw.get("families", {})),
                   programs=build(raw.get("programs", {})))
        book._caller_recipes = frozenset((raw.get("recipes") or {}).keys())
        book._family_kinds = {
            name: value.get("kind") for name, value in
            (raw.get("families") or {}).items()
            if isinstance(value, Mapping)}
        return book


class HostStaging:
    def lower(self, context):
        names, work = context.resource_names, dict(context.work)
        memory = {names["host_memory"]: work.pop(names["host_memory"])}
        compute = {names["host_vector"]: work.pop(names["host_vector"])}
        return KernelProgram((
            Stage("lookup", memory), Stage("decode", compute, after=("lookup",)),
            Stage("transfer", work, after=("decode",)),
        ), context.minimum_ms)


class RecurrentScan:
    """C1 -> vector delta -> C2 -> state update for each dependent chunk."""

    def lower(self, context):
        metadata = dict(context.op.metadata)
        count, chunk, tokens = metadata["chunks"], metadata["chunk"], metadata["tokens"]
        matrix = context.resource_names["cube"]
        vector = context.resource_names["vector"]
        heads = metadata["heads"]
        cap = {matrix: min(1, heads / context.core_count) * context.rates[matrix]}

        tile_cache = {}
        chunk_vec_total = float(metadata.get("chunk_vector_ops", 0.0))

        def tile(index):
            valid = min(chunk, tokens - index * chunk)
            if valid in tile_cache:
                return tile_cache[valid]
            fraction = valid / tokens
            # State moves once per chunk; token operands scale with real tail
            # rows. Aggregate traffic is conserved, including the last tile.
            work = {}
            for key, amount in context.work.items():
                if key == matrix:
                    work[key] = amount * fraction
                elif key == vector and 0.0 < chunk_vec_total <= amount:
                    work[key] = chunk_vec_total * (1 / count) + (amount - chunk_vec_total) * fraction
                elif key == vector:
                    work[key] = amount * fraction
                else:
                    work[key] = amount * (1 / count)
            first = {matrix: work.get(matrix, 0) / 2}
            last = dict(work)
            last[matrix] = work.get(matrix, 0) / 2
            vwork = {vector: last.pop(vector, 0)}
            res = (Stage("c1", first, caps=cap), Stage("delta", vwork, after=("c1",)),
                   Stage("c2_update", last, after=("delta",), caps=cap))
            tile_cache[valid] = res
            return res

        return PipelineProgram(count, 1, tile, context.minimum_ms, context.setup_ms)
