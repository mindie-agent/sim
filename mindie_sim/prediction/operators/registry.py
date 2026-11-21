"""Immutable operator-family registry and governed plan dispatcher.

[Role] Validate unique ownership of operator families and recipe IDs across
  governed ``operators/<family>/`` packages and dispatch each ``OpSpec`` to
  its owning family resolver.
[Invariants] Frozen after construction; duplicate family names, duplicate
  family ownership, or duplicate recipe ID ownership raises ``ModelingError``
  immediately. Lookup resolves explicit ``recipe_id`` ownership before
  ``op.family`` ownership.
[Timeouts & Locks] O(1) dictionary lookups after bounded initialization; no I/O
  or locks.
[Side-Effect Policy] Read-only lookups; never mutates operator specs or
  registered family specs.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from functools import lru_cache
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from ..mechanism import OperatorPlan
from .contracts import (
    OperatorFamilySpec,
    OperatorPlanningContext,
    ResolvedOperator,
)


class FamilyRegistry:
    """Frozen single-owner index over governed ``OperatorFamilySpec`` entries."""

    __slots__ = ("_specs", "_by_name", "_by_recipe", "_by_family", "_rewrite_rules")

    def __init__(self, specs: Iterable[OperatorFamilySpec], /) -> None:
        ordered = tuple(specs)
        by_name: dict[str, OperatorFamilySpec] = {}
        by_recipe: dict[str, OperatorFamilySpec] = {}
        by_family: dict[str, OperatorFamilySpec] = {}
        rules: list[Any] = []
        for spec in ordered:
            if not spec.name or spec.name in by_name:
                raise ModelingError(f"duplicate_operator_family_spec:{spec.name}")
            by_name[spec.name] = spec
            for recipe_id in sorted(spec.owned_recipe_ids):
                if recipe_id in by_recipe:
                    raise ModelingError(
                        f"duplicate_operator_recipe_owner:{recipe_id}:"
                        f"{by_recipe[recipe_id].name}:{spec.name}"
                    )
                by_recipe[recipe_id] = spec
            for family in sorted(spec.families):
                if family in by_family:
                    raise ModelingError(
                        f"duplicate_operator_family_owner:{family}:"
                        f"{by_family[family].name}:{spec.name}"
                    )
                by_family[family] = spec
            rules.extend(spec.rewrite_rules)
        self._specs = ordered
        self._by_name = by_name
        self._by_recipe = by_recipe
        self._by_family = by_family
        self._rewrite_rules = tuple(rules)

    def spec_for_family(self, family: str, /) -> OperatorFamilySpec | None:
        """Return the governed family spec registered for ``family``, if any."""
        return self._by_family.get(str(family or ""))

    def match(self, op: OpSpec, /) -> OperatorFamilySpec | None:
        """Return the governed family owning ``op``, or ``None`` if not governed."""
        recipe_id = str(getattr(op, "recipe", "") or "")
        if recipe_id and recipe_id in self._by_recipe:
            return self._by_recipe[recipe_id]
        family = str(getattr(op, "family", "") or "")
        spec = self._by_family.get(family)
        if spec is not None and not recipe_id:
            return spec
        return None

    def lookup(self, op: OpSpec, /) -> OperatorFamilySpec:
        """Return the owning ``OperatorFamilySpec`` or raise ``ModelingError``."""
        spec = self.match(op)
        if spec is None:
            raise ModelingError(
                f"unregistered_operator_family:{op.family}:{op.recipe}"
            )
        return spec

    def specs(self) -> tuple[OperatorFamilySpec, ...]:
        return self._specs

    def rewrite_rules(self) -> tuple[Any, ...]:
        return self._rewrite_rules


class FamilyDispatcher:
    """Resolve an ``OpSpec`` against a ``FamilyRegistry`` and apply prepared work."""

    __slots__ = ("registry",)

    def __init__(self, registry: FamilyRegistry, /) -> None:
        self.registry = registry

    def resolve_plan(
        self,
        op: OpSpec,
        ctx: OperatorPlanningContext,
        /,
    ) -> ResolvedOperator:
        spec = self.registry.lookup(op)
        resolved = spec.resolve(op, ctx)
        if isinstance(resolved.recipe, OperatorPlan):
            prepared = resolved.recipe.work.apply(resolved.op)
            if resolved.recipe.metadata:
                metadata = dict(prepared.metadata)
                metadata["program_execution"] = {
                    **dict(metadata.get("program_execution", {})),
                    **dict(resolved.recipe.metadata),
                }
                prepared = replace(prepared, metadata=tuple(sorted(metadata.items())))
            return replace(resolved, op=prepared)
        return resolved


@lru_cache(maxsize=1)
def builtin_family_registry() -> FamilyRegistry:
    """Return the frozen registry of all built-in governed operator families."""
    from . import load_builtin_families

    return FamilyRegistry(load_builtin_families())


def make_governed_recipe(spec: OperatorFamilySpec, book: Any, dispatcher: FamilyDispatcher) -> Any:
    """Build a ``PlannedRecipe`` bound inside ``operators/`` for a governed family."""
    from mindie_sim.schema.datapath import (
        ArchId, ComputeArchIdentity, DatapathCapabilitySpec, GENERIC_ARCH_ID,
    )
    from ..mechanism import PlannedRecipe
    from ..ownership import unit_group
    from ..results import OpOccurrence

    def _build(op, geometry):
        datapath = getattr(geometry, "datapath", None)
        if datapath is None and spec.name == "attention":
            from .attention import _planning_context

            geometry = _planning_context(op, geometry).geometry
            datapath = geometry.datapath
        if datapath is None:
            arch = getattr(geometry, "compute_arch", None)
            arch_id = ArchId(arch.preset) if arch is not None and arch.preset else GENERIC_ARCH_ID
            paths = arch.paths if arch is not None else ()
            aics = geometry.compute_units
            aivs = geometry.resource_units.get("vector", aics if arch is not None else 0)
            pairs = (
                tuple((aic, unit_group(aic, aics, aivs)) for aic in range(aics))
                if aics > 0 and aivs > 0 and aivs % aics == 0 else ()
            )
            datapath = DatapathCapabilitySpec(
                compute_arch=ComputeArchIdentity(arch_id, arch),
                aic_count=aics, aiv_count=aivs, aic_aiv_pairs=pairs, usable_paths=paths,
                matrix_output_paths=frozenset(path.target for path in paths if path.source == "l0c"),
                ub_peer_edges=(), stars=None, buffers=tuple(geometry.buffers.items()),
            )
        requested_abi = dict(op.metadata or ()).get("requested_abi")
        ctx = OperatorPlanningContext(
            occurrence=OpOccurrence(
                int(getattr(geometry, "replica", 0)),
                int(getattr(op, "stage", 0)),
                getattr(geometry, "rank", None),
                str(op.op_id),
            ),
            geometry=geometry, datapath=datapath, recipe_book=book,
            requested_abi=str(requested_abi) if requested_abi else None,
            stream_owners=getattr(geometry, "stream_owners", None),
            admission_memory_view=getattr(geometry, "admission_memory_view", None),
            requests=tuple(getattr(geometry, "requests", ()) or ()),
            padded_tokens=int(getattr(geometry, "padded_tokens", 0) or 0),
            layout=str(getattr(geometry, "layout", "") or ""),
        )
        return dispatcher.resolve_plan(op, ctx).recipe

    return PlannedRecipe(_build, traffic_rewrite_capable=bool(spec.traffic_rewrite_capable))


__all__ = [
    "FamilyDispatcher",
    "FamilyRegistry",
    "builtin_family_registry",
    "make_governed_recipe",
]
