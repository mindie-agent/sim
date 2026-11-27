"""Unified dispatch for shipping operators and executable template tiers.

[Role] Route declared tier requests to the existing family registry or catalog.
[Invariants] Candidate generation and lowering use the same concrete contracts.
[Timeouts & Locks] Local dispatch and a single cached builtin catalog.
[Side-Effect Policy] Imports no device runtime and performs no external work.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from mindie_sim.prediction.operators.registry import FamilyDispatcher, FamilyRegistry, builtin_family_registry
from mindie_sim.schema.common import ModelingError

from .catlass_catalog import CATLASS_SOURCE_REFS, CATLASS_TEMPLATE_IDS, enumerate_catlass_specs
from .catlass_lower import lower_catlass_spec
from .contracts import CatlassKernelSpec, DslKernelSpec, TemplateSourceRef
from .dsl_catalog import DSL_KERNEL_IDS, DSL_SOURCE_REFS, enumerate_dsl_specs
from .dsl_lower import lower_dsl_spec
from .fuse_region import compile_fuse_region

Tier = Literal["K1", "K2", "K3"]


def all_template_source_refs() -> tuple[TemplateSourceRef, ...]:
    """Return the full pinned reference inventory, independent of hardware."""
    return tuple(dict.fromkeys((*CATLASS_SOURCE_REFS, *DSL_SOURCE_REFS)))


class TemplateCatalog:
    """Public dispatcher for concrete template specs and shipping family plans."""

    __slots__ = ("_family_registry",)

    def __init__(self, family_registry: FamilyRegistry | None = None):
        self._family_registry = family_registry

    @property
    def family_registry(self) -> FamilyRegistry:
        return self._family_registry if self._family_registry is not None else builtin_family_registry()

    def ids(self, tier: Tier) -> tuple[str, ...]:
        if tier == "K1":
            return tuple(spec.name for spec in self.family_registry.specs())
        if tier == "K2":
            return CATLASS_TEMPLATE_IDS
        if tier == "K3":
            return (*DSL_KERNEL_IDS, "fuse_region")
        raise ModelingError(f"template_tier_unknown:{tier}")

    def enumerate(self, tier: Tier, problem, datapath, *, template_id=None, **options):
        """Enumerate supported specializations before any performance ranking."""
        if tier == "K1":
            if template_id is not None or options:
                raise ModelingError("shipping_family_selection_uses_operator_context")
            matched = self.family_registry.match(problem)
            return () if matched is None else (matched,)
        if tier == "K2":
            if template_id is not None:
                if "template_ids" in options:
                    raise ModelingError("catlass_template_filters_conflict")
                options["template_ids"] = (template_id,)
            return enumerate_catlass_specs(problem, datapath, **options)
        if tier == "K3" and template_id is not None and template_id != "fuse_region":
            return enumerate_dsl_specs(template_id, problem, datapath, **options)
        raise ModelingError(f"template_enumeration_requires_catalog_kernel:{tier}:{template_id}")

    def resolve_k1(self, op, planning_context):
        return FamilyDispatcher(self.family_registry).resolve_plan(op, planning_context)

    def compile_region(self, region_id, op_ids, datapath, **options):
        return compile_fuse_region(region_id, op_ids, datapath, **options)

    def lower(self, tier: Tier, spec, problem=None, datapath=None, **options):
        """Dispatch K1 plans, K2/K3 specs or a K3 FuseRegion declaration.

        K1 receives an OpSpec and OperatorPlanningContext. FuseRegion receives
        a region id and graph op ids. K2 and reference K3 kernels receive
        the catalog specialization paired with its original problem.
        """
        if tier == "K1" and datapath is None and not options:
            return self.resolve_k1(spec, problem)
        if tier == "K2" and isinstance(spec, CatlassKernelSpec):
            return lower_catlass_spec(spec, problem, datapath, **options)
        if tier == "K3":
            if isinstance(spec, DslKernelSpec):
                return lower_dsl_spec(spec, problem, datapath, **options)
            if isinstance(spec, str):
                return self.compile_region(spec, problem, datapath, **options)
        raise ModelingError(f"template_tier_spec_mismatch:{tier}:{type(spec).__name__}")

    @staticmethod
    def all_template_source_refs() -> tuple[TemplateSourceRef, ...]:
        return all_template_source_refs()


@lru_cache(maxsize=1)
def builtin_template_catalog() -> TemplateCatalog:
    return TemplateCatalog()
