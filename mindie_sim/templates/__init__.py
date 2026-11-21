"""Public API for the CPU-only three-tier kernel template catalog.

[Role] Expose specialization, compilation, and whole-step accounting records.
[Invariants] Imports are independent of device libraries and weights.
[Timeouts & Locks] Pure local imports. No I/O or runtime initialization.
[Side-Effect Policy] No filesystem or device mutation.
"""
from .catlass_catalog import CatlassProblem, enumerate_catlass_specs
from .catlass_lower import lower_catlass_spec
from .contracts import (
    CatlassKernelSpec,
    DslKernelSpec,
    EdgeRouteBinding,
    EvgVisitorNode,
    FusedRegionSummary,
    FuseRegionSpec,
    SemanticWorkItem,
    TemplateLoweringResult,
    TemplateSourceRef,
    TensorRangeObligation,
    compose_step_obligations,
)
from .dsl_catalog import (
    AttnResProblem,
    CommitRecurrentKdaProblem,
    FlashKdaProblem,
    MegaRecurrentKdaProblem,
    MixedQuantSparseMlaProblem,
    MlaPrologProblem,
    QsliProblem,
    QkvQuantMode,
    enumerate_dsl_specs,
    resolve_flash_kda_route,
)
from .dsl_lower import lower_dsl_spec
from .fuse_region import compile_fuse_region
from .registry import TemplateCatalog, all_template_source_refs, builtin_template_catalog

__all__ = [
    "AttnResProblem",
    "CatlassKernelSpec",
    "CatlassProblem",
    "CommitRecurrentKdaProblem",
    "DslKernelSpec",
    "EdgeRouteBinding",
    "EvgVisitorNode",
    "FlashKdaProblem",
    "FusedRegionSummary",
    "FuseRegionSpec",
    "MegaRecurrentKdaProblem",
    "MixedQuantSparseMlaProblem",
    "MlaPrologProblem",
    "QkvQuantMode",
    "QsliProblem",
    "SemanticWorkItem",
    "TemplateCatalog",
    "TemplateLoweringResult",
    "TemplateSourceRef",
    "TensorRangeObligation",
    "all_template_source_refs",
    "builtin_template_catalog",
    "compile_fuse_region",
    "compose_step_obligations",
    "enumerate_catlass_specs",
    "enumerate_dsl_specs",
    "lower_catlass_spec",
    "lower_dsl_spec",
    "resolve_flash_kda_route",
]
