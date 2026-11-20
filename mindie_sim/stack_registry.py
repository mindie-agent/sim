"""Unified software stack registry and lowering-base resolver.

[Role] Register audited serving software stacks (``SoftwareStackSpec``) inside
  ``SoftwareStackRegistry`` and resolve a ``Case`` + ``LoweringRequest`` into
  an immutable ``ResolvedBase`` carrying tier policy, production capture
  binding, and recipe-origin provenance.
[Invariants] Never mutates caller ``Case`` objects or ``Software.kernels``
  dictionaries. Exactly one lowering input mode (fixed ``StepRequest``,
  scheduled ``iterations``, or explicit ``declared_ops``) must be active per
  ``LoweringRequest``. Production Decode across ``unmerge``, ``mainline``, and
  ``should_achieve`` binds ``ACLGraph`` capture when ``production_capture`` is
  enabled.
[Timeouts & Locks] Pure in-memory resolution over bounded stack metadata; no
  I/O or locks.
[Side-Effect Policy] Returns new frozen ``ResolvedBase`` and ``Case`` records.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal

from mindie_sim.compose_runtime import capture_keys, finalize_compilation_spine
from mindie_sim.feature_mechanisms import MechanismContext
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import IterationPlan
from mindie_sim.stack_tracks import (
    RUNTIME_STACKS,
    RuntimeStackProfile,
    apply_resolved_mechanisms,
    resolve_runtime_stack,
    resolve_stack_track,
)
from .prediction.results import StepRequest, Tier

OriginKind = Literal["caller", "stack_profile", "body_default"]
DEFAULT_STACK_NAME = "vllm-ascend"
MIN_CAPTURE_SEQS = 16


@dataclass(frozen=True, slots=True)
class SoftwareStackSpec:
    """Declarative specification of one registered inference software stack."""

    name: str
    repository: str
    description: str
    profile: RuntimeStackProfile
    sources: tuple[SourceRef, ...] = ()
    default_track: str = "merged"
    decode_capture_mode: str = "full"


@dataclass(frozen=True, slots=True)
class RankOps:
    """One replica's immutable operator sequence."""

    replica: int
    ops: tuple[OpSpec, ...]


@dataclass(frozen=True, slots=True)
class LoweringRequest:
    """Unified lowering input covering fixed step, serving iterations, or explicit ops."""

    step: StepRequest | None = None
    iterations: tuple[tuple[int, IterationPlan], ...] = ()
    declared_ops: tuple[RankOps, ...] = ()
    tier: Tier = "mainline"
    apply_body_defaults: bool = True
    production_capture: bool = False
    stack: str | None = None

    def __post_init__(self) -> None:
        active = int(self.step is not None) + int(bool(self.iterations)) + int(bool(self.declared_ops))
        if active != 1:
            raise ModelingError(f"lowering_request_requires_exactly_one_input:{active}")
        if self.tier not in {"unmerge", "mainline", "should_achieve"}:
            raise ModelingError(f"lowering_request_invalid_tier:{self.tier}")


@dataclass(frozen=True, slots=True)
class ResolvedBase:
    """Resolved immutable case, stack binding, and provenance ready for rewrite."""

    case: Case
    request: LoweringRequest
    stack: SoftwareStackSpec
    recipe_origins: tuple[tuple[str, OriginKind], ...]
    sources: tuple[SourceRef, ...]


def _extract_origins(metadata: Mapping[str, object]) -> tuple[tuple[str, OriginKind], ...]:
    raw = metadata.get("recipe_binding_origins")
    if not isinstance(raw, Mapping):
        return ()
    entries: list[tuple[str, OriginKind]] = []
    for section, mapping in sorted(raw.items()):
        if not isinstance(mapping, Mapping):
            continue
        for key, origin in sorted(mapping.items()):
            label = str(origin)
            if label in {"caller", "stack_profile", "body_default"}:
                entries.append((f"{section}.{key}", label))  # type: ignore[arg-type]
    return tuple(entries)


def _stack_name_for_case(case: Case, requested: str | None) -> str:
    if requested:
        return str(requested).strip()
    meta = dict(case.runtime.metadata)
    report = meta.get("feature_track_report")
    if isinstance(report, Mapping) and report.get("stack"):
        return str(report["stack"]).strip()
    return DEFAULT_STACK_NAME


def _apply_production_decode_capture(
    case: Case,
    request: LoweringRequest,
    stack: SoftwareStackSpec,
) -> Case:
    if not request.production_capture or request.step is None or request.step.phase != "decode":
        return case
    if case.runtime.graph_capture_mode == stack.decode_capture_mode:
        return case
    max_seqs = max(
        MIN_CAPTURE_SEQS,
        int(case.runtime.max_num_seqs or MIN_CAPTURE_SEQS),
        request.step.sequences if isinstance(request.step.sequences, int)
        else max(request.step.sequences, default=1),
    )
    mtp = int(case.runtime.mtp_proposal_tokens or 0)
    keys = tuple(case.runtime.graph_capture_keys or capture_keys(max_seqs, mtp))
    runtime = replace(
        case.runtime,
        graph_capture_mode=stack.decode_capture_mode,
        graph_capture_keys=keys,
    )
    return replace(case, runtime=runtime)


def _apply_tier_policy(
    case: Case,
    request: LoweringRequest,
    stack: SoftwareStackSpec,
) -> Case:
    if request.tier == "mainline":
        return case
    if request.tier == "unmerge":
        kernels = deepcopy(dict(case.runtime.kernels))
        meta = dict(case.runtime.metadata)
        origins = meta.get("recipe_binding_origins")
        lowerings = dict(kernels.get("lowerings") or {})
        if isinstance(origins, Mapping) and isinstance(origins.get("lowerings"), Mapping):
            low_origins = origins["lowerings"]
            lowerings = {
                key: value for key, value in lowerings.items()
                if low_origins.get(key) == "caller"
            }
        else:
            lowerings = {}
        kernels["lowerings"] = lowerings
        policy = dict(meta.get("execution_policy") or {})
        policy["automatic_optional_fusions"] = False
        meta["execution_policy"] = policy
        runtime = replace(
            case.runtime,
            kernels=kernels,
            lowering_passes=(),
            metadata=tuple(meta.items()),
        )
        return replace(case, runtime=runtime)

    # should_achieve: enable eligible structural stack optimizations cleanly.
    rt_dict = case.runtime.to_dict()
    parallel = case.runtime.parallel
    phase = request.step.phase if request.step is not None else None
    arch = str(case.machine.arch_family or "").lower()
    ctx = MechanismContext(
        model=case.model,
        sku=str(case.machine.sku or ""),
        arch=arch,
        parallel={
            "tp": parallel.tp,
            "dp": parallel.dp,
            "pp": parallel.pp,
            "ep": parallel.ep,
            "pcp": parallel.pcp,
            "dcp": parallel.dcp,
        },
        caller_explicit=frozenset(),
        options={"phase": phase} if phase else {},
    )
    resolved = resolve_stack_track(
        stack.profile,
        track="expected_ideal",
        model_key=str(case.model.name),
        ctx=ctx,
    )
    apply_resolved_mechanisms(rt_dict, resolved, ctx)
    finalize_compilation_spine(rt_dict, raw_eager=False)
    from mindie_sim.schema.case import Software

    new_runtime = Software.from_mapping(rt_dict)
    return replace(case, runtime=new_runtime)


class SoftwareStackRegistry:
    """Registry of supported inference software stacks and tier resolution."""

    def __init__(self, specs: Iterable[SoftwareStackSpec] = (), /) -> None:
        self._specs: dict[str, SoftwareStackSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: SoftwareStackSpec, /) -> SoftwareStackSpec:
        key = spec.name.strip().lower()
        if not key:
            raise ModelingError("software_stack_name_required")
        self._specs[key] = spec
        return spec

    def get(self, name: str, /) -> SoftwareStackSpec:
        key = str(name or "").strip().lower()
        if key in self._specs:
            return self._specs[key]
        profile = resolve_runtime_stack(name)
        spec = SoftwareStackSpec(
            name=profile.name,
            repository=profile.repository,
            description=profile.description,
            profile=profile,
        )
        return spec

    def list_stacks(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def resolve_base(self, case: Case, request: LoweringRequest, /) -> ResolvedBase:
        """Resolve stack profile, tier rules, capture policy, and binding origins."""
        stack = self.get(_stack_name_for_case(case, request.stack))
        working = replace(
            case,
            runtime=replace(
                case.runtime,
                kernels=deepcopy(dict(case.runtime.kernels)),
                metadata=tuple(deepcopy(dict(case.runtime.metadata)).items()),
            ),
        )
        working = _apply_tier_policy(working, request, stack)
        working = _apply_production_decode_capture(working, request, stack)
        origins = _extract_origins(dict(working.runtime.metadata))
        return ResolvedBase(
            case=working,
            request=request,
            stack=stack,
            recipe_origins=origins,
            sources=stack.sources,
        )


@lru_cache(maxsize=1)
def default_stack_registry() -> SoftwareStackRegistry:
    """Return the built-in ``SoftwareStackRegistry`` populated from audited stacks."""
    specs = [
        SoftwareStackSpec(
            name=profile.name,
            repository=profile.repository,
            description=profile.description,
            profile=profile,
            default_track="raw_eager" if profile.name == "raw_eager" else "merged",
            decode_capture_mode="none" if profile.name == "raw_eager" else "full",
        )
        for profile in RUNTIME_STACKS.values()
    ]
    return SoftwareStackRegistry(specs)


StackRegistry = SoftwareStackRegistry

__all__ = [
    "DEFAULT_STACK_NAME",
    "LoweringRequest",
    "OriginKind",
    "RankOps",
    "ResolvedBase",
    "SoftwareStackRegistry",
    "SoftwareStackSpec",
    "StackRegistry",
    "default_stack_registry",
]
