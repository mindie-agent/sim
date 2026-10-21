"""Explicit framework work on the shared resource timeline.

Costs are user-supplied affine estimates in microseconds, not measured CPU
service or a hardware core-count model.  CPU and stream work use independent,
unit-capacity resources so their dependencies and contention remain visible in
the existing resource engine.  Device kernel dispatch is outside this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Mapping, Sequence

from .a3_resource_engine import ResourceEngine, ResourceSpec

_COEFFICIENTS = ("fixed_us", "per_request_us", "per_token_us", "per_kernel_us")
_KINDS = ("schedule", "input", "submit", "output", "stream")
_SCOPES = {"process", "replica"}


class FrameworkProfileError(ValueError):
    """Invalid explicit framework cost or resource configuration."""


def _nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FrameworkProfileError(f"framework_value_invalid:{label}")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise FrameworkProfileError(f"framework_value_invalid:{label}") from exc
    if not math.isfinite(number) or number < 0.0:
        raise FrameworkProfileError(f"framework_value_invalid:{label}")
    return number


def _mapping(value: Any, allowed: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FrameworkProfileError(f"framework_mapping_required:{label}")
    if any(key not in allowed for key in value):
        raise FrameworkProfileError(f"framework_unknown_field:{label}")
    return value


@dataclass(frozen=True)
class AffineCost:
    """Time = fixed + request/token/kernel counts times their coefficients."""

    fixed_us: float = 0.0
    per_request_us: float = 0.0
    per_token_us: float = 0.0
    per_kernel_us: float = 0.0

    def __post_init__(self) -> None:
        for key in _COEFFICIENTS:
            object.__setattr__(self, key, _nonnegative(getattr(self, key), key))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> AffineCost:
        data = _mapping(payload, set(_COEFFICIENTS), "affine_cost")
        return cls(**data)

    def to_dict(self) -> dict[str, float]:
        return {key: getattr(self, key) for key in _COEFFICIENTS}

    @property
    def is_zero(self) -> bool:
        return all(getattr(self, key) == 0.0 for key in _COEFFICIENTS)

    def duration_ms(
        self, requests: float = 0, tokens: float = 0, kernels: float = 0,
    ) -> float:
        request_count = _nonnegative(requests, "requests")
        token_count = _nonnegative(tokens, "tokens")
        kernel_count = _nonnegative(kernels, "kernels")
        duration = (
            self.fixed_us
            + self.per_request_us * request_count
            + self.per_token_us * token_count
            + self.per_kernel_us * kernel_count
        ) / 1000.0
        if not math.isfinite(duration):
            raise FrameworkProfileError("framework_duration_not_finite")
        return duration


@dataclass(frozen=True)
class FrameworkProfile:
    """Source-labelled framework costs; scopes describe simulator sharing."""

    source: str
    cpu_scope: str = "process"
    stream_scope: str = "process"
    device_launch_us: float = 0.0
    schedule: AffineCost = field(default_factory=AffineCost)
    input: AffineCost = field(default_factory=AffineCost)
    submit: AffineCost = field(default_factory=AffineCost)
    output: AffineCost = field(default_factory=AffineCost)
    stream: AffineCost = field(default_factory=AffineCost)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise FrameworkProfileError("framework_source_required")
        object.__setattr__(
            self, "device_launch_us",
            _nonnegative(self.device_launch_us, "device_launch_us"),
        )
        for key in ("cpu_scope", "stream_scope"):
            value = getattr(self, key)
            if not isinstance(value, str) or value not in _SCOPES:
                raise FrameworkProfileError(f"framework_scope_invalid:{key}")
        for kind in _KINDS:
            if not isinstance(getattr(self, kind), AffineCost):
                raise FrameworkProfileError(f"framework_cost_invalid:{kind}")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> FrameworkProfile:
        allowed = {"source", "cpu_scope", "stream_scope", "device_launch_us", *_KINDS}
        data = dict(_mapping(payload, allowed, "profile"))
        if "source" not in data:
            raise FrameworkProfileError("framework_source_required")
        for kind in _KINDS:
            if kind in data:
                data[kind] = AffineCost.from_mapping(data[kind])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "cpu_scope": self.cpu_scope,
            "stream_scope": self.stream_scope,
            "device_launch_us": self.device_launch_us,
            **{kind: getattr(self, kind).to_dict() for kind in _KINDS},
        }

    @property
    def is_zero(self) -> bool:
        return self.device_launch_us == 0.0 and all(
            getattr(self, kind).is_zero for kind in _KINDS
        )


def _replica_index(replica: int) -> int:
    if isinstance(replica, bool) or not isinstance(replica, int) or replica < 0:
        raise FrameworkProfileError("framework_replica_invalid")
    return replica


def ensure_framework_resources(
    engine: ResourceEngine, profile: FrameworkProfile, replica: int,
) -> dict[str, str]:
    """Register compatible scopes atomically without replacing live resources."""
    index = _replica_index(replica)
    if not isinstance(profile, FrameworkProfile):
        raise FrameworkProfileError("framework_profile_required")
    specs: dict[str, ResourceSpec] = {}
    for alias, suffix, scope in (
        ("cpu", "cpu", profile.cpu_scope),
        ("stream", "dma", profile.stream_scope),
    ):
        name = suffix if scope == "process" else f"r{index}:{suffix}"
        specs[alias] = ResourceSpec(name, 1.0, "ms", scope=scope)
    for spec in specs.values():
        existing = engine.resources.get(spec.name)
        # The framework's process-wide PrepareSpec registers "cpu"/"dma"
        # with the older generic "shared" label.  They are the same resources, so
        # retain its object and service counters rather than create a second
        # CPU lane.  Other scope/quantity/policy conflicts remain errors.
        compatible_shared = (
            existing is not None
            and spec.name in {"cpu", "dma"}
            and spec.scope == "process"
            and existing == ResourceSpec(spec.name, 1.0, "ms", scope="shared")
        )
        if existing is not None and existing != spec and not compatible_shared:
            raise FrameworkProfileError(f"framework_resource_conflict:{spec.name}")
    for spec in specs.values():
        if spec.name not in engine.resources:
            engine.resources[spec.name] = spec
            engine.serviced[spec.name] = 0.0
    return {alias: spec.name for alias, spec in specs.items()}


def add_framework_task(
    engine: ResourceEngine,
    profile: FrameworkProfile,
    kind: str,
    *,
    requests: float,
    tokens: float,
    kernels: float,
    prefix: str,
    replica: int,
    dependencies: Sequence[str] = (),
    release_ms: float = 0.0,
    name: str | None = None,
) -> str:
    """Append one framework task, including zero-work dependency gates."""
    if not isinstance(profile, FrameworkProfile):
        raise FrameworkProfileError("framework_profile_required")
    if not isinstance(kind, str) or kind not in _KINDS:
        raise FrameworkProfileError("framework_kind_invalid")
    if not isinstance(prefix, str) or not prefix.strip():
        raise FrameworkProfileError("framework_prefix_required")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise FrameworkProfileError("framework_task_name_invalid")
    if (
        not isinstance(dependencies, Sequence)
        or isinstance(dependencies, (str, bytes))
        or any(not isinstance(dep, str) or not dep for dep in dependencies)
    ):
        raise FrameworkProfileError("framework_dependencies_invalid")
    release = _nonnegative(release_ms, "release_ms")
    cost = getattr(profile, kind)
    duration = cost.duration_ms(requests=requests, tokens=tokens, kernels=kernels)
    names = ensure_framework_resources(engine, profile, replica)
    task_id = f"{prefix}:framework:{name or kind}"
    resource = names["stream" if kind == "stream" else "cpu"]
    engine.add_task(
        task_id, {resource: duration}, dependencies=dependencies,
        release_ms=release, owner=f"{prefix}:framework",
        metadata={
            "source": profile.source, "kind": kind,
            "framework_duration_ms": duration,
            "framework_cost": cost.to_dict(),
            "framework_counts": {
                "requests": requests, "tokens": tokens, "kernels": kernels,
            },
        },
    )
    return task_id
