"""Load accepted T06 resource/phase profiles from outside the packages.

[Role]
  Validate one user resource-service profile through the same T06
  ``load_resource_profile`` path used for built-in profiles, keep its formulas
  and provenance, and expose the normalized services to the public execution
  facade.

[Invariants]
  * A user profile is a resource/phase service profile, not a full-operator
    duration table.  Primitive samples remain in the observations/replay store;
    the old whole-operator primitive profile is rejected here.
  * The T06 normalizer resolves the supply representation exactly once.
  * A request materializes an immutable profile wrapper; service selection is
    explicit or unambiguous by declared conditions.
  * No model, request, target duration or hidden fallback is synthesized.

[Timeouts & Locks]
  One bounded local file read through T06 (under 50 MiB), then pure validation.

[Side-Effect Policy]
  Reads the explicit source; ``export`` writes only a requested path.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from mindie_sim.observations.calibration import (
    export_resource_profile, load_resource_profile, service_conditions_match,
)
from mindie_sim.observations.resource_services import (
    RESOURCE_SERVICE_PROFILE_SCHEMA, NormalizedResourceService,
    normalize_resource_service,
)

from .errors import ExtensionError
from .services import ProfileServiceRecipe
from .values import canonical_json, text

PROFILE_SCHEMA = RESOURCE_SERVICE_PROFILE_SCHEMA
USER_PROFILE_SCHEMA = "mindie-sim.user-profile/2"
_MAX_PROFILE_BYTES = 50 * 1024 ** 2
_OLD_PRIMITIVE_SCHEMA = "mindie-sim-user-profile/1"


def _source_schema(source: Any) -> str:
    if isinstance(source, Mapping):
        value = source.get("schema")
        return value if isinstance(value, str) else ""
    if not isinstance(source, (str, Path)):
        return ""
    path = Path(source).expanduser()
    try:
        if not path.is_file() or path.stat().st_size > _MAX_PROFILE_BYTES:
            return ""
        value = json.loads(path.read_text(encoding="utf-8"))
        return value.get("schema", "") if isinstance(value, dict) else ""
    except TimeoutError:
        raise
    except (OSError, ValueError):
        return ""


@dataclass(frozen=True)
class UserProfile:
    """One immutable T06 profile plus its normalized service records."""

    id: str
    version: str
    profile: Mapping[str, Any]
    services: tuple[NormalizedResourceService, ...]
    source: Mapping[str, Any] = field(default_factory=dict)
    digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", text(self.id, "user_profile_id_invalid"))
        object.__setattr__(self, "version",
                           text(self.version, "user_profile_version_invalid"))
        services = tuple(self.services)
        if not services or not all(
                isinstance(item, NormalizedResourceService) for item in services):
            raise ExtensionError("user_profile_requires_normalized_services")
        identifiers = [item.service_id for item in services]
        if len(set(identifiers)) != len(identifiers):
            raise ExtensionError("user_profile_service_ids_duplicate")
        object.__setattr__(self, "services", services)
        if not isinstance(self.profile, Mapping):
            raise ExtensionError("user_profile_profile_requires_mapping")
        if not isinstance(self.source, Mapping):
            raise ExtensionError("user_profile_source_requires_mapping")
        digest = self.digest or sha256(canonical_json(
            self.profile, "user_profile_digest").encode("utf-8")).hexdigest()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ExtensionError("user_profile_digest_invalid")
        object.__setattr__(self, "digest", digest)

    def service(self, service_id: str | None = None, *,
                conditions_context: Mapping[str, Any] | None = None
                ) -> NormalizedResourceService:
        if service_id is not None:
            selected = [item for item in self.services
                        if item.service_id == service_id]
            if not selected:
                raise ExtensionError("user_profile_service_missing:"
                                     + str(service_id))
            return selected[0]
        if len(self.services) == 1:
            return self.services[0]
        if conditions_context is not None:
            selected = [item for item in self.services
                        if service_conditions_match(
                            item, conditions_context)["applies"]]
            if len(selected) == 1:
                return selected[0]
            if not selected:
                raise ExtensionError(
                    "user_profile_service_conditions_have_no_match")
            raise ExtensionError(
                "user_profile_service_ambiguous:"
                + ",".join(item.service_id for item in selected))
        raise ExtensionError(
            "user_profile_service_ambiguous:"
            + ",".join(item.service_id for item in self.services))

    def recipe(self, service_id: str | None = None, *,
               conditions_context: Mapping[str, Any] | None = None
               ) -> ProfileServiceRecipe:
        return ProfileServiceRecipe(self.service(
            service_id, conditions_context=conditions_context))

    def describe(self) -> dict[str, Any]:
        services = []
        for item in self.services:
            services.append({
                "id": item.service_id, "scope": item.scope,
                "resource": item.resource, "resources": list(item.resources),
                "phase": item.phase, "work_unit": item.work_unit,
                "boundary": item.boundary, "fixed_us": item.fixed_us,
                "effective_rate": item.effective_rate,
                "rate_scope": item.rate_scope, "active_units": item.active_units,
                "conditions": deepcopy(item.conditions),
                "formula": item.formula, "model_id": item.model_id,
            })
        return {
            "schema": USER_PROFILE_SCHEMA, "id": self.id,
            "version": self.version, "digest": self.digest,
            "source": deepcopy(dict(self.source)),
            "profile_schema": self.profile.get("schema"),
            "service_ids": [item.service_id for item in self.services],
            "formulas": {item.service_id: item.formula for item in self.services},
            "services": services,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.describe()
        payload["profile"] = deepcopy(dict(self.profile))
        return payload

    def export(self, *, path: str | Path | None = None):
        """Export the normalized T06 profile and formulas through T06."""
        return export_resource_profile(self.profile, path=path)


def load_user_profile(source: Any, *, profile_id: str | None = None,
                      nominal_rates: Mapping[str, Any] | None = None,
                      hardware: str | None = None) -> UserProfile:
    """Load one resource/phase profile; a primitive full-op profile is rejected."""
    if isinstance(source, UserProfile):
        return source
    schema = _source_schema(source)
    if schema == _OLD_PRIMITIVE_SCHEMA:
        raise ExtensionError(
            "user_profile_primitive_duration_not_supported:"
            "use_resource_service_profile")
    try:
        profile = load_resource_profile(
            source, nominal_rates=nominal_rates, hardware=hardware)
    except ValueError as exc:
        raise ExtensionError("user_profile_load_failed:" + str(exc)) from exc
    services = tuple(normalize_resource_service(item)
                     for item in profile.get("services", ()))
    digest = sha256(canonical_json(profile, "user_profile_digest").encode(
        "utf-8")).hexdigest()
    source_data = profile.get("source") if isinstance(
        profile.get("source"), Mapping) else {}
    identifier = profile_id or profile.get("id") or ("user:" + digest[:12])
    version = str(profile.get("version") or "1")
    return UserProfile(
        id=str(identifier), version=version, profile=deepcopy(dict(profile)),
        services=services, source=deepcopy(dict(source_data)), digest=digest,
    )


__all__ = [
    "PROFILE_SCHEMA", "USER_PROFILE_SCHEMA", "UserProfile", "load_user_profile",
]
