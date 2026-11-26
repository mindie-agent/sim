"""Public extension declarations and one-shot discovery.

[Role]
  Let an installed wheel or a local workspace directory declare providers that
  reuse the existing operator, native-call, graph, model, service, profile and
  comparison modules.  A request receives one immutable context; no process
  global active registry is created.

[Invariants]
  * A workspace source is one ``mindie_ext.py`` file (or that file's path).
  * An installed source is a ``mindie_sim.extensions`` entry point selected by
    distribution name, or an explicitly supplied entry point.
  * Extension ID, provider key, default-provider and profile-ID collisions are
    errors.  Resolution never depends on load order: one provider is
    unambiguous, an explicit default wins, and several unnamed providers are
    an ambiguity error that requires a name (or ``provider="core"``).
  * The fingerprint covers declaration versions, source code digests and user
    profile content so a cache key can separate incompatible environments.
  * A workspace's sibling modules are scoped to that import: top-level sibling
    imports remain valid for the loaded provider functions, while a second
    workspace cannot inherit the first workspace's ``sys.modules`` entries.
  * Loading executes ordinary trusted local code.  It does not sandbox, sign,
    hash-admit or silently fall back to a different provider.

[Timeouts & Locks]
  Bounded local module/metadata reads under 50 MiB.  No network, device context,
  subprocess, sleep or lock is used.

[Side-Effect Policy]
  Imports the explicit sources and may create Python bytecode caches beside a
  workspace module, as ordinary imports do.  No persistent registry is written.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from importlib import metadata
from importlib.abc import Loader, MetaPathFinder
from importlib.util import module_from_spec, spec_from_loader
import os
from pathlib import Path
import sys
from threading import RLock
from tokenize import open as open_source
from types import MappingProxyType
from typing import Any

from .errors import ExtensionError
from .values import (
    digest, digest_bytes, frozen_mapping, json_value, name as value_name, text,
)

INTERFACE_VERSION = "mindie-sim-extension/1"
ENTRY_POINT_GROUP = "mindie_sim.extensions"
WORKSPACE_MODULE = "mindie_ext.py"
CORE_PROVIDER = "core"
CAPABILITIES = frozenset({
    "operator", "native", "graph", "model", "service",
    "profile", "compare", "hardware", "profile_loader",
})
RESERVED_PROVIDER_NAMES = frozenset({CORE_PROVIDER})
_MAX_SOURCE_BYTES = 50 * 1024 ** 2
_WORKSPACE_IMPORT_LOCK = RLock()


def _entry_point(value: Any) -> metadata.EntryPoint:
    if isinstance(value, metadata.EntryPoint):
        return value
    rendered = str(value)
    if not rendered or ":" not in rendered:
        raise ExtensionError("extension_entry_point_requires_module_colon_factory")
    module, factory = rendered.split(":", 1)
    if not module or not factory:
        raise ExtensionError("extension_entry_point_requires_module_colon_factory")
    return metadata.EntryPoint(
        name=factory.rsplit(".", 1)[-1], value=rendered, group=ENTRY_POINT_GROUP)


def _sources(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Path, Mapping, metadata.EntryPoint)) \
            or hasattr(value, "describe"):
        return (value,)
    return tuple(value)


def _source_digest(path: str | Path | None) -> str:
    if not path:
        return ""
    try:
        candidate = Path(path)
        if not candidate.is_file() or candidate.stat().st_size > _MAX_SOURCE_BYTES:
            return ""
        return digest_bytes(candidate.read_bytes())
    except TimeoutError:
        raise
    except OSError:
        return ""


@dataclass(frozen=True)
class Provider:
    """One named implementation of an existing public capability."""

    capability: str
    name: str
    factory: Callable[..., Any]
    purpose: str = "report"
    version: str = ""
    default: bool = False
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        capability = text(self.capability, "extension_provider_capability_invalid")
        if capability not in CAPABILITIES:
            raise ExtensionError("extension_provider_capability_unknown:" + capability)
        object.__setattr__(self, "capability", capability)
        object.__setattr__(self, "name", value_name(
            self.name, "extension_provider_name_invalid"))
        if self.name in RESERVED_PROVIDER_NAMES:
            raise ExtensionError("extension_provider_name_reserved:" + self.name)
        if not callable(self.factory):
            raise ExtensionError("extension_provider_factory_not_callable:"
                                 + self.capability + ":" + self.name)
        if self.purpose not in {"report", "builder"}:
            raise ExtensionError("extension_provider_purpose_invalid:"
                                 + self.capability + ":" + self.name)
        if not isinstance(self.default, bool):
            raise ExtensionError("extension_provider_default_requires_bool")
        if not isinstance(self.version, str) or not isinstance(self.description, str):
            raise ExtensionError("extension_provider_metadata_requires_string")
        object.__setattr__(self, "metadata", frozen_mapping(
            self.metadata, "extension_provider_metadata_invalid"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability, "name": self.name,
            "purpose": self.purpose,
            "version": self.version, "default": self.default,
            "description": self.description,
            "metadata": json_value(self.metadata, "extension_provider_metadata_invalid"),
        }

    @classmethod
    def from_value(cls, value: Any, reason: str) -> Provider:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ExtensionError(reason)
        allowed = {"capability", "name", "factory", "purpose", "version",
                   "default", "description", "metadata"}
        unknown = set(value) - allowed
        if unknown:
            raise ExtensionError(reason + ":unknown_fields:" + ",".join(sorted(unknown)))
        return cls(
            capability=value.get("capability"),
            name=value.get("name"),
            factory=value.get("factory"),
            purpose=value.get("purpose", "report"),
            version=value.get("version", ""),
            default=value.get("default", False),
            description=value.get("description", ""),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True)
class Extension:
    """One extension declaration, optionally filled with discovery provenance."""

    id: str
    version: str
    providers: tuple[Provider, ...] = ()
    requires: tuple[str, ...] = ()
    interface: str = INTERFACE_VERSION
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    source_kind: str = ""
    source: str = ""
    code_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", value_name(self.id, "extension_id_invalid"))
        object.__setattr__(self, "version", text(self.version, "extension_version_invalid"))
        if self.interface != INTERFACE_VERSION:
            raise ExtensionError(
                "extension_interface_incompatible:" + self.id + ":" + str(self.interface))
        provider_values = tuple(Provider.from_value(
            item, "extension_provider_invalid:" + self.id) for item in self.providers)
        keys = [(item.capability, item.name) for item in provider_values]
        duplicate = sorted({key for key in keys if keys.count(key) > 1})
        if duplicate:
            raise ExtensionError("extension_provider_duplicate:"
                                 + self.id + ":" + repr(duplicate[0]))
        requires = tuple(text(item, "extension_require_invalid:" + self.id)
                         for item in self.requires)
        if len(set(requires)) != len(requires):
            raise ExtensionError("extension_require_duplicate:" + self.id)
        if self.id in requires:
            raise ExtensionError("extension_self_dependency:" + self.id)
        if self.source_kind not in {"", "workspace", "entry_point"}:
            raise ExtensionError("extension_source_kind_invalid:" + self.source_kind)
        if not isinstance(self.source, str):
            raise ExtensionError("extension_source_requires_string")
        code_sha256 = self.code_sha256 or ""
        if code_sha256 and (len(code_sha256) != 64
                            or any(ch not in "0123456789abcdef" for ch in code_sha256)):
            raise ExtensionError("extension_code_digest_invalid:" + self.id)
        object.__setattr__(self, "providers", tuple(sorted(
            provider_values, key=lambda item: (item.capability, item.name))))
        object.__setattr__(self, "requires", requires)
        object.__setattr__(self, "metadata", frozen_mapping(
            self.metadata, "extension_metadata_invalid:" + self.id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "version": self.version, "interface": self.interface,
            "requires": list(self.requires),
            "providers": [item.to_dict() for item in self.providers],
            "metadata": json_value(self.metadata, "extension_metadata_invalid:" + self.id),
            "source_kind": self.source_kind, "source": self.source,
            "code_sha256": self.code_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Extension:
        allowed = {"id", "version", "interface", "interface_version", "requires",
                   "providers", "metadata"}
        unknown = set(value) - allowed
        if unknown:
            raise ExtensionError("extension_declaration_unknown_fields:"
                                 + ",".join(sorted(unknown)))
        interface = value.get("interface", value.get("interface_version",
                                                      INTERFACE_VERSION))
        if "interface" in value and "interface_version" in value \
                and value["interface"] != value["interface_version"]:
            raise ExtensionError("extension_interface_declarations_conflict")
        providers = value.get("providers", ())
        if not isinstance(providers, (list, tuple)):
            raise ExtensionError("extension_providers_require_sequence")
        return cls(
            id=value.get("id"), version=value.get("version"),
            providers=tuple(providers), requires=tuple(value.get("requires", ())),
            interface=interface, metadata=value.get("metadata", {}),
        )


def _declaration(value: Any, source: str) -> Extension:
    if isinstance(value, Extension):
        return value
    if isinstance(value, Mapping):
        return Extension.from_mapping(value)
    if hasattr(value, "EXTENSION"):
        return _declaration(value.EXTENSION, source)
    if hasattr(value, "extension"):
        factory = value.extension
        try:
            extension = factory() if callable(factory) else factory
        except TypeError as exc:
            raise ExtensionError(
                "extension_factory_failed:" + source + ":" + str(exc)) from exc
        return _declaration(extension, source)
    if callable(value):
        try:
            extension = value()
        except TypeError as exc:
            raise ExtensionError(
                "extension_factory_failed:" + source + ":" + str(exc)) from exc
        return _declaration(extension, source)
    raise ExtensionError("extension_declaration_invalid:" + source)


def _workspace_module_file(module: Any, root: Path) -> bool:
    filename = getattr(module, "__file__", None)
    if not filename:
        return False
    try:
        return Path(filename).resolve().is_relative_to(root)
    except TimeoutError:
        raise
    except (OSError, ValueError):
        return False


def _workspace_snapshot(root: Path) -> dict[str, bytes]:
    """Read one bounded source snapshot used for both identity and execution."""
    snapshot: dict[str, bytes] = {}
    total = 0
    files = sorted(
        path for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and path.is_file())
    for path in files:
        data = path.read_bytes()
        total += len(data)
        if total > _MAX_SOURCE_BYTES:
            raise ExtensionError(
                "extension_workspace_source_tree_exceeds_50_mib:" + str(root))
        snapshot[path.relative_to(root).as_posix()] = data
    return snapshot


def _snapshot_digest(snapshot: Mapping[str, bytes]) -> str:
    digest_value = sha256()
    for relative in sorted(snapshot):
        data = snapshot[relative]
        encoded = relative.encode("utf-8")
        digest_value.update(len(encoded).to_bytes(4, "big"))
        digest_value.update(encoded)
        digest_value.update(len(data).to_bytes(8, "big"))
        digest_value.update(data)
    return digest_value.hexdigest()


class _SnapshotLoader(Loader):
    def __init__(self, fullname: str, origin: Path, data: bytes,
                 *, is_package: bool):
        self.fullname = fullname
        self.origin = origin
        self.data = data
        self.is_package = is_package

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        module.__file__ = str(self.origin)
        module.__loader__ = self
        if self.is_package:
            module.__package__ = self.fullname
            module.__path__ = [str(self.origin.parent)]
        else:
            module.__package__ = self.fullname.rpartition(".")[0]
        exec(compile(self.data, str(self.origin), "exec"), module.__dict__)


class _SnapshotFinder(MetaPathFinder):
    def __init__(self, root: Path, snapshot: Mapping[str, bytes]):
        self.root = root
        self.snapshot = snapshot

    def find_spec(self, fullname, path=None, target=None):
        relative = fullname.replace(".", "/")
        for candidate, is_package in ((relative + ".py", False),
                                      (relative + "/__init__.py", True)):
            data = self.snapshot.get(candidate)
            if data is None:
                continue
            origin = self.root / candidate
            spec = spec_from_loader(
                fullname,
                _SnapshotLoader(fullname, origin, data,
                                is_package=is_package),
                is_package=is_package)
            if spec is not None:
                spec.origin = str(origin)
                if is_package:
                    spec.submodule_search_locations = [str(origin.parent)]
            return spec
        return None


def _snapshot_names(snapshot: Mapping[str, bytes]) -> tuple[str, ...]:
    names = set()
    for relative in snapshot:
        parts = relative.split("/")
        module_parts = list(parts)
        if module_parts[-1] == "__init__.py":
            module_parts = module_parts[:-1]
        else:
            module_parts[-1] = module_parts[-1][:-3]
        for end in range(1, len(module_parts) + 1):
            candidate = module_parts[:end]
            if all(part.isidentifier() for part in candidate):
                names.add(".".join(candidate))
    return tuple(sorted(names))


def _import_workspace_module(path: Path) -> tuple[Any, str]:
    with _WORKSPACE_IMPORT_LOCK:
        root = path.parent.resolve()
        snapshot = _workspace_snapshot(root)
        module_name = "_mindie_ext_" + digest(
            str(path.resolve()), "extension_module_identity")[:16]
        entry = path.resolve().relative_to(root).as_posix()
        if entry not in snapshot:
            raise ExtensionError(
                "extension_workspace_entry_missing_from_snapshot:" + str(path))
        spec = spec_from_loader(
            module_name,
            _SnapshotLoader(module_name, path, snapshot[entry],
                            is_package=False),
            is_package=False)
        if spec is None:
            raise ExtensionError("extension_workspace_import_unavailable:"
                                 + str(path))
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        parent = str(root)
        original_path = list(sys.path)
        original_meta = list(sys.meta_path)
        displaced = {}
        for name in _snapshot_names(snapshot):
            current = sys.modules.get(name)
            if current is not None and not _workspace_module_file(current, root):
                displaced[name] = sys.modules.pop(name)
        for name, current in list(sys.modules.items()):
            if name != module_name and _workspace_module_file(current, root):
                sys.modules.pop(name, None)
        sys.path.insert(0, parent)
        sys.meta_path.insert(0, _SnapshotFinder(root, snapshot))
        try:
            spec.loader.exec_module(module)
        except TimeoutError:
            sys.modules.pop(module_name, None)
            raise
        except ExtensionError:
            sys.modules.pop(module_name, None)
            raise
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise ExtensionError(
                "extension_workspace_import_failed:" + str(path) + ":"
                + str(exc)) from exc
        finally:
            sys.path[:] = original_path
            sys.meta_path[:] = original_meta
            for name, current in list(sys.modules.items()):
                if name != module_name and _workspace_module_file(current, root):
                    sys.modules.pop(name, None)
            for name, current in displaced.items():
                if name not in sys.modules:
                    sys.modules[name] = current
        return module, _snapshot_digest(snapshot)


def _workspace(path: Any) -> Extension:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / WORKSPACE_MODULE
    if not candidate.is_file() or candidate.suffix != ".py":
        raise ExtensionError("extension_workspace_requires_mindie_ext_py:" + str(path))
    module, code_sha256 = _import_workspace_module(candidate)
    declaration = _declaration(module, str(candidate))
    return replace(declaration, source_kind="workspace",
                   source=str(candidate.resolve()), code_sha256=code_sha256)


def _loaded_digest(value: Any, entry_point: metadata.EntryPoint) -> str:
    module = value if hasattr(value, "__file__") else sys.modules.get(
        getattr(value, "__module__", ""))
    if module is None:
        module = sys.modules.get(getattr(entry_point, "module", ""))
    return _source_digest(getattr(module, "__file__", None))


def _installed(value: Any, *, package: str = "") -> Extension:
    entry_point = _entry_point(value)
    try:
        loaded = entry_point.load()
    except TimeoutError:
        raise
    except ExtensionError:
        raise
    except Exception as exc:
        raise ExtensionError(
            "extension_entry_point_load_failed:" + entry_point.value + ":" + str(exc)
        ) from exc
    declaration = _declaration(loaded, entry_point.value)
    distribution = getattr(entry_point, "dist", None)
    provenance = entry_point.group + ":" + entry_point.name + "=" + entry_point.value
    if distribution is not None:
        provenance += "@" + str(getattr(distribution, "name", "")) + "==" + str(
            getattr(distribution, "version", ""))
    elif package:
        provenance += "@" + package
    return replace(declaration, source_kind="entry_point", source=provenance,
                   code_sha256=_loaded_digest(loaded, entry_point))


def _package(name: str) -> tuple[Extension, ...]:
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError as exc:
        raise ExtensionError("extension_package_not_installed:" + name) from exc
    entry_points = sorted(
        (item for item in distribution.entry_points
         if item.group == ENTRY_POINT_GROUP),
        key=lambda item: (item.name, item.value),
    )
    if not entry_points:
        raise ExtensionError("extension_package_has_no_entry_points:" + name)
    return tuple(_installed(item, package=name) for item in entry_points)


def _default_conflicts(extensions: Sequence[Extension]) -> None:
    defaults: dict[str, list[str]] = {}
    for extension in extensions:
        for provider in extension.providers:
            if provider.default:
                defaults.setdefault(provider.capability, []).append(
                    extension.id + ":" + provider.name)
    conflicts = {capability: sorted(values) for capability, values
                 in defaults.items() if len(values) > 1}
    if conflicts:
        capability = sorted(conflicts)[0]
        raise ExtensionError("extension_default_provider_ambiguous:"
                             + capability + ":" + ",".join(conflicts[capability]))


def _fingerprint(extensions: Sequence[Extension],
                 profiles: Sequence[Any]) -> str:
    payload = {
        "interface": INTERFACE_VERSION,
        "extensions": [item.to_dict() for item in sorted(extensions, key=lambda x: x.id)],
        "profiles": [item.describe() for item in sorted(profiles, key=lambda x: x.id)],
    }
    return "sha256:" + digest(payload, "extension_fingerprint")


@dataclass(frozen=True)
class ExtensionContext:
    """An immutable, request-scoped set of extensions, providers and profiles."""

    extensions: tuple[Extension, ...] = ()
    providers: Mapping[tuple[str, str], Provider] = field(
        default_factory=lambda: MappingProxyType({}))
    profiles: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    fingerprint: str = ""

    def __post_init__(self) -> None:
        extensions = tuple(sorted(self.extensions, key=lambda item: item.id))
        for extension in extensions:
            if not isinstance(extension, Extension):
                raise ExtensionError("extension_context_requires_extension_records")
        object.__setattr__(self, "extensions", extensions)
        providers = dict(self.providers)
        for key, provider in providers.items():
            if (not isinstance(key, tuple) or len(key) != 2
                    or not isinstance(provider, Provider)
                    or key != (provider.capability, provider.name)):
                raise ExtensionError("extension_context_provider_key_invalid")
        object.__setattr__(self, "providers", MappingProxyType(providers))
        profiles = dict(self.profiles)
        for key, profile in profiles.items():
            if not isinstance(key, str) or not hasattr(profile, "describe"):
                raise ExtensionError("extension_context_profile_invalid")
        object.__setattr__(self, "profiles", MappingProxyType(profiles))
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint",
                               _fingerprint(extensions, tuple(profiles.values())))
        if not isinstance(self.fingerprint, str) \
                or not self.fingerprint.startswith("sha256:"):
            raise ExtensionError("extension_context_fingerprint_invalid")
        _default_conflicts(extensions)

    def __enter__(self) -> ExtensionContext:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def extension(self, extension_id: str) -> Extension:
        for extension in self.extensions:
            if extension.id == extension_id:
                return extension
        raise ExtensionError("extension_id_not_loaded:" + str(extension_id))

    def provider(self, capability: str, name: str | None = None) -> Provider:
        if capability not in CAPABILITIES:
            raise ExtensionError("extension_provider_capability_unknown:"
                                 + str(capability))
        if name is not None:
            name = value_name(name, "extension_provider_name_invalid")
            selected = self.providers.get((capability, name))
            if selected is None:
                raise ExtensionError("extension_provider_missing:"
                                     + capability + ":" + name)
            return selected
        candidates = self.providers_for(capability)
        if not candidates:
            raise ExtensionError("extension_provider_missing:" + capability)
        if len(candidates) > 1:
            raise ExtensionError("extension_provider_ambiguous:" + capability + ":"
                                 + ",".join(item.name for item in candidates))
        return candidates[0]

    def providers_for(self, capability: str) -> tuple[Provider, ...]:
        if capability not in CAPABILITIES:
            raise ExtensionError("extension_provider_capability_unknown:"
                                 + str(capability))
        return tuple(sorted(
            (item for item in self.providers.values()
             if item.capability == capability), key=lambda item: item.name))

    def default_provider(self, capability: str) -> Provider | None:
        candidates = tuple(item for item in self.providers_for(capability)
                           if item.default)
        if len(candidates) > 1:
            raise ExtensionError("extension_default_provider_ambiguous:"
                                 + capability + ":"
                                 + ",".join(item.name for item in candidates))
        return candidates[0] if candidates else None

    def select(self, capability: str) -> Provider | None:
        """Resolve the unambiguous provider for an unqualified request.

        One provider is unambiguous even without ``default=True``.  An explicit
        default wins when several providers exist.  Several providers without a
        default require the caller to name one; load order never decides.
        """
        providers = self.providers_for(capability)
        if not providers:
            return None
        defaults = tuple(item for item in providers if item.default)
        if len(defaults) > 1:
            raise ExtensionError("extension_default_provider_ambiguous:"
                                 + capability + ":"
                                 + ",".join(item.name for item in defaults))
        if defaults:
            return defaults[0]
        if len(providers) == 1:
            return providers[0]
        raise ExtensionError("extension_provider_ambiguous:" + capability + ":"
                             + ",".join(item.name for item in providers))

    def profile(self, profile_id: str) -> Any:
        profile = self.profiles.get(profile_id)
        if profile is None:
            raise ExtensionError("extension_profile_missing:" + str(profile_id))
        return profile

    def scoped_fingerprint(self, *, capabilities: Sequence[str] = (),
                           profile_ids: Sequence[str] = ()) -> str:
        """Fingerprint only the capabilities/profiles that affect a request.

        This keeps an unrelated loaded extension from invalidating a cache
        comparison while covering the selected code, its declared ``requires``
        dependency closure (including data-only extensions) and parameters.
        """
        capabilities = tuple(sorted(set(capabilities)))
        profile_ids = tuple(sorted(set(profile_ids)))
        if not capabilities and not profile_ids:
            return self.fingerprint
        for capability in capabilities:
            if capability not in CAPABILITIES:
                raise ExtensionError("extension_provider_capability_unknown:"
                                     + str(capability))
        by_id = {extension.id: extension for extension in self.extensions}
        selected_ids: set[str] = set()

        def include(extension: Extension) -> None:
            if extension.id in selected_ids:
                return
            selected_ids.add(extension.id)
            for dependency in extension.requires:
                dependency_extension = by_id.get(dependency)
                if dependency_extension is not None:
                    include(dependency_extension)

        for extension in self.extensions:
            if any(provider.capability in capabilities
                   for provider in extension.providers):
                include(extension)
        selected = []
        for extension in sorted(
                (by_id[identifier] for identifier in selected_ids),
                key=lambda item: item.id):
            record = extension.to_dict()
            # The selected capability's providers are the implementation under
            # test; a data dependency contributes its metadata/version even
            # when it declares no providers.
            if any(provider.capability in capabilities
                   for provider in extension.providers):
                record["providers"] = [
                    provider.to_dict() for provider in extension.providers
                    if provider.capability in capabilities]
            selected.append(record)
        profile_rows = [self.profile(profile_id).describe()
                        for profile_id in profile_ids]
        payload = {
            "interface": INTERFACE_VERSION, "extensions": selected,
            "profiles": profile_rows,
        }
        return "sha256:" + digest(payload, "extension_scoped_fingerprint")

    def cache_key(self, *parts: Any, capabilities: Sequence[str] = (),
                  profile_ids: Sequence[str] = ()) -> str:
        scope = self.scoped_fingerprint(
            capabilities=capabilities, profile_ids=profile_ids)
        payload = {"scope": scope,
                   "parts": json_value(parts, "extension_cache_key_invalid")}
        return "mindie-ext:" + digest(payload, "extension_cache_key")

    def describe(self, extension_id: str | None = None) -> dict[str, Any]:
        selected = self.extensions
        if extension_id is not None:
            selected = tuple(item for item in self.extensions
                             if item.id == extension_id)
            if not selected:
                raise ExtensionError("extension_id_not_loaded:" + str(extension_id))
        return {
            "schema": "mindie-sim.extensions/1",
            "interface_version": INTERFACE_VERSION,
            "fingerprint": self.fingerprint,
            "extensions": [item.to_dict() for item in selected],
            "profiles": [item.describe() for item in sorted(
                self.profiles.values(), key=lambda item: item.id)],
        }

    def __len__(self) -> int:
        return len(self.extensions)


def load_extensions(
    *,
    paths: Any = (),
    packages: Any = (),
    entry_points: Any = (),
    extensions: Any = (),
    profiles: Any = (),
    discover_installed: bool = False,
) -> ExtensionContext:
    """Discover, validate and freeze one request's extension environment.

    ``paths`` accepts workspace directories (or an explicit ``mindie_ext.py``).
    ``packages`` accepts installed distribution names with entry points in
    ``mindie_sim.extensions``.  ``entry_points`` accepts explicit entry-point
    objects or ``module:factory`` strings.  ``extensions`` accepts explicit
    in-memory ``Extension`` declarations (such as Layer-2 bindings returned by
    ``bind_kernel_implementation``).  ``profiles`` accepts a local user
    calibration profile path or mapping.  All sources are loaded once; callers
    hold the returned immutable context for the whole request.
    """
    if not isinstance(discover_installed, bool):
        raise ExtensionError("extension_discover_installed_requires_bool")
    loaded: list[Extension] = []
    for path in sorted(_sources(paths), key=lambda item: os.fspath(item)):
        loaded.append(_workspace(path))
    for package in sorted(_sources(packages), key=lambda item: str(item)):
        loaded.extend(_package(str(package)))
    explicit = _sources(entry_points)
    if explicit:
        loaded.extend(_installed(item) for item in explicit)
    if extensions:
        raw_exts = (extensions,) if isinstance(extensions, (Extension, Mapping)) else tuple(extensions)
        loaded.extend(_declaration(item, "inline") for item in raw_exts)
    if discover_installed:
        discovered = sorted(metadata.entry_points(group=ENTRY_POINT_GROUP),
                            key=lambda item: (item.name, item.value))
        loaded.extend(_installed(item) for item in discovered)

    by_id: dict[str, Extension] = {}
    for extension in loaded:
        existing = by_id.get(extension.id)
        if existing is not None:
            raise ExtensionError("extension_id_duplicate:" + extension.id + ":"
                                 + existing.source + ":" + extension.source)
        by_id[extension.id] = extension
    for extension in by_id.values():
        missing = [item for item in extension.requires if item not in by_id]
        if missing:
            raise ExtensionError("extension_dependency_missing:" + extension.id
                                 + ":" + ",".join(missing))

    providers: dict[tuple[str, str], Provider] = {}
    provider_owner: dict[tuple[str, str], str] = {}
    for extension in sorted(by_id.values(), key=lambda item: item.id):
        for provider in extension.providers:
            key = provider.capability, provider.name
            if key in providers:
                raise ExtensionError(
                    "extension_provider_duplicate:" + key[0] + ":" + key[1] + ":"
                    + provider_owner[key] + ":" + extension.id)
            providers[key] = provider
            provider_owner[key] = extension.id
    _default_conflicts(tuple(by_id.values()))

    from .profiles import load_user_profile
    selected_profiles = []
    for item in _sources(profiles):
        selected_profiles.append(item if hasattr(item, "describe")
                                 else load_user_profile(item))
    profile_map: dict[str, Any] = {}
    for profile in selected_profiles:
        if profile.id in profile_map:
            raise ExtensionError("extension_profile_id_duplicate:" + profile.id)
        profile_map[profile.id] = profile

    extensions = tuple(sorted(by_id.values(), key=lambda item: item.id))
    return ExtensionContext(
        extensions=extensions, providers=providers, profiles=profile_map,
        fingerprint=_fingerprint(extensions, tuple(profile_map.values())),
    )
