"""Pluggable runtime-agnostic inference stack registry and built-in adapters.

[Role] Expose the unified ``RuntimeStackRegistry`` API (``register_stack``,
  ``unregister_stack``, ``get_stack``, ``list_stacks``,
  ``resolve_stack_for_scenario``, ``load_manifest_stack``) alongside the
  built-in ``ideal-target``, ``raw-eager``, and ``diffusion-native`` adapters.
[Invariants] Zero hard static dependencies on optional plugins such as
  ``vllm-ascend`` (discovered lazily via ``importlib`` in ``registry.py``).
  Never imports ``export/`` or ``compose/``, with zero SCC cycles.
[Timeouts & Locks] Pure re-export module; no I/O or locks on import.
[Side-Effect Policy] Zero side effects on import; discovery happens lazily on
  first registry access.
"""
from __future__ import annotations

from .diffusion_native import DiffusionNativeStackAdapter, DiffusionWorkloadConfig
from .ideal_target import IdealTargetStackAdapter
from .raw_eager import RawEagerStackAdapter
from .registry import (
    DEFAULT_DOMAIN_WORK_UNITS,
    MAX_MANIFEST_BYTES,
    STACK_ENTRY_POINT_GROUP,
    STACK_MANIFEST_SCHEMA,
    ManifestStackAdapter,
    RuntimeStackRegistry,
    default_registry,
    get_stack,
    list_stacks,
    load_manifest_stack,
    normalize_hardware_backend,
    register_stack,
    reset_default_registry,
    resolve_stack_for_scenario,
    unregister_stack,
)

__all__ = [
    "DEFAULT_DOMAIN_WORK_UNITS",
    "MAX_MANIFEST_BYTES",
    "STACK_ENTRY_POINT_GROUP",
    "STACK_MANIFEST_SCHEMA",
    "DiffusionNativeStackAdapter",
    "DiffusionWorkloadConfig",
    "IdealTargetStackAdapter",
    "ManifestStackAdapter",
    "RawEagerStackAdapter",
    "RuntimeStackRegistry",
    "default_registry",
    "get_stack",
    "list_stacks",
    "load_manifest_stack",
    "normalize_hardware_backend",
    "register_stack",
    "reset_default_registry",
    "resolve_stack_for_scenario",
    "unregister_stack",
]
