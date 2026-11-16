"""Public SDK: independent entries, one-shot extension loading and user profiles.

[Role]
  Publish the stable Python boundary for existing simulation levels plus the
  thin provider/profile loader.  Business modules remain authoritative.

[Invariants]
  * Importing this package performs no discovery, I/O, device work or registry
    mutation.  Sources load only through an explicit ``load_extensions`` call.
  * One context is immutable for a request; cache users include its fingerprint.
  * No second engine or IR is introduced by this facade.

[Timeouts & Locks]
  Import-time declarations only.  Bounded reads occur in explicit load calls.

[Side-Effect Policy]
  Pure declarations at import; explicit loading imports selected sources.
"""
from __future__ import annotations

from .api import (
    simulate_graph, simulate_model, simulate_native_call,
    simulate_operator, simulate_service,
)
from .contracts import (
    CAPABILITIES, CORE_PROVIDER, ENTRY_POINT_GROUP, INTERFACE_VERSION,
    WORKSPACE_MODULE, Extension, ExtensionContext, Provider, load_extensions,
)
from .environment import ExecutionEnvironment
from .errors import GAP_SCHEMA, CapabilityGap, ExtensionError
from .profiles import (
    PROFILE_SCHEMA, USER_PROFILE_SCHEMA, UserProfile, load_user_profile,
)

__all__ = [
    "CAPABILITIES", "CORE_PROVIDER", "ENTRY_POINT_GROUP", "INTERFACE_VERSION",
    "WORKSPACE_MODULE", "PROFILE_SCHEMA", "USER_PROFILE_SCHEMA",
    "GAP_SCHEMA",
    "ExecutionEnvironment",
    "Extension", "ExtensionContext", "Provider", "UserProfile",
    "ExtensionError", "CapabilityGap", "load_extensions", "load_user_profile",
    "simulate_operator", "simulate_native_call",
    "simulate_graph", "simulate_model", "simulate_service",
]
