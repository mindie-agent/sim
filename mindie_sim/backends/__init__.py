"""[Role] Translate native hardware metadata into neutral resource declarations.
[Invariants] Importing this package performs no simulation or device work.
[Timeouts & Locks] No waits or locks.
[Side-Effect Policy] No I/O.
"""
from functools import lru_cache

from mindie_sim.schema.common import ModelingError
from .ascend import materialize as ascend, normalize_ascend_machine
from .ascend import profiles as ascend_profiles

ADAPTERS = {"ascend": normalize_ascend_machine}
_PRESET_CACHE_MAXSIZE = 64


@lru_cache(maxsize=_PRESET_CACHE_MAXSIZE)
def _registered_preset(sku: str, arch_family: str) -> str | None:
    try:
        profile = ascend_profiles().get(sku)
    except TimeoutError:
        raise
    except (OSError, ValueError, KeyError):
        return None
    if not isinstance(profile, dict) or profile.get("architecture") != arch_family:
        return None
    declared = (profile.get("device") or {}).get("compute_arch") or {}
    preset = declared.get("preset") if isinstance(declared, dict) else None
    return str(preset) if preset else None


def registered_capability(sku: str, arch_family: str):
    """Return the architecture capability declared by a registered SKU.

    This is the neutral backend boundary for pre-capability hardware
    documents: a case may already materialize a registered SKU's device rates
    before ``device.compute_arch`` existed. The adapter returns only the
    capability preset; the caller keeps the case's own rates and peaks.
    Unknown SKUs and unmatched architecture families return ``None`` so the
    caller can raise an explicit capability error instead of inventing one.
    """
    if not sku or not arch_family:
        return None
    preset = _registered_preset(sku, arch_family)
    return {"preset": preset} if preset else None


def materialize(raw):
    backend = raw.get("backend")
    if not backend:
        return raw
    try:
        adapter = ADAPTERS[backend]
    except KeyError as exc:
        raise ModelingError(f"hardware_backend_unknown:{backend}:declare_neutral_resources") from exc
    return adapter(raw)
