"""Device kernel timing declarations, independent of host submission.

[Role] Parse declared startup and inclusive completion constraints.
[Invariants] Nonnegative milliseconds; provenance is retained, never a proof
  of universality. Empty declarations introduce no inferred device overhead.
[Timeouts & Locks] Pure finite metadata; no I/O or locks.
[Side-Effect Policy] Copies caller-owned provenance.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import math
from numbers import Real


def _declared_ms(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"invocation_{name}_requires_finite_nonnegative_number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invocation_{name}_requires_finite_nonnegative_number")
    return value


@dataclass(frozen=True)
class InvocationTiming:
    startup_ms: float = 0.0
    minimum_ms: float = 0.0
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "startup_ms", _declared_ms("startup_ms", self.startup_ms))
        object.__setattr__(self, "minimum_ms", _declared_ms("minimum_ms", self.minimum_ms))
        if not isinstance(self.provenance, Mapping):
            raise TypeError("invocation_provenance_requires_mapping")
        object.__setattr__(self, "provenance", deepcopy(dict(self.provenance)))

    @classmethod
    def from_mapping(cls, data: Mapping) -> "InvocationTiming":
        if not isinstance(data, Mapping):
            raise TypeError("invocation_timing_requires_mapping")
        unknown = data.keys() - {"startup_ms", "minimum_ms", "provenance"}
        if unknown:
            raise ValueError("invocation_timing_unknown_fields:" + ",".join(sorted(unknown)))
        return cls(**data)

    def to_dict(self) -> dict:
        return {
            "startup_ms": self.startup_ms,
            "minimum_ms": self.minimum_ms,
            "provenance": deepcopy(dict(self.provenance)),
        }


