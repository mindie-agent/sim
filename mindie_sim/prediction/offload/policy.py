"""Policy validation and transfer resources for hierarchical offload.

[Role] Validate case policy and submit shared FIFO transfer work.
[Invariants] The scheduler's Memory owns placement when it is available.
[Timeouts & Locks] In-memory work and event submission. No I/O, waits or locks.
[Side-Effect Policy] Caches execution Memory and submits transfer events.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from mindie_sim.prediction.capacity import Memory
from mindie_sim.schema.case import Case as CaseSpec


_DEFAULT_DDR_BANDWIDTH_GBPS = 200.0
_DEFAULT_DDR_BANDWIDTH_UTILIZATION = 0.70
_DEFAULT_DDR_LATENCY_US = 5.0
_DEFAULT_SSD_BANDWIDTH_GBPS = 14.0
_DEFAULT_SSD_BANDWIDTH_UTILIZATION = 0.70
_DEFAULT_SSD_LATENCY_US = 80.0
_PREFETCH_VERSIONS = ("v1", "v2", "v3", "v4")


@dataclass(frozen=True)
class OffloadPolicy:
    """Validated host-DDR/SSD offload and prefetch policy."""

    expert_offload_enabled: bool = False
    force_expert_offload: bool = False
    expert_ddr_ratio: float = 0.0
    ddr_bw_gbps: float = _DEFAULT_DDR_BANDWIDTH_GBPS
    ddr_bw_util: float = _DEFAULT_DDR_BANDWIDTH_UTILIZATION
    ddr_latency_us: float = _DEFAULT_DDR_LATENCY_US
    ssd_bw_gbps: float = _DEFAULT_SSD_BANDWIDTH_GBPS
    ssd_bw_util: float = _DEFAULT_SSD_BANDWIDTH_UTILIZATION
    ssd_latency_us: float = _DEFAULT_SSD_LATENCY_US
    kvc_ddr_ratio: float = 0.0
    kvc_ssd_ratio: float = 0.0
    kvc_prefetch_version: str = "v3"
    mtp_ffn_hit_rate: float = 0.0
    mtp_kvc_hit_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in ("expert_offload_enabled", "force_expert_offload"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"offload_policy_boolean_required:{name}")
        for name in ("expert_ddr_ratio", "kvc_ddr_ratio", "kvc_ssd_ratio",
                     "mtp_ffn_hit_rate", "mtp_kvc_hit_rate"):
            _ratio(name, getattr(self, name))
        for name in ("ddr_bw_util", "ssd_bw_util"):
            _utilization(name, getattr(self, name))
        for name in ("ddr_bw_gbps", "ssd_bw_gbps"):
            _positive(name, getattr(self, name))
        for name in ("ddr_latency_us", "ssd_latency_us"):
            _nonnegative(name, getattr(self, name))
        if self.kvc_prefetch_version not in _PREFETCH_VERSIONS:
            raise ValueError(f"offload_policy_invalid_prefetch_version:{self.kvc_prefetch_version}")
        if self.kvc_ddr_ratio + self.kvc_ssd_ratio > 1.0 + 1e-12:
            raise ValueError("offload_policy_kvc_ratios_exceed_unity")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None = None) -> "OffloadPolicy":
        """Build a policy with strict ratio, bandwidth, and enum validation."""
        if mapping is None:
            return cls()
        if isinstance(mapping, cls):
            return mapping
        if not isinstance(mapping, Mapping):
            raise ValueError("offload_policy_mapping_required")
        unknown = set(mapping) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("offload_policy_unknown_fields:" + ",".join(sorted(unknown)))
        return cls(**dict(mapping))


def _ratio(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"offload_policy_invalid_ratio:{name}:{value}")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"offload_policy_invalid_ratio:{name}:{value}")
    return result


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"offload_policy_must_be_positive:{name}:{value}")
    return float(value)


def _utilization(name: str, value: Any) -> float:
    result = _positive(name, value)
    if result > 1.0:
        raise ValueError(f"offload_policy_invalid_utilization:{name}:{value}")
    return result


def _nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or float(value) < 0.0:
        raise ValueError(f"offload_policy_must_be_nonnegative:{name}:{value}")
    return float(value)


def _resolve_policy(policy: OffloadPolicy | Mapping[str, Any] | None) -> OffloadPolicy:
    if policy is None:
        return OffloadPolicy()
    if isinstance(policy, OffloadPolicy):
        return policy
    return OffloadPolicy.from_mapping(policy)


def resolve_case_offload_policy(case: CaseSpec, memory_policy: Any = None) -> OffloadPolicy | None:
    """Resolve active offload policy from case metadata and memory policy."""
    raw_meta = dict(case.runtime.metadata).get("offload", {})
    if raw_meta is None:
        raw_meta = {}
    if not isinstance(raw_meta, Mapping):
        raise ValueError("memory_offload_metadata_mapping_required")
    merged: dict[str, Any] = dict(raw_meta)
    mem_cfg = dict(case.runtime.memory or {})
    for key in ("expert_ddr_ratio", "kvc_ddr_ratio", "kvc_ssd_ratio"):
        if mem_cfg.get(key, 0.0):
            merged[key] = mem_cfg[key]
        if memory_policy is not None and getattr(memory_policy, key, 0.0):
            merged[key] = getattr(memory_policy, key)
    if not merged:
        return None
    policy = OffloadPolicy.from_mapping(merged)
    if (policy.expert_ddr_ratio <= 0.0 and policy.kvc_ddr_ratio <= 0.0
            and policy.kvc_ssd_ratio <= 0.0 and not policy.force_expert_offload):
        return None
    return policy


def offload_resources(policy: OffloadPolicy | None, namespace: str, dp: int, pp: int) -> list[Any]:
    """Build host DDR and SSD transfer resources for Execution when offload is active."""
    if policy is None:
        return []
    from mindie_sim.prediction.engine import Resource
    ddr_rate = policy.ddr_bw_gbps * policy.ddr_bw_util * 1e6
    ssd_rate = policy.ssd_bw_gbps * policy.ssd_bw_util * 1e6
    resources = []
    for replica in range(max(1, dp)):
        for stage in range(max(1, pp)):
            resources.append(Resource(f"{namespace}offload_ddr@r{replica}:s{stage}",
                                      ddr_rate, "bytes", "fifo"))
            resources.append(Resource(f"{namespace}offload_ssd@r{replica}:s{stage}",
                                      ssd_rate, "bytes", "fifo"))
    return resources


def _submit_offload_transfer(
    engine: Any, name: str, work: Mapping[str, float], *,
    after: tuple[Any, ...], latency_us: float, owner: str,
) -> Any:
    startup_ms = float(latency_us) / 1000.0
    if startup_ms > 0.0:
        start = engine.submit(f"{name}:startup", after=after, latency=startup_ms, owner=owner)
        return engine.submit(name, work, after=(start,), owner=owner)
    return engine.submit(name, work, after=after, owner=owner)


def _offload_memory_for(execution: Any) -> Memory:
    """The one Memory shared with the service Scheduler when available."""
    memory = getattr(execution, "memory", None)
    if memory is None:
        memory = getattr(execution, "_offload_memory", None)
    if memory is None:
        memory = Memory(execution.case)
    execution._offload_memory = memory
    return memory
