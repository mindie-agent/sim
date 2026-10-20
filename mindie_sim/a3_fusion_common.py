"""Finite fusion stage/buffer witness construction.

[Role]
  Build the explicit task and buffer lifetime plan used by fusion rules.  A
  plan owns ordered stage dictionaries (compatible with ``add_costed_ops``
  ``fusion_stages``) plus the resident-tensor allocation/release records that
  make every deleted intermediate a real on-chip value.

[Invariants]
  * Every resident buffer has a creating stage and a final-consumer release
    stage.  A later task that overwrites the same physical slot must depend on
    that release, directly or through another stage dependency.
  * Stage resource fields are conserved into the fused OpSpec totals; the plan
    never invents free bytes or drops a tail.  Extra recurrence traffic/work is
    charged explicitly.
  * Admission uses the declared unified scratch or per-pool budget.  A unified
    byte count is not evidence of a physical L1/UB/L0 pool, and the scope text
    travels with the plan.
  * Stage extents and dtype/shape records are data, not prose.  Peak live bytes
    can be reconstructed from the buffer records independently of task times.

[Timeouts & Locks]
  Pure in-memory construction bounded by emitted stage and buffer count.

[Side-Effect Policy]
  No I/O, device, global mutation, or scheduler state.  Callers own admission.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def ceil_div(value: int, divisor: int) -> int:
    return (max(0, int(value)) + max(1, int(divisor)) - 1) // max(1, int(divisor))


def tile_sizes(total: int, preferred: Sequence[int]) -> list[int]:
    """Distinct positive tile sizes not exceeding ``total``, descending."""
    total = max(1, int(total))
    values = {total}
    for item in preferred:
        size = min(total, max(1, int(item)))
        values.add(size)
    for size in list(values):
        if size > 1:
            values.add(max(1, size // 2))
    return sorted({value for value in values if 1 <= value <= total}, reverse=True)


@dataclass(frozen=True)
class MemoryBudget:
    """Declared local-memory budget, unified or per physical pool."""

    unified_bytes: int | None = None
    pools: Mapping[str, int] = field(default_factory=dict)
    scope: str = "unified_declared_scratch"

    @classmethod
    def from_declared(
        cls,
        scratch_capacity_bytes: int | None,
        pool_capacities: Mapping[str, int] | None,
    ) -> "MemoryBudget":
        pools = {
            str(key).strip().lower(): max(0, int(value))
            for key, value in dict(pool_capacities or {}).items()
            if value is not None
        }
        return cls(
            unified_bytes=None if scratch_capacity_bytes is None
            else max(0, int(scratch_capacity_bytes)),
            pools=pools,
            scope=("declared_pools" if pools else
                   "unified_capacity_only_not_physical_pool_proof"),
        )

    def details(self, required_by_pool: Mapping[str, float]) -> dict[str, Any]:
        required = {
            str(key): float(max(0.0, value))
            for key, value in dict(required_by_pool).items()
            if float(max(0.0, value)) > 0
        }
        total = float(sum(required.values()))
        per_pool: dict[str, Any] = {}
        fits = True
        pool_covered = True
        for name, amount in sorted(required.items()):
            capacity = self.pools.get(name)
            if capacity is None:
                pool_covered = False
                per_pool[name] = {
                    "required_bytes": amount,
                    "capacity_bytes": None,
                    "status": "not_declared",
                }
                continue
            ok = amount <= float(capacity) + 1e-9
            fits = fits and ok
            per_pool[name] = {
                "required_bytes": amount,
                "capacity_bytes": int(capacity),
                "status": "fits" if ok else "exceeds",
            }
        if not pool_covered and self.unified_bytes is not None:
            ok = total <= float(self.unified_bytes) + 1e-9
            fits = fits and ok
        elif pool_covered:
            # Per-pool capacities are physically disjoint; total is advisory.
            pass
        elif self.unified_bytes is None and not self.pools:
            fits = True
        scope = self.scope
        if self.pools and not pool_covered:
            scope = "mixed_declared_pools_and_unified_fallback"
        return {
            "scope": scope,
            "required_by_pool": required,
            "required_total_bytes": total,
            "declared_unified_bytes": self.unified_bytes,
            "declared_pools": dict(self.pools),
            "pool_checks": per_pool,
            "fits": bool(fits),
            "pool_coverage_complete": bool(pool_covered),
            "note": (
                "unified/L2 capacity is not physical L1/UB/L0 proof; "
                "per-pool capacities must be declared by the caller"
            ),
        }

    def fits(self, required_by_pool: Mapping[str, float]) -> bool:
        return bool(self.details(required_by_pool)["fits"])

    def declares_positive(self, names: Sequence[str]) -> bool:
        """True only when every named physical pool is declared and positive."""
        return all(
            name in self.pools and float(self.pools[name]) > 0.0
            for name in names
        )


@dataclass
class PipelinePlan:
    """Ordered finite stages plus resident-buffer lifetimes."""

    stages: list[dict[str, Any]] = field(default_factory=list)
    buffers: list[dict[str, Any]] = field(default_factory=list)
    required_by_pool: dict[str, float] = field(default_factory=dict)
    kind: str = ""
    notes: dict[str, Any] = field(default_factory=dict)
    _buffer_index: dict[str, dict[str, Any]] = field(
        default_factory=dict, repr=False, compare=False,
    )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def add_buffer(
        self,
        buffer_id: str,
        *,
        size_bytes: float,
        pool: str,
        owner: str,
        shape: Sequence[int],
        dtype: str,
        slot: int,
        category: str,
    ) -> str:
        if buffer_id in self._buffer_index:
            raise ValueError(f"duplicate_fusion_buffer:{buffer_id}")
        payload = {
            "id": buffer_id,
            "bytes": float(max(0.0, size_bytes)),
            "pool": str(pool),
            "owner": str(owner),
            "shape": tuple(int(value) for value in shape),
            "dtype": str(dtype),
            "slot": int(slot),
            "category": str(category),
            "created_stage": None,
            "consumers": [],
            "release_stage": None,
        }
        self.buffers.append(payload)
        self._buffer_index[buffer_id] = payload
        return buffer_id

    def add_stage(
        self,
        name: str,
        family: str,
        *,
        deps: Sequence[int] = (),
        launch: bool = False,
        reads: Sequence[str] = (),
        writes: Sequence[str] = (),
        creates: Sequence[str] = (),
        releases: Sequence[str] = (),
        kind: str = "",
        extents: Mapping[str, Any] | None = None,
        tile_shape: Sequence[int] = (),
        resources: Mapping[str, float] | None = None,
        **work: Any,
    ) -> int:
        index = len(self.stages)
        payload: dict[str, Any] = {
            "stage_index": index,
            "name": str(name),
            "family": str(family),
            "deps": tuple(int(item) for item in deps),
            "launch": bool(launch),
            "kind": str(kind or family),
            "extents": dict(extents or {}),
            "tile_shape": tuple(int(value) for value in tile_shape),
        }
        for key in ("flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes"):
            payload[key] = float(max(0.0, float(work.get(key, 0.0) or 0.0)))
        # Standard work travels in flops/vector_ops/hbm_bytes/mte2_bytes/
        # mte3_bytes.  ``resources`` carries only extra physical demands such
        # as mte1, so a bridge that sums both cannot double-count the standard
        # fields.
        payload["resources"] = {
            key: float(max(0.0, value))
            for key, value in dict(resources or {}).items()
            if float(max(0.0, value)) > 0
        }
        actions: dict[str, list[str]] = {}
        for action, ids in (
            ("create", creates), ("read", reads), ("write", writes),
            ("release", releases),
        ):
            clean = [str(item) for item in ids]
            if clean:
                actions[action] = clean
        payload["buffers"] = actions
        self.stages.append(payload)
        for buffer_id in creates:
            buffer = self._buffer_index[buffer_id]
            if buffer["created_stage"] is not None:
                raise ValueError(f"buffer_created_twice:{buffer_id}")
            buffer["created_stage"] = index
        touched = {
            buffer_id
            for ids in (reads, writes, releases)
            for buffer_id in ids
        }
        for buffer_id in touched:
            buffer = self._buffer_index[buffer_id]
            buffer["consumers"].append(index)
            if buffer_id in releases:
                buffer["release_stage"] = index
        return index

    def retarget_release(self, buffer_id: str, stage_index: int) -> None:
        """Move a buffer's release to its actual final-consumer stage."""
        index = int(stage_index)
        if index < 0 or index >= len(self.stages):
            raise ValueError(f"buffer_release_stage_invalid:{buffer_id}:{index}")
        buffer = self._buffer_index.get(buffer_id)
        if buffer is None:
            raise ValueError(f"buffer_unknown:{buffer_id}")
        previous = buffer["release_stage"]
        if previous is not None:
            previous = int(previous)
            old_stage = self.stages[previous]
            old_stage["buffers"]["release"] = [
                item for item in old_stage["buffers"].get("release", [])
                if item != buffer_id
            ]
            if not any(
                buffer_id in old_stage["buffers"].get(action, ())
                for action in ("create", "read", "write")
            ):
                buffer["consumers"] = [
                    item for item in buffer["consumers"] if item != previous
                ]
        self.stages[index]["buffers"].setdefault("release", []).append(buffer_id)
        buffer["release_stage"] = index
        if index not in buffer["consumers"]:
            buffer["consumers"].append(index)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------
    def finish(self, *, required_by_pool: Mapping[str, float] | None = None) -> "PipelinePlan":
        by_id = self._buffer_index
        for stage in self.stages:
            for action in ("create", "read", "write", "release"):
                ids = stage["buffers"].get(action, [])
                stage["buffers"][action] = [
                    {
                        "id": buffer_id,
                        "bytes": by_id[buffer_id]["bytes"],
                        "pool": by_id[buffer_id]["pool"],
                        "slot": by_id[buffer_id]["slot"],
                        "owner": by_id[buffer_id]["owner"],
                        "dtype": by_id[buffer_id]["dtype"],
                        "category": by_id[buffer_id]["category"],
                    }
                    for buffer_id in ids
                ]
        for buffer in self.buffers:
            created = buffer["created_stage"]
            release = buffer["release_stage"]
            if created is None:
                raise ValueError(f"buffer_without_creator:{buffer['id']}")
            if release is None:
                raise ValueError(f"buffer_without_release:{buffer['id']}")
            if int(release) < max(buffer["consumers"]):
                raise ValueError(f"buffer_release_before_last_consumer:{buffer['id']}")
            buffer["first_stage"] = int(created)
            buffer["last_consumer_stage"] = max(buffer["consumers"])
            buffer["lifetime"] = [int(created), int(release)]
        required = {
            str(key): float(max(0.0, value))
            for key, value in dict(required_by_pool or self.required_by_pool).items()
            if float(max(0.0, value)) > 0
        }
        self.required_by_pool = required
        self.notes["peak_live_bytes"] = float(sum(required.values()))
        return self

    def totals(self) -> dict[str, float]:
        return {
            key: float(sum(float(stage.get(key, 0.0) or 0.0) for stage in self.stages))
            for key in ("flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes")
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "fusion_stages": tuple(dict(stage) for stage in self.stages),
            "buffer_plan": tuple(dict(buffer) for buffer in self.buffers),
            "peak_live_bytes": int(round(float(sum(self.required_by_pool.values())))),
            "required_by_pool": {
                key: int(round(value)) for key, value in sorted(self.required_by_pool.items())
            },
            "pipeline_kind": self.kind,
            "pipeline_notes": dict(self.notes),
            "stage_count": len(self.stages),
            "buffer_count": len(self.buffers),
        }


def stage_has_work(stage: Mapping[str, Any]) -> bool:
    return any(
        float(stage.get(key, 0.0) or 0.0) > 0.0
        for key in ("flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes")
    ) or any(float(value) > 0.0 for value in dict(stage.get("resources", {})).values())


def assign_stage_cores(
    plan: PipelinePlan, *, core_count: int | None,
    per_core_rates: Mapping[str, float] | None = None,
    core_base: int | None = None,
) -> dict[str, Any] | None:
    """Bind one whole fused pipeline to one fixed worker core.

    Producer and consumer of the same local L1/UB/L0 buffer must stay on the
    same physical worker; therefore every stage in this plan uses one fixed
    core assignment.  Independent fused pipelines may be given different
    ``core_base`` values by the caller.  A stage never claims a whole die or a
    multi-core partition.
    """
    if core_count is None:
        return None
    cores = max(1, int(core_count))
    base = 0 if core_base is None else int(core_base)
    worker_cores: dict[int, int] = {}
    rates = {
        str(key): float(value) for key, value in dict(per_core_rates or {}).items()
        if float(value) > 0.0
    }
    assigned = 0
    for stage in plan.stages:
        worker = int(stage.get("worker_id", 0) or 0)
        worker_core = (base + worker) % cores
        worker_cores[worker] = worker_core
        aic = 1 if float(stage.get("flops", 0.0) or 0.0) > 0.0 else 0
        aiv = 1 if float(stage.get("vector_ops", 0.0) or 0.0) > 0.0 else 0
        width = {
            "aic_cores": aic,
            "aiv_cores": aiv,
            "scope": "per_die",
            "partition": "single_core_tile",
        }
        stage["execution_width"] = width
        if not stage_has_work(stage):
            stage["core_assignment"] = {
                "core_ids": (), "count": 0, "partition": "zero_work_join",
                "per_core_rates": rates, "execution_width": width,
            }
            continue
        stage["core_assignment"] = {
            "core_ids": (worker_core,),
            "count": 1,
            "partition": "single_core_tile",
            "per_core_rates": rates,
            "scope": "whole_worker_pipeline_fixed_core",
            "worker_id": worker,
            "execution_width": width,
        }
        assigned += 1
    contract = {
        "declared_cores": cores,
        "worker_cores": {
            str(worker): core for worker, core in sorted(worker_cores.items())
        },
        "all_stages_same_worker": len(worker_cores) <= 1,
        "workers_with_work": len(worker_cores),
        "assigned_work_stages": assigned,
        "stage_core_count": 1,
        "per_core_rates": rates,
        "execution_width_scope": "per_die",
        "execution_width_partition": "single_core_tile",
        "aggregate_scope": (
            "one fused pipeline is bound to one worker core; independent "
            "pipelines with private buffers may use distinct core_base values"
        ),
        "multi_core_partitions": 0,
    }
    plan.notes["core_contract"] = contract
    return contract

def stage_core_service_ms(stage: Mapping[str, Any],
                          per_core_rates: Mapping[str, float]) -> float:
    """Declared scheduling oracle: one-core tile service, no die-wide divide."""
    assignment = dict(stage.get("core_assignment", {}) or {})
    if int(assignment.get("count", 0)) <= 0:
        return 0.0
    work = {
        "cube": float(stage.get("flops", 0.0) or 0.0),
        "vector": float(stage.get("vector_ops", 0.0) or 0.0),
        "mte2": float(stage.get("mte2_bytes", 0.0) or 0.0),
        "mte3": float(stage.get("mte3_bytes", 0.0) or 0.0),
        "hbm": float(stage.get("hbm_bytes", 0.0) or 0.0),
    }
    values = [
        value / max(1e-12, float(per_core_rates[key]))
        for key, value in work.items()
        if value > 0.0 and key in per_core_rates
    ]
    return float(max(values, default=0.0))


def core_group_makespan_ms(plan: PipelinePlan,
                           per_core_rates: Mapping[str, float]) -> float:
    """Conservative declared one-core service makespan for assigned stages.

    Each work stage is one core tile.  Stages assigned to distinct cores can
    overlap; stages assigned to the same core are summed.  This is a scheduling
    contract oracle, not an A3 measurement.
    """
    loads: dict[int, float] = {}
    for stage in plan.stages:
        assignment = dict(stage.get("core_assignment", {}) or {})
        ids = tuple(int(value) for value in assignment.get("core_ids", ()))
        if not ids:
            continue
        duration = stage_core_service_ms(stage, per_core_rates)
        loads[ids[0]] = loads.get(ids[0], 0.0) + duration
    return float(max(loads.values(), default=0.0))


def pressure(
    slots: Mapping[str, float], slot_count: int,
    groups: Mapping[str, float] | None = None, group_count: int = 1,
) -> dict[str, float]:
    """Conservative per-pool pressure for a ring of finite slots."""
    result = {
        str(key): float(max(0.0, value)) * max(1, int(slot_count))
        for key, value in dict(slots).items()
    }
    for key, value in dict(groups or {}).items():
        result[str(key)] = result.get(str(key), 0.0) + (
            float(max(0.0, value)) * max(1, int(group_count))
        )
    return {key: value for key, value in result.items() if value > 0}


def add_release_join(
    plan: PipelinePlan, *, name: str, family: str,
    consumers: Sequence[int], releases: Sequence[str],
    kind: str = "slot_release_join",
    extents: Mapping[str, Any] | None = None,
) -> int:
    """Join every consumer of a resident buffer before releasing its slot.

    Index order is not completion order when consumers run in parallel, so a
    slot release must be a zero-work successor of *all* fan-out consumers.
    """
    deps = tuple(dict.fromkeys(int(item) for item in consumers))
    return plan.add_stage(
        name, family, deps=deps, kind=kind, extents=dict(extents or {}),
        releases=tuple(str(item) for item in releases),
    )


def sum_fields(stages: Iterable[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, float]:
    return {
        key: float(sum(float(stage.get(key, 0.0) or 0.0) for stage in stages))
        for key in keys
    }


__all__ = [
    "MemoryBudget", "PipelinePlan", "add_release_join", "assign_stage_cores",
    "ceil_div", "core_group_makespan_ms", "pressure",
    "stage_core_service_ms", "stage_has_work", "sum_fields", "tile_sizes",
]
