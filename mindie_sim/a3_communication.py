"""Measured communication lane calibration and analytic fallbacks.

The curve is expressed in per-call normalized work; the engine receives total
loop work and the caller's requested delay once.  Missing curves produce an
explicit machine-gap or analytic status, never a calibrated zero.

[Invariants]
  * Hardware identity is an exact join.  A query that names a machine key only
    sees observations for that same key; an observation imported without a key
    (or for another machine) is refused and counted in the result's ``identity``
    block, never pooled across SKUs.
  * Placement is a declared fabric token.  An unknown token is kept as declared
    so it cannot match another fabric's measured curve.
  * No measured curve means the analytic link/topology estimate, with the
    reason attached.

[Timeouts & Locks]
  Pure in-memory arithmetic and one bounded resource-engine run; no I/O, locks,
  device or sleep.

[Side-Effect Policy]
  No filesystem/network/global writes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .a3_resources import ResourceEngine, ResourceSpec
from .a3_types import A3InputError, MachineSpec

if TYPE_CHECKING:
    from .a3_calibration_store import SQLiteCalibrationStore


@dataclass(frozen=True)
class CommunicationExecutionSpec:
    """Typed request/observed communication identity.

    Requested configuration is deliberately separate from observed execution:
    a requested expansion mode or a generic kernel name never fills an observed
    engine.  Stage records are the common executable event interface.
    """
    operation: str = "all_reduce"
    reduction: str = "sum"
    dtype: str = "bf16"
    group_size: int = 2
    group_rank_order: tuple[int, ...] = ()
    topology_class: str = ""
    selected_routes: tuple[str, ...] = ()
    env_expansion: str = "unspecified"
    communicator_expansion: str = "unspecified"
    effective_requested_expansion: str = "unspecified"
    requested_aiv_cores: int = 0
    observed_execution: str = "unknown"
    selected_algorithm: str = ""
    algorithm_regime: str = ""
    observed_aiv_cores: int = 0
    source_of_selection: str = ""
    timing_completion: str = "completion"
    in_place: bool = False
    stages: tuple[Mapping[str, Any], ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation, "reduction": self.reduction,
            "dtype": self.dtype, "group_size": self.group_size,
            "group_rank_order": list(self.group_rank_order),
            "topology_class": self.topology_class,
            "selected_routes": list(self.selected_routes),
            "env_expansion": self.env_expansion,
            "communicator_expansion": self.communicator_expansion,
            "effective_requested_expansion": self.effective_requested_expansion,
            "requested_aiv_cores": self.requested_aiv_cores,
            "observed_execution": self.observed_execution,
            "selected_algorithm": self.selected_algorithm,
            "algorithm_regime": self.algorithm_regime,
            "observed_aiv_cores": self.observed_aiv_cores,
            "source_of_selection": self.source_of_selection,
            "timing_completion": self.timing_completion,
            "in_place": self.in_place,
            "stages": [dict(stage) for stage in self.stages],
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class CommCondition:
    placement: str
    mode: str
    size_bytes: int
    median_ms: float
    samples: int
    hardware_key: str = ""
    timing_scope: str = ""
    ranks: int = 2
    dtype: str = ""
    algorithm: str = ""
    execution_backend: str = "unspecified"
    observed_execution: str = "unknown"
    collective: str = ""
    reduction: str = ""
    group_size: int = 0
    execution_stages: tuple[Mapping[str, Any], ...] = ()
    topology_class: str = ""
    route: str = ""
    timing_completion: str = "completion"
    in_place: bool = False
    requested_aiv_cores: int = 0
    observed_aiv_cores: int = 0
    algorithm_regime: str = ""


#: Declared fabric tokens.  A token maps to itself or to the canonical spelling
#: of the same physical fabric; anything else is *not* mapped (an unknown token
#: stays unknown so it can never borrow a measured curve from another fabric).
COMM_PLACEMENT_ALIASES: dict[str, str] = {
    "sio": "sio", "d2d_intra_package": "sio", "intra_package": "sio",
    "die_pair": "sio",
    "hccssw": "hccssw", "hccs_sw": "hccssw", "hccs_switch": "hccssw",
    "hccs": "hccs", "intra_node": "hccs", "d2d_intra_node": "hccs",
    "roce": "roce", "inter_node": "roce", "d2d_inter_node": "roce",
    "ub": "ub", "ub_mesh": "ub", "ubmesh": "ub",
}


def communication_placement_token(placement: str) -> str:
    """Canonical fabric token for a declared placement.

    The previous rule collapsed everything that did not contain ``sio`` onto
    ``hccssw``, which silently mapped an A5 UB-Mesh fabric (and RoCE) onto
    A3-measured HCCS switch curves.  An unrecognised token is preserved instead.
    """
    key = str(placement or "").strip().lower()
    return COMM_PLACEMENT_ALIASES.get(key, key)


class CommunicationCalibration:
    """Affine size curves + shared-engine contention from measured training."""

    def __init__(self, conditions: Sequence[CommCondition]) -> None:
        self.conditions = list(conditions)
        self._by = {(item.placement, item.mode, item.size_bytes): item for item in self.conditions}
        self._placements = sorted({item.placement for item in self.conditions})

    @staticmethod
    def _strip_placement_suffix(key: str, placement: str) -> str:
        """Remove a redundant ``<machine>:<placement>`` suffix.

        The comm importer writes ``<device>:<placement>``.  The placement is
        already a first-class condition field and is an exact query filter, so
        the suffix is redundant identity, not a wildcard: stripping it makes the
        join *more* specific (same machine **and** same fabric), never broader.
        A suffix that is not this condition's placement is left intact and can
        therefore only match a query that names it verbatim.
        """
        text = str(key or "").strip()
        placement = str(placement or "").strip()
        if placement and text.endswith(":" + placement):
            return text[: -len(placement) - 1]
        return text

    @classmethod
    def observation_machine_key(cls, item: "CommCondition") -> str:
        """Machine identity of one observation; empty means unbound."""
        return cls._strip_placement_suffix(
            str(item.hardware_key or ""), str(item.placement or ""),
        )

    @classmethod
    def _identity_matches(cls, item: "CommCondition", hardware_key: str) -> bool:
        """Exact machine join.  An unkeyed query is an explicit fleet query."""
        key = str(hardware_key or "").strip()
        if not key:
            return True
        query = cls._strip_placement_suffix(key, str(item.placement or ""))
        identity = cls.observation_machine_key(item)
        return bool(identity) and identity == query

    def _identity_audit(
        self, placement: str, modes: Sequence[str], hardware_key: str,
    ) -> dict[str, Any] | None:
        """Conditions refused for this machine, with the reason each was refused."""
        key = str(hardware_key or "").strip()
        if not key:
            return None
        scope = [
            item for item in self.conditions
            if item.placement == placement and item.mode in modes
        ]
        refused = [item for item in scope if not self._identity_matches(item, key)]
        if not refused:
            return None
        unbound = [item for item in refused if not self.observation_machine_key(item)]
        return {
            "machine_key": key,
            "refused_observations": {
                "count": len(refused),
                "unbound": len(unbound),
                "mismatched": len(refused) - len(unbound),
                "examples": [
                    {
                        "mode": item.mode,
                        "size_bytes": int(item.size_bytes),
                        "hardware_key": str(item.hardware_key or ""),
                        "reason": (
                            "hardware_key_unbound"
                            if not self.observation_machine_key(item)
                            else "hardware_key_mismatch"
                        ),
                    }
                    for item in refused[:8]
                ],
            },
        }

    @classmethod
    def from_store(cls, store: SQLiteCalibrationStore) -> "CommunicationCalibration":
        conditions = []
        expected_family = {
            "all_reduce_only": "all_reduce",
            "p2p_only": "p2p",
            "all_reduce_plus_p2p": "comm_span",
        }
        for row in store.statistics(split="train"):
            if row["phase"] != "communication":
                continue
            features = row["features"]
            placement = str(features.get("placement", ""))
            mode = str(features.get("mode", ""))
            if expected_family.get(mode) != row["op_family"]:
                continue
            size = int(row["shape"].get("bytes", 0))
            if not placement or size <= 0 or int(row["calibratable_count"]) <= 0:
                continue
            conditions.append(CommCondition(
                placement=placement, mode=mode, size_bytes=size,
                median_ms=float(row["duration_median"]),
                samples=int(row["calibratable_count"]),
                hardware_key=str(row.get("hardware_key", "")),
                timing_scope=str(row.get("timing_scope", "")),
                ranks=int(features.get("ranks", 2)),
                dtype=str(features.get("dtype", "")),
                algorithm=str(features.get("algorithm", "")),
                execution_backend=str(
                    features.get("execution_backend")
                    or features.get("requested_expansion")
                    or features.get("effective_requested_expansion")
                    or "unspecified"
                ),
                observed_execution=str(
                    features.get("observed_execution")
                    or features.get("actual_execution")
                    or "unknown"
                ),
                collective=str(
                    features.get("collective") or features.get("operation") or ""
                ),
                reduction=str(features.get("reduction", "")),
                group_size=int(features.get("group_size", features.get("ranks", 0)) or 0),
                execution_stages=tuple(
                    dict(stage) for stage in (
                        features.get("execution_stages")
                        or features.get("execution_path")
                        or ()
                    )
                ),
                topology_class=str(features.get("topology_class", "")),
                route=str(features.get("route", "")),
                timing_completion=str(
                    features.get("timing_completion",
                                 features.get("completion_basis", "completion"))
                ),
                in_place=bool(features.get("in_place", False)),
                requested_aiv_cores=int(features.get("requested_aiv_cores", 0) or 0),
                observed_aiv_cores=int(features.get("observed_aiv_cores", 0) or 0),
                algorithm_regime=str(features.get("algorithm_regime", "")),
            ))
        return cls(conditions)

    def _curve_points(
        self, placement: str, mode: str, *, hardware_key: str = "",
        timing_scope: str = "", ranks: int | None = None,
        dtype: str = "", algorithm: str = "", execution_backend: str = "",
        observed_execution: str = "", collective: str = "",
        reduction: str = "", group_size: int | None = None,
        topology_class: str = "", route: str = "", timing_completion: str = "",
        in_place: bool | None = None, requested_aiv_cores: int | None = None,
        observed_aiv_cores: int | None = None, algorithm_regime: str = "",
        require_no_stages: bool = False,
    ) -> list[tuple[int, float]]:
        return sorted(
            (item.size_bytes, item.median_ms) for item in self.conditions
            if item.placement == placement and item.mode == mode
            # Identity is an exact join; an unbound observation is refused for
            # every keyed query instead of being pooled across SKUs.
            and self._identity_matches(item, hardware_key)
            and (not timing_scope or not item.timing_scope or item.timing_scope == timing_scope)
            and (ranks is None or int(item.ranks) == int(ranks))
            and (not dtype or not item.dtype or item.dtype == dtype)
            and (not algorithm or not item.algorithm or item.algorithm == algorithm)
            and (not execution_backend
                 or str(item.execution_backend) == execution_backend)
            and (not observed_execution
                 or str(item.observed_execution) == observed_execution)
            and (not collective or not item.collective
                 or item.collective == collective)
            and (not reduction or not item.reduction
                 or item.reduction == reduction)
            and (group_size is None or not item.group_size
                 or int(item.group_size) == int(group_size))
            and (not topology_class or not item.topology_class
                 or item.topology_class == topology_class)
            and (not route or not item.route or item.route == route)
            and (not timing_completion
                 or item.timing_completion == timing_completion)
            and (in_place is None or not item.in_place or item.in_place == in_place)
            and (requested_aiv_cores is None or not item.requested_aiv_cores
                 or item.requested_aiv_cores == requested_aiv_cores)
            and (observed_aiv_cores is None or not item.observed_aiv_cores
                 or item.observed_aiv_cores == observed_aiv_cores)
            and (not algorithm_regime or not item.algorithm_regime
                 or item.algorithm_regime == algorithm_regime)
            and (not require_no_stages or not item.execution_stages)
        )

    @staticmethod
    def _fit_points(points: Sequence[tuple[int, float]]) -> tuple[float, float]:
        if not points:
            return 0.0, 0.0
        if len(points) == 1:
            _, duration = points[0]
            return max(0.0, duration), 0.0
        n = len(points)
        mean_x = sum(point[0] for point in points) / n
        mean_y = sum(point[1] for point in points) / n
        var = sum((point[0] - mean_x) ** 2 for point in points)
        if var <= 0:
            return mean_y, 0.0
        slope = sum((point[0] - mean_x) * (point[1] - mean_y) for point in points) / var
        intercept = mean_y - slope * mean_x
        if intercept < 0:
            # Constrained nonnegative origin fit: a bandwidth slope with no
            # invented negative fixed cost.  A constant mean_y is not a curve.
            slope_origin = sum(
                point[0] * point[1] for point in points
            ) / max(1.0, sum(point[0] * point[0] for point in points))
            return 0.0, max(0.0, slope_origin)
        return max(0.0, intercept), slope

    def _curve(
        self, placement: str, mode: str, *, hardware_key: str = "",
        timing_scope: str = "", ranks: int | None = None,
        dtype: str = "", algorithm: str = "", execution_backend: str = "",
        observed_execution: str = "", collective: str = "",
        reduction: str = "", group_size: int | None = None,
        topology_class: str = "", route: str = "", timing_completion: str = "",
        in_place: bool | None = None, requested_aiv_cores: int | None = None,
        observed_aiv_cores: int | None = None, algorithm_regime: str = "",
        require_no_stages: bool = False,
    ) -> tuple[float, float]:
        return self._fit_points(self._curve_points(
            placement, mode, hardware_key=hardware_key,
            timing_scope=timing_scope, ranks=ranks, dtype=dtype,
            algorithm=algorithm, execution_backend=execution_backend,
            observed_execution=observed_execution, collective=collective,
            reduction=reduction, group_size=group_size,
            topology_class=topology_class, route=route,
            timing_completion=timing_completion, in_place=in_place,
            requested_aiv_cores=requested_aiv_cores,
            observed_aiv_cores=observed_aiv_cores,
            algorithm_regime=algorithm_regime,
            require_no_stages=require_no_stages,
        ))

    def _lane_ms(
        self, placement: str, mode: str, size_bytes: int, *,
        hardware_key: str = "", timing_scope: str = "",
        ranks: int | None = None, dtype: str = "", algorithm: str = "",
        execution_backend: str = "", observed_execution: str = "",
        collective: str = "", reduction: str = "", group_size: int | None = None,
        topology_class: str = "", route: str = "", timing_completion: str = "",
        in_place: bool | None = None, requested_aiv_cores: int | None = None,
        observed_aiv_cores: int | None = None, algorithm_regime: str = "",
    ) -> float:
        intercept, slope = self._curve(
            placement, mode, hardware_key=hardware_key, timing_scope=timing_scope,
            ranks=ranks, dtype=dtype, algorithm=algorithm,
            execution_backend=execution_backend,
            observed_execution=observed_execution, collective=collective,
            reduction=reduction, group_size=group_size,
            topology_class=topology_class, route=route,
            timing_completion=timing_completion, in_place=in_place,
            requested_aiv_cores=requested_aiv_cores,
            observed_aiv_cores=observed_aiv_cores,
            algorithm_regime=algorithm_regime,
        )
        return max(0.0, intercept + slope * size_bytes)

    def link_capacity_bytes_per_ms(
        self, *, placement: str, size_bytes: int, mode: str = "p2p_only",
    ) -> float | None:
        duration = self._lane_ms(placement, mode, size_bytes)
        if duration <= 0:
            return None
        return float(size_bytes) / duration

    def _lane_repeated_ms(
        self, placement: str, mode: str, size_bytes: int, repetitions: int,
        *, hardware_key: str = "", timing_scope: str = "",
        ranks: int | None = None, dtype: str = "", algorithm: str = "",
        execution_backend: str = "", observed_execution: str = "",
        collective: str = "", reduction: str = "", group_size: int | None = None,
        topology_class: str = "", route: str = "", timing_completion: str = "",
        in_place: bool | None = None, requested_aiv_cores: int | None = None,
        observed_aiv_cores: int | None = None, algorithm_regime: str = "",
        require_no_stages: bool = False,
    ) -> float:
        intercept, slope = self._curve(
            placement, mode, hardware_key=hardware_key, timing_scope=timing_scope,
            ranks=ranks, dtype=dtype, algorithm=algorithm,
            execution_backend=execution_backend,
            observed_execution=observed_execution, collective=collective,
            reduction=reduction, group_size=group_size,
            topology_class=topology_class, route=route,
            timing_completion=timing_completion, in_place=in_place,
            requested_aiv_cores=requested_aiv_cores,
            observed_aiv_cores=observed_aiv_cores,
            algorithm_regime=algorithm_regime,
            require_no_stages=require_no_stages,
        )
        # Imported lane durations are total-loop divided by repetitions, so the
        # fitted intercept is part of the PER-CALL normalized cost.  Repetition
        # count scales both per-call work and per-call fixed/launch terms; the
        # separate requested loop delay is added once by the caller/engine.
        return max(0.0, (intercept + slope * size_bytes) * max(1, repetitions))

    def _analytic_prediction(
        self, placement: str, size_bytes: int, requested_delay_ms: float,
        repetitions: int, mode: str, machine: MachineSpec | None,
    ) -> dict[str, Any]:
        request = {
            "op_family": "all_reduce" if mode == "all_reduce_only"
            else "p2p" if mode == "p2p_only" else "comm_span",
            "phase": "communication",
            "recipe": f"two_rank_{mode}_{placement}",
            "shape": {"bytes": int(size_bytes), "ranks": 2},
            "features": {"placement": placement, "mode": mode},
        }
        if machine is None:
            return {
                "status": "uncalibrated_needs_machine",
                "reason": "missing_curve_and_no_machine_for_analytic_prediction",
                "placement": placement, "mode": mode, "size_bytes": int(size_bytes),
                "span_ms": None, "measurement_request": request,
                "assumptions": ["no zero-duration calibration claim"],
            }
        from .a3_resources import (
            ResourceEngine, ResourceSpec, select_link_for_ranks,
        )
        link = select_link_for_ranks(machine, range(2))
        if link is None:
            raise A3InputError(f"communication_link_missing:{placement}")
        capacity = float(link.capacity_bytes_per_ms)
        latency = float(link.latency_us) / 1000.0
        reps = max(1, repetitions)
        per_call_work = size_bytes / capacity if capacity > 0 else 0.0
        p2p = latency + reps * per_call_work
        # Two-rank ring allreduce sends one payload per rank; use an explicit
        # topology assumption instead of an unexplained 1.5 multiplier.
        all_reduce = latency + reps * per_call_work
        if mode == "all_reduce_only":
            span = all_reduce
        elif mode == "p2p_only":
            span = p2p
        else:
            engine = ResourceEngine({"lane": ResourceSpec("lane", 1.0)})
            engine.add_task("all_reduce", {"lane": all_reduce - latency})
            engine.add_task("p2p", {"lane": p2p - latency},
                            release_ms=max(0.0, float(requested_delay_ms)))
            result = engine.run()
            span = latency + result["makespan_ms"]
        return {
            "status": "uncalibrated_analytic_shared_lane",
            "reason": "missing_measured_curve; analytic link/topology estimate",
            "placement": placement, "mode": mode, "size_bytes": int(size_bytes),
            "span_ms": float(span), "all_reduce_ms": float(all_reduce),
            "p2p_ms": float(p2p), "measurement_request": request,
            "assumptions": [
                "declared link capacity/latency",
                "two-rank ring allreduce wire bytes approximated by one payload",
                "per-call fixed latency repeated per call; requested delay added once",
            ],
        }

    @staticmethod
    def _scale_stages(
        stages: Sequence[Mapping[str, Any]], *, source_size: int,
        query_size: int,
    ) -> tuple[dict[str, Any], ...]:
        """Scale variable stage work, preserving explicit fixed/startup work.

        A nearest payload template is not reused unchanged.  Work fields are
        variable by default because isolated lane work scales with bytes.  A
        stage can declare ``fixed_work``/``work_per_byte`` (or set
        ``scale_with_size=False``) when its startup/control component is not
        proportional.
        """
        if not stages:
            return ()
        scale = (
            float(query_size) / float(source_size)
            if source_size > 0 and query_size > 0 else 1.0
        )
        scaled: list[dict[str, Any]] = []
        for raw in stages:
            stage = dict(raw)
            work = dict(stage.get("work", {}) or {})
            fixed = dict(stage.get("fixed_work", {}) or {})
            per_byte = dict(
                stage.get("per_byte_work", stage.get("work_per_byte", {})) or {}
            )
            preserve = bool(stage.get("scale_with_size", True)) is False
            output: dict[str, float] = {}
            for key, value in work.items():
                amount = float(value or 0.0)
                lower = str(key).lower()
                if key in fixed:
                    output[key] = float(fixed[key])
                elif key in per_byte:
                    output[key] = float(per_byte[key]) * max(0, int(query_size))
                elif preserve or lower.startswith((
                    "fixed_", "startup_", "control_", "launch",
                )):
                    output[key] = amount
                else:
                    output[key] = amount * scale
            for key, value in fixed.items():
                if key not in output:
                    output[key] = float(value or 0.0)
            for key, value in per_byte.items():
                if key not in output:
                    output[key] = float(value or 0.0) * max(0, int(query_size))
            stage["work"] = output
            stage["source_size_bytes"] = int(source_size)
            stage["size_scale"] = scale
            scaled.append(stage)
        return tuple(scaled)

    def predict(
        self, *, placement: str, size_bytes: int, requested_delay_ms: float = 0.0,
        repetitions: int = 1, mode: str = "all_reduce_plus_p2p",
        machine: MachineSpec | None = None,
        hardware_key: str = "",
        timing_scope: str = "",
        ranks: int | None = None,
        dtype: str = "",
        algorithm: str = "",
        execution_backend: str = "unspecified",
        observed_execution: str = "unknown",
        collective: str = "",
        reduction: str = "",
        group_size: int | None = None,
        topology_class: str = "",
        route: str = "",
        timing_completion: str = "",
        in_place: bool | None = None,
        requested_aiv_cores: int | None = None,
        observed_aiv_cores: int | None = None,
        algorithm_regime: str = "",
    ) -> dict[str, Any]:
        placement = communication_placement_token(placement)
        if machine is not None and not hardware_key:
            hardware_key = machine.hardware_key
        if not timing_scope:
            timing_scope = "communication_lane"
        identity = self._identity_audit(
            placement,
            ("all_reduce_only", "p2p_only", "all_reduce_plus_p2p"),
            hardware_key,
        )

        def finish(result: dict[str, Any]) -> dict[str, Any]:
            """Attach the machine-identity audit to every answer.

            Refused observations are reported, never silently pooled: an
            uncalibrated answer caused by identity keeps its own reason so a
            caller can tell "never measured" from "measured on another part".
            """
            if identity is None:
                return result
            out = dict(result)
            out["identity"] = identity
            if not str(out.get("status", "")).startswith("calibrated"):
                out["identity_reason"] = (
                    "measured_conditions_exist_only_for_another_machine_key"
                )
            return out
        c_iso = self._lane_ms(
            placement, "all_reduce_only", size_bytes,
            hardware_key=hardware_key, timing_scope=timing_scope,
            ranks=ranks, dtype=dtype, algorithm=algorithm,
            execution_backend=execution_backend,
            observed_execution=observed_execution, collective=collective,
            reduction=reduction, group_size=group_size,
            topology_class=topology_class, route=route,
            timing_completion=timing_completion, in_place=in_place,
            requested_aiv_cores=requested_aiv_cores,
            observed_aiv_cores=observed_aiv_cores,
            algorithm_regime=algorithm_regime,
        )
        v_iso = self._lane_ms(
            placement, "p2p_only", size_bytes,
            hardware_key=hardware_key, timing_scope=timing_scope,
            ranks=ranks, dtype=dtype, algorithm=algorithm,
            execution_backend=execution_backend,
            observed_execution=observed_execution, collective=collective,
            reduction=reduction, group_size=group_size,
            topology_class=topology_class, route=route,
            timing_completion=timing_completion, in_place=in_place,
            requested_aiv_cores=requested_aiv_cores,
            observed_aiv_cores=observed_aiv_cores,
            algorithm_regime=algorithm_regime,
        )
        c_repeat = self._lane_repeated_ms(
            placement, "all_reduce_only", size_bytes, repetitions,
            hardware_key=hardware_key, timing_scope=timing_scope,
            ranks=ranks, dtype=dtype, algorithm=algorithm,
            execution_backend=execution_backend,
            observed_execution=observed_execution, collective=collective,
            reduction=reduction, group_size=group_size,
            topology_class=topology_class, route=route,
            timing_completion=timing_completion, in_place=in_place,
            requested_aiv_cores=requested_aiv_cores,
            observed_aiv_cores=observed_aiv_cores,
            algorithm_regime=algorithm_regime,
        )
        v_repeat = self._lane_repeated_ms(
            placement, "p2p_only", size_bytes, repetitions,
            hardware_key=hardware_key, timing_scope=timing_scope,
            ranks=ranks, dtype=dtype, algorithm=algorithm,
            execution_backend=execution_backend,
            observed_execution=observed_execution, collective=collective,
            reduction=reduction, group_size=group_size,
            topology_class=topology_class, route=route,
            timing_completion=timing_completion, in_place=in_place,
            requested_aiv_cores=requested_aiv_cores,
            observed_aiv_cores=observed_aiv_cores,
            algorithm_regime=algorithm_regime,
        )
        stage_conditions = [
            item for item in self.conditions
            if item.placement == placement and item.mode == mode
            and (not execution_backend or item.execution_backend == execution_backend)
            and (not observed_execution or item.observed_execution == observed_execution)
            and (not collective or not item.collective or item.collective == collective)
            and (not reduction or not item.reduction or item.reduction == reduction)
            and (group_size is None or not item.group_size
                 or int(item.group_size) == int(group_size))
            and (not dtype or not item.dtype or item.dtype == dtype)
            and (ranks is None or item.ranks == int(ranks))
            and (not algorithm or not item.algorithm
                 or item.algorithm == algorithm)
            and self._identity_matches(item, hardware_key)
            and (not timing_scope or not item.timing_scope
                 or item.timing_scope == timing_scope)
            and (not topology_class or not item.topology_class
                 or item.topology_class == topology_class)
            and (not route or not item.route or item.route == route)
            and (not timing_completion
                 or item.timing_completion == timing_completion)
            and (in_place is None or not item.in_place or item.in_place == in_place)
            and (requested_aiv_cores is None or not item.requested_aiv_cores
                 or item.requested_aiv_cores == requested_aiv_cores)
            and (observed_aiv_cores is None or not item.observed_aiv_cores
                 or item.observed_aiv_cores == observed_aiv_cores)
            and (not algorithm_regime or not item.algorithm_regime
                 or item.algorithm_regime == algorithm_regime)
            and item.execution_stages
        ]
        nearest_distance = (
            min(abs(item.size_bytes - size_bytes) for item in stage_conditions)
            if stage_conditions else None
        )
        nearest = (
            [item for item in stage_conditions
             if abs(item.size_bytes - size_bytes) == nearest_distance]
            if nearest_distance is not None else []
        )
        descriptor_query = any((
            route, timing_completion, requested_aiv_cores is not None,
            observed_aiv_cores is not None, algorithm_regime,
            topology_class,
        ))
        semantics = {
            (item.route, item.timing_completion, item.requested_aiv_cores,
             item.observed_aiv_cores, item.algorithm_regime,
             item.topology_class, item.observed_execution)
            for item in nearest
        }
        typed_execution = any(
            item.observed_execution in {"aiv", "aicpu", "other"}
            for item in nearest
        )
        if typed_execution and len(semantics) > 1 and not descriptor_query:
            analytic = self._analytic_prediction(
                placement, size_bytes, requested_delay_ms, repetitions, mode,
                machine,
            )
            return finish({
                **analytic,
                "status": "uncalibrated_ambiguous_execution",
                "reason": "same-size rows differ in route/timing/cores/regime; "
                          "query descriptors are required",
                "required_query_fields": [
                    "route", "timing_completion", "requested_aiv_cores",
                    "observed_aiv_cores", "algorithm_regime",
                ],
                "candidate_routes": sorted(
                    {item.route for item in nearest if item.route}
                ),
                "stages": [],
            })
        selected_stage_condition = (
            min(nearest, key=lambda item: (
                abs(item.size_bytes - size_bytes), item.size_bytes,
                item.route, item.timing_completion,
                item.observed_aiv_cores, item.algorithm_regime,
            ))
            if nearest else None
        )
        stages = self._scale_stages(
            selected_stage_condition.execution_stages
            if selected_stage_condition is not None else (),
            source_size=(
                selected_stage_condition.size_bytes
                if selected_stage_condition is not None else int(size_bytes)
            ),
            query_size=int(size_bytes),
        )
        if mode == "all_reduce_only" and c_iso <= 0:
            return finish(self._analytic_prediction(
                placement, size_bytes, requested_delay_ms, repetitions, mode, machine,
            ))
        if mode == "p2p_only" and v_iso <= 0:
            return finish(self._analytic_prediction(
                placement, size_bytes, requested_delay_ms, repetitions, mode, machine,
            ))
        def size_support_for(mode_name: str) -> dict[str, Any]:
            points = self._curve_points(
                placement, mode_name,
                hardware_key=hardware_key, timing_scope=timing_scope,
                ranks=ranks, dtype=dtype, algorithm=algorithm,
                execution_backend=execution_backend,
                observed_execution=observed_execution, collective=collective,
                reduction=reduction, group_size=group_size,
                topology_class=topology_class, route=route,
                timing_completion=timing_completion, in_place=in_place,
                requested_aiv_cores=requested_aiv_cores,
                observed_aiv_cores=observed_aiv_cores,
                algorithm_regime=algorithm_regime,
            )
            sizes = sorted({int(point[0]) for point in points})
            inside = bool(sizes) and sizes[0] <= int(size_bytes) <= sizes[-1]
            exact = any(int(point[0]) == int(size_bytes) for point in points)
            singleton = len(sizes) == 1
            if exact:
                engine_selection = "measured_size"
            elif inside:
                engine_selection = "interpolated_size"
            else:
                engine_selection = "unknown_outside_measured_points"
            return {
                "measured_size_bytes": sizes,
                "query_size_bytes": int(size_bytes),
                "inside_envelope": bool(inside),
                "singleton_measurement": singleton,
                "extrapolated": not inside,
                "engine_selection": engine_selection,
                "support_basis": (
                    "measured size envelope; no threshold or engine-selection "
                    "inference outside measured points"
                ),
            }

        def missing_condition_for(mode_name: str, family: str) -> dict[str, Any]:
            return {
                "family": family, "phase": "communication",
                "recipe": f"two_rank_{mode_name}_{placement}",
                "shape": {"bytes": int(size_bytes), "ranks": int(ranks or 2)},
                "hardware_key": (
                    hardware_key or (machine.hardware_key if machine else "")
                ),
                "features": {
                    "placement": placement, "mode": mode_name,
                    "execution_backend": execution_backend,
                    "observed_execution": observed_execution,
                    "collective": collective, "reduction": reduction,
                    "group_size": group_size, "algorithm": algorithm,
                    "topology_class": topology_class, "route": route,
                    "timing_completion": timing_completion,
                    "in_place": in_place,
                },
                "reason": "query_size_outside_measured_support",
            }

        def support_fields(mode_name: str, family: str) -> dict[str, Any]:
            support = size_support_for(mode_name)
            fields = {
                "size_support": support,
                "engine_selection": support["engine_selection"],
                "in_envelope": support["inside_envelope"],
                "accuracy_scope": (
                    "measured_size_or_interpolated_envelope"
                    if support["inside_envelope"] else
                    "extrapolated_numeric_estimate_not_in_envelope"
                ),
            }
            if support["extrapolated"]:
                fields["missing_condition"] = missing_condition_for(
                    mode_name, family,
                )
            return fields

        def calibrated_status(base: str, support: dict[str, Any]) -> str:
            return base + "_extrapolated" if support["extrapolated"] else base

        typed_scope = (
            str(observed_execution) in {"aiv", "aicpu", "other"} and not stages
        )
        scope_fields = {
            "calibrated_scope": (
                "typed_whole_span" if typed_scope else
                ("typed_stages" if stages else "affine_curve")
            ),
            "occupancy": "unknown" if not stages else "typed_stage_work",
        }
        if mode not in {"all_reduce_only", "p2p_only"} and (c_iso <= 0 or v_iso <= 0):
            typed_span = self._lane_repeated_ms(
                placement, "all_reduce_plus_p2p", size_bytes, repetitions,
                hardware_key=hardware_key, timing_scope=timing_scope,
                ranks=ranks, dtype=dtype, algorithm=algorithm,
                execution_backend=execution_backend,
                observed_execution=observed_execution, collective=collective,
                reduction=reduction, group_size=group_size,
                topology_class=topology_class, route=route,
                timing_completion=timing_completion, in_place=in_place,
                requested_aiv_cores=requested_aiv_cores,
                observed_aiv_cores=observed_aiv_cores,
                algorithm_regime=algorithm_regime,
                require_no_stages=True,
            )
            if typed_scope and typed_span > 0:
                span_support = size_support_for("all_reduce_plus_p2p")
                return {
                    "status": calibrated_status(
                        "calibrated_typed_whole_span", span_support,
                    ),
                    **support_fields("all_reduce_plus_p2p", "comm_span"),
                    "placement": placement, "mode": mode,
                    "size_bytes": int(size_bytes),
                    "execution_backend": execution_backend,
                    "observed_execution": observed_execution,
                    "stages": [],
                    "calibrated_scope": "typed_whole_span",
                    "occupancy": "unknown",
                    "span_ms": typed_span,
                    "per_repetition": {
                        "span_ms": typed_span / max(1, repetitions),
                    },
                    "phases": {
                        "all_reduce": "analytic_unknown_occupancy",
                        "p2p": "analytic_unknown_occupancy",
                    },
                    "assumptions": [
                        "measured typed total span envelope only",
                        "internal phase occupancy/stage work remains unknown; no fabricated stages",
                    ],
                }
            return finish(self._analytic_prediction(
                placement, size_bytes, requested_delay_ms, repetitions, mode, machine,
            ))
        if mode in {"all_reduce_only"}:
            lane_support = size_support_for("all_reduce_only")
            return finish({
                "status": calibrated_status("calibrated_affine", lane_support),
                **support_fields("all_reduce_only", "all_reduce"),
                "placement": placement,
                "mode": mode, "size_bytes": int(size_bytes),
                "execution_backend": execution_backend,
                "observed_execution": observed_execution,
                "stages": list(stages),
                **scope_fields,
                "span_ms": c_repeat,
                "per_repetition": {"span_ms": c_iso, "all_reduce_ms": c_iso},
                "assumptions": ["affine isolated-lane curve from measured training"],
            })
        if mode in {"p2p_only"}:
            delay = max(0.0, float(requested_delay_ms))
            lane_support = size_support_for("p2p_only")
            return finish({
                "status": calibrated_status("calibrated_affine", lane_support),
                **support_fields("p2p_only", "p2p"),
                "placement": placement,
                "mode": mode, "size_bytes": int(size_bytes),
                "execution_backend": execution_backend,
                "observed_execution": observed_execution,
                "stages": list(stages),
                **scope_fields,
                "span_ms": delay + v_repeat,
                "per_repetition": {"span_ms": v_iso, "p2p_ms": v_iso},
                "assumptions": ["affine isolated-lane curve from measured training"],
            })
        joint_intercept, joint_slope = self._curve(
            placement, "all_reduce_plus_p2p",
            hardware_key=hardware_key, timing_scope=timing_scope,
            ranks=ranks, dtype=dtype, algorithm=algorithm,
            execution_backend=execution_backend,
            observed_execution=observed_execution, collective=collective,
            reduction=reduction, group_size=group_size,
            topology_class=topology_class, route=route,
            timing_completion=timing_completion, in_place=in_place,
            requested_aiv_cores=requested_aiv_cores,
            observed_aiv_cores=observed_aiv_cores,
            algorithm_regime=algorithm_regime,
        )
        joint_target = max(0.0, joint_intercept + joint_slope * size_bytes)
        probe = ResourceEngine({"lane": ResourceSpec("lane", 1.0)})
        probe.add_task("all_reduce", {"lane": c_iso})
        probe.add_task("p2p", {"lane": v_iso})
        base_span = probe.run()["makespan_ms"]
        scale = joint_target / base_span if base_span > 0 and joint_target > 0 else 1.0
        engine = ResourceEngine({"lane": ResourceSpec("lane", 1.0)})
        engine.add_task("all_reduce", {"lane": c_repeat * scale})
        engine.add_task("p2p", {"lane": v_repeat * scale},
                        release_ms=max(0.0, requested_delay_ms))
        result = engine.run()
        completed = result["completed_at_ms"]
        c_end = completed["all_reduce"]
        v_start = max(0.0, float(requested_delay_ms))
        v_end = completed["p2p"]
        v_duration = max(0.0, v_end - v_start)
        span = result["makespan_ms"]
        joint_support = size_support_for("all_reduce_plus_p2p")
        return finish({
            "status": calibrated_status(
                "calibrated_affine_shared_lane", joint_support,
            ),
            **support_fields("all_reduce_plus_p2p", "comm_span"),
            "placement": placement,
            "mode": mode,
            "execution_backend": execution_backend,
            "observed_execution": observed_execution,
            "stages": list(stages),
            **scope_fields,
            "size_bytes": int(size_bytes),
            "requested_delay_ms": float(requested_delay_ms),
            "repetitions": int(repetitions),
            "all_reduce_ms": c_end,
            "p2p_ms": v_duration,
            "c_ms": c_end,
            "v_ms": v_duration,
            "c_duration_ms": c_end,
            "v_duration_ms": v_duration,
            "span_ms": span,
            "per_repetition": {
                "all_reduce_ms": c_end / max(1, repetitions),
                "p2p_ms": v_duration / max(1, repetitions),
                "span_ms": span / max(1, repetitions),
            },
            "joint_target_ms": joint_target,
            "joint_scale": scale,
            "assumptions": [
                "isolated lane curves use measured training samples and affine size extrapolation",
                "joint span curve is fitted from measured training; delay is scheduled separately",
                "shared-lane remaining work is advanced by ResourceEngine at the requested start offset",
                "measured raw offsets/spans remain in the store; not used as condition identity",
            ],
        })


__all__ = [
    "COMM_PLACEMENT_ALIASES", "CommCondition", "CommunicationCalibration",
    "CommunicationExecutionSpec", "communication_placement_token",
]
