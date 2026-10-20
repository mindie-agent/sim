"""Predictive analytic cost model and measurement calibration for A3.

[Role]
  Convert explicit op work into resource demands: Cube/Vector compute, shared
  HBM, local MTE2/MTE3, launch, and D2D link bytes.  Apply shape/tiling
  efficiency per op family and an optional calibration residual learned from
  caller-supplied observations.

[Invariants]
  * Every cost keeps the analytic work and the final estimate separate.  A
    calibration factor is never labelled measured; only observations explicitly
    marked ``measured`` can calibrate.  Synthetic/unknown records remain
    metadata and are ignored by the estimator.
  * Revision/commit fields are excluded from the semantic calibration key.
    Changing a source revision for an identical semantic case cannot make the
    estimate unavailable.
  * GEMM work is ``2*M*N*K``.  A one-query attention against KV length ``L``
    uses ``L`` in its work, not ``1``.
  * Padding, graph buckets, TP/EP shards and launch are already present in the
    input op; this module never re-adds a whole-kernel memory term.
  * Missing calibration is not a reason to return null.  The analytic estimate
    remains available with an explicit ``uncalibrated_analytic`` source.

[Timeouts & Locks]
  Pure computation.  No timeouts, locks, I/O, subprocesses or provider imports.

[Side-Effect Policy]
  No filesystem/network/global-state writes.  Calibration data is read-only.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .a3_curve_binding import build_curve_base_query, query_curve_for_op
from .a3_operator_curve_protocol import canonical_protocol, protocol_equal
from .a3_types import A3InputError, MachineSpec, dtype_bytes
from .a3_workload import IterationWork, OpSpec

COMPUTE_FAMILIES = {
    "attention_qk", "attention_pv", "qkv_projection", "o_projection",
    "gate_up_projection", "down_projection", "mla_q_a", "mla_kv_a",
    "mla_q_b", "mla_kv_b", "moe_gate_up", "moe_down", "shared_expert_gate_up",
    "logits", "gemm", "vision_patch_embed", "vision_projector",
}
GEMM_TRANSFER_FAMILIES = {
    "qkv_projection", "o_projection", "gate_up_projection", "down_projection",
    "mla_q_a", "mla_kv_a", "mla_q_b", "mla_kv_b", "moe_gate_up",
    "moe_down", "shared_expert_gate_up", "shared_expert_down", "logits", "gemm",
}
VECTOR_FAMILIES = {
    "rmsnorm", "attention_softmax", "embedding", "sampling", "dequant",
}
COMM_FAMILIES = {
    "allreduce", "ep_dispatch", "ep_combine", "pp_p2p", "dcp_allgather",
    "moe_comm_fused", "kv_transfer", "engram_rh2d",
}
_TILE_M = 64
_TILE_N = 128
_TILE_K = 64
_MIB = 1024 * 1024


def gemm_flops(m: int, n: int, k: int) -> float:
    """Independent oracle for the GEMM work formula."""
    return 2.0 * max(0, int(m)) * max(0, int(n)) * max(0, int(k))


def attention_flops(
    q_len: int, kv_len: int, heads: int, head_dim: int, *,
    causal: bool = False, context: int = 0,
) -> float:
    """QK+PV work with an explicit query/KV distinction.

    For a decode query, ``q_len=1`` and ``kv_len=context``.  For a causal
    prefill chunk, prior context is fully visible and the in-chunk part is
    triangular.
    """
    q_len = max(0, int(q_len))
    heads = max(0, int(heads))
    head_dim = max(0, int(head_dim))
    if causal and q_len > 1:
        pairs = q_len * max(0, int(context)) + q_len * (q_len + 1) / 2.0
    else:
        pairs = q_len * max(0, int(kv_len))
    # One QK multiply-add and one PV multiply-add; each is 2 FLOPs/MAC.
    return 4.0 * heads * head_dim * pairs


@dataclass(frozen=True)
class CalibrationObservation:
    op_family: str
    phase: str
    recipe: str
    shape: tuple[tuple[str, float], ...]
    duration_ms: float
    analytic_ms: float
    source_class: str = "unknown"
    source: str = ""
    revision: str = ""
    machine_key: str = ""
    synthetic: bool = False
    metadata: tuple[tuple[str, str], ...] = ()
    dtype: str = ""
    layout: str = ""
    features: tuple[tuple[str, str], ...] = ()
    timing_scope: str = ""
    protocol: str = ""

    @property
    def usable(self) -> bool:
        return (
            self.source_class in {"measured", "observed"}
            and not self.synthetic
            and self.duration_ms > 0
            and self.analytic_ms > 0
        )

    @property
    def factor(self) -> float:
        return self.duration_ms / self.analytic_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_family": self.op_family, "phase": self.phase, "recipe": self.recipe,
            "shape": dict(self.shape), "dtype": self.dtype, "layout": self.layout,
            "features": dict(self.features), "timing_scope": self.timing_scope,
            "protocol": (
                json.loads(self.protocol) if self.protocol else {}
            ),
            "duration_ms": self.duration_ms,
            "analytic_ms": self.analytic_ms, "source_class": self.source_class,
            "source": self.source, "revision": self.revision,
            "synthetic": self.synthetic, "usable": self.usable,
            "factor": self.factor if self.analytic_ms else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CalibrationResult:
    status: str
    factor: float
    observations_used: int
    inside_envelope: bool | None
    extrapolated: bool
    clamped: bool
    source: str
    exact_matches: int = 0
    nearest_distance: float | None = None
    reason: str | None = None
    evidence: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "factor": self.factor,
            "observations_used": self.observations_used,
            "inside_envelope": self.inside_envelope,
            "extrapolated": self.extrapolated, "clamped": self.clamped,
            "source": self.source, "exact_matches": self.exact_matches,
            "nearest_distance": self.nearest_distance, "reason": self.reason,
            "evidence": dict(self.evidence) if self.evidence else None,
        }


class CalibrationStore:
    """Residual calibration keyed by semantics, not source revision."""

    def __init__(self, observations: Sequence[CalibrationObservation] = ()) -> None:
        self._observations = list(observations)

    @classmethod
    def from_mapping(cls, value: Any) -> "CalibrationStore":
        if value is None:
            return cls()
        if isinstance(value, CalibrationStore):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items = value
        elif isinstance(value, Mapping):
            items = value.get("observations", ())
        else:
            raise A3InputError("calibration_invalid")
        observations: list[CalibrationObservation] = []
        for index, item in enumerate(items):
            raw = item if isinstance(item, Mapping) else None
            if raw is None:
                raise A3InputError(f"calibration.observation_invalid:{index}")
            shape_raw = raw.get("shape", {})
            if isinstance(shape_raw, Mapping):
                shape = tuple(sorted((str(k), float(v)) for k, v in shape_raw.items()))
            else:
                shape = ()
            source_class = str(raw.get("source_class", raw.get("source_type", "unknown"))).lower()
            synthetic = bool(raw.get("synthetic", False))
            if "synthetic" in source_class or "fixture" in source_class:
                synthetic = True
            try:
                observations.append(CalibrationObservation(
                    op_family=str(raw.get("op_family", raw.get("family", ""))),
                    phase=str(raw.get("phase", "")),
                    recipe=str(raw.get("recipe", "")),
                    shape=shape,
                    duration_ms=float(raw.get("duration_ms", raw.get("measured_ms", 0.0))),
                    analytic_ms=float(raw.get("analytic_ms", raw.get("predicted_ms", 0.0))),
                    source_class=source_class,
                    source=str(raw.get("source", "")),
                    revision=str(raw.get("revision", raw.get("commit", ""))),
                    machine_key=str(raw.get("machine_key", "")),
                    synthetic=synthetic,
                    metadata=tuple(sorted((str(k), str(v)) for k, v in (raw.get("metadata") or {}).items())),
                    dtype=str(raw.get("dtype", "")),
                    layout=str(raw.get("layout", "")),
                    features=tuple(sorted((str(k), str(v)) for k, v in (raw.get("features") or {}).items())),
                    timing_scope=str(raw.get("timing_scope", "")),
                    protocol=json.dumps(
                        canonical_protocol(raw.get("protocol") or {}),
                        sort_keys=True, default=str,
                    ),
                ))
            except (TypeError, ValueError) as exc:
                raise A3InputError(f"calibration.observation_invalid:{index}") from exc
        return cls(observations)

    def observations(self) -> tuple[CalibrationObservation, ...]:
        return tuple(self._observations)

    def factor_for(
        self,
        *,
        op_family: str,
        phase: str,
        recipe: str,
        shape: Mapping[str, float],
        machine_key: str = "",
        dtype: str = "",
        layout: str = "",
        features: Mapping[str, Any] | None = None,
        competition: Mapping[str, Any] | None = None,
        protocol: Mapping[str, Any] | None = None,
        timing_scope: str = "",
        allow_cross_recipe: bool = False,
    ) -> CalibrationResult:
        query_features = {str(k): str(v) for k, v in (features or {}).items()}
        query_competition = {str(k): str(v) for k, v in (competition or {}).items()}
        ignored_feature_keys = {
            "source_family", "source_recipe", "coverage_scope",
            "native_precision_claim", "analytic_baseline_omitted",
            "call_path", "measurement_scope", "repeats_are_not_independent_trials",
        }

        def compatible_features(obs: Mapping[str, str], query: Mapping[str, str]) -> bool:
            if not all(query.get(key) == value for key, value in obs.items()
                       if key not in ignored_feature_keys):
                return False
            if not all(key in obs for key in query):
                return False
            return all(obs.get(key) == value for key, value in query.items())

        def same_map(obs: Mapping[str, str], query: Mapping[str, str]) -> bool:
            if not obs and not query:
                return True
            # Conditions must be declared on both sides to be comparable.  A
            # query that omits cache/competition conditions cannot claim a
            # measurement made under different conditions.
            return all(query.get(key) == value for key, value in obs.items()) and all(
                key in obs for key in query
            )

        query_protocol = canonical_protocol(protocol)

        def observation_protocol(item: "CalibrationObservation") -> dict[str, Any]:
            return canonical_protocol(
                json.loads(item.protocol) if item.protocol else {}
            )

        # Physical protocol only means anything inside the actual business
        # candidates.  Filter family/phase/hardware/precision/layout/features
        # and recipe before considering protocol signatures, so an unrelated
        # measurement cannot create ambiguity for a valid exact match.
        relevant = [
            item for item in self._observations
            if item.usable
            and item.op_family == op_family
            and item.phase == phase
            and (not machine_key or not item.machine_key
                 or item.machine_key == machine_key)
            and (not dtype or not item.dtype or item.dtype == dtype)
            and (not layout or not item.layout or item.layout == layout)
            and compatible_features(dict(item.features), query_features)
            and same_map(dict(item.metadata), query_competition)
            and str(dict(item.features).get("execution_kind", "primitive"))
                not in {"fused_kernel", "reference_decomposition"}
            and str(dict(item.features).get("target_match", "true")).lower()
                not in {"false", "0", "no"}
            and not str(dict(item.features).get("coverage_scope", "")).startswith(
                "native_operator_semantic"
            )
            and (not timing_scope or not item.timing_scope
                 or item.timing_scope == timing_scope)
        ]
        if not relevant:
            return CalibrationResult(
                "uncalibrated_analytic", 1.0, 0, None, False, False,
                "no_usable_measured_observation",
                reason="no_compatible_measured_observation",
            )
        exact_recipe = [item for item in relevant if item.recipe == recipe]
        if exact_recipe:
            candidates = exact_recipe
        elif allow_cross_recipe:
            candidates = relevant
        else:
            return CalibrationResult(
                "uncalibrated_recipe_mismatch", 1.0, 0, None, False, False,
                "no_usable_measured_observation", reason="cross_recipe_requires_explicit_model",
            )
        query = {str(key): float(value) for key, value in shape.items()}
        exact = [
            item for item in candidates
            if set(dict(item.shape)) == set(query)
            and all(abs(query[key] - value) < 1e-9
                    for key, value in dict(item.shape).items())
        ]
        neighbor_rows: list[tuple[float, "CalibrationObservation", set[str]]] = []
        if exact:
            shape_candidates = exact
        else:
            # Neighbors must share at least one semantic shape dimension.  An
            # empty intersection is not distance zero.
            for item in candidates:
                obs = dict(item.shape)
                common = set(query) & set(obs)
                if not common:
                    continue
                distance = sum(
                    abs(math.log((abs(query[key]) + 1.0) / (abs(obs[key]) + 1.0)))
                    for key in common
                )
                neighbor_rows.append((distance, item, common))
            if not neighbor_rows:
                return CalibrationResult(
                    "uncalibrated_semantic_key_mismatch", 1.0, 0, None, False,
                    False, "no_usable_measured_observation",
                    reason="no_common_shape_dimension",
                )
            shape_candidates = [item for _, item, _ in neighbor_rows]

        protocol_signatures = {
            json.dumps(
                observation_protocol(item), sort_keys=True, separators=(",", ":"),
            )
            for item in shape_candidates
        }
        if query_protocol:
            selected = [
                item for item in shape_candidates
                if protocol_equal(observation_protocol(item), query_protocol)
            ]
            if not selected:
                return CalibrationResult(
                    "uncalibrated_analytic", 1.0, 0, None, False, False,
                    "no_usable_measured_observation",
                    reason="physical_protocol_mismatch",
                )
        else:
            selected = [
                item for item in shape_candidates
                if not observation_protocol(item)
            ]
            if not selected:
                if len(protocol_signatures) > 1:
                    return CalibrationResult(
                        "uncalibrated_ambiguous_protocol", 1.0, 0, None,
                        False, False, "no_usable_measured_observation",
                        reason="multiple_physical_protocols_require_explicit_selection",
                    )
                return CalibrationResult(
                    "uncalibrated_protocol_required", 1.0, 0, None, False,
                    False, "no_usable_measured_observation",
                    reason="measured_physical_protocol_requires_explicit_selection",
                )
        if exact:
            factor = sum(item.factor for item in selected) / len(selected)
            return CalibrationResult(
                "calibrated_exact", factor, len(selected), True, False, False,
                "measured_residual_excluding_revision",
                exact_matches=len(selected), nearest_distance=0.0,
            )
        selected_ids = {id(item) for item in selected}
        neighbor_rows = [
            row for row in neighbor_rows if id(row[1]) in selected_ids
        ]
        common_keys: set[str] = set()
        for _, _, common in neighbor_rows:
            common_keys |= common
        envelope: dict[str, tuple[float, float]] = {}
        for key in common_keys:
            values = [float(dict(item.shape)[key]) for _, item, common in neighbor_rows if key in common]
            envelope[key] = (min(values), max(values))
        inside = all(
            lo <= query[key] <= hi for key, (lo, hi) in envelope.items() if key in query
        ) and set(query).issubset(common_keys)
        weighted = 0.0
        weight_sum = 0.0
        nearest = math.inf
        for distance, item, _ in neighbor_rows:
            weight = 1.0 / (1.0 + distance) ** 2
            weighted += weight * item.factor
            weight_sum += weight
            nearest = min(nearest, distance)
        factor = weighted / weight_sum if weight_sum else 1.0
        return CalibrationResult(
            "calibrated_interpolated" if inside else "calibrated_extrapolated",
            factor, len(neighbor_rows), inside, not inside, False,
            "measured_residual_excluding_revision",
            nearest_distance=nearest,
        )


@dataclass(frozen=True)
class CostedOp:
    op: OpSpec
    demands: tuple[tuple[str, float], ...]
    analytic_ms: float
    estimated_ms: float
    calibration: CalibrationResult
    coverage: str = "decomposed"
    curve_result: Mapping[str, Any] | None = None

    def demand_map(self) -> dict[str, float]:
        return dict(self.demands)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op.op_id, "family": self.op.family,
            "demands": dict(self.demands),
            "analytic_ms": self.analytic_ms, "estimated_ms": self.estimated_ms,
            "calibration": self.calibration.to_dict(),
            "curve_result": dict(self.curve_result) if self.curve_result else None,
            "coverage": self.coverage, "recipe": self.op.recipe,
        }


def _gemm_efficiency(op: OpSpec, aic_count: int) -> tuple[float, dict[str, float]]:
    shape = op.launch_shape
    if op.family in {"attention_qk", "attention_pv", "attention_fused"} and len(shape) >= 4:
        m, n, k = shape[0], shape[1], shape[3]
    elif len(shape) >= 3:
        m, n, k = shape[-3], shape[-2], shape[-1]
    elif len(shape) == 2:
        m, n, k = shape[0], shape[1], 1
    else:
        m, n, k = max(1, shape[0] if shape else 1), 1, 1
    tiles = max(1, math.ceil(max(1, m) / _TILE_M)) * max(1, math.ceil(max(1, n) / _TILE_N))
    waves = max(1, math.ceil(tiles / max(1, aic_count)))
    wave_eff = tiles / (waves * max(1, aic_count))
    k_eff = max(0.5, min(1.0, max(1, k) / (max(1, k) + 2 * _TILE_K)))
    return max(0.02, wave_eff * k_eff), {
        "tiles": float(tiles), "waves": float(waves), "wave_efficiency": wave_eff,
        "k_pipeline_efficiency": k_eff,
    }


def _shape_efficiency(op: OpSpec, machine: MachineSpec) -> tuple[float, float, dict[str, Any]]:
    # The AIC core count is a schedule parameter, not a hidden efficiency.  A
    # 24-core tile reference is used only for the wave term; the raw FLOPs stay
    # in ``demands['cube']`` so a smaller/larger core count cannot erase work.
    if op.flops > 0 and op.family not in COMM_FAMILIES:
        eff, detail = _gemm_efficiency(op, 24)
        return eff, 1.0, detail
    if op.family in COMM_FAMILIES:
        return 1.0, 1.0, {"comm": True}
    ops = max(1.0, op.vector_ops)
    eff = min(1.0, 0.5 + 0.5 * ops / 4096.0)
    bytes_total = max(1.0, op.hbm_bytes)
    mem_eff = min(1.0, 0.65 + 0.35 * bytes_total / (2 * _MIB))
    return eff, mem_eff, {"vector_efficiency": eff, "memory_efficiency": mem_eff}


from .a3_precision import (  # noqa: E402  (public aliases kept for callers)
    physical_layout as _physical_layout,
    physical_quantization as _physical_quantization,
)


@dataclass
class PredictiveCostModel:
    machine: MachineSpec
    calibration: CalibrationStore = field(default_factory=CalibrationStore)
    compute_dtype: str = "bf16"
    link_scope: str = "d2d"
    weight_dtype: str = ""
    activation_dtype: str = ""
    quantization: str = ""
    timing_scope: str = "device_event_warm"
    accumulation_dtype: str = "backend_default"
    communication_calibration: Any = None
    communication_placement: str = ""
    communication_ranks: int = 2
    communication_backend: str = "unspecified"
    communication_observed_backend: str = "unknown"
    communication_route: str = ""
    communication_timing_completion: str = ""
    communication_requested_aiv_cores: int = 0
    communication_observed_aiv_cores: int = 0
    communication_algorithm_regime: str = ""
    communication_topology_class: str = ""
    communication_in_place: bool = False
    cv_calibration: Any = None
    cv_query: tuple[int, int, int, int] | None = None
    curve_artifact: Mapping[str, Any] | None = None
    curve_allow_primitive_transfer: bool = False
    graph_mode: str = "auto"
    calibration_protocol: Mapping[str, Any] | None = None

    def calibration_features(self) -> dict[str, str]:
        return {
            "weight_dtype": self.weight_dtype or self.compute_dtype,
            "activation_dtype": self.activation_dtype or self.compute_dtype,
            "compute_dtype": self.compute_dtype,
            "accumulation_dtype": self.accumulation_dtype,
            "quantization": _physical_quantization(self.quantization),
        }

    def _link(self) -> Any:
        for link in self.machine.links:
            if link.scope == self.link_scope or link.name == self.link_scope:
                return link
        return self.machine.links[0] if self.machine.links else None

    def machine_key(self) -> str:
        # A real measurement is keyed by the hardware identity, not by the
        # simulator's configured device count.  An explicit fixture name/key is
        # usable directly.
        return self.machine.hardware_key or self.machine.sku or self.machine.name

    def shape_features(self, op: OpSpec) -> dict[str, float]:
        shape = op.launch_shape
        if op.family in {"attention_qk", "attention_pv", "attention_fused"} and len(shape) >= 4:
            return {"q": float(shape[0]), "kv": float(shape[1]), "heads": float(shape[2]), "dim": float(shape[3])}
        if len(shape) >= 3:
            return {"m": float(shape[-3]), "n": float(shape[-2]), "k": float(shape[-1])}
        if len(shape) == 2:
            return {"elements": float(shape[0] * shape[1]), "dim": float(shape[1])}
        return {"elements": float(shape[0] if shape else 1)}

    def _curve_base_query(self, op: OpSpec) -> dict[str, Any]:
        meta = dict(op.metadata)
        shape = meta.get("curve_shape")
        if not isinstance(shape, Mapping):
            shape = self.shape_features(op)
        return build_curve_base_query(self, op, shape)

    def curve_prediction(self, op: OpSpec) -> dict[str, Any] | None:
        meta = dict(op.metadata)
        shape = meta.get("curve_shape")
        if not isinstance(shape, Mapping):
            shape = self.shape_features(op)
        return query_curve_for_op(self, op, shape)

    def calibration_for(self, op: OpSpec) -> CalibrationResult:
        primary = self.calibration.factor_for(
            op_family=op.family, phase=op.phase, recipe=op.recipe,
            shape=self.shape_features(op), machine_key=self.machine_key(),
            dtype=op.dtype or self.compute_dtype,
            layout=_physical_layout(op.layout),
            features=self.calibration_features(),
            protocol=self.calibration_protocol,
            timing_scope=self.timing_scope,
        )
        if primary.status in {
            "calibrated_exact", "calibrated_interpolated", "calibrated_extrapolated",
        }:
            return primary
        if op.family in GEMM_TRANSFER_FAMILIES:
            # Generic torch.mm measurements are an explicit transfer model for
            # projection/MLP GEMMs, not an implicit family-wide residual.
            transfer = self.calibration.factor_for(
                op_family="gemm", phase="operator", recipe="generic_transfer",
                shape=self.shape_features(op), machine_key=self.machine_key(),
                dtype=op.dtype or self.compute_dtype,
                layout=_physical_layout(op.layout),
                features=self.calibration_features(),
                protocol=self.calibration_protocol,
                timing_scope=self.timing_scope,
                allow_cross_recipe=True,
            )
            if transfer.status.startswith("calibrated"):
                return CalibrationResult(
                    "calibrated_transfer_gemm", transfer.factor,
                    transfer.observations_used, transfer.inside_envelope,
                    transfer.extrapolated, transfer.clamped,
                    "generic_gemm_measurement_transfer",
                    exact_matches=transfer.exact_matches,
                    nearest_distance=transfer.nearest_distance,
                )
            return transfer
        return primary

    def cost(self, op: OpSpec) -> CostedOp:
        op_metadata = dict(op.metadata)
        if (bool(op_metadata.get("witness_only"))
                or bool(op_metadata.get("not_execution"))):
            return CostedOp(
                op=op, demands=(), analytic_ms=0.0, estimated_ms=0.0,
                calibration=CalibrationResult(
                    "witness_not_execution", 1.0, 0, None, False, False,
                    "catalog_structural_witness_excluded",
                ),
                coverage="witness",
            )
        die = self.machine.dies[0] if self.machine.dies else None
        if die is None:
            raise A3InputError("machine_has_no_dies")
        factor_status = self.calibration_for(op)
        factor = factor_status.factor
        typed_stage_ms = 0.0
        # Compute uncalibrated and calibrated demands from the same shape model.
        def build(scale: float) -> dict[str, float]:
            demands: dict[str, float] = {}
            if op.flops > 0 and op.family not in COMM_FAMILIES:
                eff, _, _ = _shape_efficiency(op, self.machine)
                demands["cube"] = op.flops * scale / max(0.02, eff)
            if op.vector_ops > 0:
                eff = max(0.05, min(1.0, 0.5 + 0.5 * max(1.0, op.vector_ops) / 4096.0))
                demands["vector"] = op.vector_ops * scale / eff
            if op.hbm_bytes > 0:
                mem_eff = min(1.0, 0.65 + 0.35 * op.hbm_bytes / (2 * _MIB))
                demands["hbm"] = op.hbm_bytes * scale / mem_eff
            if op.mte2_bytes > 0:
                mte_eff = min(1.0, 0.5 + 0.5 * op.mte2_bytes / _MIB)
                demands["mte2"] = op.mte2_bytes * scale / mte_eff
            if op.mte3_bytes > 0:
                mte_eff = min(1.0, 0.5 + 0.5 * op.mte3_bytes / _MIB)
                demands["mte3"] = op.mte3_bytes * scale / mte_eff
            if op.link_bytes > 0:
                demands["link"] = op.link_bytes * scale
            launch_ms = die.launch_us / 1000.0
            link = self._link()
            if op.link_bytes > 0 and link is not None:
                launch_ms += link.latency_us / 1000.0
            if launch_ms > 0 and op.family not in {"pp_p2p"}:
                # A measured residual covers the whole kernel duration, including
                # launch gaps.  The uncalibrated baseline keeps launch at 1x.
                demands["launch"] = launch_ms * scale
            return demands

        capacity = {
            "cube": self.machine.peak_flops_for(self.compute_dtype) / 1000.0,
            "vector": die.vector_ops_per_s / 1000.0,
            "hbm": die.hbm_bw_bytes_per_s / 1000.0,
            "mte2": die.mte_bw_bytes_per_s / 1000.0,
            "mte3": die.mte_bw_bytes_per_s / 1000.0,
            "link": (self._link().capacity_bytes_per_ms if self._link() is not None else 1.0),
            "launch": 1.0, "cpu": 1.0, "serial": 1.0, "aicpu": 1.0,
        }
        baseline = build(1.0)
        calibrated = build(factor)
        calibration_status = factor_status
        # Integrate measured communication curves through the primitive
        # transport semantics, not blind family-key equality.
        if (self.communication_calibration is not None and op.link_bytes > 0
                and op.family in COMM_FAMILIES):
            link = self._link()
            mode = ("all_reduce_only" if op.family == "allreduce"
                    else "p2p_only")
            placement = self.communication_placement
            if not placement and link is not None:
                name = str(getattr(link, "name", "")).lower()
                placement = "sio" if "sio" in name else (
                    "hccssw" if "hccs" in name else ""
                )
            # Unknown topology stays unspecified; never infer placement
            # merely because one measured placement is available.
            if placement:
                elements = 1
                for dim in op.semantic_shape:
                    elements *= max(1, int(dim))
                try:
                    payload_bytes = int(elements * dtype_bytes(op.dtype or self.compute_dtype))
                except Exception:
                    payload_bytes = int(op.link_bytes)
                ranks = int(self.communication_ranks) if op.family == "allreduce" else 2
                comm = self.communication_calibration.predict(
                    placement=placement, size_bytes=max(1, payload_bytes),
                    mode=mode, repetitions=1, machine=self.machine,
                    ranks=ranks, dtype=op.dtype or self.compute_dtype,
                    algorithm=mode,
                    execution_backend=self.communication_backend,
                    observed_execution=self.communication_observed_backend,
                    collective=(
                        "all_reduce" if op.family == "allreduce"
                        else "send_recv"
                    ),
                    reduction="sum" if op.family == "allreduce" else "",
                    group_size=ranks,
                    route=self.communication_route,
                    timing_completion=self.communication_timing_completion,
                    requested_aiv_cores=(
                        self.communication_requested_aiv_cores or None
                    ),
                    observed_aiv_cores=(
                        self.communication_observed_aiv_cores or None
                    ),
                    algorithm_regime=self.communication_algorithm_regime,
                    topology_class=self.communication_topology_class,
                    in_place=(
                        self.communication_in_place
                        if self.communication_in_place else None
                    ),
                )
                if not isinstance(comm, Mapping):
                    raise A3InputError("communication_prediction_invalid")
                span = comm.get("span_ms")
                typed_observed = str(
                    comm.get("observed_execution", "unknown")
                ) in {"aiv", "aicpu", "other"}
                calibrated_scope = str(comm.get("calibrated_scope") or "")
                if (typed_observed and not comm.get("stages")
                        and calibrated_scope != "typed_whole_span"):
                    comm = {
                        **comm,
                        "status": "uncalibrated_typed_execution_without_stages",
                    }
                if (span is not None and float(span) > 0 and link is not None
                        and str(comm.get("status", "")).startswith("calibrated")):
                    stages = tuple(comm.get("stages", ()) or ())
                    if stages:
                        calibrated.pop("link", None)
                        op = replace(op, metadata=(
                            *op.metadata, ("comm_stages", stages),
                            ("comm_execution_backend",
                             str(comm.get("execution_backend", "unspecified"))),
                        ))
                        stage_elapsed = 0.0
                        for stage in stages:
                            stage_work = dict(stage.get("work", {}) or {})
                            stage_ms = 0.0
                            for logical, resource_key in (
                                ("aicpu_service", "aicpu"),
                                ("vector_ops", "vector"),
                                ("mte2_bytes", "mte2"),
                                ("mte3_bytes", "mte3"),
                                ("link_edge_bytes", "link"),
                            ):
                                amount = float(stage_work.get(logical, 0.0) or 0.0)
                                if amount > 0:
                                    stage_ms = max(
                                        stage_ms,
                                        amount / max(1e-9, capacity[resource_key]),
                                    )
                            hbm_amount = sum(
                                float(stage_work.get(key, 0.0) or 0.0)
                                for key in (
                                    "local_hbm_read", "local_hbm_write",
                                    "peer_hbm_read", "peer_hbm_write",
                                )
                            )
                            if hbm_amount > 0:
                                stage_ms = max(
                                    stage_ms,
                                    hbm_amount / max(1e-9, capacity["hbm"]),
                                )
                            stage_elapsed += stage_ms
                        calibration_status = CalibrationResult(
                            str(comm.get("status", "calibrated_communication")),
                            1.0, 0, None, False, False,
                            "communication_typed_stage_work",
                        )
                        typed_stage_ms = stage_elapsed
                    else:
                        # An opaque measured whole span is a per-op completion
                        # envelope, not physical link/AIV/AICPU occupancy and
                        # not a shared serial resource.  The known analytic
                        # payload/link work stays in analytic_ms and evidence,
                        # so it cannot fabricate concurrency in the engine.
                        analytic_link_ms = (
                            float(op.link_bytes) / max(
                                1e-9, link.capacity_bytes_per_ms,
                            ) if link is not None else 0.0
                        )
                        calibrated["completion_envelope_ms"] = float(span)
                        comm_evidence = {
                            key: comm.get(key) for key in (
                                "size_support", "engine_selection",
                                "in_envelope", "accuracy_scope",
                                "missing_condition",
                            ) if comm.get(key) is not None
                        }
                        comm_evidence.update({
                            "source_status": str(comm.get(
                                "status", "calibrated_whole_span_envelope",
                            )),
                            "analytic_link_ms": analytic_link_ms,
                        })
                        support = dict(comm.get("size_support") or {})
                        calibration_status = CalibrationResult(
                            str(comm.get(
                                "status", "calibrated_whole_span_envelope",
                            )),
                            1.0, 0,
                            support.get("inside_envelope"),
                            bool(support.get("extrapolated", False)),
                            False,
                            "observed_whole_span_envelope_only",
                            reason=(
                                "per_op_completion_envelope;"
                                f"source_status={comm_evidence['source_status']};"
                                f"engine_selection={comm.get('engine_selection', 'unknown')}"
                            ),
                            evidence=comm_evidence,
                        )
                    estimated_ms = max(
                        (
                            work / max(1e-9, capacity[key])
                            for key, work in calibrated.items()
                            if key != "completion_envelope_ms"
                        ),
                        default=0.0,
                    )
                    estimated_ms = max(
                        estimated_ms,
                        float(calibrated.get("completion_envelope_ms", 0.0)),
                    )
        # CV contention is applied only when the caller declares a matching
        # C/V query; it never silently pools an unrelated measured condition.
        if (self.cv_calibration is not None and self.cv_query is not None
                and op.flops > 0 and len(op.launch_shape) == 3
                and op.family not in {"attention_fused", "attention_qk", "attention_pv"}):
            # Op launch shape is (M,N,K); CV conditions use (M,K,N).
            m, n, k = (int(v) for v in op.launch_shape[-3:])
            c_shape = (m, k, n)
            qc = tuple(int(v) for v in self.cv_query[:3])
            if c_shape == qc:
                cv = self.cv_calibration.predict(
                    c_shape_mkn=qc, v_elements=int(self.cv_query[3]),
                    requested_v_delay_ms=0.0, machine=self.machine,
                    dtype=op.dtype or self.compute_dtype,
                    hardware_key=self.machine_key(),
                )
                if cv.get("status", "").startswith("calibrated"):
                    scale = float(cv.get("slowdowns", {}).get("joint_scale", 1.0))
                    if scale > 0:
                        calibrated["hbm"] = calibrated.get("hbm", 0.0) * scale
                        calibrated["vector"] = calibrated.get("vector", 0.0) * scale
                        calibration_status = CalibrationResult(
                            "calibrated_cv_contention", scale, 1, None,
                            False, False, "measured_cv_joint_contention",
                        )
                        estimated_ms = max(
                            (work / max(1e-9, capacity[key])
                             for key, work in calibrated.items()), default=0.0,
                        )
        analytic_ms = max(
            (work / max(1e-9, capacity[key]) for key, work in baseline.items()),
            default=0.0,
        )
        curve = self.curve_prediction(op)
        transfer_scope = str((curve or {}).get("transfer_scope") or "")
        native_candidate = (
            curve is not None
            and curve.get("target_match") is False
            and bool(self.curve_allow_primitive_transfer)
            and transfer_scope == "explicit_native_primitive_transfer"
            and curve.get("production_match") is False
        )
        if (
            curve is not None
            and curve.get("status") in {"ok", "extrapolation"}
            and curve.get("prediction_ms") is not None
            and (curve.get("target_match") is not False or native_candidate)
            and analytic_ms > 0
            and not calibration_status.status.startswith("calibrated")
        ):
            prediction_ms = float(curve["prediction_ms"])
            if prediction_ms > 0:
                curve_factor = prediction_ms / analytic_ms
                calibrated = build(curve_factor)
                if transfer_scope == "explicit_native_primitive_transfer":
                    source = (
                        "operator_train_only_piecewise_curve_native_primitive_candidate"
                    )
                    reason = (
                        "approximate_primitive_transfer;candidate;"
                        "production_match=false"
                    )
                elif transfer_scope:
                    source = "operator_train_only_piecewise_curve_generic_bf16_transfer"
                    reason = "approximate_primitive_transfer;production_match=false"
                else:
                    source = "operator_train_only_piecewise_curve"
                    reason = str(curve.get("status"))
                calibration_status = CalibrationResult(
                    "calibrated_piecewise_curve", curve_factor,
                    int(curve.get("train_support", {}).get("point_count", 0) or 0),
                    curve.get("region") == "interpolation", False, False,
                    source, reason=reason,
                )
        elif (
            curve is not None and curve.get("status") == "transition_unresolved"
            and not calibration_status.status.startswith("calibrated")
        ):
            calibration_status = CalibrationResult(
                "curve_transition_unresolved", 1.0, 0, None, False, False,
                "operator_train_only_piecewise_curve",
                reason=str(curve.get("reason") or "transition_unresolved"),
            )
        estimated_ms = max(
            typed_stage_ms,
            max(
                (
                    work / max(1e-9, capacity[key])
                    for key, work in calibrated.items()
                    if key != "completion_envelope_ms"
                ),
                default=0.0,
            ),
            float(calibrated.get("completion_envelope_ms", 0.0)),
        )
        return CostedOp(
            op=op, demands=tuple(sorted(calibrated.items())),
            analytic_ms=analytic_ms, estimated_ms=estimated_ms,
            calibration=calibration_status, coverage="decomposed",
            curve_result=curve,
        )

    def cost_iteration(self, work: IterationWork) -> list[CostedOp]:
        if not work.ops:
            raise A3InputError("iteration_work_empty")
        return [self.cost(op) for op in work.ops]

    def hardware_lower_bound_ms(self, work: IterationWork) -> float:
        """Raw resource roofline for this exact DAG; not a normative target."""
        die = self.machine.dies[0]
        flops = sum(op.flops for op in work.ops)
        vector = sum(op.vector_ops for op in work.ops)
        hbm = sum(op.hbm_bytes for op in work.ops)
        link_bytes = sum(op.link_bytes for op in work.ops)
        link = self._link()
        capacities = [
            flops / (die.cube_flops_per_s / 1000.0) if flops else 0.0,
            vector / (die.vector_ops_per_s / 1000.0) if vector else 0.0,
            hbm / (die.hbm_bw_bytes_per_s / 1000.0) if hbm else 0.0,
            link_bytes / (link.capacity_bytes_per_ms if link else 1.0) if link_bytes else 0.0,
        ]
        return max(capacities) if capacities else 0.0


__all__ = [
    "COMPUTE_FAMILIES", "VECTOR_FAMILIES", "COMM_FAMILIES",
    "gemm_flops", "attention_flops", "CalibrationObservation",
    "CalibrationResult", "CalibrationStore", "CostedOp", "PredictiveCostModel",
]
