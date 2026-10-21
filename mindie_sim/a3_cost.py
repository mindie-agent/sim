"""Predictive analytic cost model and measurement calibration for A3/A2.

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
  * A measured duration is dispatch gap **plus** kernel service, so
    ``estimated_ms = launch_ms + max(resource service times)``.  Launch is an
    explicit additive term (``launch_amortization`` = 1.0 for eager per-op
    submission), never one more resource competing in a ``max``.
  * The wave quantization uses the machine's own ``die.aic_count``; the literal
    24 survives only as a documented fallback for a machine that declares no
    AIC count at all.
  * No invented multiplicative "efficiency" floors: a demand is raw work
    divided by the resource ceiling.  Terms that could not be justified
    physically were deleted, and the record says so.
  * Attention QK and PV use different contraction axes: QK is
    ``(q, kv, dim)``, PV is ``(q, dim, kv)``.
  * A link is resolved per op from its declared fabric scope; an unresolvable
    scope is an error, never ``links[0]``.  A collective's calibration placement
    is the fabric whose protocol the machine declares (SIO pair, HCCS switch,
    RoCE); a fabric with no measured placement token (an A5 UB-Mesh link) is
    explicitly uncalibrated rather than priced with another fabric's curve.
  * The fitted physical layer, when supplied, owns the duration: its fitted
    intercept *is* the launch term and is never added a second time.  An op the
    artifact does not model falls back to the analytic/curve path with the
    artifact's own reason, and a family whose fitted parameters are labelled
    unidentified keeps the analytic path and says so.
  * Revision/commit fields are excluded from the semantic calibration key.
    Changing a source revision for an identical semantic case cannot make the
    estimate unavailable.
  * Hardware identity is required to consume a measured residual: a query
    without a ``machine_key`` and an observation with an empty ``machine_key``
    never match, so A3 residuals cannot leak onto A2.
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
#: Declared k-block width.  The tile model has no k dimension (a known
#: remaining structural gap, W1/B12): k-blocking is not charged, and the old
#: ``k/(k+128)`` discount that used this value has been deleted.
_TILE_K = 64
#: AIC-core reference used *only* when the machine declares no AIC count at
#: all.  It is the Ascend910_9391 count and is reported in the cost detail as
#: ``aic_count_source=legacy_fallback_24`` whenever it is used.
_LEGACY_AIC_REFERENCE = 24
_ATTENTION_FAMILIES = {"attention_qk", "attention_pv", "attention_fused"}


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
        query_machine = str(machine_key or "").strip()
        if not query_machine:
            # Hardware identity is not optional: without it a residual from any
            # machine would match.  Refuse instead of wildcarding.
            return CalibrationResult(
                "uncalibrated_machine_key_missing", 1.0, 0, None, False, False,
                "no_usable_measured_observation",
                reason="machine_key_required_for_calibration",
            )
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
            # Identity is an exact join.  An observation imported without a
            # hardware key is unusable for every machine, which is visible in
            # the returned reason rather than silently pooled across SKUs.
            and item.machine_key == query_machine
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
    #: The additive dispatch term used by ``estimated_ms`` and the serial kernel
    #: service time it is added to.  Reported separately so a physical
    #: calibration can later fit the intercept instead of a ratio.
    launch_ms: float = 0.0
    service_ms: float = 0.0
    #: Fitted physical-calibration provenance for this op: the artifact's
    #: prediction (or its refusal reason), the policy decision, and the fitted
    #: parameter quality/level.  ``None`` when no artifact was supplied, so the
    #: analytic/curve record is exactly what it was before.
    physical: Mapping[str, Any] | None = None

    def demand_map(self) -> dict[str, float]:
        return dict(self.demands)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op.op_id, "family": self.op.family,
            "demands": dict(self.demands),
            "analytic_ms": self.analytic_ms, "estimated_ms": self.estimated_ms,
            "launch_ms": self.launch_ms, "service_ms": self.service_ms,
            "calibration": self.calibration.to_dict(),
            "curve_result": dict(self.curve_result) if self.curve_result else None,
            "coverage": self.coverage, "recipe": self.op.recipe,
            "physical": dict(self.physical) if self.physical else None,
        }


def _gemm_efficiency(op: OpSpec, aic_count: int) -> tuple[float, dict[str, float]]:
    """Wave quantization against the machine's declared AIC count.

    ``waves = ceil(tiles / aic)``; a partially filled last wave is charged in
    full by using ``tiles / (waves * aic)`` as the throughput factor.  The
    previous ``k / (k + 2*tile_k)`` "pipeline fill" multiplier (floored at 0.5)
    was an invented saturating discount: pipeline fill is additive latency, not
    a multiplicative loss, and the term was ~1 for every production ``k`` while
    collapsing all ``k <= 128`` shapes onto the same floor.  It is deleted; the
    raw FLOPs stay in the demand so the resource ceiling is the bound.

    Attention QK and PV are different GEMMs over the same 4-D
    ``(q, kv, heads, dim)`` launch shape: QK contracts ``dim``
    (``m=q, n=kv, k=dim``), PV contracts ``kv`` (``m=q, n=dim, k=kv``).
    """
    shape = op.launch_shape
    mapping = "generic:m=shape[-3],n=shape[-2],k=shape[-1]"
    if op.family in _ATTENTION_FAMILIES and len(shape) >= 4:
        q, kv, dim = max(1, int(shape[0])), max(1, int(shape[1])), max(1, int(shape[3]))
        qk_tiles = (
            math.ceil(q / _TILE_M) * math.ceil(kv / _TILE_N)
        )
        pv_tiles = (
            math.ceil(q / _TILE_M) * math.ceil(dim / _TILE_N)
        )
        if op.family == "attention_pv":
            tiles = pv_tiles
            mapping = "attention_pv:m=q,n=dim,k=kv"
        elif op.family == "attention_fused":
            # A fused kernel performs both matmuls in sequence; its cube work is
            # the union of both tile sets.
            tiles = qk_tiles + pv_tiles
            mapping = "attention_fused:qk_tiles+pv_tiles"
        else:
            tiles = qk_tiles
            mapping = "attention_qk:m=q,n=kv,k=dim"
    else:
        if len(shape) >= 3:
            m, n, k = shape[-3], shape[-2], shape[-1]
        elif len(shape) == 2:
            m, n, k = shape[0], shape[1], 1
        else:
            m, n, k = (max(1, shape[0]) if shape else 1), 1, 1
        tiles = (
            max(1, math.ceil(max(1, m) / _TILE_M))
            * max(1, math.ceil(max(1, n) / _TILE_N))
        )
    cores = max(1, int(aic_count or 0))
    waves = max(1, math.ceil(tiles / cores))
    wave_eff = min(1.0, tiles / (waves * cores))
    return wave_eff, {
        "tiles": float(tiles), "waves": float(waves),
        "wave_efficiency": wave_eff,
        # Kept as a named diagnostic for cost records that already carry it;
        # the pipeline-fill multiplier no longer exists.
        "k_pipeline_efficiency": 1.0,
        "contraction_mapping": mapping,
    }


def _shape_efficiency(op: OpSpec, machine: MachineSpec) -> tuple[float, float, dict[str, Any]]:
    # The AIC core count is a schedule parameter, not a hidden efficiency.  It
    # comes from the machine; 24 is only the explicit legacy fallback for a
    # fixture that declares no AIC count (probe-verified baseline-neutral for
    # such fixtures, and it changes A2's 20-AIC decode waves exactly as the
    # declared silicon requires).
    if op.flops > 0 and op.family not in COMM_FAMILIES:
        die = machine.dies[0] if machine.dies else None
        declared = int(getattr(die, "aic_count", 0) or 0) if die is not None else 0
        eff, detail = _gemm_efficiency(
            op, declared if declared > 0 else _LEGACY_AIC_REFERENCE,
        )
        detail["aic_count"] = float(declared if declared > 0 else _LEGACY_AIC_REFERENCE)
        detail["aic_count_source"] = (
            "die.aic_count" if declared > 0 else "legacy_fallback_24"
        )
        return eff, 1.0, detail
    if op.family in COMM_FAMILIES:
        return 1.0, 1.0, {"comm": True}
    # The previous non-GEMM branch here returned ``0.5 + 0.5*ops/4096`` and
    # ``0.65 + 0.35*bytes/2MiB``; both saturate at 1.0 for every production op
    # (a single rmsnorm of an 8k prefill is >3e7 element-ops, every weight read
    # is >2 MiB), so the branch was fake physics -- and unreachable from the
    # only call site anyway.  Deleted rather than re-invented.
    return 1.0, 1.0, {
        "vector_efficiency": 1.0, "memory_efficiency": 1.0,
        "deleted_terms": "saturating_vector_and_memory_multipliers",
    }


def _elapsed_ms(demands: Mapping[str, float], capacity: Mapping[str, float]) -> float:
    """Additive per-op elapsed time from an explicit demand/capacity map.

    ``launch`` is a dispatch gap that precedes kernel service, so it is added
    to the busiest resource's service time instead of competing with it in a
    ``max``.  HBM/MTE/Cube/Vector/link all run concurrently inside one kernel,
    so those remain a ``max``.  A whole-span completion envelope is also a
    duration, not a resource.
    """
    launch = float(demands.get("launch", 0.0) or 0.0)
    envelope = float(demands.get("completion_envelope_ms", 0.0) or 0.0)
    service = max(
        (
            float(work) / max(1e-9, float(capacity[key]))
            for key, work in demands.items()
            if key not in {"launch", "completion_envelope_ms"}
        ),
        default=0.0,
    )
    return launch + max(service, envelope)


#: Placement token of a communication *fabric*, keyed by the protocol the
#: machine declares for that transfer channel.  A measured comm curve belongs to
#: one fabric identity, so the token may only be derived from a declared
#: protocol -- never from a substring of a link name.
COMM_PLACEMENT_BY_PROTOCOL: dict[str, str] = {
    "SIO": "sio",
    "HCCS": "hccs",
    "HCCS_SW": "hccssw",
    "HCCS_SWITCH": "hccssw",
}
#: Channels whose name already *is* the fabric identity (a die pair is only ever
#: SIO, an inter-node channel is only RoCE).  ``d2d_intra_node`` is deliberately
#: absent: it is HCCS on A2/A3 and UB-Mesh on A5, so the channel name alone is
#: not evidence of a fabric.
COMM_PLACEMENT_BY_CHANNEL: dict[str, str] = {
    "d2d_intra_package": "sio",
    "d2d_inter_node": "roce",
}


def communication_placement_for_link(link: Any) -> str:
    """Placement token of the fabric a resolved link actually is.

    The token is a physical fabric, so it comes from the declared
    channel/protocol, not from a link name (audit B6).  A protocol this table
    does not know -- e.g. an A5 ``UB_MESH`` intra-node fabric -- returns ``""``
    (explicit unmodeled) instead of being mapped onto the A2/A3 HCCS placement
    and priced with HCCS-measured curves (a5-audit B-2).
    """
    if link is None:
        return ""
    protocol = str(getattr(link, "protocol", "") or "").strip().upper()
    if protocol in COMM_PLACEMENT_BY_PROTOCOL:
        return COMM_PLACEMENT_BY_PROTOCOL[protocol]
    if protocol.startswith("ROCE"):
        # RoCEv2_400G / RoCEv2_200G are declared variants of one fabric family.
        return "roce"
    if protocol:
        # The machine declared a fabric with no measured placement token.
        return ""
    channel = str(getattr(link, "channel", "") or "").strip().lower()
    return COMM_PLACEMENT_BY_CHANNEL.get(channel, "")


def communication_placement_for_machine(
    machine: MachineSpec, *, link: Any = None, ranks: Sequence[int] | None = None,
) -> str:
    """Derive the collective placement from the machine's declared channels.

    ``machine.transfer_channels`` is the declaration this module trusts: when it
    carries a channel matching the resolved link, that channel's declared
    protocol decides the fabric token.  An explicit fixture that declares only
    links still has its own protocol/``channel``.  An unresolvable topology
    returns ``""`` so the comm calibration is explicitly unmodeled instead of
    guessing a placement (and therefore another fabric's measured curve).
    """
    resolved = link
    if resolved is None and ranks is not None:
        from .a3_resources import select_link_for_ranks

        resolved = select_link_for_ranks(machine, ranks)
    if resolved is None:
        return ""
    declared = machine.channel(getattr(resolved, "channel", "") or "")
    protocol = str(getattr(declared, "protocol", "") or "").strip().upper()
    if protocol in COMM_PLACEMENT_BY_PROTOCOL:
        return COMM_PLACEMENT_BY_PROTOCOL[protocol]
    if protocol.startswith("ROCE"):
        return "roce"
    if protocol:
        return ""
    return communication_placement_for_link(resolved)


# ---------------------------------------------------------------------------
# Fitted physical calibration (``a3_physical_calibration.py``)
# ---------------------------------------------------------------------------
#: The fitted physical layer answered for this op: ``estimated_ms`` is the
#: artifact's duration (fitted launch intercept + fitted roofline), not a
#: multiplicative correction of the analytic formula.
PHYSICAL_CALIBRATED = "physical_calibrated"
#: The artifact has a row but flags its own parameters as unidentified
#: (``quality`` tells which); the analytic/curve path stays primary and the
#: fitted number is reported for diagnosis only.
PHYSICAL_LOW_CONFIDENCE = "physical_low_confidence"
#: The artifact has no row for this identity (unknown family / recipe / timing
#: scope / dtype / hardware key).  The analytic/curve path stays primary and the
#: artifact's own reason is carried through.
PHYSICAL_UNMODELED = "physical_unmodeled"

#: Quality labels that mean "the number exists but its parameters are not
#: identified by the data".
PHYSICAL_UNIDENTIFIED_QUALITIES = frozenset({"low_confidence", "transferred"})
#: Families whose unidentified *rates* must not replace the analytic estimate.
#: ``add``/``copy`` are noise-dominated in the corpora that exist (only the
#: intercept is real, and the artifact itself says so), so their fitted number is
#: reported but the analytic/curve path stays primary.
PHYSICAL_LOW_CONFIDENCE_ANALYTIC_FAMILIES: frozenset[str] = frozenset(
    {"add", "copy", "copy_bytes"}
)
#: Family-level opt-out from the fitted layer, used when a family's own fit
#: quality is acceptable but the analytic/curve path is the declared primary
#: (empty today).
PHYSICAL_ANALYTIC_PREFERRED_FAMILIES: frozenset[str] = frozenset()

# ---------------------------------------------------------------------------
# Declared physical transfer map
# ---------------------------------------------------------------------------
#: The artifact keys its rows by the *measurement* family/recipe
#: (``gemm/torch_npu_managed_runner``, ``rmsnorm/npu_rms_norm``,
#: ``softmax/torch_softmax``), while the simulated graph names kernel recipes
#: (``qkv_projection/prefill_causal:qkv``).  A projection GEMM is the same
#: physical kernel as the measured ``gemm`` family, so a cost-model family with
#: no row of its own may be priced by a *declared* transfer target.  The transfer
#: is always recorded on the op (``physical.transfer_from``) and counted in the
#: provenance; it is never silent, and families with no fitted model are absent
#: on purpose (``attention_qk``/``attention_pv``/``attention_fused`` are
#: attention kernels, ``embedding``/``sampling``/``swiglu`` have no fit).
#:
#: Recipes are selected by the declared timing scope: a ``batched`` scope lists
#: the batched-protocol recipes first, everything else the per-call recipes first.
#: Each entry is an **ordered candidate list**, because the two SKUs do not name
#: the same measured recipe for the same physical family: A2's vector rows are
#: ``elementwise_add_reference`` / ``npu_swiglu`` / ``npu_dynamic_quant``, A3's
#: are ``torch_npu_managed_runner``.  The cost model asks the artifact for the
#: candidates in order and the first fitted row wins; the recipe is a protocol
#: identity, so the order is declared here and never guessed per shape.
PHYSICAL_TRANSFER_TARGETS: dict[str, dict[str, tuple[str, ...]]] = {
    "gemm": {
        "batched": ("torch_mm_preallocated", "torch_npu_managed_runner"),
        "per_call": ("torch_npu_managed_runner", "torch_mm_preallocated"),
    },
    "rmsnorm": {"batched": ("npu_rms_norm",), "per_call": ("npu_rms_norm",)},
    "attention_softmax": {"batched": ("torch_softmax",), "per_call": ("torch_softmax",)},
    "add": {
        "batched": ("elementwise_add_reference", "torch_npu_managed_runner"),
        "per_call": ("elementwise_add_reference", "torch_npu_managed_runner"),
    },
    "copy": {
        "batched": ("copy_rows_reference", "torch_npu_managed_runner"),
        "per_call": ("torch_npu_managed_runner", "copy_rows_reference"),
    },
    "swiglu": {"batched": ("npu_swiglu",), "per_call": ("npu_swiglu",)},
    "quantize": {
        "batched": ("npu_dynamic_quant",), "per_call": ("npu_dynamic_quant",),
    },
}
#: Artifact family each transfer target belongs to (``attention_softmax`` is the
#: measured ``softmax`` family; every GEMM projection family is ``gemm``).
PHYSICAL_TRANSFER_ARTIFACT_FAMILY: dict[str, str] = {
    "gemm": "gemm",
    "rmsnorm": "rmsnorm",
    "attention_softmax": "softmax",
    "add": "add",
    "copy": "copy",
    "swiglu": "swiglu",
    "quantize": "quantize",
}
#: Cost-model aliases that are the same physical family.
PHYSICAL_TRANSFER_ALIASES: dict[str, str] = {"copy_bytes": "copy"}


def physical_transfer_target(
    family: str, timing_scope: str,
) -> tuple[tuple[str, str], ...] | None:
    """Ordered ``(artifact family, artifact recipe)`` candidates for a family.

    ``None`` means "no fitted model exists for this family": attention
    (``attention_qk``/``attention_pv``/``attention_fused`` is not a GEMM),
    ``embedding``, ``sampling`` and the communication families stay unmodeled with
    the artifact's own reason.  The first candidate is the recipe the declared
    timing scope names; the rest are the other SKU's declared spelling of the same
    physical family, tried in order.
    """
    key = PHYSICAL_TRANSFER_ALIASES.get(str(family or ""), str(family or ""))
    if key in GEMM_TRANSFER_FAMILIES:
        key = "gemm"
    target = PHYSICAL_TRANSFER_TARGETS.get(key)
    if target is None:
        return None
    scope = str(timing_scope or "").lower()
    recipes = target["batched"] if "batched" in scope else target["per_call"]
    artifact_family = PHYSICAL_TRANSFER_ARTIFACT_FAMILY.get(key, key)
    return tuple((artifact_family, recipe) for recipe in recipes)


def physical_prediction_policy(explained: Any) -> tuple[str, str]:
    """Decide whether a fitted physical prediction replaces the analytic path.

    One rule, one place, so the policy can be changed without touching
    ``cost()``:

    * the artifact refused the op (unknown family/recipe/scope/dtype/key) ->
      ``physical_unmodeled``; the analytic/curve path is primary and the
      artifact's reason is reported verbatim;
    * the fitted row exists but its parameters are labelled unidentified
      (:data:`PHYSICAL_UNIDENTIFIED_QUALITIES`): a *GEMM* row is still the only
      fitted GEMM model and is physically fitted (wave-quantized window,
      refit-per-fold CV), so it wins and reports its quality; an ``add``/``copy``
      row
      (:data:`PHYSICAL_LOW_CONFIDENCE_ANALYTIC_FAMILIES`) reports
      ``physical_low_confidence`` and the analytic/curve path stays primary,
      because the artifact says its rates are unidentified;
    * otherwise the fitted duration wins (``physical_calibrated``).

    Returns ``(status, detail)`` where ``detail`` is the reported reason.
    """
    if not getattr(explained, "modeled", False):
        return PHYSICAL_UNMODELED, str(getattr(explained, "reason", "") or "unmodeled")
    parameters = dict(getattr(explained, "parameters", {}) or {})
    quality = str(parameters.get("quality") or "")
    family = str(getattr(explained, "family", "") or "")
    if quality in PHYSICAL_UNIDENTIFIED_QUALITIES:
        if family in PHYSICAL_LOW_CONFIDENCE_ANALYTIC_FAMILIES:
            return PHYSICAL_LOW_CONFIDENCE, (
                f"physical_parameters_{quality}_unidentified_rates;"
                "analytic_curve_path_retained"
            )
        return PHYSICAL_CALIBRATED, (
            f"{getattr(explained, 'reason', 'physical_calibration')};"
            f"quality={quality};fitted_row_applied"
        )
    if family in PHYSICAL_ANALYTIC_PREFERRED_FAMILIES:
        return PHYSICAL_LOW_CONFIDENCE, (
            f"family_{family}_declared_analytic_primary;analytic_curve_path_retained"
        )
    return PHYSICAL_CALIBRATED, str(
        getattr(explained, "reason", "") or "physical_calibration"
    )


def physical_calibration_provenance(
    calibration: Any, *, timing_scope: str = "",
) -> dict[str, Any]:
    """Public provenance block for a loaded physical-calibration artifact.

    Additive record: the artifact's hardware key, schema version, model kind and
    fitted training sample count, plus the effective timing scope.  It is
    reported even when no artifact is supplied (``status = not_configured``) so
    "no physical layer" is visible instead of absent.
    """
    if calibration is None:
        return {
            "status": "not_configured",
            "hardware_key": "",
            "schema_version": "",
            "model_kind": "",
            "training_sample_count": 0,
            "timing_scope": timing_scope,
            "families": [],
        }
    families = dict(getattr(calibration, "families", {}) or {})
    rows = []
    for key, parameters in sorted(families.items()):
        rows.append({
            "key": key,
            "family": parameters.family,
            "recipe": parameters.recipe,
            "timing_scope": parameters.timing_scope,
            "model_kind": parameters.model_kind,
            "model_form": parameters.model_form,
            "fit_level": parameters.fit_level,
            "quality": parameters.quality,
            "samples": parameters.samples,
            "train_wape_pct": parameters.train_wape_pct,
            "cv_wape_pct": parameters.cv_wape_pct,
        })
    provenance = dict(getattr(calibration, "provenance", {}) or {})
    return {
        "status": "configured",
        "hardware_key": str(getattr(calibration, "hardware_key", "") or ""),
        "schema_version": str(getattr(calibration, "schema_version", "") or ""),
        "model_kind": "fitted_physical_roofline",
        "arch_family": str(getattr(calibration, "arch_family", "") or ""),
        "timing_scope": timing_scope,
        "training_sample_count": sum(int(row["samples"] or 0) for row in rows),
        "condition_count": int(provenance.get("condition_count", 0) or 0),
        "raw_event_count": int(provenance.get("raw_event_count", 0) or 0),
        "fitted_utc": str(provenance.get("fitted_utc", "") or ""),
        "training_hash": str(provenance.get("training_hash", "") or ""),
        "heldout_guard_status": str(
            dict(provenance.get("heldout_guard") or {}).get("status", "") or ""
        ),
        "families": rows,
    }


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
    #: Fraction of the per-op dispatch intercept charged to one op.  1.0 is
    #: eager per-op submission.  Graph replay submits a whole bucket at once,
    #: so a caller that knows the bucket can declare the amortized fraction
    #: here; this module does not guess it.
    launch_amortization: float = 1.0
    #: Fitted physical-calibration layer (``a3_physical_calibration.py``): a
    #: ``PhysicalCalibration`` instance, or ``None`` for the analytic/curve path.
    #: This module duck-types the object (``predict_op_explained``) so importing
    #: the cost model never drags in the fitter.
    physical_calibration: Any = None
    #: Default measurement recipe for the physical lookup when an op does not
    #: declare ``physical_recipe`` in its metadata.  The artifact keys its fits by
    #: (family, recipe, timing scope); a caller that measured with one runner
    #: recipe declares it once here instead of per op.
    physical_recipe: str = ""
    #: Allow the declared family transfer
    #: (:data:`PHYSICAL_TRANSFER_TARGETS`) when the artifact has no row for the
    #: op's own family/recipe.  The transfer is always recorded on the op and
    #: counted in the provenance; set ``False`` to demand an exact row.
    physical_allow_family_transfer: bool = True

    def calibration_features(self) -> dict[str, str]:
        return {
            "weight_dtype": self.weight_dtype or self.compute_dtype,
            "activation_dtype": self.activation_dtype or self.compute_dtype,
            "compute_dtype": self.compute_dtype,
            "accumulation_dtype": self.accumulation_dtype,
            "quantization": _physical_quantization(self.quantization),
        }

    def op_link_scope(self, op: OpSpec) -> str:
        """The fabric token this op's traffic uses.

        A collective's declared placement is its fabric (a measured SIO
        all-reduce ran on the SIO pair, not on the switch).  Otherwise an op
        that names a real channel/scope wins; the generic ``d2d`` token means
        "the model's default fabric".
        """
        if op.family in COMM_FAMILIES and op.link_bytes > 0:
            placement = str(self.communication_placement or "").strip()
            if placement:
                return placement
        declared = str(op.link_scope or "").strip()
        if declared and declared.lower() not in {"d2d", "default", "shared"}:
            return declared
        return str(self.link_scope or "d2d").strip()

    def _link(self, op: OpSpec | None = None) -> Any:
        if not self.machine.links:
            return None
        return self.machine.link_for_scope(self.op_link_scope(op))

    def launch_ms(self, op: OpSpec, link: Any = None) -> float:
        """Explicit additive dispatch term for one op, in milliseconds.

        ``die.launch_us`` is the declared dispatch intercept (the physical
        calibration layer will later replace it with a fitted value per
        hardware key).  A link op also pays the fabric's fixed setup latency,
        which is part of its dispatch, not of its steady-state bandwidth.
        """
        die = self.machine.dies[0] if self.machine.dies else None
        dispatch = (
            float(getattr(die, "launch_us", 0.0) or 0.0) / 1000.0
            * max(0.0, float(self.launch_amortization))
        )
        if op.link_bytes > 0 and link is not None:
            dispatch += float(link.latency_us) / 1000.0
        return dispatch

    def machine_key(self) -> str:
        # A real measurement is keyed by the hardware identity, not by the
        # simulator's configured device count.  An explicit fixture name/key is
        # usable directly.
        key = str(
            self.machine.hardware_key or self.machine.sku or self.machine.name
        ).strip()
        if not key:
            raise A3InputError("machine_identity_missing")
        return key

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

    # -- fitted physical calibration --------------------------------------
    def physical_query(self, op: OpSpec) -> dict[str, Any] | None:
        """The declared identity/shape the fitted physical layer is asked for.

        Only fields this module can state are passed: the family and recipe the
        caller declared (``physical_recipe`` metadata, else the model default,
        else ``op.recipe``), the model's timing scope (a physical protocol
        identity, never inferred), and the shape/work vector.  An unknown family
        stays unknown -- ``attention_qk`` is never relabelled ``gemm``.
        """
        if self.physical_calibration is None:
            return None
        meta = dict(op.metadata)
        features = self.shape_features(op)
        shape: dict[str, float] = {}
        for key in ("m", "n", "k"):
            if key in features:
                shape[key] = float(features[key])
        declared_shape = meta.get("physical_shape")
        if isinstance(declared_shape, Mapping):
            shape.update({str(key): float(value) for key, value in declared_shape.items()})
        elements = float(features.get("elements") or 0.0)
        if elements <= 0 and op.vector_ops > 0:
            elements = float(op.vector_ops)
        recipe = str(
            meta.get("physical_recipe") or self.physical_recipe or op.recipe or ""
        )
        return {
            "family": op.family,
            "recipe": recipe,
            "timing_scope": str(self.timing_scope or ""),
            "protocol_signature": str(meta.get("physical_protocol_signature") or ""),
            "shape": shape,
            "elements": elements,
            "flops": float(op.flops),
            "hbm_bytes": float(op.hbm_bytes),
            "dtype": op.dtype or self.compute_dtype,
        }

    def _physical_ask(self, query: Mapping[str, Any], op: OpSpec) -> Any:
        return self.physical_calibration.predict_op_explained(
            query, {"hardware_key": self.machine.hardware_key},
            compute_dtype=op.dtype or self.compute_dtype,
        )

    def physical_explained(
        self, op: OpSpec,
    ) -> tuple[dict[str, Any] | None, Any, dict[str, Any] | None, dict[str, Any] | None]:
        """Ask the artifact for this op, then for a declared transfer target.

        Returns ``(query, explained, transfer, refused_transfer)``:

        * ``query``/``explained`` are the op's own identity answer;
        * ``transfer`` is set when the own-family answer was unmodeled and the
          declared map (:data:`PHYSICAL_TRANSFER_TARGETS`) had a fitted row --
          ``explained``/``query`` are then the transfer target's;
        * ``refused_transfer`` records an attempted target that the artifact also
          refused, so a "why is this not modeled" answer names both attempts.

        ``(None, None, None, None)`` when no artifact is configured.
        """
        query = self.physical_query(op)
        if query is None:
            return None, None, None, None
        explained = self._physical_ask(query, op)
        if explained.modeled:
            return query, explained, None, None
        if not self.physical_allow_family_transfer:
            return query, explained, None, None
        candidates = physical_transfer_target(op.family, self.timing_scope)
        if not candidates:
            return query, explained, None, None
        origin = {
            "from_family": str(query.get("family") or ""),
            "from_recipe": str(query.get("recipe") or ""),
        }
        attempts: list[dict[str, Any]] = []
        for target_family, target_recipe in candidates:
            transfer_query = {
                **query, "family": target_family, "recipe": target_recipe,
            }
            transferred = self._physical_ask(transfer_query, op)
            if transferred.modeled:
                return transfer_query, transferred, {
                    **origin,
                    "to_family": target_family,
                    "to_recipe": target_recipe,
                }, None
            attempts.append({
                "family": target_family,
                "recipe": target_recipe,
                "reason": str(transferred.reason or ""),
            })
        return query, explained, None, {**origin, "attempts": attempts}

    def _physical_record(
        self, query: Mapping[str, Any], explained: Any, status: str, detail: str,
        *, applied: bool, superseded: str = "",
        transfer: Mapping[str, Any] | None = None,
        refused_transfer: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = dict(explained.to_dict())
        record["status"] = status
        record["artifact_status"] = str(explained.status or "")
        record["detail"] = detail
        record["applied"] = applied
        record["hardware_key"] = str(self.machine.hardware_key or "")
        record["query"] = {
            key: query.get(key)
            for key in (
                "family", "recipe", "timing_scope", "protocol_signature", "shape",
            )
        }
        if transfer is not None:
            # A transferred row is not the op's own family: say so on the op.
            record["transfer_from"] = {
                "family": transfer["from_family"],
                "recipe": transfer["from_recipe"],
            }
            record["transfer_to"] = {
                "family": transfer["to_family"],
                "recipe": transfer["to_recipe"],
            }
        if refused_transfer is not None:
            record["refused_transfer"] = dict(refused_transfer)
        if superseded:
            record["superseded_calibration_status"] = superseded
        if status == PHYSICAL_UNMODELED:
            reason = str(explained.reason or "")
            if reason == "unmodeled_recipe_or_protocol":
                families = dict(getattr(self.physical_calibration, "families", {}) or {})
                rows = [
                    item for item in families.values()
                    if item.family == str(explained.family)
                ]
                record["available_recipes_for_family"] = sorted({
                    item.recipe for item in rows
                })
                # A scope mismatch is the same failure class as a recipe
                # mismatch, and the two intercepts are never interchangeable:
                # name the scopes the artifact does model for this family.
                record["available_timing_scopes_for_family"] = sorted({
                    item.timing_scope for item in rows
                })
        return record

    def _apply_physical(
        self, explained: Any, baseline: Mapping[str, float],
        capacity: Mapping[str, float],
    ) -> dict[str, float]:
        """Analytic busy-resource work re-priced to the fitted duration.

        The artifact owns the *duration* (fitted intercept + fitted roofline),
        so the fitted intercept replaces the declared dispatch term and the
        analytic service demands are scaled to the fitted service time.  The
        analytic roofline still decides the relative busy-resource shares inside
        the service part, which is what the engine schedules; the fitted
        breakdown itself is reported on ``CostedOp.physical``.  Launch is never
        added twice: the fitted intercept is the launch term, it is not an extra
        term on top of ``explained.duration_ms``.
        """
        fitted_ms = float(explained.duration_ms or 0.0)
        breakdown = dict(getattr(explained, "breakdown", {}) or {})
        fitted_launch = float(breakdown.get("launch_ms") or 0.0)
        analytic_launch = float(baseline.get("launch", 0.0) or 0.0)
        analytic_service = max(
            (
                float(work) / max(1e-9, float(capacity[key]))
                for key, work in baseline.items()
                if key not in {"launch", "completion_envelope_ms"}
            ),
            default=0.0,
        )
        fitted_service = max(0.0, fitted_ms - fitted_launch)
        scale = fitted_service / analytic_service if analytic_service > 0 else 0.0
        applied = {
            key: float(work) * scale
            for key, work in baseline.items()
            if key not in {"launch", "completion_envelope_ms"}
        }
        if fitted_launch > 0 or analytic_launch > 0:
            applied["launch"] = fitted_launch
        return applied

    def physical_provenance(self) -> dict[str, Any]:
        return physical_calibration_provenance(
            self.physical_calibration, timing_scope=str(self.timing_scope or ""),
        )

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
        link = self._link(op)
        if op.link_bytes > 0 and link is None:
            raise A3InputError(
                f"machine_link_missing:{op.op_id}:{self.op_link_scope(op)}"
            )
        factor_status = self.calibration_for(op)
        factor = factor_status.factor
        typed_stage_ms = 0.0
        # Compute uncalibrated and calibrated demands from the same shape model.
        # Every demand is raw work: the resource ceiling in ``capacity`` is the
        # only bound.  The former multiplicative floors (0.02 cube, 0.05 vector,
        # 0.65/0.5 memory and MTE) were invented saturating discounts -- 1.0 for
        # every production shape -- and are deleted instead of re-invented.
        def build(scale: float) -> dict[str, float]:
            demands: dict[str, float] = {}
            if op.flops > 0 and op.family not in COMM_FAMILIES:
                eff, _, _ = _shape_efficiency(op, self.machine)
                demands["cube"] = op.flops * scale / max(1e-9, eff)
            if op.vector_ops > 0:
                demands["vector"] = op.vector_ops * scale
            if op.hbm_bytes > 0:
                demands["hbm"] = op.hbm_bytes * scale
            if op.mte2_bytes > 0:
                demands["mte2"] = op.mte2_bytes * scale
            if op.mte3_bytes > 0:
                demands["mte3"] = op.mte3_bytes * scale
            if op.link_bytes > 0:
                demands["link"] = op.link_bytes * scale
            dispatch_ms = self.launch_ms(op, link)
            if dispatch_ms > 0 and op.family not in {"pp_p2p"}:
                # A measured residual covers the whole kernel duration, which
                # is dispatch + service.  The term is additive in ``estimated``
                # and stays explicit in the record for physical calibration.
                demands["launch"] = dispatch_ms * scale
            return demands

        capacity = {
            "cube": self.machine.peak_flops_for(self.compute_dtype) / 1000.0,
            "vector": die.vector_ops_per_s / 1000.0,
            "hbm": die.hbm_bw_bytes_per_s / 1000.0,
            "mte2": die.mte_bw_bytes_per_s / 1000.0,
            "mte3": die.mte_bw_bytes_per_s / 1000.0,
            "link": (link.capacity_bytes_per_ms if link is not None else 1.0),
            "launch": 1.0, "cpu": 1.0, "serial": 1.0, "aicpu": 1.0,
        }
        baseline = build(1.0)
        calibrated = build(factor)
        calibration_status = factor_status
        # Integrate measured communication curves through the primitive
        # transport semantics, not blind family-key equality.
        if (self.communication_calibration is not None and op.link_bytes > 0
                and op.family in COMM_FAMILIES):
            mode = ("all_reduce_only" if op.family == "allreduce"
                    else "p2p_only")
            placement = self.communication_placement
            if not placement and link is not None:
                # A collective's fabric is the machine's declared transfer
                # channel (SIO pair vs HCCS switch vs RoCE), never a substring of
                # a link name and never the generic ``d2d`` token.
                placement = communication_placement_for_machine(
                    self.machine, link=link,
                )
                if not placement:
                    # Unspecified topology stays unmodeled, explicitly: the
                    # analytic link estimate remains, but no measured curve of
                    # another fabric is applied to it.
                    calibration_status = CalibrationResult(
                        "uncalibrated_comm_placement", 1.0, 0, None, False, False,
                        "no_measured_comm_curve",
                        reason=(
                            "comm_placement_unresolved:"
                            f"{getattr(link, 'protocol', '') or getattr(link, 'channel', '')}"
                        ),
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
        analytic_ms = _elapsed_ms(baseline, capacity)
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
        launch_ms = float(calibrated.get("launch", 0.0))
        service_ms = max(
            (
                work / max(1e-9, capacity[key])
                for key, work in calibrated.items()
                if key not in {"launch", "completion_envelope_ms"}
            ),
            default=0.0,
        )
        # Fitted physical calibration: the artifact owns the duration when it
        # models this identity.  A measured whole-span envelope (comm) already
        # *is* a duration, so it wins and the fitted layer reports why it did not
        # apply instead of silently replacing a measurement.
        physical_record: dict[str, Any] | None = None
        (
            physical_query, physical_explained, physical_transfer,
            physical_refused_transfer,
        ) = self.physical_explained(op)
        if physical_explained is not None:
            decision, detail = physical_prediction_policy(physical_explained)
            applied = False
            if decision == PHYSICAL_CALIBRATED and not calibrated.get(
                "completion_envelope_ms"
            ):
                superseded = str(calibration_status.status)
                calibrated = self._apply_physical(physical_explained, baseline, capacity)
                physical_parameters = dict(physical_explained.parameters or {})
                transfer_note = ""
                if physical_transfer is not None:
                    transfer_note = (
                        ";transfer_from="
                        f"{physical_transfer['from_family']}/"
                        f"{physical_transfer['from_recipe']}"
                    )
                calibration_status = CalibrationResult(
                    PHYSICAL_CALIBRATED, 1.0,
                    int(physical_parameters.get("samples") or 0), None, False, False,
                    (
                        "physical_train_only_fitted_roofline"
                        if physical_transfer is None else
                        "physical_train_only_fitted_roofline_family_transfer"
                    ),
                    reason=(
                        f"{detail};quality={physical_parameters.get('quality', '')};"
                        f"fit_level={physical_explained.fit_level};"
                        f"model_form={physical_explained.model_form};"
                        f"supersedes={superseded}{transfer_note}"
                    ),
                )
                applied = True
                launch_ms = float(calibrated.get("launch", 0.0))
                service_ms = max(
                    (
                        work / max(1e-9, capacity[key])
                        for key, work in calibrated.items()
                        if key not in {"launch", "completion_envelope_ms"}
                    ),
                    default=0.0,
                )
            elif decision == PHYSICAL_CALIBRATED:
                decision, detail = PHYSICAL_UNMODELED, (
                    "physical_not_applied_measured_span_envelope_wins"
                )
            physical_record = self._physical_record(
                physical_query or {}, physical_explained, decision, detail,
                applied=applied,
                superseded=str(calibration_status.status)
                if not applied else "",
                transfer=physical_transfer, refused_transfer=physical_refused_transfer,
            )
        # Additive composition: a kernel cannot start before its dispatch and
        # cannot finish before its service.  The engine may still overlap two
        # ops' dispatch and service; this per-op estimate is the serial bound.
        envelope_ms = float(calibrated.get("completion_envelope_ms", 0.0))
        service_ms = max(service_ms, typed_stage_ms)
        if envelope_ms > 0:
            # An opaque measured whole span *is* a complete duration that
            # already contains dispatch; adding the intercept on top would
            # double-count it.
            estimated_ms = max(launch_ms + service_ms, envelope_ms)
        else:
            estimated_ms = launch_ms + service_ms
        return CostedOp(
            op=op, demands=tuple(sorted(calibrated.items())),
            analytic_ms=analytic_ms, estimated_ms=estimated_ms,
            calibration=calibration_status, coverage="decomposed",
            curve_result=curve,
            launch_ms=launch_ms, service_ms=service_ms,
            physical=physical_record,
        )

    def cost_iteration(self, work: IterationWork) -> list[CostedOp]:
        if not work.ops:
            raise A3InputError("iteration_work_empty")
        return [self.cost(op) for op in work.ops]

    def hardware_lower_bound_ms(self, work: IterationWork) -> float:
        """Raw resource roofline for this exact DAG; not a normative target.

        Uses the same compute peak as ``cost()`` (``peak_flops_for`` honours a
        declared dtype peak; reading ``die.cube_flops_per_s`` here made the
        roofline and the cost model disagree for declared fp8 parts) and the
        per-op resolved link, so heterogeneous fabrics are bounded by the
        busiest wire rather than one arbitrary link.
        """
        die = self.machine.dies[0]
        flops = sum(op.flops for op in work.ops)
        vector = sum(op.vector_ops for op in work.ops)
        hbm = sum(op.hbm_bytes for op in work.ops)
        cube_peak = self.machine.peak_flops_for(self.compute_dtype) / 1000.0
        link_ms: dict[str, float] = {}
        for op in work.ops:
            if op.link_bytes <= 0:
                continue
            link = self._link(op)
            if link is None:
                raise A3InputError(
                    f"machine_link_missing:{op.op_id}:{self.op_link_scope(op)}"
                )
            link_ms[link.name] = link_ms.get(link.name, 0.0) + (
                op.link_bytes / link.capacity_bytes_per_ms
            )
        capacities = [
            flops / cube_peak if flops else 0.0,
            vector / (die.vector_ops_per_s / 1000.0) if vector else 0.0,
            hbm / (die.hbm_bw_bytes_per_s / 1000.0) if hbm else 0.0,
            max(link_ms.values()) if link_ms else 0.0,
        ]
        return max(capacities) if capacities else 0.0


__all__ = [
    "COMPUTE_FAMILIES", "VECTOR_FAMILIES", "COMM_FAMILIES",
    "gemm_flops", "attention_flops", "CalibrationObservation",
    "CalibrationResult", "CalibrationStore", "CostedOp", "PredictiveCostModel",
    "COMM_PLACEMENT_BY_PROTOCOL", "COMM_PLACEMENT_BY_CHANNEL",
    "communication_placement_for_link", "communication_placement_for_machine",
    "PHYSICAL_CALIBRATED", "PHYSICAL_LOW_CONFIDENCE", "PHYSICAL_UNMODELED",
    "physical_prediction_policy", "physical_calibration_provenance",
    "PHYSICAL_TRANSFER_TARGETS", "PHYSICAL_TRANSFER_ARTIFACT_FAMILY",
    "PHYSICAL_LOW_CONFIDENCE_ANALYTIC_FAMILIES", "physical_transfer_target",
]
