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
  * A graph family name is mapped onto a fitted family only through the declared
    transfer table, and the mapping is physical: ``elementwise`` is the ``add``
    kernel, ``router_softmax`` is the ``softmax`` kernel, and a projection named
    ``linear_*_projection`` is the measured ``gemm`` kernel.  An attention
    kernel, the grouped expert GEMM, an embedding and a sampling op are never
    written off as a dense GEMM.
  * MC2 is a fused kernel, so it is costed as one unit: its dispatch/combine
    volume is ``tokens*top_k*hidden*dtype_bytes`` per replica, scaled by the EP
    group factor ``(ep-1)/ep``; its compute is the grouped-matmul duration (the
    fitted row when the artifact has one, else the analytic roofline), and its
    duration is ``launch + compute + comm - overlap*min(compute, comm)`` with an
    explicit overlap fraction -- ``0`` is the serial bound and ``1`` is
    ``launch + max(compute, comm)``.  Both bounds are reported, and an unverified
    overlap keeps the serial bound as the primary estimate instead of being
    replaced by a midpoint.  The dense-GEMM transfer is never applied to MC2 on
    top of that.
  * A grouped matmul pays one dispatch intercept per ``tensorList`` launch: the
    list API caps a launch at :data:`GROUPED_LAUNCH_EXPERT_CAP` experts, so a
    256- or 384-expert layer is two or three launches, never one.
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
    # Gated/linear attention blocks name their projections ``linear_*``.  They
    # are the measured GEMM kernel when the op declares GEMM work
    # (:data:`PHYSICAL_GEMM_SHAPED_ONLY` keeps the vector gates out).
    "linear_q_projection", "linear_k_projection", "linear_v_projection",
    "linear_o_projection", "linear_gate", "linear_output_gate",
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
    #: MC2 fused-kernel record for an op that belongs to an MC2 chunk: the
    #: dispatch/combine bytes and lane, the grouped-matmul compute term, the
    #: serial/overlapped bounds and the overlap provenance.  ``None`` for every
    #: op that is not part of an MC2 chunk.
    mc2: Mapping[str, Any] | None = None
    #: Why the dispatch term is more than one launch (the grouped-matmul
    #: tensor-list expert cap), or ``None`` for a single-launch op.
    launch_detail: Mapping[str, Any] | None = None
    #: The kernel's *service* work, before any launch folding.  ``demands`` is what
    #: the engine schedules (the dispatch intercept folded into the service window
    #: when the composition is serial); this is the same work without it, so a
    #: consumer that re-prices a fused kernel (MC2 chunk composition) scales from
    #: the service and not from an already-folded demand.
    service_demands: tuple[tuple[str, float], ...] = ()

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
            "mc2": dict(self.mc2) if self.mc2 else None,
            "launch_detail": (
                dict(self.launch_detail) if self.launch_detail else None
            ),
            "service_demands": dict(self.service_demands),
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

#: Wiring status of an op whose duration came from a fitted row that belongs to a
#: *different* artifact family (a declared transfer target).  It is a calibration
#: status only in the loose sense: the op's own identity was not measured, and the
#: wiring says so next to ``physical.status``.
PHYSICAL_TRANSFER = "physical_transfer"
#: Families whose duration dominates a large-model iteration.  Their wiring is
#: reported separately from the launch-band tail, because a large operator that
#: silently stays analytic changes the whole-network prediction while a small one
#: does not.  ``moe_gate_up``/``moe_down``/``grouped_matmul``/``mc2`` are the MoE
#: path; ``attention_*`` is the attention path; the projections are the GEMM path.
LARGE_OPERATOR_FAMILIES: frozenset[str] = frozenset({
    "mc2", "grouped_matmul", "moe_gate_up", "moe_down",
    "shared_expert_gate_up", "shared_expert_down",
    "attention_fused", "attention_qk", "attention_pv",
    "gemm", "qkv_projection", "o_projection", "gate_up_projection",
    "down_projection", "mla_q_a", "mla_kv_a", "mla_q_b", "mla_kv_b",
    "linear_q_projection", "linear_k_projection", "linear_v_projection",
    "linear_o_projection", "logits",
})
#: Operating-class token reported next to the wiring status, so a reader can tell
#: which large-operator family a status belongs to without a family table.
LARGE_OPERATOR_CLASSES: dict[str, str] = {
    "mc2": "mc2", "moe_gate_up": "grouped_matmul", "moe_down": "grouped_matmul",
    "grouped_matmul": "grouped_matmul", "shared_expert_gate_up": "grouped_matmul",
    "shared_expert_down": "grouped_matmul",
    "attention_fused": "attention", "attention_qk": "attention",
    "attention_pv": "attention",
}


def operator_class(family: str) -> str:
    """Declared operating class of a cost-model family."""
    key = str(family or "")
    if key in LARGE_OPERATOR_CLASSES:
        return LARGE_OPERATOR_CLASSES[key]
    if key in GEMM_TRANSFER_FAMILIES or key in {"logits"}:
        return "gemm"
    if key in {"allreduce", "ep_dispatch", "ep_combine", "pp_p2p", "kv_transfer",
               "dcp_allgather"}:
        return "communication"
    return "vector"


def physical_wiring(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """Large-operator physical/wiring status of one costed op.

    The vocabulary is closed and small so a trace reader (or a test) can assert
    it: ``physical_calibrated`` (the artifact's own row ran), ``physical_transfer``
    (a declared sibling row ran), ``mc2_overlap_unverified`` (the op is an MC2
    chunk member whose fused overlap has no measured verification),
    ``mc2_overlap_calibrated`` (a fitted chain factor on a measured lane) and
    ``unmodeled`` (everything else, including "no artifact configured" and "a
    fitted row exists but does not replace the estimate").  The MC2 token names
    are kept for provenance stability: ``*_unverified`` now means the fused
    chain's factor is a declared assumption, not a measured one.  The artifact's
    own status stays in ``physical_status`` for anyone who needs the finer label.
    """
    if not record:
        return {
            "status": PHYSICAL_UNMODELED, "physical_status": "",
            "reason": "no_physical_record",
        }
    status = str(record.get("status") or "")
    applied = bool(record.get("applied"))
    base = {
        "physical_status": status,
        "artifact_family": str(dict(record.get("query") or {}).get("family") or ""),
    }
    # A row reached through the declared timing-scope family is a protocol sibling,
    # not an exact measurement: name the hop on every branch that reports it.
    scope_alias = dict(record.get("scope_alias") or {})
    if scope_alias:
        base = {
            **base,
            "scope_alias_from": str(scope_alias.get("from") or ""),
            "scope_alias_to": str(scope_alias.get("to") or ""),
        }
    if applied and record.get("transfer_from"):
        return {
            **base,
            "status": PHYSICAL_TRANSFER,
            "reason": str(record.get("detail") or "declared_family_transfer"),
            "artifact_family": str(
                dict(record.get("transfer_to") or {}).get("family") or ""
            ),
            "artifact_recipe": str(
                dict(record.get("transfer_to") or {}).get("recipe") or ""
            ),
        }
    if applied and status == PHYSICAL_CALIBRATED:
        return {
            **base,
            "status": PHYSICAL_CALIBRATED,
            "reason": str(record.get("detail") or "fitted_row"),
        }
    attempts = [
        dict(row) for row in
        (record.get("refused_transfer") or {}).get("attempts", ()) or ()
    ]
    reason = str(record.get("reason") or record.get("detail") or status)
    outcome = {
        **base,
        "status": PHYSICAL_UNMODELED,
        "reason": reason,
    }
    if attempts:
        # The artifact's own refusal per attempted target: a mapping that exists
        # but finds no row (wrong scope / fp32 op against a bf16 fit) is a
        # different finding from a family that was never mapped at all.
        outcome["transfer_attempts"] = attempts
        outcome["reason"] = reason + ";transfer_attempts=" + ",".join(
            f"{row.get('family')}/{row.get('recipe') or '*'}:{row.get('reason')}"
            for row in attempts
        )
    return outcome


def op_family(query: Mapping[str, Any]) -> str:
    """Cost-model family named by a physical query (``""`` when absent)."""
    return str(query.get("family") or "")


def physical_transfer_eligible(op: Any) -> bool:
    """Whether a declared family transfer may price this op.

    A ``linear_gate``/``linear_output_gate`` op is a matrix projection in a gated
    attention block but a *vector* gate kernel in a linear-attention block; only
    the GEMM-shaped op may be priced by the measured dense GEMM.  Every other
    declared transfer target is shape-agnostic (an op with no ``m/n/k`` is refused
    by the artifact itself, with its own reason).
    """
    if str(getattr(op, "family", "") or "") not in PHYSICAL_GEMM_SHAPED_ONLY:
        return True
    shape = tuple(getattr(op, "launch_shape", ()) or ())
    return float(getattr(op, "flops", 0.0) or 0.0) > 0 and len(shape) >= 3



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
#: provenance; it is never silent.  Families with no fitted model are absent on
#: purpose (``attention_qk``/``attention_pv`` are half kernels, ``embedding``,
#: ``sampling`` and the collectives have no fit).
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
    # The grouped expert GEMM is its own measured kernel (``npu_grouped_matmul``
    # list API), not a stack of dense GEMMs.  The A3 heldout corpus measures it
    # both under the routed-family name and under the pooled ``grouped_matmul``
    # name, so both fitted spellings are declared candidates.
    "grouped_matmul": {
        "batched": ("npu_grouped_matmul_list_bf16",),
        "per_call": ("npu_grouped_matmul_list_bf16",),
    },
    "moe_gate_up": {
        "batched": ("npu_grouped_matmul_list_bf16",),
        "per_call": ("npu_grouped_matmul_list_bf16",),
    },
    "moe_down": {
        "batched": ("npu_grouped_matmul_list_bf16",),
        "per_call": ("npu_grouped_matmul_list_bf16",),
    },
    # A fused attention kernel is measured with ``npu_fusion_attention`` (which
    # is a whole QK+PV call).  Only the *fused* family may claim that row: the
    # separate QK and PV halves are never priced by a whole-kernel measurement.
    "attention_fused": {
        "batched": ("npu_fusion_attention",),
        "per_call": ("npu_fusion_attention",),
    },
    # The halves are their own fitted attention families.  They resolve to the
    # same-named artifact family only: a QK op is never answered by the PV or the
    # fused row, and no attention op is ever answered by a GEMM row.
    "attention_qk": {
        "batched": ("npu_fusion_attention",),
        "per_call": ("npu_fusion_attention",),
    },
    "attention_pv": {
        "batched": ("npu_fusion_attention",),
        "per_call": ("npu_fusion_attention",),
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
    "grouped_matmul": "grouped_matmul",
    "moe_gate_up": "moe_gate_up",
    "moe_down": "moe_down",
    "attention_fused": "attention_fused",
    "attention_qk": "attention_qk",
    "attention_pv": "attention_pv",
}
#: Ordered *further* targets of the same physical kernel, each keeping its own
#: artifact family and recipe spelling.  A grouped expert GEMM is fitted under its
#: routed family or under the pooled ``grouped_matmul`` family; the dense GEMM row
#: is the last declared approximation for a *non-MC2* grouped GEMM (it is the same
#: ``torch.mm``/``aclnn`` cube work at the same total geometry, and the existing
#: acceptance corpus counts it as applied).  An MC2 op refuses the dense-GEMM
#: target: its damage is priced by the fused kernel, not by a second GEMM model.
PHYSICAL_TRANSFER_FALLBACK_TARGETS: dict[str, tuple[str, ...]] = {
    "moe_gate_up": ("grouped_matmul", "gemm"),
    "moe_down": ("grouped_matmul", "gemm"),
}
#: Cost-model aliases that are the same physical family.  ``elementwise`` is the
#: graph name of the measured add family and ``router_softmax`` is a softmax over
#: the expert axis (the ``softmax`` artifact family, aliased through the declared
#: ``attention_softmax`` entry).
PHYSICAL_TRANSFER_ALIASES: dict[str, str] = {
    "copy_bytes": "copy",
    "elementwise": "add",
    "router_softmax": "attention_softmax",
}
#: Families that are the measured GEMM kernel only when the op actually declares
#: GEMM work.  ``linear_gate``/``linear_output_gate`` name a matrix projection in
#: a gated attention block but a *vector* gate kernel in a linear-attention
#: block; a 1-D op is not a GEMM and must not be priced as one.
PHYSICAL_GEMM_SHAPED_ONLY: frozenset[str] = frozenset(
    {"linear_gate", "linear_output_gate"}
)
#: Artifact family of the dense-GEMM transfer target, named once so the MC2
#: suppression rule cannot silently stop matching.
PHYSICAL_GEMM_ARTIFACT_FAMILY = "gemm"
#: Measured kernels this cost model cannot represent, mapped to the reason.  The
#: campaign measures a fused ``npu_grouped_matmul_swiglu_quant`` kernel: pricing it
#: as a plain grouped matmul *plus* a separate activation would double count the
#: fused epilogue, and pricing it as a plain grouped matmul would drop the
#: activation work.  Such an op is reported unmodeled with that reason instead of
#: being silently priced as something it is not.
PHYSICAL_UNEXPRESSIBLE_RECIPES: dict[str, str] = {
    "npu_grouped_matmul_swiglu_quant":
        "fused_grouped_matmul_swiglu_quant_epilogue_not_expressible",
}
#: Recipe spelling of an artifact row that declares *no* runner recipe.  Such a
#: row is a fit keyed by family/timing scope only (the routed-expert rows of the
#: A3 artifact are spelled that way); it is the last declared candidate for its own
#: family and is always reported as the transfer target it is.
PHYSICAL_TRANSFER_ANY_RECIPE = ""


def physical_transfer_target(
    family: str, timing_scope: str, *, allow_dense_gemm: bool = True,
) -> tuple[tuple[str, str], ...] | None:
    """Ordered ``(artifact family, artifact recipe)`` candidates for a family.

    ``None`` means "no fitted model exists for this family": the QK/PV attention
    halves (a half kernel is not the measured fused call), ``embedding``,
    ``sampling`` and the communication families stay unmodeled with the artifact's
    own reason.  The first candidate is the artifact family and recipe the
    declared timing scope names; the rest are the other declared spellings of the
    same physical kernel, tried in order, ending with the recipe-agnostic row of
    the same family.

    ``allow_dense_gemm=False`` drops the dense-GEMM fallback target.  That is the
    MC2 rule: an MC2 chunk's compute is priced inside the fused kernel, so a second
    GEMM model may not be applied on top of it.
    """
    declared = str(family or "")
    alias = PHYSICAL_TRANSFER_ALIASES.get(declared)
    key = alias or declared
    if key not in PHYSICAL_TRANSFER_TARGETS and key in GEMM_TRANSFER_FAMILIES:
        # No entry of its own: a projection/MLP GEMM is the measured gemm kernel.
        key = "gemm"
    scope = str(timing_scope or "").lower()
    keys = (key, *PHYSICAL_TRANSFER_FALLBACK_TARGETS.get(key, ()))
    candidates: list[tuple[str, str]] = []
    # A graph family that resolves to the *pooled* dense-GEMM declaration may still
    # have measured rows of its own (``qkv_projection``, ``mla_q_a``, ...), so its
    # own name is asked first and the pooled rows follow.  A family whose artifact
    # family is itself (``rmsnorm``, ``softmax``, ``attention_fused``) or that is a
    # declared alias (``elementwise`` -> ``add``) needs no extra hop.
    primary_artifact_family = PHYSICAL_TRANSFER_ARTIFACT_FAMILY.get(key, key)
    if (
        declared and not alias
        and primary_artifact_family == PHYSICAL_GEMM_ARTIFACT_FAMILY
        and declared != PHYSICAL_GEMM_ARTIFACT_FAMILY
    ):
        target = PHYSICAL_TRANSFER_TARGETS.get(key) or PHYSICAL_TRANSFER_TARGETS["gemm"]
        own_recipes = target["batched"] if "batched" in scope else target["per_call"]
        candidates.extend((declared, recipe) for recipe in own_recipes)
        candidates.append((declared, PHYSICAL_TRANSFER_ANY_RECIPE))
    for target_key in keys:
        target = PHYSICAL_TRANSFER_TARGETS.get(target_key)
        if target is None:
            continue
        artifact_family = PHYSICAL_TRANSFER_ARTIFACT_FAMILY.get(target_key, target_key)
        if not allow_dense_gemm and artifact_family == PHYSICAL_GEMM_ARTIFACT_FAMILY:
            continue
        recipes = target["batched"] if "batched" in scope else target["per_call"]
        candidates.extend((artifact_family, recipe) for recipe in recipes)
        candidates.append((artifact_family, PHYSICAL_TRANSFER_ANY_RECIPE))
    return tuple(candidates) or None


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


# ---------------------------------------------------------------------------
# MC2: dispatch/combine traffic overlapped with the grouped matmul
# ---------------------------------------------------------------------------
#: MC2 is a fused kernel: one launch covers dispatch, the grouped expert GEMMs
#: and combine.  A chunk's compute and its communication therefore run inside the
#: same kernel, so the op decomposition the graph declares (dispatch -> GEMM ->
#: combine) is folded back into one overlap-aware estimate
#: (:meth:`PredictiveCostModel.compose_mc2`).
MC2_OVERLAP_UNVERIFIED = "mc2_overlap_unverified"
MC2_OVERLAP_CALIBRATED = "mc2_overlap_calibrated"
MC2_NO_COMMUNICATION = "mc2_no_communication"
#: Recipes the MoE lowering emits for an MC2 layer.  ``moe_mc2_swiglu`` is
#: deliberately absent: the activation is its own vector kernel and is not part of
#: the fused dispatch/GMM/combine window.
MC2_RECIPES: frozenset[str] = frozenset(
    {"moe_mc2", "mc2", "mc2_dispatch", "mc2_combine"}
)
MC2_FAMILIES: frozenset[str] = frozenset({"mc2"})
MC2_DISPATCH_FAMILIES: frozenset[str] = frozenset({"ep_dispatch"})
MC2_COMBINE_FAMILIES: frozenset[str] = frozenset({"ep_combine"})
#: Families whose duration is for the measured ``npu_grouped_matmul`` kernel.
#: The list API accepts at most :data:`GROUPED_LAUNCH_EXPERT_CAP` experts per
#: ``tensorList``, so a routed group above the cap needs ceil(experts/cap)
#: sequential launches -- a 256-expert layer is two launches, not one.
GROUPED_MATMUL_FAMILIES: frozenset[str] = frozenset(
    {"moe_gate_up", "moe_down", "grouped_matmul"}
)
GROUPED_LAUNCH_EXPERT_CAP = 128
#: Keys the lowering uses to declare how many experts one grouped call covers,
#: most specific first (``group_size`` is the per-op expert count; ``local_experts``
#: is the layer's total and only answers when nothing narrower is declared).
GROUPED_EXPERT_METADATA_KEYS: tuple[str, ...] = (
    "group_size", "group_experts", "experts", "num_experts_local", "local_experts",
)
#: Families whose duration is the fused kernel's *compute* term (the grouped
#: matmul).  Everything else in a chunk is communication or bookkeeping.
MC2_COMPUTE_FAMILIES: frozenset[str] = frozenset(
    {"moe_gate_up", "moe_down", "grouped_matmul", "mc2"}
)
#: Families that declare the layer's batch geometry as a ``(tokens, hidden)``
#: shape.  The MC2 barrier is the one the lowering always emits next to the
#: chunk; a norm/allreduce/add carrier is accepted for graphs that omit it.
MC2_TOKEN_CARRIER_FAMILIES: frozenset[str] = frozenset(
    {"barrier", "allreduce", "dcp_allgather", "elementwise", "cache_store"}
)
#: Measured spread of the fused chain against the sum of its legs (A3 0.95-1.18x,
#: A2 ~1.00x, 17 campaign jobs / 686 rows).  ``chain_factor`` is bounded by this
#: declared range: below 1.0 would claim an overlap the measurement does not show,
#: above 1.2 would claim a penalty it does not show either.
MC2_CHAIN_FACTOR_BOUNDS: tuple[float, float] = (1.0, 1.2)

#: The comm lane an MC2 all-to-all is priced on: a two-rank point-to-point lane
#: (the same lane the existing EP dispatch/combine path prices).  Group scaling
#: is the ``(ep-1)/ep`` volume factor, not a per-rank lane change.
MC2_LANE_MODE = "p2p_only"
MC2_LANE_TIMING_SCOPE = "communication_lane"
MC2_LANE_RANKS = 2
MC2_BYTES_FORMULA = "tokens*top_k*hidden*dtype_bytes"


def is_mc2_op(op: Any) -> bool:
    """True for an op the MoE lowering marked as part of an MC2 fused layer."""
    family = str(getattr(op, "family", "") or "")
    recipe = str(getattr(op, "recipe", "") or "")
    return family in MC2_FAMILIES or recipe in MC2_RECIPES


def _mc2_int(value: Any) -> int:
    """Declared integer metadata, or ``0`` when the field is absent/unparsable."""
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def grouped_expert_count(op: Any) -> int:
    """Experts one grouped-matmul op covers, from its declared grouping."""
    metadata = dict(getattr(op, "metadata", ()) or {})
    for key in GROUPED_EXPERT_METADATA_KEYS:
        declared = _mc2_int(metadata.get(key))
        if declared > 0:
            return declared
    per_expert = metadata.get("expert_tokens")
    if isinstance(per_expert, (tuple, list)) and per_expert:
        return len(per_expert)
    return 1


def grouped_matmul_launch_count(op: Any) -> int:
    """Sequential ``npu_grouped_matmul`` launches this op costs.

    The list API caps one ``tensorList`` at :data:`GROUPED_LAUNCH_EXPERT_CAP`
    experts; a wider routed group is issued as ceil(experts/cap) calls, each
    paying the declared dispatch intercept.  A 256- or 384-expert layer must not
    be priced as a single launch.
    """
    if str(getattr(op, "family", "") or "") not in GROUPED_MATMUL_FAMILIES:
        return 1
    experts = max(1, grouped_expert_count(op))
    return max(1, math.ceil(experts / GROUPED_LAUNCH_EXPERT_CAP))


def mc2_chunk_expert_count(ops: Sequence[OpSpec]) -> int:
    """Local experts the MC2 chunk's fused grouped matmul covers.

    The lowering declares the chunk's expert ids on its dispatch op; a chunk that
    does not declare them falls back to the widest per-op grouping.  The fused
    kernel issues one grouped matmul over that whole list, so the tensor-list cap
    applies to the chunk's expert count, not to one expert at a time.
    """
    for op in ops:
        declared = dict(getattr(op, "metadata", ()) or ()).get("experts")
        if isinstance(declared, (tuple, list)) and declared:
            return len(declared)
    return max([grouped_expert_count(op) for op in ops] or [1])


def mc2_chunk_key(op: Any) -> tuple[int, int, int] | None:
    """``(stage, layer, chunk)`` identity of an MC2 op, else ``None``.

    The chunk is the fused kernel's unit: one dispatch, one grouped GEMM per
    local expert and one combine.  Ops of independent chunks never share a
    kernel, and ops of different layers never do either.
    """
    if not is_mc2_op(op):
        return None
    metadata = dict(getattr(op, "metadata", ()) or ())
    try:
        chunk = int(metadata.get("mc2_chunk", 0))
    except (TypeError, ValueError):
        chunk = 0
    return (int(getattr(op, "stage", 0) or 0), int(getattr(op, "layer", 0) or 0), chunk)


def mc2_comm_bytes(
    *, tokens: float, top_k: int, hidden: float, dtype_bytes_value: float,
    ep_size: int,
) -> tuple[float, float]:
    """``(payload_bytes, link_bytes)`` of one MC2 direction.

    ``payload_bytes`` is the routed activation a replica exchanges in one
    direction (``tokens*top_k*hidden*dtype_bytes``).  An all-to-all sends
    ``(ep-1)/ep`` of it to other ranks, which is the D2D link volume; an EP group
    of one has no traffic.
    """
    payload = (
        max(0.0, float(tokens)) * max(0, int(top_k))
        * max(0.0, float(hidden)) * max(0.0, float(dtype_bytes_value))
    )
    group = max(1, int(ep_size))
    return payload, payload * (group - 1) / group


def mc2_time_bounds(
    *, launch_ms: float, compute_ms: float, comm_ms: float,
    chain_factor: float = 1.0,
) -> dict[str, float]:
    """Bounds of one fused MC2 kernel under the *measured* chain contract.

    The completed A3/A2 campaign (17 jobs, 686 rows) measured the fused chain
    against the sum of its independently measured legs: 0.95-1.18x on A3
    (ep2/ep4/ep8, 64/128 experts) and ~1.00x on A2.  The chain is therefore
    ``launch + (compute + comm) * chain_factor`` -- it never collapses to
    ``launch + max(compute, comm)``, which the campaign refutes by ~35 % at
    ep2/e64:

    * ``legs_ms`` = ``launch + compute + comm`` -- the campaign's own baseline
      (``chain_factor = 1.0``); this is the default and the primary estimate;
    * ``upper_ms`` = ``launch + (compute + comm) * 1.2`` -- the declared campaign
      ceiling (:data:`MC2_CHAIN_FACTOR_BOUNDS`), reported so a reader sees the
      measured spread rather than one number;
    * ``max_bound_ms`` = ``launch + max(compute, comm)`` -- the full-overlap bound,
      reported for reference only and flagged ``max_bound_refuted_by_measurement``;
      it is never the estimate.
    """
    launch = max(0.0, float(launch_ms))
    compute = max(0.0, float(compute_ms))
    comm = max(0.0, float(comm_ms))
    factor = min(
        MC2_CHAIN_FACTOR_BOUNDS[1],
        max(MC2_CHAIN_FACTOR_BOUNDS[0], float(chain_factor)),
    )
    return {
        "launch_ms": launch,
        "compute_ms": compute,
        "comm_ms": comm,
        "chain_factor": factor,
        "legs_ms": launch + compute + comm,
        "serial_ms": launch + compute + comm,
        "upper_ms": launch + (compute + comm) * MC2_CHAIN_FACTOR_BOUNDS[1],
        "max_bound_ms": launch + max(compute, comm),
        "max_bound_refuted_by_measurement": True,
        "estimated_ms": launch + (compute + comm) * factor,
    }


def communication_lane_curve(
    calibration: Any, *, placement: str, hardware_key: str, dtype: str,
    model: "PredictiveCostModel", ranks: int = MC2_LANE_RANKS,
) -> tuple[float, float, str]:
    """``(alpha_ms, beta_ms_per_byte, note)`` of the measured comm lane.

    ``alpha``/``beta`` are the intercept and slope of the affine lane fit the
    communication calibration already applies (``span = alpha + beta*bytes``).
    The lane is returned only when a *measured* curve exists for this machine and
    fabric: ``note`` names why it does not, and the caller then prices the lane
    analytically instead of inventing a curve.
    """
    curve = getattr(calibration, "_curve", None)
    if not callable(curve):
        return 0.0, 0.0, "comm_curve_api_unavailable"
    intercept, slope = curve(
        placement, MC2_LANE_MODE,
        hardware_key=hardware_key, timing_scope=MC2_LANE_TIMING_SCOPE,
        ranks=int(ranks), dtype=dtype, algorithm=MC2_LANE_MODE,
        execution_backend=model.communication_backend,
        observed_execution=model.communication_observed_backend,
        collective="send_recv", group_size=int(ranks),
        route=model.communication_route,
        timing_completion=model.communication_timing_completion,
        requested_aiv_cores=model.communication_requested_aiv_cores or None,
        observed_aiv_cores=model.communication_observed_aiv_cores or None,
        algorithm_regime=model.communication_algorithm_regime,
        topology_class=model.communication_topology_class,
        in_place=(
            model.communication_in_place if model.communication_in_place else None
        ),
    )
    intercept = max(0.0, float(intercept))
    slope = max(0.0, float(slope))
    if intercept + slope <= 0:
        return 0.0, 0.0, "no_measured_lane_curve_for_fabric"
    return intercept, slope, ""


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
    #: Measured chain factor of the MC2 fused kernel:
    #: ``t = launch + (compute + comm) * chain_factor``.  The completed campaign
    #: measured the chain at 0.95-1.18x the sum of its legs on A3 and ~1.00x on A2
    #: (17 jobs, 686 rows) -- there is no overlap to claim, and
    #: ``launch + max(compute, comm)`` is refuted by ~35 % at ep2/e64.  The declared
    #: default is ``1.0`` (the campaign's baseline) and the value is bounded by
    #: :data:`MC2_CHAIN_FACTOR_BOUNDS`; ``1.2`` is reported as the upper bound on
    #: every MC2 op.  A value is only called *verified* when it came from a fit and
    #: the fabric lane is measured.
    mc2_chain_factor: float = 1.0
    #: Provenance of :attr:`mc2_chain_factor`: ``declared`` (a model assumption) or
    #: ``fitted`` (a fitted chain factor supplied by a measurement campaign).
    mc2_chain_factor_source: str = "declared"
    #: Declared MC2 EP group size.  ``0`` means "read it from the op metadata, else
    #: from the graph's declared routed volume"; it is never guessed as 2.
    mc2_ep_size: int = 0
    #: How the per-kernel dispatch intercept composes with the kernel's service.
    #: ``concurrent`` is the declared default and matches the module invariant
    #: ``estimated_ms = launch + service`` with the launch carried as its own demand
    #: (the engine serves a task's demands concurrently, so an eager per-op submit
    #: lets a kernel's dispatch overlap its own service).  ``serial`` folds the
    #: intercept into the service demand so the engine cannot drop it;
    #: ``cost_iteration`` selects it for an iteration the workload actually
    #: *replayed* (``work.graph_bucket`` is not ``None``), because a replayed bucket
    #: issues its kernels' device dispatch gaps in sequence even though the host
    #: submit is amortized.
    launch_composition: str = "concurrent"

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
        A grouped matmul pays one intercept *per tensor-list launch*: the list API
        caps a launch at :data:`GROUPED_LAUNCH_EXPERT_CAP` experts, so a wider
        routed group is several sequential calls (:meth:`grouped_launch_detail`).
        """
        die = self.machine.dies[0] if self.machine.dies else None
        dispatch = (
            float(getattr(die, "launch_us", 0.0) or 0.0) / 1000.0
            * max(0.0, float(self.launch_amortization))
        )
        dispatch *= grouped_matmul_launch_count(op)
        if op.link_bytes > 0 and link is not None:
            dispatch += float(link.latency_us) / 1000.0
        return dispatch

    def grouped_launch_detail(self, op: OpSpec) -> dict[str, Any] | None:
        """Why a grouped matmul costs more than one launch (``None`` otherwise).

        The declared per-expert tensor-list cap is a hardware contract of the
        ``npu_grouped_matmul`` list API, so it is reported instead of being folded
        silently into the dispatch number.
        """
        launches = grouped_matmul_launch_count(op)
        if launches <= 1 and str(op.family) not in GROUPED_MATMUL_FAMILIES:
            return None
        die = self.machine.dies[0] if self.machine.dies else None
        per_launch = (
            float(getattr(die, "launch_us", 0.0) or 0.0) / 1000.0
            * max(0.0, float(self.launch_amortization))
        )
        return {
            "launches": int(launches),
            "experts": int(grouped_expert_count(op)),
            "expert_cap_per_launch": GROUPED_LAUNCH_EXPERT_CAP,
            "per_launch_ms": per_launch,
            "reason": (
                "single_launch" if launches == 1 else
                "npu_grouped_matmul_tensorlist_expert_cap"
            ),
            "source": "declared_die.launch_us",
        }

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

        The large-operator families carry axes a dense GEMM does not: an attention
        kernel is identified by ``(q, kv, heads, dim)`` and a grouped expert GEMM
        by ``(tokens, experts, n, k)``.  Those axes come from the op's own launch
        shape and its declared expert grouping, so the artifact can identify the
        row instead of refusing a shape it could have been given.
        """
        if self.physical_calibration is None:
            return None
        meta = dict(op.metadata)
        features = self.shape_features(op)
        shape: dict[str, float] = {}
        for key in ("m", "n", "k"):
            if key in features:
                shape[key] = float(features[key])
        if op.family in _ATTENTION_FAMILIES and len(op.launch_shape) >= 4:
            q_len, kv_len, heads, dim = (int(value) for value in op.launch_shape[:4])
            shape.update({
                "q": float(q_len), "kv": float(kv_len),
                "heads": float(heads), "dim": float(dim),
            })
            # A multi-query prefill chunk is causal; a single query sees the whole
            # context either way, so it declares the non-causal form.
            shape["causal"] = 1.0 if q_len > 1 else 0.0
        if op.family in {"moe_gate_up", "moe_down", "grouped_matmul"}:
            declared = meta.get("expert_tokens")
            if isinstance(declared, (tuple, list)):
                tokens = float(sum(int(value) for value in declared))
                experts = float(len(declared) or 1)
            else:
                tokens = float(
                    declared if declared is not None
                    else (op.launch_shape[0] if op.launch_shape else 0)
                )
                experts = 1.0
            group = meta.get("group_size")
            if group is not None:
                try:
                    experts = float(int(group))
                except (TypeError, ValueError):
                    experts = 1.0
            if tokens > 0 and experts > 0:
                shape.update({"tokens": tokens, "experts": experts})
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

        An MC2 op never takes the dense-GEMM fallback target: its duration is the
        fused kernel (:meth:`mc2_chunk_estimate`), so a second GEMM model applied
        on top of it would double count the same cube work.
        """
        query = self.physical_query(op)
        if query is None:
            return None, None, None, None
        unexpressible = PHYSICAL_UNEXPRESSIBLE_RECIPES.get(str(op.recipe or ""))
        if unexpressible:
            # The op is a fused kernel this module cannot decompose (activation and
            # quantisation inside the grouped matmul).  Refuse it explicitly rather
            # than price it as a plain GEMM and a separate activation.
            return query, self._physical_refusal(query, op, unexpressible), None, None
        explained = self._physical_ask(query, op)
        if explained.modeled:
            return query, explained, None, None
        if not self.physical_allow_family_transfer:
            return query, explained, None, None
        if not physical_transfer_eligible(op):
            # A ``linear_gate``/``linear_output_gate`` op that carries no GEMM
            # work is a vector gate kernel, not a projection: transferring it onto
            # the measured dense GEMM would price a vector op as a matmul.
            return query, explained, None, None
        candidates = physical_transfer_target(
            op.family, self.timing_scope, allow_dense_gemm=not is_mc2_op(op),
        )
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

    def _physical_refusal(
        self, query: Mapping[str, Any], op: OpSpec, reason: str,
    ) -> Any:
        """An unmodeled answer this module owes for an unexpressible kernel.

        Shaped like ``a3_physical_calibration.OpPrediction`` so the record,
        policy and provenance paths are identical; ``duration_ms`` is ``None``, so
        the analytic/curve path stays primary and the reason is carried through.
        """
        from .a3_physical_calibration import OpPrediction

        return OpPrediction(
            duration_ms=None,
            status="unmodeled",
            reason=reason,
            family=str(query.get("family") or ""),
            recipe=str(query.get("recipe") or ""),
            timing_scope=str(query.get("timing_scope") or ""),
            machine_source="cost_model_declared",
        )

    def _physical_record(
        self, query: Mapping[str, Any], explained: Any, status: str, detail: str,
        *, applied: bool, family: str = "", superseded: str = "",
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
        # Timing-scope identity.  ``timing_scope`` is the scope that actually
        # priced the op (the fitted row's own scope when the artifact answered
        # through its declared scope *family*, e.g. a case that declares
        # ``device_event_warm`` and gets the ``device_event_warm_operator``
        # attention row).  The declared scope and the alias are reported next to
        # it so a reader can tell "measured at this protocol" from "measured at a
        # sibling protocol of the same family".
        declared_scope = str((query or {}).get("timing_scope") or "")
        parameters = dict(getattr(explained, "parameters", {}) or {})
        fitted_scope = str(getattr(explained, "timing_scope", "") or "")
        record["declared_timing_scope"] = declared_scope
        record["fitted_timing_scope"] = fitted_scope
        record["timing_scope"] = fitted_scope or declared_scope
        scope_from = str(parameters.get("scope_alias_from") or "")
        scope_to = str(parameters.get("scope_alias_to") or "")
        if scope_from:
            record["scope_alias"] = {"from": scope_from, "to": scope_to}
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
        # The coarse wiring label a trace reader (and ``predict``'s provenance)
        # aggregates; the artifact's own status stays in ``record["status"]``.
        record["wiring"] = physical_wiring(record)
        record["wiring"]["operator_class"] = operator_class(family or op_family(
            record.get("query") or {},
        ))
        record["wiring"]["large_operator"] = (
            str(family or "") in LARGE_OPERATOR_FAMILIES
        )
        return record

    def _apply_physical(
        self, explained: Any, baseline: Mapping[str, float],
        capacity: Mapping[str, float], *, launch_multiplicity: int = 1,
    ) -> dict[str, float]:
        """Analytic busy-resource work re-priced to the fitted duration.

        The artifact owns the *duration* (fitted intercept + fitted roofline),
        so the fitted intercept replaces the declared dispatch term and the
        analytic service demands are scaled to the fitted service time.  The
        analytic roofline still decides the relative busy-resource shares inside
        the service part, which is what the engine schedules; the fitted
        breakdown itself is reported on ``CostedOp.physical``.  Launch is never
        added twice: the fitted intercept is the launch term, it is not an extra
        term on top of ``explained.duration_ms``.  A grouped matmul whose expert
        count exceeds the tensor-list cap pays the fitted intercept once per
        launch (``launch_multiplicity``), because every launch of the list API
        repeats that intercept.
        """
        fitted_ms = float(explained.duration_ms or 0.0)
        breakdown = dict(getattr(explained, "breakdown", {}) or {})
        fitted_launch = float(breakdown.get("launch_ms") or 0.0)
        if launch_multiplicity > 1 and fitted_launch > 0:
            # The fit measured one tensor-list launch; the extra launches repeat
            # exactly that intercept.  The scaled launch is carried into the
            # fitted duration so estimated_ms stays launch + service.
            fitted_launch *= int(launch_multiplicity)
            fitted_ms += float(breakdown.get("launch_ms") or 0.0) * (
                int(launch_multiplicity) - 1
            )
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

    # -- MC2 fused dispatch/combine + grouped matmul -----------------------
    def capacity_map(self, link: Any = None) -> dict[str, float]:
        """Per-resource ceilings (work per millisecond) for this machine.

        One declaration shared by :meth:`cost` and :meth:`compose_mc2`, so an MC2
        member's fused duration is re-derived with exactly the capacities that
        priced it.
        """
        die = self.machine.dies[0] if self.machine.dies else None
        if die is None:
            raise A3InputError("machine_has_no_dies")
        return {
            "cube": self.machine.peak_flops_for(self.compute_dtype) / 1000.0,
            "vector": die.vector_ops_per_s / 1000.0,
            "hbm": die.hbm_bw_bytes_per_s / 1000.0,
            "mte2": die.mte_bw_bytes_per_s / 1000.0,
            "mte3": die.mte_bw_bytes_per_s / 1000.0,
            "link": (link.capacity_bytes_per_ms if link is not None else 1.0),
            "launch": 1.0, "cpu": 1.0, "serial": 1.0, "aicpu": 1.0,
        }

    def _resource_capacity(self, name: str) -> float:
        return float(self.capacity_map().get(name, 1.0))

    def mc2_link(self, ep_size: int) -> tuple[Any, str, str]:
        """``(link, scope, note)`` the MC2 traffic is priced on.

        The EP group decides the fabric: a group that spans nodes uses the
        declared ``d2d_inter_node`` channel when the machine has one, otherwise
        the ``d2d_intra_node`` fabric, otherwise the machine's single declared
        fabric.  A machine that declares neither is an explicit error -- MC2
        traffic is never priced on an arbitrary wire.
        """
        ranks_per_node = max(1, int(getattr(self.machine, "ranks_per_node", 1) or 1))
        order: list[str] = []
        if int(ep_size) > ranks_per_node:
            order.append("d2d_inter_node")
        order.extend(("d2d_intra_node", "d2d"))
        note = ""
        for scope in order:
            try:
                resolved = self.machine.link_for_scope(scope)
            except A3InputError:
                continue
            if resolved is None:
                continue
            if scope == "d2d_intra_node" and int(ep_size) > ranks_per_node:
                note = (
                    "ep_group_spans_nodes_but_machine_declares_no_d2d_inter_node_"
                    "fabric;price_on_intra_node"
                )
            return resolved, scope, note
        raise A3InputError(
            f"mc2_link_unresolved:ep={int(ep_size)}:"
            f"links={[link.name for link in self.machine.links]}"
        )

    def mc2_geometry(
        self, ops: Sequence[OpSpec], *, layer_ops: Sequence[OpSpec] | None = None,
    ) -> dict[str, Any]:
        """Declared geometry of one MC2 chunk.

        Every input of :data:`MC2_BYTES_FORMULA` is reported with where it came
        from (``*_source``): the op metadata first, then the layer's
        ``(tokens, hidden)`` carrier op, then the graph's own declared routed
        volume (:attr:`OpSpec.link_bytes`), then the routed row count.  The
        chunk's payload is ``tokens*top_k*hidden*dtype_bytes / chunks``; an EP
        group of ``ep`` sends ``(ep-1)/ep`` of it over the fabric.
        """
        chunk_ops = tuple(ops)
        if not chunk_ops:
            raise A3InputError("mc2_chunk_empty")
        metadata: dict[str, Any] = {}
        for op in chunk_ops:
            for key, value in dict(op.metadata or ()).items():
                metadata.setdefault(str(key), value)
        dispatch = next(
            (op for op in chunk_ops if op.family in MC2_DISPATCH_FAMILIES), None,
        )
        combine = next(
            (op for op in chunk_ops if op.family in MC2_COMBINE_FAMILIES), None,
        )
        anchor = dispatch or combine or chunk_ops[0]
        shape = tuple(anchor.semantic_shape or ())
        hidden = _mc2_int(metadata.get("mc2_hidden")) or _mc2_int(
            metadata.get("hidden")
        )
        hidden_source = "declared_metadata" if hidden else ""
        if not hidden and len(shape) >= 3:
            hidden, hidden_source = int(shape[2]), "dispatch_shape"
        if not hidden and len(shape) == 2:
            hidden, hidden_source = int(shape[1]), "combine_shape"
        if not hidden:
            # No dispatch/combine shape to read: the grouped gate/up GEMM's k is
            # the hidden width (the down GEMM's k is the expert width, so it is
            # only a last resort).
            for family in ("moe_gate_up", "grouped_matmul", "moe_down", "mc2"):
                compute_op = next(
                    (
                        op for op in chunk_ops
                        if op.family == family
                        and len(op.launch_shape or ()) >= 3
                    ),
                    None,
                )
                if compute_op is not None:
                    hidden = int(compute_op.launch_shape[-1])
                    hidden_source = "grouped_matmul_k"
                    break
        top_k = _mc2_int(metadata.get("mc2_top_k")) or _mc2_int(
            metadata.get("top_k")
        )
        top_k_source = "declared_metadata" if top_k else ""
        if not top_k:
            if len(shape) >= 3:
                top_k, top_k_source = int(shape[1]), "dispatch_shape"
            else:
                top_k, top_k_source = 1, "no_routed_dimension"
        top_k = max(1, top_k)
        hidden = max(0, hidden)
        chunks = max(1, _mc2_int(metadata.get("mc2_chunks")) or 1)
        dtype = str(anchor.dtype or self.compute_dtype or "")
        per_element = dtype_bytes(dtype)
        declared = max(
            float(getattr(combine, "link_bytes", 0.0) or 0.0),
            float(getattr(dispatch, "link_bytes", 0.0) or 0.0),
        )
        ep_size = self.mc2_ep_size or _mc2_int(metadata.get("mc2_ep_size")) or _mc2_int(
            metadata.get("ep_size")
        ) or _mc2_int(metadata.get("ep"))
        ep_source = "declared" if ep_size else ""
        tokens = _mc2_int(metadata.get("mc2_tokens")) or _mc2_int(
            metadata.get("tokens")
        )
        tokens_source = "declared_metadata" if tokens else ""
        if not tokens and layer_ops is not None and hidden > 0:
            best = 0
            for op in layer_ops:
                if op.family not in MC2_TOKEN_CARRIER_FAMILIES:
                    continue
                candidate = tuple(op.semantic_shape or ())
                if len(candidate) != 2 or int(candidate[1]) != hidden:
                    continue
                best = max(best, int(candidate[0]))
            if best > 0:
                tokens, tokens_source = best, "layer_token_carrier"
        # A routed volume the graph already declares pins the batch geometry
        # exactly: link = tokens*top_k*hidden*bytes*(ep-1)/ep/chunks.
        if not tokens and declared > 0 and ep_size > 1 and hidden > 0:
            payload_per_chunk = declared * ep_size / ((ep_size - 1) * chunks)
            tokens = int(round(payload_per_chunk / (top_k * hidden * per_element)))
            tokens_source = "inverted_from_declared_link_bytes"
        if ep_size <= 1 and declared > 0 and tokens > 0 and hidden > 0:
            payload_per_chunk = tokens * top_k * hidden * per_element / chunks
            if payload_per_chunk > 0:
                ratio = declared / payload_per_chunk
                if 0.0 < ratio < 1.0:
                    candidate = 1.0 / (1.0 - ratio)
                    rounded = int(round(candidate))
                    if rounded >= 2 and abs(candidate - rounded) <= 0.02:
                        ep_size, ep_source = rounded, "inverted_from_declared_link_bytes"
        if not tokens:
            rows = int(shape[0]) if shape else 0
            if rows > 0:
                tokens = max(1, int(round(rows / top_k)))
                tokens_source = "routed_rows_over_top_k"
            else:
                tokens, tokens_source = 0, "undeclared"
        ep_size = max(1, ep_size)
        if not ep_source:
            ep_source = "undeclared_single_rank"
        payload, grouped_link = mc2_comm_bytes(
            tokens=tokens, top_k=top_k, hidden=hidden,
            dtype_bytes_value=per_element, ep_size=ep_size,
        )
        payload_per_chunk = payload / chunks
        formula_link = grouped_link / chunks
        if formula_link > 0:
            link_bytes, link_source = formula_link, "formula"
        elif declared > 0:
            link_bytes, link_source = declared, "graph_declared_routed_volume"
        else:
            link_bytes, link_source = 0.0, "no_routed_volume"
        link_scope: str = ""
        link_note = ""
        link = None
        if link_bytes > 0:
            link, link_scope, link_note = self.mc2_link(ep_size)
        return {
            "bytes_formula": MC2_BYTES_FORMULA,
            "tokens": float(tokens),
            "tokens_source": tokens_source,
            "top_k": int(top_k),
            "top_k_source": top_k_source,
            "hidden": float(hidden),
            "hidden_source": hidden_source,
            "dtype": dtype,
            "dtype_bytes": float(per_element),
            "ep_size": int(ep_size),
            "ep_source": ep_source,
            "ep_group_factor": (ep_size - 1) / ep_size,
            "chunks": int(chunks),
            "chunk_split": "equal_by_declared_chunk_count",
            "payload_bytes": float(payload),
            "payload_bytes_per_chunk": float(payload_per_chunk),
            "link_bytes": float(link_bytes),
            "link_bytes_source": link_source,
            "declared_link_bytes": float(declared),
            "declared_over_formula": (
                float(declared / formula_link) if formula_link > 0 else None
            ),
            "link_scope": link_scope,
            "link_name": str(getattr(link, "name", "") or ""),
            "link_protocol": str(getattr(link, "protocol", "") or ""),
            "link_note": link_note,
        }

    def mc2_lane(
        self, *, link: Any, link_bytes: float, dtype: str, ep_size: int = 0,
    ) -> dict[str, Any]:
        """One MC2 direction's lane: ``alpha + beta*bytes`` when measured.

        ``alpha``/``beta`` come from the communication calibration fitted for this
        machine and fabric.  With no measured lane the wire's declared bandwidth
        is used (``alpha = 0``) and the record says so, so a zero-``alpha`` lane is
        never mistaken for a measurement.
        """
        placement = str(self.communication_placement or "").strip()
        if not placement:
            placement = communication_placement_for_machine(self.machine, link=link)
        alpha = 0.0
        beta = 0.0
        note = "no_communication_calibration"
        measured_ranks = MC2_LANE_RANKS
        if self.communication_calibration is not None and placement:
            # The campaign measures the all-to-all per EP group, so the group's own
            # lane is asked first; the two-rank lane is the declared fallback for a
            # machine whose calibration only carries the pair.
            for ranks in (int(ep_size) if int(ep_size) > 1 else None, MC2_LANE_RANKS):
                if ranks is None:
                    continue
                alpha, beta, note = communication_lane_curve(
                    self.communication_calibration, placement=placement,
                    hardware_key=str(self.machine.hardware_key or ""),
                    dtype=dtype, model=self, ranks=int(ranks),
                )
                if alpha + beta > 0 and not note:
                    measured_ranks = int(ranks)
                    break
        measured = alpha + beta > 0 and not note
        if measured:
            lane_ms = max(0.0, alpha + beta * max(0.0, float(link_bytes)))
            source = "measured_comm_curve"
        else:
            capacity = float(getattr(link, "capacity_bytes_per_ms", 0.0) or 0.0)
            lane_ms = float(link_bytes) / capacity if capacity > 0 else 0.0
            beta = 1.0 / capacity if capacity > 0 else 0.0
            alpha = 0.0
            source = "analytic_link_capacity"
            note = note or "no_measured_lane_curve_for_fabric"
        return {
            "placement": placement,
            "mode": MC2_LANE_MODE,
            "measured_ranks": int(measured_ranks),
            "requested_ranks": int(ep_size) if int(ep_size) > 1 else int(MC2_LANE_RANKS),
            "alpha_ms": float(alpha),
            "beta_ms_per_byte": float(beta),
            "link_capacity_bytes_per_ms": float(
                getattr(link, "capacity_bytes_per_ms", 0.0) or 0.0
            ),
            "lane_ms": float(lane_ms),
            "source": source,
            "note": note,
        }

    def mc2_chunk_estimate(
        self, ops: Sequence[OpSpec], costed: Sequence[CostedOp],
        *, layer_ops: Sequence[OpSpec] | None = None,
    ) -> dict[str, Any]:
        """The fused MC2 kernel's duration for one chunk.

        ``compute`` is the grouped-matmul duration (the fitted duration of each
        expert GEMM when the artifact models it, else its analytic roofline) and
        ``comm`` is the measured-or-analytic dispatch + combine lane time.  The
        duration is ``launch + (compute + comm) * chain_factor``: the measured
        campaign chain is 1.0-1.2x the sum of its legs, so the legs bound and the
        campaign ceiling are both reported and the full-overlap
        ``launch + max(compute, comm)`` is carried only as a refuted reference.
        """
        geometry = self.mc2_geometry(ops, layer_ops=layer_ops)
        compute_members = [
            (op, cost) for op, cost in zip(ops, costed)
            if op.family in MC2_COMPUTE_FAMILIES
        ]
        compute_ms = sum(max(0.0, float(cost.service_ms)) for _, cost in compute_members)
        fitted = sum(
            1 for _, cost in compute_members
            if cost.physical is not None and bool(cost.physical.get("applied"))
        )
        fitted_families = sorted({
            str(dict(cost.physical.get("query") or {}).get("family") or "")
            for _, cost in compute_members
            if cost.physical is not None and bool(cost.physical.get("applied"))
        })
        if not compute_members:
            compute_source = "no_grouped_matmul_member"
        elif fitted == len(compute_members):
            compute_source = "fitted_physical"
        elif fitted == 0:
            compute_source = "analytic_roofline"
        else:
            compute_source = "mixed_fitted_and_analytic"
        link = None
        if geometry["link_bytes"] > 0:
            link, _, _ = self.mc2_link(geometry["ep_size"])
        lane = self.mc2_lane(
            link=link, link_bytes=geometry["link_bytes"],
            dtype=str(ops[0].dtype or self.compute_dtype or ""),
            ep_size=int(geometry["ep_size"]),
        )
        dispatch_ms = lane["lane_ms"]
        combine_ms = lane["lane_ms"]
        comm_ms = dispatch_ms + combine_ms
        measured_lane = lane["source"] == "measured_comm_curve"
        die = self.machine.dies[0] if self.machine.dies else None
        # The fused kernel is one launch, except that the grouped matmul inside it
        # is still capped at one tensor list per GROUPED_LAUNCH_EXPERT_CAP experts:
        # the chunk's routed expert list decides the launch count.
        chunk_experts = max(1, mc2_chunk_expert_count(ops))
        launches = max(1, math.ceil(chunk_experts / GROUPED_LAUNCH_EXPERT_CAP))
        launch_ms = (
            float(getattr(die, "launch_us", 0.0) or 0.0) / 1000.0
            * max(0.0, float(self.launch_amortization))
            * launches
        )
        if not measured_lane and link is not None:
            # The analytic lane has no fitted intercept, so the fabric's declared
            # setup latency is the dispatch/combine term.  A measured alpha
            # already contains it and is never added a second time.
            launch_ms += 2.0 * float(getattr(link, "latency_us", 0.0) or 0.0) / 1000.0
        bounds = mc2_time_bounds(
            launch_ms=launch_ms, compute_ms=compute_ms, comm_ms=comm_ms,
            chain_factor=self.mc2_chain_factor,
        )
        fitted_factor = str(self.mc2_chain_factor_source) == "fitted"
        verified = measured_lane and fitted_factor
        if comm_ms <= 0:
            status = MC2_NO_COMMUNICATION
            reason = f"mc2_no_communication:{geometry['link_bytes_source']}"
        elif verified:
            status = MC2_OVERLAP_CALIBRATED
            reason = (
                "mc2_chain_calibrated:fitted_chain_factor_on_measured_lane;"
                f"lane={lane['placement']};chain_factor={bounds['chain_factor']}"
            )
        else:
            status = MC2_OVERLAP_UNVERIFIED
            missing = []
            if not measured_lane:
                missing.append(f"measured_comm_lane({lane['note']})")
            if not fitted_factor:
                missing.append("fitted_chain_factor")
            reason = (
                "chain_unverified:missing=" + "+".join(missing)
                + ";legs_and_upper_bounds_reported"
                + ";measured_campaign_chain_is_1.0-1.2x_the_leg_sum"
            )
        return {
            "status": status,
            "wiring_status": status,
            "operator_class": "mc2",
            "reason": reason,
            "chunk": {
                "stage": int(getattr(ops[0], "stage", 0) or 0),
                "layer": int(getattr(ops[0], "layer", 0) or 0),
                "chunk": int(
                    dict(ops[0].metadata or ()).get("mc2_chunk", 0) or 0
                ),
                "chunks": geometry["chunks"],
            },
            "geometry": geometry,
            "compute": {
                "compute_ms": float(compute_ms),
                "source": compute_source,
                "grouped_matmul_members": len(compute_members),
                "fitted_members": int(fitted),
                "fitted_artifact_families": fitted_families,
                # The chunk's fused duration already contains the cube work; the
                # dense-GEMM transfer is not applied on top of it.
                "dense_gemm_transfer": "suppressed_for_mc2",
            },
            "comm": {
                "comm_ms": float(comm_ms),
                "dispatch_ms": float(dispatch_ms),
                "combine_ms": float(combine_ms),
                "directions": 2,
                "lane": lane,
                "group_scaling": "volume_only",
            },
            "chain": {
                "factor": bounds["chain_factor"],
                "bounds": list(MC2_CHAIN_FACTOR_BOUNDS),
                "source": str(self.mc2_chain_factor_source),
                "verified": bool(verified),
                "measured_lane": bool(measured_lane),
                "max_bound_refuted_by_measurement": True,
            },
            "bounds": {
                "launch_ms": bounds["launch_ms"],
                "launch_count": int(launches),
                "legs_ms": bounds["legs_ms"],
                "serial_ms": bounds["serial_ms"],
                "upper_ms": bounds["upper_ms"],
                "max_bound_ms": bounds["max_bound_ms"],
                "estimated_ms": bounds["estimated_ms"],
                "primary": "legs" if bounds["chain_factor"] <= 1.0 else "chain",
            },
        }

    def compose_mc2(
        self, ops: Sequence[OpSpec], costed: Sequence[CostedOp],
    ) -> list[CostedOp]:
        """Fold every MC2 chunk's op decomposition into one fused-kernel cost.

        The graph must keep its dispatch -> expert GEMM -> combine ops (they are
        real dependencies), but MC2 runs them in one kernel: the chunk's launch is
        charged once, the grouped-matmul service times are the compute term, the
        dispatch/combine lane times are the communication term, and the members
        share the fused duration in proportion to their serial share.

        The communication rides on the *compute* members: their per-resource times
        become ``compute + (1-overlap)*comm`` and ``comm + (1-overlap)*compute``,
        whose ``max`` is exactly the fused duration.  The dispatch/combine ops
        themselves are marked ``absorbed`` with zero marginal duration, so no byte
        or launch of the same kernel is charged twice.
        """
        result = list(costed)
        groups: dict[tuple[int, int, int], list[int]] = {}
        for index, op in enumerate(ops):
            key = mc2_chunk_key(op)
            if key is not None:
                groups.setdefault(key, []).append(index)
        if not groups:
            return result
        by_layer: dict[tuple[int, int], list[OpSpec]] = {}
        for op in ops:
            by_layer.setdefault((op.stage, op.layer), []).append(op)
        for key, indices in groups.items():
            chunk_ops = [ops[index] for index in indices]
            chunk_costed = [costed[index] for index in indices]
            estimate = self.mc2_chunk_estimate(
                chunk_ops, chunk_costed, layer_ops=by_layer.get((key[0], key[1])),
            )
            geometry = estimate["geometry"]
            compute_indices = [
                index for index, op in zip(indices, chunk_ops)
                if op.family in MC2_COMPUTE_FAMILIES
            ]
            absorbed = [
                index for index, op in zip(indices, chunk_ops)
                if op.family not in MC2_COMPUTE_FAMILIES
            ]
            total_compute = sum(
                max(0.0, float(costed[index].service_ms)) for index in compute_indices
            )
            if estimate["status"] == MC2_NO_COMMUNICATION or total_compute <= 0:
                # No peer traffic means no fusion to model: the graph's own
                # decomposition (one launch per op) is the estimate, and nothing is
                # absorbed.  The record still says the chunk was seen.
                for index in indices:
                    current = costed[index]
                    is_compute = index in compute_indices
                    result[index] = replace(
                        current, mc2={
                            **estimate,
                            "role": "compute" if is_compute else "no_communication",
                            "note": "no MC2 communication to overlap",
                            "share": 1.0 if is_compute and total_compute <= 0 else 0.0,
                            "serial_analytic_ms": float(current.estimated_ms),
                        },
                    )
                continue
            for index in absorbed:
                current = costed[index]
                record = (
                    dict(current.physical) if current.physical is not None else None
                )
                if record is not None and record.get("applied"):
                    record["applied"] = False
                    record["superseded_by"] = "mc2_fused_chunk"
                    record["wiring"] = physical_wiring(record)
                result[index] = replace(
                    current, demands=(), estimated_ms=0.0, launch_ms=0.0,
                    service_ms=0.0, physical=record,
                    mc2={
                        **estimate,
                        "role": "absorbed",
                        "note": (
                            "dispatch/combine traffic and launch of this op are "
                            "charged to the MC2 chunk's fused compute members"
                        ),
                        "share": 0.0,
                        "serial_analytic_ms": float(current.estimated_ms),
                    },
                )
            link = None
            if geometry["link_bytes"] > 0:
                link, _, _ = self.mc2_link(geometry["ep_size"])
            capacity = float(getattr(link, "capacity_bytes_per_ms", 0.0) or 0.0)
            factor = float(estimate["chain"]["factor"])
            launch_ms = float(estimate["bounds"]["launch_ms"])
            comm_ms = float(estimate["comm"]["comm_ms"])
            anchor = compute_indices[0]
            for index in compute_indices:
                current = costed[index]
                service = max(0.0, float(current.service_ms))
                share = service / total_compute if total_compute > 0 else 0.0
                comm_share = comm_ms * share
                launch_share = launch_ms if index == anchor else 0.0
                # The chain is the *sum* of the legs times the chain factor: the
                # fused kernel does not overlap compute with its own dispatch.  The
                # member's local resource carries the whole serial window (leg plus
                # comm share) so the task's duration is the chain, while the fabric
                # carries only the real comm share.
                local_ms = factor * (service + comm_share)
                link_ms = factor * comm_share
                target = launch_share + local_ms
                # Scale from the *service* work (never from an already-folded
                # demand), so the chunk's own launch is charged exactly once.
                base_demands = (
                    current.service_demands
                    or tuple(
                        (name, work) for name, work in current.demands
                        if name not in {"launch", "completion_envelope_ms"}
                    )
                )
                scale = (
                    local_ms / service if service > 0 and local_ms > 0 else 0.0
                )
                window = local_ms + launch_share
                # Serial launch composition: the fused kernel's dispatch intercept
                # is folded into the (one) member that carries it, exactly as
                # ``cost`` folds a single op's intercept.
                fold = window / local_ms if launch_share > 0 and local_ms > 0 else 1.0
                demands: dict[str, float] = {
                    name: float(work) * scale * fold
                    for name, work in base_demands
                    if float(work) * scale * fold > 0
                }
                demands["launch"] = 0.0
                if link is not None and link_ms > 0 and capacity > 0:
                    demands["link"] = link_ms * capacity
                service_ms = max(
                    (
                        work / max(1e-9, self._resource_capacity(name))
                        for name, work in demands.items() if name != "launch"
                    ),
                    default=0.0,
                )
                result[index] = replace(
                    current,
                    demands=tuple(sorted(demands.items())),
                    estimated_ms=target,
                    launch_ms=launch_share,
                    service_ms=service_ms,
                    mc2={
                        **estimate,
                        "role": "compute",
                        "share": float(share),
                        "serial_analytic_ms": float(current.estimated_ms),
                        "member": {
                            "compute_ms": float(service),
                            "comm_ms": float(comm_share),
                            "local_resource_ms": float(local_ms),
                            "link_resource_ms": float(link_ms),
                            "estimated_ms": float(target),
                        },
                    },
                )
        return result

    def cost(
        self, op: OpSpec, *, launch_composition: str | None = None,
    ) -> CostedOp:
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

        capacity = self.capacity_map(link)
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
                calibrated = self._apply_physical(
                    physical_explained, baseline, capacity,
                    launch_multiplicity=grouped_matmul_launch_count(op),
                )
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
                applied=applied, family=str(op.family or ""),
                superseded=str(calibration_status.status)
                if not applied else "",
                transfer=physical_transfer, refused_transfer=physical_refused_transfer,
            )
        # Additive composition: a kernel cannot start before its dispatch and
        # cannot finish before its service.  The engine may still overlap two
        # ops' dispatch and service; this per-op estimate is the serial bound.
        envelope_ms = float(calibrated.get("completion_envelope_ms", 0.0))
        service_ms = max(service_ms, typed_stage_ms)
        composition = str(launch_composition or self.launch_composition or "concurrent")
        launch_detail = self.grouped_launch_detail(op)
        service_demands = tuple(sorted(
            (key, float(work)) for key, work in calibrated.items()
            if key not in {"launch", "completion_envelope_ms"} and float(work) > 0
        ))
        if composition == "serial" and envelope_ms <= 0 and launch_ms > 0:
            # A replayed graph pays each kernel's dispatch gap in sequence: the host
            # submit is amortized across the bucket, the device dispatch is not.  The
            # engine serves a task's demands concurrently, so the intercept is folded
            # into the busiest service demand to keep it additive; ``estimated_ms`` is
            # unchanged and ``launch_ms`` still reports the term.
            folded = {
                key: float(work)
                for key, work in calibrated.items()
                if key not in {"launch", "completion_envelope_ms"}
            }
            if folded and service_ms > 0:
                scale = (launch_ms + service_ms) / service_ms
                calibrated = {
                    key: value * scale for key, value in folded.items()
                }
                calibrated["launch"] = 0.0
                launch_detail = {
                    **(launch_detail or {
                        "launches": 1, "per_launch_ms": launch_ms,
                        "reason": "single_launch",
                    }),
                    "composition": "serial",
                    "folded_into_service_ms": launch_ms,
                }
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
            launch_detail=launch_detail,
            service_demands=service_demands,
        )

    def cost_iteration(self, work: IterationWork) -> list[CostedOp]:
        """Cost every op, then fold each MC2 chunk into its fused-kernel estimate.

        The MoE lowering keeps dispatch -> grouped GEMM -> combine as separate
        dependency nodes because they are real scheduling dependencies.  MC2 runs
        them inside one kernel, so the chunk's overlap has to be computed across
        those nodes (:meth:`compose_mc2`); a per-op ``cost`` can only report the
        serial decomposition.
        """
        if not work.ops:
            raise A3InputError("iteration_work_empty")
        # A kernel stream pays its dispatch gaps in sequence when the kernels are
        # short and back to back (a decode step) or when the bucket is replayed; a
        # long prefill kernel's dispatch can overlap its own service instead.
        replayed = getattr(work, "graph_bucket", None) is not None
        decode = str(getattr(work, "mode", "") or "") == "decode"
        composition = "serial" if (replayed or decode) else self.launch_composition
        costed = [self.cost(op, launch_composition=composition) for op in work.ops]
        return self.compose_mc2(work.ops, costed)

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
    "PHYSICAL_TRANSFER", "physical_prediction_policy",
    "physical_calibration_provenance", "physical_wiring",
    "PHYSICAL_TRANSFER_TARGETS", "PHYSICAL_TRANSFER_ARTIFACT_FAMILY",
    "PHYSICAL_TRANSFER_FALLBACK_TARGETS", "PHYSICAL_TRANSFER_ALIASES",
    "PHYSICAL_GEMM_ARTIFACT_FAMILY",
    "PHYSICAL_GEMM_SHAPED_ONLY", "physical_transfer_eligible",
    "PHYSICAL_LOW_CONFIDENCE_ANALYTIC_FAMILIES", "physical_transfer_target",
    "LARGE_OPERATOR_FAMILIES", "operator_class",
    "MC2_OVERLAP_UNVERIFIED", "MC2_OVERLAP_CALIBRATED", "MC2_NO_COMMUNICATION",
    "MC2_RECIPES", "MC2_FAMILIES", "MC2_COMPUTE_FAMILIES",
    "MC2_BYTES_FORMULA", "MC2_LANE_RANKS", "MC2_CHAIN_FACTOR_BOUNDS",
    "PHYSICAL_UNEXPRESSIBLE_RECIPES",
    "is_mc2_op", "mc2_chunk_key", "mc2_comm_bytes",
    "mc2_time_bounds", "communication_lane_curve",
    "GROUPED_MATMUL_FAMILIES", "GROUPED_LAUNCH_EXPERT_CAP",
    "grouped_expert_count", "grouped_matmul_launch_count",
]
