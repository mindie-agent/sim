"""Public interfaces for graph rewrite plans, verification, and replay.

[Role] Publish the canonical rewrite algebra and executable oracle.
[Invariants] Re-export the defining modules' public names without wrappers.
[Timeouts & Locks] Imports use normal Python module initialization.
[Side-Effect Policy] No execution, device access, or external writes on import.
"""
from .algebra import (
    AlgebraicCandidate, AlgebraicRewriteResult, AlgebraicTrait,
    OpTraitInfo, SemanticAuditTrigger, classify_op_traits,
    discover_algebraic_rewrites, discover_algebraic_rewrites_for_ops,
)
from .apply import (
    AppliedPlan, CollectiveRewriteRecord, EditReceipt, PrefetchLease, ResolvedScenario,
    TemplateBinding, TileBinding, apply, apply_plan, resolve_scenario,
)
from .oracle import (
    BatchEvaluation, CandidateEvaluation, EvalBudget, FullPlanEvaluation, OracleCache,
    ScenarioDossier, ScenarioHandle, ScenarioRequest, StepArchetype, evaluate_rewrites,
    finalize_plan, intern_step_archetypes, prepare_scenario,
)
from .schema import (
    COLLECTIVE_TRANSFORM_MODES, MAIN_STREAM, REWRITE_PLAN_SCHEMA, CollectiveMode,
    CollectiveTransform, Edit, FuseRegion, FusionTier,
    HoistOrDeduplicate, KernelBoundary, OpRef, PipelineTiles, PortRef, QuantizeContract,
    RewriteOp, RewritePlan, SelectTemplate, StreamCoreBudget, StreamEvent, StreamId,
    StreamLane, StreamSchedule, TemplateTier, TilePipelineSpec, VariantRef, WeightPrefetch,
    decode_rewrite_plan, empty_plan, encode_rewrite_plan, load_rewrite_plan, plan_sha256,
)
from .verify import (
    ACCEL_MAX_L, ACCEL_MAX_R, ACCEL_MAX_T, ACCEL_MAX_U, FAKE_OVERLAP_THRESHOLD_PCT,
    SHIPPING_REGRESSION_TOLERANCE_NS, GateReceipt,
    ShippingRegression, SoundnessInversionError, VerificationReceipt, compute_overlap_pct,
    verify_plan,
)

__all__ = [
    "AlgebraicCandidate", "AlgebraicRewriteResult", "AlgebraicTrait",
    "OpTraitInfo", "SemanticAuditTrigger", "classify_op_traits",
    "discover_algebraic_rewrites", "discover_algebraic_rewrites_for_ops",
    "AppliedPlan", "CollectiveRewriteRecord", "EditReceipt", "PrefetchLease",
    "ResolvedScenario", "TemplateBinding", "TileBinding", "apply", "apply_plan",
    "resolve_scenario", "BatchEvaluation", "CandidateEvaluation", "EvalBudget",
    "FullPlanEvaluation", "OracleCache", "ScenarioDossier", "ScenarioHandle",
    "ScenarioRequest", "StepArchetype", "evaluate_rewrites", "finalize_plan",
    "intern_step_archetypes", "prepare_scenario", "COLLECTIVE_TRANSFORM_MODES",
    "CollectiveMode", "MAIN_STREAM", "REWRITE_PLAN_SCHEMA",
    "CollectiveTransform", "Edit", "FuseRegion", "FusionTier", "HoistOrDeduplicate",
    "KernelBoundary", "OpRef", "PipelineTiles", "PortRef", "QuantizeContract",
    "RewriteOp", "RewritePlan", "SelectTemplate", "StreamCoreBudget", "StreamEvent",
    "StreamId", "StreamLane", "StreamSchedule", "TemplateTier", "TilePipelineSpec",
    "VariantRef", "WeightPrefetch", "decode_rewrite_plan", "empty_plan",
    "encode_rewrite_plan", "load_rewrite_plan", "plan_sha256", "ACCEL_MAX_L", "ACCEL_MAX_R", "ACCEL_MAX_T",
    "ACCEL_MAX_U", "FAKE_OVERLAP_THRESHOLD_PCT", "SHIPPING_REGRESSION_TOLERANCE_NS",
    "GateReceipt", "ShippingRegression",
    "SoundnessInversionError", "VerificationReceipt", "compute_overlap_pct", "verify_plan",
]
