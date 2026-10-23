"""Explicit configuration and a finite, model-independent operator DAG report.

[Role] Assemble opt-in estimators and execute declared operator DAGs.
[Invariants] Observations keep exact conditions; existing calibration wins.
[Timeouts & Locks] Pure orchestration and bounded provider lookups; no devices.
[Side-Effect Policy] Read only explicitly selected inputs; no writes.
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import math
from typing import Any, Mapping

from .a3_cost import PredictiveCostModel
from .a3_ops import OpSpec
from .a3_resources import ResourceEngine, add_costed_ops, build_resource_specs
from .a3_types import A3InputError, MachineSpec
from .operator_plan import CompositeEstimator, EstimateContext


def create_operator_estimator(config: Mapping[str, Any]) -> CompositeEstimator:
    """Providers are optional, statically selected and constructed only on opt-in."""
    allowed = {"vector_profile", "gemm_profile", "cannbot_repo", "cannbot_revision", "strict", "profiled_kernels"}
    if not isinstance(config, Mapping) or set(config)-allowed:
        raise A3InputError("operator_estimator_config_invalid")
    if not isinstance(config.get("strict", False), bool):
        raise A3InputError("operator_estimator_strict_invalid")
    providers = []
    if config.get("profiled_kernels"):
        from .profiled_kernel import ProfiledKernelProvider
        providers.append(ProfiledKernelProvider(config["profiled_kernels"]))
    if config.get("cannbot_repo"):
        from .operator_vector import CannbotVectorProvider, CANNBOT_REVISION
        providers.append(CannbotVectorProvider(
            config["cannbot_repo"], revision=config.get("cannbot_revision", CANNBOT_REVISION),
        ))
    vector = config.get("vector_profile")
    if vector is not None and vector is not False:
        from .operator_vector import VectorInstructionProvider, VectorProfile
        if vector is not True and not isinstance(vector, Mapping):
            raise A3InputError("operator_vector_profile_invalid")
        profile = None if vector is True else VectorProfile.from_mapping(vector)
        providers.append(VectorInstructionProvider(profile))
    gemm = config.get("gemm_profile")
    if gemm is not None:
        from .operator_gemm import GemmProvider, GemmProfile
        if not isinstance(gemm, Mapping):
            raise A3InputError("operator_gemm_profile_invalid")
        providers.append(GemmProvider(GemmProfile.from_mapping(gemm)))
    if not providers:
        raise A3InputError("operator_estimators_empty")
    return CompositeEstimator(tuple(providers), strict=config.get("strict", False))


def _operator(raw: Mapping[str, Any]) -> OpSpec:
    if not isinstance(raw, Mapping) or set(raw)-{f.name for f in fields(OpSpec)}:
        raise A3InputError("operator_input_invalid")
    data = dict(raw)
    for key in ("op_id", "family", "launch_shape"):
        if not data.get(key):
            raise A3InputError(f"operator_input_missing:{key}")
    for key in ("semantic_shape", "local_shape", "padded_shape"):
        data.setdefault(key, data["launch_shape"])
    for key in ("semantic_shape", "local_shape", "padded_shape", "launch_shape"):
        value = data[key]
        if not isinstance(value, (list, tuple)) or any(
            isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in value
        ):
            raise A3InputError(f"operator_shape_invalid:{key}")
        data[key] = tuple(value)
    data.setdefault("phase", "prefill")
    data.setdefault("stage", 0)
    data.setdefault("layer", 0)
    data.setdefault("requests", ())
    data["requests"] = tuple(data["requests"])
    data["deps"] = tuple(data.get("deps", ()))
    data["notes"] = tuple(data.get("notes", ()))
    metadata = data.get("metadata", {})
    data["metadata"] = tuple(metadata.items()) if isinstance(metadata, Mapping) else tuple(metadata)
    amounts = []
    for key in ("flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes", "link_bytes"):
        value = data.get(key, 0)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise A3InputError(f"operator_work_invalid:{key}")
        amounts.append(value)
    declared_metadata = dict(data["metadata"])
    if not any(amounts) and not (
        declared_metadata.get("witness_only") or declared_metadata.get("not_execution")
        or declared_metadata.get("serial_ms", 0) > 0
        or declared_metadata.get("profile_variant")
    ):
        raise A3InputError("operator_baseline_work_required")
    if isinstance(data["stage"], bool) or not isinstance(data["stage"], int) or data["stage"] < 0:
        raise A3InputError("operator_stage_invalid")
    return OpSpec(**data)


def operator_plan_report(payload: Mapping[str, Any], *, include_task_ids: bool = False) -> dict[str, Any]:
    allowed = {"machine", "operators", "estimators", "context", "name", "source"}
    if not isinstance(payload, Mapping) or set(payload)-allowed:
        raise A3InputError("operator_report_input_invalid")
    machine = MachineSpec.from_mapping(payload.get("machine"))
    context = EstimateContext(**dict(payload.get("context", {})))
    provider = create_operator_estimator(payload.get("estimators", {}))
    operators = tuple(_operator(raw) for raw in payload.get("operators", ()))
    if not operators:
        raise A3InputError("operators_empty")
    seen: set[str] = set()
    for op in operators:
        if op.op_id in seen or any(dep not in seen for dep in op.deps):
            raise A3InputError(f"operator_dependency_or_duplicate:{op.op_id}")
        seen.add(op.op_id)
    model = PredictiveCostModel(
        machine, compute_dtype=context.compute_dtype,
        weight_dtype=context.weight_dtype, activation_dtype=context.activation_dtype,
        timing_scope=context.timing_scope, op_estimator=provider,
        quantization=context.quantization,
    )
    costs = [model.cost(op) for op in operators]
    resources, stage_map = build_resource_specs(
        machine, stages=max(op.stage for op in operators)+1,
        compute_dtype=context.compute_dtype,
    )
    engine = ResourceEngine(resources)
    ids = add_costed_ops(engine, costs, stage_map, prefix="operators", start_ms=0.0)
    execution = engine.run(until=tuple(ids.values()))
    report = {
        "status": "modeled", "command": "operator-plan",
        "name": payload.get("name", "operator-dag"), "source": payload.get("source", "declared"),
        "input_sha256": hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest(),
        "effective_input": dict(payload),
        "scope": "declared finite operator DAG; observed kernel costs and/or analytic plans; not independent accuracy acceptance",
        "timing_scope": context.timing_scope,
        "elapsed_ms": max(execution["completed_at_ms"][key] for key in ids.values()),
        "operators": [cost.to_dict() for cost in costs],
        "execution": execution,
    }
    if include_task_ids:
        report["task_ids"] = dict(ids)
    return report
