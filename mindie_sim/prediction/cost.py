"""One model-independent pricing path for lowered operations.

[Role] Combine physical work with optional primitive calibration parameters.
[Invariants] Physical FLOPs/bytes are never inflated to fit a measured latency.
  A measured duration is a separate kernel completion constraint. Missing
  calibration leaves the mechanism's declared parameters unchanged.
[Timeouts & Locks] Pure arithmetic and indexed, cached PrimitiveStore reads.
[Side-Effect Policy] Pricing does not record measurements or start device work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.case import Case as CaseSpec
from .primitives import PrimitiveKey, PrimitiveStore, PrimitiveTiming


@dataclass(frozen=True)
class OpCost:
    op: OpSpec
    key: PrimitiveKey | None
    work: Mapping[str, float]
    timing: PrimitiveTiming | None

    @property
    def basis(self) -> str:
        return "measured_primitive" if self.timing is not None else "analytic_work"


def primitive_key(case: CaseSpec, op: OpSpec, replay_scope: str, *, hardware=None) -> PrimitiveKey:
    metadata = dict(op.metadata)
    implementation = {
        "runtime": case.runtime.vllm_impl,
        "revision": case.runtime.vllm_version,
        "format": case.runtime.kernel_format,
        "kernel_bindings": case.runtime.kernels,
        "graph": replay_scope or "none",
        "quantization": metadata.get("weight_format", op.dtype),
        # Lowering may declare extra implementation axes. They belong to the
        # kernel signature, whereas request/layer ids and model labels do not.
        "axes": metadata.get("implementation", {}),
    }
    if "peer_traffic" in metadata:
        implementation["peer_traffic"] = metadata["peer_traffic"]
    return PrimitiveKey.from_mapping({
        "hardware": hardware or {
            "identity": case.machine.hardware_key or case.machine.sku,
            "dies": [die.to_dict() for die in case.machine.dies],
            "compute_peaks": dict(case.machine.compute_peaks),
        },
        "architecture": case.machine.arch_family,
        "implementation": implementation,
        "kernel": metadata.get("kernel", metadata.get("kernel_name", op.recipe)),
        "family": op.family,
        "recipe": op.recipe,
        "phase": op.phase,
        "dtype": op.dtype or case.model.compute_dtype,
        "layout": op.layout,
        "shape": {
            "semantic": op.semantic_shape, "local": op.local_shape,
            "padded": op.padded_shape, "launch": op.launch_shape,
        },
        "work": {
            "flops": op.flops, "vector_ops": op.vector_ops,
            **({"matrix_flops": dict(op.matrix_flops)} if op.matrix_flops else {}),
            "hbm_bytes": op.hbm_bytes, "mte2_bytes": op.mte2_bytes,
            "mte3_bytes": op.mte3_bytes, "link_bytes": op.link_bytes,
            "resource_work": metadata.get("resource_work", {}),
        },
        "scope": "resource_phase" if metadata.get("device_kernel_group") else "device_kernel",
    })


class CostModel:
    def __init__(self, case: CaseSpec, primitives: PrimitiveStore | None = None):
        self.case = case
        self.primitives = primitives
        self.hardware = {
            "identity": case.machine.hardware_key or case.machine.sku,
            "dies": [die.to_dict() for die in case.machine.dies],
            "compute_peaks": dict(case.machine.compute_peaks),
        }
        self.counts = {"measured_primitive": 0, "analytic_work": 0}
        self.services = {}
        self.mappings = {}
        self.unrecorded_mappings = 0
        self.invocations = {}
        self.regimes = {}
        self.joint_native = {}

    def price(self, op: OpSpec, replay_scope: str) -> OpCost:
        if op.matrix_flops:
            import math
            if not math.isclose(sum(v for _, v in op.matrix_flops), op.flops, rel_tol=1e-12):
                raise ValueError("matrix_precision_work_must_sum_to_operator_flops")
        key = primitive_key(self.case, op, replay_scope, hardware=self.hardware) if self.primitives else None
        timing = self.primitives.lookup(key) if self.primitives is not None else None
        work = {
            "cube": op.flops, "vector": op.vector_ops, "hbm": op.hbm_bytes,
            "mte2": op.mte2_bytes, "mte3": op.mte3_bytes, "link": op.link_bytes,
        }
        metadata = dict(op.metadata)
        for resource, field in (("mte1", "mte1_bytes"), ("fixpipe", "fixpipe_bytes")):
            value = float(metadata.get(field, 0.0) or 0.0)
            if value:
                work[resource] = value
        for resource, amount in dict(metadata.get("resource_work", {})).items():
            work[resource] = work.get(resource, 0) + amount
        result = OpCost(op, key, {k: float(v) for k, v in work.items() if v}, timing)
        self.counts[result.basis] += 1
        return result

    def record_service(self, curve, basis, *, capacity_limited=False):
        if curve.model_id not in self.services:
            data = curve.to_dict()
            self.services[curve.model_id] = {
                "model_id": curve.model_id, "resource": curve.resource,
                "work_unit": curve.work_unit, "identity": data["identity"],
                "boundary": curve.boundary, "active_units": curve.active_units,
                "provenance": data["provenance"], "queries": {},
                "capacity_limited_queries": 0,
            }
        record = self.services[curve.model_id]
        record["queries"][basis] = record["queries"].get(basis, 0) + 1
        record["capacity_limited_queries"] += int(capacity_limited)

    def record_mapping(self, mapping):
        import json
        key = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
        if key in self.mappings:
            self.mappings[key]["invocations"] += 1
        elif len(self.mappings) < 32:
            self.mappings[key] = {**mapping, "invocations": 1}
        else:
            self.unrecorded_mappings += 1

    def record_invocation(self, record):
        key = record["id"]
        if key not in self.invocations:
            self.invocations[key] = {**record, "invocations": 0}
        self.invocations[key]["invocations"] += 1

    def record_regime(self, model, key, basis, *, duration_us=None):
        record = self.regimes.setdefault(key, {
            "model_id": key, **model, "queries": {}, "response_us_min": None,
            "response_us_max": None})
        record["queries"][basis] = record["queries"].get(basis, 0) + 1
        if duration_us is not None:
            for field, select in (("response_us_min", min), ("response_us_max", max)):
                record[field] = duration_us if record[field] is None else select(record[field], duration_us)

    def record_joint_native(self, declaration, reference_cost):
        self.counts[reference_cost.basis] -= 1
        self.counts["joint_native_service"] = self.counts.get("joint_native_service", 0) + 1
        key = declaration["native_kernel"], declaration["resource"]
        row = self.joint_native.setdefault(key, {
            **declaration, "invocations": 0, "analytical_reference_work": {},
            "replaced_primitive_count": 0,
            "execution_basis": "declared_joint_native_service_queue",
            "unmodeled": ["internal_resource_occupancy", "internal_buffer_occupancy"]})
        row["invocations"] += 1
        row["replaced_primitive_count"] += int(reference_cost.timing is not None)
        for role, value in reference_cost.work.items():
            row["analytical_reference_work"][role] = row["analytical_reference_work"].get(role, 0) + value

    def summary(self) -> dict[str, Any]:
        sources = dict(self.case.runtime.metadata).get("primitive_parameter_sources", ())
        calibration = []
        if self.primitives and self.counts["measured_primitive"]:
            calibration.append("primitive_completion_parameters")
        if sources:
            calibration.append("declared_primitive_service_parameters")
        if self.services:
            calibration.append("resource_service_curves")
        if self.regimes:
            calibration.append("observable_kernel_regimes")
        declared_regimes = [row["id"] for row in self.case.runtime.kernels.get("service_regimes", ())]
        return {
            "operator_occurrences": dict(self.counts),
            "calibration": "+".join(calibration) or "none",
            "declared_parameter_sources": sources,
            **({"invocation_models": list(self.invocations.values())} if self.invocations else {}),
            **({"service_models": list(self.services.values())} if self.services else {}),
            **({"service_regimes": list(self.regimes.values())} if self.regimes else {}),
            **({"joint_native_services": list(self.joint_native.values()),
                "work_basis": "logical_and_analytical_reference; joint_native_queues_replace_selected_service"}
               if self.joint_native else {}),
            **({"regime_selection": {"declared": declared_regimes,
                "unmatched": sorted(set(declared_regimes) - {r["id"] for r in self.regimes.values()})}}
               if declared_regimes else {}),
            **({"matrix_mappings": list(self.mappings.values()),
                "unrecorded_mapping_invocations": self.unrecorded_mappings}
               if self.mappings else {}),
        }
