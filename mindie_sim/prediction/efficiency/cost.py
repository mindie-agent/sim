"""One model-independent pricing path for lowered operations.

[Role] Combine physical work with the declared resource/phase supply.  Primitive
  observations are explicit diagnostics, never a production completion floor.
[Invariants] Physical FLOPs/bytes are never inflated to fit a measured latency.
  Ordinary ``price`` consumes physical work only.  A PrimitiveStore lookup is
  available only through ``lookup_diagnostic`` and never enters ``OpCost.timing``
  or ``minimum_ms``.  A normalized T06 profile attached to the Case is exposed
  for the executor/recipe owner; this module does not invent a second service.
[Timeouts & Locks] Pure arithmetic; diagnostic store reads are indexed/cached.
[Side-Effect Policy] Pricing does not record measurements or start device work.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from ..evaluation import FrozenHardwareView

from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.prediction.program.primitives import PrimitiveKey, PrimitiveStore, PrimitiveTiming

MAX_RECORDED_MAPPINGS = 32
MATRIX_FLOPS_REL_TOL = 1e-9


@dataclass(frozen=True)
class OpCost:
    op: OpSpec
    key: PrimitiveKey | None
    work: Mapping[str, float]
    #: Always ``None`` in production.  Kept for callers that read the old
    #: attribute; a diagnostic lookup is returned by ``CostModel.lookup_diagnostic``.
    timing: PrimitiveTiming | None = None

    @property
    def basis(self) -> str:
        return "analytic_work"


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
    if metadata.get("pr_source"):
        implementation["pr_source"] = metadata["pr_source"]
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


def _resource_service_context(context):
    """Map a real T04 RecipeContext (or a mapping) to applicability fields.

    T04 owns the execution context.  This adapter only presents its actual
    service identity, resource binding, units and phase to the T06 condition
    matcher.  An A5/T09 domain may be attached as ``service_domain``; missing
    declared fields remain missing and produce the normal explicit rejection.
    """
    if isinstance(context, Mapping):
        return dict(context)
    identity = dict(getattr(context, "service_identity", {}) or {})
    hardware = dict(identity.get("hardware") or {})
    op = getattr(context, "op", None)
    metadata = dict(getattr(op, "metadata", ()) or ())
    domain = dict(getattr(context, "service_domain", {}) or {})
    if not domain:
        domain = dict(metadata.get("service_domain") or {})
    view = {
        "service_identity": identity,
        "hardware": hardware.get("key"),
        "platform": {
            "key": hardware.get("key"),
            "family": hardware.get("architecture"),
        },
        "runtime": identity.get("runtime"),
        "revision": identity.get("revision"),
        "format": identity.get("format"),
        "dtype": identity.get("dtype"),
        "layout": identity.get("layout"),
        "axes": dict(identity.get("axes") or {}),
        "resource_names": dict(getattr(context, "resource_names", {}) or {}),
        "resource_units": dict(getattr(context, "resource_units", {}) or {}),
        "work_units": dict(getattr(context, "work_units", {}) or {}),
        "work": dict(getattr(context, "work", {}) or {}),
        "phase": getattr(op, "phase", None),
        "op_id": getattr(op, "op_id", None),
        "op_family": getattr(op, "family", None),
        "op_stage": getattr(op, "stage", None),
    }
    if domain:
        view = {**view, **deepcopy(domain)}
    return view


class CostLedger:
    """Mutable diagnostic records owned by exactly one isolated execution.

    ``CostModel`` supplies immutable hardware/service declarations and pricing
    logic.  Counts of what was priced, recorded mappings/invocations/regimes
    and diagnostic primitive lookups live here, so an isolated row's ``basis``
    contains only its own records and the main graph cannot absorb later
    isolated-analysis counts.
    """

    __slots__ = (
        "counts", "services", "mappings", "unrecorded_mappings",
        "invocations", "regimes", "joint_native",
        "diagnostic_primitive_lookups", "diagnostic_primitive_hits",
    )

    def __init__(self) -> None:
        self.counts = {"measured_primitive": 0, "analytic_work": 0}
        self.services: dict[str, dict[str, Any]] = {}
        self.mappings: dict[str, dict[str, Any]] = {}
        self.unrecorded_mappings = 0
        self.invocations: dict[str, dict[str, Any]] = {}
        self.regimes: dict[str, dict[str, Any]] = {}
        self.joint_native: dict[Any, dict[str, Any]] = {}
        self.diagnostic_primitive_lookups = 0
        self.diagnostic_primitive_hits = 0

    def account(self, basis: str) -> None:
        self.counts[basis] = self.counts.get(basis, 0) + COUNT_STEP

    def summary(self, model: "CostModel") -> dict[str, Any]:
        return model.summary(self)


COUNT_STEP = 1


class CostModel:
    def __init__(
        self, case: CaseSpec, primitives: PrimitiveStore | None = None,
        *, hardware: "FrozenHardwareView | None" = None, ledger: CostLedger | None = None,
        primitive_table: Any = None,
    ):
        self.case = case
        # The store is retained only for explicit diagnostic reads.  It is not
        # consulted by price() and cannot set a production completion floor.
        self.primitives = primitives
        self.primitive_table = primitive_table
        if hardware is None:
            from mindie_sim.prediction.evaluation import FrozenHardwareView
            hardware = FrozenHardwareView.from_machine(case.machine)
        self.hardware_view = hardware
        self.hardware = hardware.payload()
        self.ledger = ledger if ledger is not None else CostLedger()
        self.resource_service_profile = None
        self.resource_services = ()
        self._load_resource_services()

    @property
    def counts(self):
        return self.ledger.counts

    @property
    def services(self):
        return self.ledger.services

    @property
    def mappings(self):
        return self.ledger.mappings

    @property
    def unrecorded_mappings(self):
        return self.ledger.unrecorded_mappings

    @property
    def invocations(self):
        return self.ledger.invocations

    @property
    def regimes(self):
        return self.ledger.regimes

    @property
    def joint_native(self):
        return self.ledger.joint_native

    @property
    def diagnostic_primitive_lookups(self):
        return self.ledger.diagnostic_primitive_lookups

    @property
    def diagnostic_primitive_hits(self):
        return self.ledger.diagnostic_primitive_hits

    def _load_resource_services(self):
        metadata = dict(getattr(self.case.runtime, "metadata", ()) or ())
        profile = metadata.get("resource_service_profile")
        if profile is None:
            return
        if not isinstance(profile, Mapping):
            raise ValueError("resource_service_profile_requires_mapping")
        # Delegate to the one public profile loader: schema validation,
        # top-level hardware propagation, file/mapping consistency, frozen
        # reload and provenance all stay in the observations layer.
        from mindie_sim.observations.calibration import load_resource_profile
        loaded = load_resource_profile(profile)
        from mindie_sim.observations.resource_services import (
            normalize_resource_service,
        )
        normalized = tuple(
            normalize_resource_service(item) for item in loaded["services"])
        identifiers = [item.service_id for item in normalized]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("resource_service_profile_requires_distinct_ids")
        self.resource_service_profile = loaded
        self.resource_services = normalized

    def lookup_diagnostic(self, op: OpSpec, replay_scope: str, *, ledger: CostLedger | None = None):
        """Explicit primitive-evidence read; never a production price or floor."""
        if self.primitives is None:
            return None
        ledger = self.ledger if ledger is None else ledger
        key = primitive_key(self.case, op, replay_scope, hardware=self.hardware)
        ledger.diagnostic_primitive_lookups += COUNT_STEP
        timing = self.primitives.lookup(key)
        if timing is not None:
            ledger.diagnostic_primitive_hits += COUNT_STEP
        return {"key": key, "timing": timing}

    def select_resource_service(self, context: Mapping[str, Any], *,
                                service_id: str | None = None):
        """Select one declared T06 service for a real placement context.

        The executor supplies the actual stage rates/units/owner/software
        context.  A missing or ambiguous declaration is an explicit input
        error, not a fallback to a nominal rate.
        """
        from mindie_sim.observations.calibration import (
            select_resource_service_candidate,
        )
        context = _resource_service_context(context)
        return select_resource_service_candidate(
            self.resource_services, context, service_id=service_id)

    def bind_operator_resource_service(self, op: OpSpec, context: Mapping[str, Any],
                                       *, service_id: str | None = None,
                                       resource_role: str | None = None,
                                       phase: str | None = None):
        """Select a profile service and generate its exact op metadata binding.

        T14 calls this after choosing the actual program/phase/context; the
        returned OpSpec is what T04's existing resource-service stage cap/fixed
        path consumes.  Selection and binding are one operation, not a
        separate unconsumed lookup.
        """
        from mindie_sim.observations.calibration import (
            bind_operator_resource_service as _bind,
        )
        context = _resource_service_context(context)
        service = self.select_resource_service(context, service_id=service_id)
        return _bind(
            op, service, resource_role=resource_role, phase=phase or op.phase,
            work_unit=service.work_unit, conditions=context)

    def price(self, op: OpSpec, replay_scope: str, *, ledger: CostLedger | None = None) -> OpCost:
        matrix_sum = (sum(float(value) for _, value in op.matrix_flops)
                      if op.matrix_flops else 0.0)
        declared_flops = float(op.flops)
        if (op.matrix_flops and declared_flops > 0.0
                and not math.isclose(matrix_sum, declared_flops,
                                     rel_tol=MATRIX_FLOPS_REL_TOL)):
            raise ValueError("matrix_precision_work_must_sum_to_operator_flops")
        cube_flops = matrix_sum if op.matrix_flops else declared_flops
        work = {
            "cube": cube_flops, "vector": op.vector_ops, "hbm": op.hbm_bytes,
            "mte2": op.mte2_bytes, "mte3": op.mte3_bytes, "link": op.link_bytes,
        }
        metadata = dict(op.metadata)
        for resource, field in (("mte1", "mte1_bytes"), ("fixpipe", "fixpipe_bytes")):
            value = float(metadata.get(field, 0.0) or 0.0)
            if value:
                work[resource] = value
        for resource, amount in dict(metadata.get("resource_work", {})).items():
            if resource == "cube" and cube_flops > 0.0:
                continue
            work[resource] = work.get(resource, 0) + amount
        result = OpCost(op, None, {k: float(v) for k, v in work.items() if v})
        (self.ledger if ledger is None else ledger).account(result.basis)
        return result

    def record_service(self, curve, basis, *, capacity_limited=False, ledger: CostLedger | None = None):
        ledger = self.ledger if ledger is None else ledger
        services = ledger.services
        if curve.model_id not in services:
            data = curve.to_dict()
            services[curve.model_id] = {
                "model_id": curve.model_id, "resource": curve.resource,
                "work_unit": curve.work_unit, "identity": data["identity"],
                "boundary": curve.boundary, "active_units": curve.active_units,
                "provenance": data["provenance"], "queries": {},
                "capacity_limited_queries": 0,
            }
        record = services[curve.model_id]
        record["queries"][basis] = record["queries"].get(basis, 0) + 1
        record["capacity_limited_queries"] += int(capacity_limited)

    def record_mapping(self, mapping, *, ledger: CostLedger | None = None):
        import json
        ledger = self.ledger if ledger is None else ledger
        key = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
        mappings = ledger.mappings
        if key in mappings:
            mappings[key]["invocations"] += COUNT_STEP
        elif len(mappings) < MAX_RECORDED_MAPPINGS:
            mappings[key] = {**mapping, "invocations": COUNT_STEP}
        else:
            ledger.unrecorded_mappings += COUNT_STEP

    def record_invocation(self, record, *, ledger: CostLedger | None = None):
        ledger = self.ledger if ledger is None else ledger
        key = record["id"]
        invocations = ledger.invocations
        if key not in invocations:
            invocations[key] = {**record, "invocations": 0}
        invocations[key]["invocations"] += COUNT_STEP

    def record_regime(self, model, key, basis, *, duration_us=None, coordinate_values=None,
                      response_basis=None, mismatched_axes=(), continuity_group=None,
                      execution_mode=None, exterior_policy=None, issued_work=None,
                      rate_flops_per_s=None, active_owners=None, stage=None,
                      ledger: CostLedger | None = None):
        ledger = self.ledger if ledger is None else ledger
        record = ledger.regimes.setdefault(key, {
            "model_id": key, **model, "queries": {}, "response_us_min": None,
            "response_us_max": None})
        record["queries"][basis] = record["queries"].get(basis, 0) + 1
        if response_basis is not None:
            counts = record.setdefault("response_relations", {})
            counts[response_basis] = counts.get(response_basis, 0) + 1
            bases = record.setdefault("response_bases", {})
            bases[response_basis] = bases.get(response_basis, 0) + 1
        if continuity_group is not None:
            groups = record.setdefault("continuity_groups", {})
            groups[continuity_group] = groups.get(continuity_group, 0) + 1
        if execution_mode is not None:
            modes = record.setdefault("execution_modes", {})
            modes[execution_mode] = modes.get(execution_mode, 0) + 1
        if exterior_policy is not None:
            policies = record.setdefault("exterior_policies", {})
            policies[exterior_policy] = policies.get(exterior_policy, 0) + 1
        if issued_work is not None:
            record["issued_work"] = max(
                float(record.get("issued_work", 0.0)), float(issued_work))
        if rate_flops_per_s is not None:
            record["rate_flops_per_s"] = float(rate_flops_per_s)
        if active_owners is not None:
            record["active_owners"] = int(active_owners)
        if stage is not None:
            record["stage"] = str(stage)
        for axis in mismatched_axes:
            counts = record.setdefault("mismatched_axes", {})
            counts[axis] = counts.get(axis, 0) + 1
        for name, value in (coordinate_values or {}).items():
            bounds = record.setdefault("coordinate_ranges", {}).setdefault(
                name, {"min": value, "max": value})
            bounds["min"], bounds["max"] = min(bounds["min"], value), max(bounds["max"], value)
        if duration_us is not None:
            for field, select in (("response_us_min", min), ("response_us_max", max)):
                record[field] = duration_us if record[field] is None else select(record[field], duration_us)
            record["response_ms_total"] = record.get("response_ms_total", 0.0) + duration_us / 1000
            record["response_ms_max"] = max(record.get("response_ms_max", 0.0), duration_us / 1000)

    def record_joint_native(self, selection, reference_cost, *, ledger: CostLedger | None = None):
        ledger = self.ledger if ledger is None else ledger
        declaration = dict(selection.execution or {})
        counts = ledger.counts
        counts[reference_cost.basis] -= COUNT_STEP
        counts["joint_native_service"] = counts.get(
            "joint_native_service", 0) + COUNT_STEP
        key = declaration["native_kernel"], declaration["resource"]
        row = ledger.joint_native.setdefault(key, {
            **declaration, "invocations": 0, "analytical_reference_work": {},
            "replaced_primitive_count": 0,
            "execution_basis": "declared_joint_native_service_queue",
            "unmodeled": ["internal_resource_occupancy", "internal_buffer_occupancy"]})
        row["invocations"] += 1
        has_primitive = reference_cost.timing is not None
        if not has_primitive and self.primitives is not None:
            for scope in dict.fromkeys((self.case.runtime.graph_capture_mode, "none")):
                if self.primitives.lookup(primitive_key(self.case, reference_cost.op, scope, hardware=self.hardware)) is not None:
                    has_primitive = True
                    break
        row["replaced_primitive_count"] += int(has_primitive)
        rule_counts = dict(row.get("rule_counts", {}))
        if selection.rule_id is not None:
            rule_counts[selection.rule_id] = rule_counts.get(selection.rule_id, 0) + 1
        row["rule_counts"] = {k: rule_counts[k] for k in sorted(rule_counts)}
        row["rule_ids"] = sorted(rule_counts)
        row["rule_id"] = row["rule_ids"][0] if len(row["rule_ids"]) == 1 else None
        row["continuity_group"] = (selection.continuity_group if row["invocations"] == 1
                                   or row.get("continuity_group") == selection.continuity_group
                                   else None)
        row["execution_mode"] = selection.mode
        bases = dict(row.get("response_bases", {}))
        if selection.response_basis is not None:
            bases[selection.response_basis] = bases.get(selection.response_basis, 0) + 1
        row["response_bases"] = {k: bases[k] for k in sorted(bases)}
        row["response_basis"] = next(iter(row["response_bases"])) if len(row["response_bases"]) == 1 else None
        coords = dict(selection.coordinate_values or {})
        row["coordinate_values"] = (coords if row["invocations"] == 1
                                    or row.get("coordinate_values") == coords else None)
        ranges = row.setdefault("coordinate_ranges", {})
        for name, value in coords.items():
            bounds = ranges.setdefault(name, {"min": value, "max": value})
            bounds["min"], bounds["max"] = min(bounds["min"], value), max(bounds["max"], value)
        duration = selection.duration_ms
        row["response_duration_ms"] = (duration if row["invocations"] == 1
                                       or row.get("response_duration_ms") == duration else None)
        row["response_ms_min"] = min(row.get("response_ms_min", duration), duration)
        row["response_ms_max"] = max(row.get("response_ms_max", duration), duration)
        row["response_ms_total"] = row.get("response_ms_total", 0.0) + duration
        for role, value in reference_cost.work.items():
            row["analytical_reference_work"][role] = row["analytical_reference_work"].get(role, 0) + value

    def summary(self, ledger: CostLedger | None = None) -> dict[str, Any]:
        ledger = self.ledger if ledger is None else ledger
        counts, services = ledger.counts, ledger.services
        regimes, invocations = ledger.regimes, ledger.invocations
        mappings, joint_native = ledger.mappings, ledger.joint_native
        sources = dict(self.case.runtime.metadata).get("primitive_parameter_sources", ())
        calibration = []
        pt = self.primitive_table
        pt_warnings = tuple(getattr(pt, "warnings", ())) if pt is not None else ()
        rescaled = "primitive_table_rescaled_to_declared_die" in pt_warnings
        pt_calibrated = bool(
            pt is not None and getattr(getattr(pt, "provenance", None), "calibrated", False)
        )
        if pt is not None:
            calibration.append(
                "calibrated_primitive_table_rescaled" if rescaled
                else "calibrated_primitive_table" if pt_calibrated
                else "uncalibrated_primitive_table"
            )
        if ledger.diagnostic_primitive_hits:
            calibration.append("diagnostic_primitive_evidence_not_production")
        if sources:
            calibration.append("declared_primitive_service_parameters")
        if services:
            calibration.append("resource_service_curves")
        selected_regimes = {
            r["id"] for r in regimes.values()
            if (r["queries"].get("observable_response")
                or r["queries"].get("joint_native_service")
                or r["queries"].get("linear_outside_support"))}
        queried_regimes = {r["id"] for r in regimes.values()
                           if set(r["queries"]) - {"physical_identity_mismatch"}}
        if any(r["id"] in selected_regimes and r["response"].get("kind") != "roofline"
               for r in regimes.values()):
            calibration.append("observable_kernel_regimes")
        declared_regimes = [row["id"] for row in self.case.runtime.kernels.get("service_regimes", ())]
        declared_sku = str(self.case.machine.sku or "")
        pt_sku = getattr(pt, "sku", None)
        pt_cann = getattr(pt, "cann_version", None)
        pt_sha = getattr(pt, "envelope_sha256", None)
        calibration_basis = {
            "mode": (
                "calibrated_primitive_table_rescaled" if (pt is not None and rescaled)
                else "calibrated_primitive_table" if pt_calibrated
                else "uncalibrated_primitive_table" if pt is not None
                else "datasheet_declaration"
            ),
            "calibrated": pt_calibrated,
            "rescaled": rescaled,
            "declared_sku": declared_sku,
            "primitive_table_sku": str(pt_sku) if pt_sku is not None else None,
            "cann_version": str(pt_cann) if pt_cann is not None else None,
            "envelope_sha256": str(pt_sha) if pt_sha is not None else None,
            "warnings": (
                list(pt_warnings) if pt is not None
                else (["uncalibrated_datasheet_declaration"] if declared_sku else [])
            ),
        }
        return {
            "operator_occurrences": dict(counts),
            "production_price_basis": "physical_work_only",
            "diagnostic_primitive_lookups": ledger.diagnostic_primitive_lookups,
            "diagnostic_primitive_hits": ledger.diagnostic_primitive_hits,
            "calibration": "+".join(calibration) or "none",
            "calibration_basis": calibration_basis,
            "declared_parameter_sources": sources,
            **({"resource_services": [item.service_id
                                      for item in self.resource_services],
                "resource_service_profile_source": deepcopy(
                    self.resource_service_profile.get("source"))
                if isinstance(self.resource_service_profile, Mapping) else None}
               if self.resource_services else {}),
            **({"invocation_models": list(invocations.values())} if invocations else {}),
            **({"service_models": list(services.values())} if services else {}),
            **({"service_regimes": list(regimes.values())} if regimes else {}),
            **({"joint_native_services": list(joint_native.values()),
                "work_basis": "logical_and_analytical_reference; joint_native_queues_replace_selected_service"}
               if joint_native else {}),
            **({"regime_selection": {"declared": declared_regimes,
                "unmatched": sorted(set(declared_regimes) - queried_regimes),
                "unselected": sorted(set(declared_regimes) - selected_regimes)}}
               if declared_regimes else {}),
            **({"matrix_mappings": list(mappings.values()),
                "unrecorded_mapping_invocations": ledger.unrecorded_mappings}
               if mappings else {}),
        }
