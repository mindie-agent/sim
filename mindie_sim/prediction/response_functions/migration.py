"""Normalize service migration plans and their provenance.

[Role] Validate work and fixed references, scopes and provenance roles.
[Invariants] Preserve caller-declared service identities and program boundaries.
[Timeouts & Locks] Bounded by the supplied plan and provenance. No waits or locks.
[Side-Effect Policy] Returns normalized declarations without fitting or executing them.
"""
from collections.abc import Mapping
import math

from mindie_sim.schema.common import ModelingError

from .evaluation import positive


MIGRATION_PLAN_FIELDS = frozenset({
    "replacement_kind", "scope", "condition_context", "slice_conditions",
    "fixed", "work", "program", "context", "occurrences", "native_boundary"})
MIGRATION_WORK_FIELDS = frozenset({
    "term_index", "service_id", "stage", "resource", "work_unit",
    "work_scale", "multiplicity", "phase", "native_boundary", "work_scope"})
MIGRATION_FIXED_FIELDS = frozenset({"service_id", "stage", "native_boundary"})
MIGRATION_REPLACEMENT_KINDS = frozenset({
    "independent_service", "joint_phase_service"})
MIGRATION_SCOPES = frozenset({"complete_rule", "slice"})
_WORK_SCOPES = frozenset({"aggregate", "per_owner"})
_TRAINING_FIELDS = ("training_sources", "training", "train_sources")
_VALIDATION_FIELDS = ("validation_sources", "validation", "heldout",
                      "heldout_sources", "diagnostic_sources", "diagnostic")
_OTHER_SOURCE_FIELDS = ("source", "sources", "source_path", "sha256",
                        "source_sha256", "profile_source", "observation_source",
                        "calibration_source", "train_source")


def _plan_text(value, code):
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(code)
    return value.strip()


def _role_tokens(value):
    tokens = set()

    def walk(node):
        if isinstance(node, str):
            if node.strip():
                tokens.add(node.strip())
        elif isinstance(node, Mapping):
            for item in node.values():
                walk(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(value)
    return tokens


def provenance_roles(provenance):
    """Read T06 production source roles without merging training with validation."""
    if not isinstance(provenance, Mapping):
        provenance = {}
    fit = provenance.get("fit") if isinstance(provenance.get("fit"), Mapping) else {}
    roles = {}
    for role, fields in (("training", _TRAINING_FIELDS),
                         ("validation", _VALIDATION_FIELDS),
                         ("other", _OTHER_SOURCE_FIELDS)):
        tokens = set()
        for field in fields:
            if field in provenance:
                tokens |= _role_tokens(provenance[field])
            if field in fit:
                tokens |= _role_tokens(fit[field])
        roles[role] = sorted(tokens)
    roles["training_or_other"] = sorted(set(roles["training"]) | set(roles["other"]))
    return roles


def same_declared_value(left, right):
    """Compare declared condition values with numeric tolerance and structure."""
    if (isinstance(left, (int, float)) and not isinstance(left, bool)
            and isinstance(right, (int, float)) and not isinstance(right, bool)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (left.keys() == right.keys()
                and all(same_declared_value(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return (len(left) == len(right)
                and all(same_declared_value(a, b) for a, b in zip(left, right)))
    return left == right


def _program_view(program):
    if program is None:
        return None, (), None
    stages = getattr(program, "stages", None)
    if stages is None and isinstance(program, (list, tuple)):
        stages = program
    if stages is None:
        raise ModelingError("migration_plan_program_requires_stages")
    stages = tuple(stages)
    names, view = set(), []
    for stage in stages:
        name = getattr(stage, "name", None)
        work = getattr(stage, "work", None)
        if not isinstance(name, str) or not name or name in names or not isinstance(work, Mapping):
            raise ModelingError("migration_plan_program_stage_invalid:" + str(name))
        names.add(name)
        view.append({
            "name": name, "work": dict(work),
            "work_units": dict(getattr(stage, "work_units", {}) or {}),
            "after": list(getattr(stage, "after", ()) or ()),
            "previous": list(getattr(stage, "previous", ()) or ()),
            "acquire": [getattr(item, "name", str(item))
                        for item in getattr(stage, "acquire", ()) or ()],
            "free": [getattr(item, "name", str(item))
                     for item in getattr(stage, "free", ()) or ()],
            "placement": {key: list(value) for key, value in dict(
                getattr(stage, "placement", {}) or {}).items()},
            "fragment": getattr(getattr(stage, "fragment", None), "name", None),
            "repeats": getattr(getattr(stage, "fragment", None), "repeats", None),
        })
    return program, stages, view


def _fragment_evidence(fragment):
    work = getattr(fragment, "work", None)
    items = tuple(work.issued()) if work is not None and hasattr(work, "issued") else ()
    return (tuple((item.resource, float(item.amount), item.unit, item.kind)
                  for item in items),
            int(getattr(fragment, "repeats", 1)))


def normalize_migration_plan(plan, decomposition):
    """Validate a migration plan against real program/stage/context references.

    ``program``/``context`` are read-only references to the actual selected
    execution objects. Old ``term_index`` remains as a diagnostic link to the
    legacy formula; ``stage`` is the actual occurrence being replaced.
    """
    if not isinstance(plan, Mapping):
        raise ModelingError("migration_plan_requires_mapping")
    unknown = set(plan) - MIGRATION_PLAN_FIELDS
    if unknown:
        raise ModelingError("migration_plan_unknown_fields:"
                            + ",".join(sorted(unknown)))
    kind = _plan_text(plan.get("replacement_kind"), "migration_plan_requires_replacement_kind")
    if kind not in MIGRATION_REPLACEMENT_KINDS:
        raise ModelingError("migration_plan_replacement_kind_unknown:" + kind)
    scope = _plan_text(plan.get("scope"), "migration_plan_requires_scope")
    if scope not in MIGRATION_SCOPES:
        raise ModelingError("migration_plan_scope_unknown:" + scope)
    context = plan.get("condition_context")
    if context is not None and not isinstance(context, Mapping):
        raise ModelingError("migration_plan_condition_context_requires_mapping")
    slice_conditions = plan.get("slice_conditions") or {}
    if not isinstance(slice_conditions, Mapping):
        raise ModelingError("migration_plan_slice_conditions_requires_mapping")
    if scope == "slice" and not slice_conditions:
        raise ModelingError("migration_plan_slice_requires_conditions")
    if scope == "complete_rule" and slice_conditions:
        raise ModelingError("migration_plan_complete_rule_does_not_accept_slice_conditions")
    program, stages, view = _program_view(plan.get("program"))
    stage_names = {stage["name"] for stage in view} if view else set()
    occurrences = plan.get("occurrences") or {}
    if not isinstance(occurrences, Mapping):
        raise ModelingError("migration_plan_occurrences_requires_mapping")
    for name, fragment in occurrences.items():
        if not isinstance(name, str) or not name:
            raise ModelingError("migration_plan_occurrence_stage_requires_name")
        if view and name not in stage_names:
            raise ModelingError("migration_plan_occurrence_stage_not_in_program:" + name)
        if not hasattr(fragment, "work") or not hasattr(fragment.work, "issued"):
            raise ModelingError("migration_plan_occurrence_requires_program_fragment")
    fragments = {}
    for stage in stages:
        builtin = getattr(stage, "fragment", None)
        external = occurrences.get(getattr(stage, "name", ""))
        if builtin is not None and external is not None:
            if _fragment_evidence(builtin) != _fragment_evidence(external):
                raise ModelingError(
                    "migration_plan_occurrence_conflicts_with_builtin_fragment:"
                    + getattr(stage, "name", ""))
        fragments[getattr(stage, "name", "")] = external or builtin
    actual_context = plan.get("context")
    if actual_context is not None and not any(
            hasattr(actual_context, name) for name in (
                "native_work", "work_units", "rates", "resource_names")):
        raise ModelingError("migration_plan_context_requires_recipe_context_view")
    fixed_raw = plan.get("fixed")
    if fixed_raw is None:
        fixed_items = []
    elif isinstance(fixed_raw, Mapping):
        fixed_items = [fixed_raw]
    elif isinstance(fixed_raw, (list, tuple)):
        fixed_items = list(fixed_raw)
    else:
        raise ModelingError("migration_plan_fixed_requires_mapping_or_sequence")
    fixed = []
    for item in fixed_items:
        if not isinstance(item, Mapping) or set(item) - MIGRATION_FIXED_FIELDS:
            raise ModelingError("migration_plan_fixed_item_unknown_fields")
        sid = _plan_text(item.get("service_id"), "migration_plan_fixed_requires_service_id")
        stage = item.get("stage")
        if stage is not None:
            stage = _plan_text(stage, "migration_plan_fixed_stage_requires_name")
            if view and stage not in stage_names:
                raise ModelingError("migration_plan_fixed_stage_not_in_program:" + stage)
        native = item.get("native_boundary")
        if native is not None:
            native = _plan_text(native, "migration_plan_fixed_native_boundary_requires_text")
        fixed.append({"service_id": sid, "stage": stage, "native_boundary": native})
    declared = {term["index"]: term for term in decomposition.get("terms", ())}
    work = plan.get("work") or ()
    if not isinstance(work, (list, tuple)):
        raise ModelingError("migration_plan_work_requires_sequence")
    normalized = []
    for item in work:
        if not isinstance(item, Mapping):
            raise ModelingError("migration_plan_work_item_requires_mapping")
        unknown = set(item) - MIGRATION_WORK_FIELDS
        if unknown:
            raise ModelingError("migration_plan_work_item_unknown_fields:"
                                + ",".join(sorted(unknown)))
        stage = item.get("stage")
        if stage is not None:
            stage = _plan_text(stage, "migration_plan_work_stage_requires_name")
            if view and stage not in stage_names:
                raise ModelingError("migration_plan_work_stage_not_in_program:" + stage)
        index = item.get("term_index")
        if index is not None:
            if isinstance(index, bool) or not isinstance(index, int) or index not in declared:
                raise ModelingError("migration_plan_work_item_requires_declared_term_index")
            if declared[index].get("kind") == "zero_slope":
                raise ModelingError("migration_plan_cannot_bind_zero_slope_term:" + str(index))
        elif stage is None:
            raise ModelingError("migration_plan_work_item_requires_term_or_stage")
        resource = item.get("resource")
        if resource is None and index is not None:
            resource = declared[index].get("resource")
        if resource is not None:
            resource = _plan_text(resource, "migration_plan_work_resource_requires_text")
        work_unit = item.get("work_unit")
        if work_unit is not None:
            work_unit = _plan_text(work_unit, "migration_plan_work_unit_requires_text")
        phase = _plan_text(item.get("phase"), "migration_plan_work_item_requires_phase")
        multiplicity = item.get("multiplicity", 1)
        if (isinstance(multiplicity, bool) or not isinstance(multiplicity, int)
                or multiplicity < 1):
            raise ModelingError("migration_plan_work_item_requires_positive_multiplicity")
        scale = item.get("work_scale")
        if scale is not None:
            scale = positive(scale)
        elif stage is None:
            raise ModelingError("migration_plan_term_binding_requires_work_scale")
        else:
            scale = 1.0
        work_scope = item.get("work_scope")
        if work_scope is not None:
            work_scope = _plan_text(work_scope, "migration_plan_work_scope_unknown")
            if work_scope not in _WORK_SCOPES:
                raise ModelingError("migration_plan_work_scope_unknown:" + work_scope)
        boundary = item.get("native_boundary")
        if boundary is not None:
            boundary = _plan_text(boundary, "migration_plan_native_boundary_requires_text")
        normalized.append({
            "term_index": index, "service_id": _plan_text(
                item.get("service_id"), "migration_plan_work_item_requires_service_id"),
            "stage": stage, "resource": resource, "work_unit": work_unit,
            "work_scale": scale, "multiplicity": multiplicity, "phase": phase,
            "native_boundary": boundary, "work_scope": work_scope})
    plan_native = plan.get("native_boundary")
    if plan_native is not None:
        plan_native = _plan_text(plan_native, "migration_plan_native_boundary_requires_text")
    return {
        "replacement_kind": kind, "scope": scope,
        "condition_context": dict(context) if context is not None else None,
        "slice_conditions": dict(slice_conditions), "fixed": fixed, "work": normalized,
        "program": program, "stages": stages, "program_view": view,
        "occurrences": dict(occurrences), "fragments": fragments,
        "context": actual_context, "native_boundary": plan_native,
    }
