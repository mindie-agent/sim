"""Explicit phase and batch geometry domains for implementation bindings.

[Role] Resolve declared alternatives before constructing a target graph.
[Invariants] Domain matching precedes lowering. A planner failure never selects
  another implementation. Mixed batches have both phases; valid rows and graph
  padding are distinct. Analytical work requires an explicit declaration.
[Timeouts & Locks] O(configured cases + active requests); no history or I/O.
[Side-Effect Policy] Returns an immutable runtime replacement and audit records.
"""
from dataclasses import dataclass, replace
from itertools import islice

from mindie_sim.schema.common import ModelingError


_GEOMETRY = {"valid_tokens", "padded_tokens", "sequences", "min_query_tokens", "max_query_tokens",
             "max_context_tokens", "min_context_tokens"}
_STATIC_SELECTORS = {"mla_prolog", "mx_matrix"}
_CATEGORIES = {"phases": {"prefill", "decode"},
               "replay_scopes": {"none", "full", "piecewise"},
               "model_scopes": {"target", "draft"}}


AUTOMATIC_LOWERINGS_KEY = "automatic_lowerings"
RECIPE_BINDING_ORIGINS_KEY = "recipe_binding_origins"
ALLOWED_BINDING_ORIGINS = frozenset({"caller", "stack_profile", "body_default"})
_ORIGIN_PRECEDENCE = {"body_default": 0, "stack_profile": 1, "caller": 2}


def record_binding_origin(metadata, section: str, name: str, origin: str):
    """Record whether a kernel binding came from ``caller``, ``stack_profile`` or ``body_default``."""
    if origin not in ALLOWED_BINDING_ORIGINS:
        raise ModelingError(f"recipe_binding_origin_invalid:{origin}")
    values = dict(metadata or ())
    table = dict(values.get(RECIPE_BINDING_ORIGINS_KEY) or {})
    key = f"{section}.{name}"
    existing = table.get(key)
    if existing is None or _ORIGIN_PRECEDENCE.get(origin, 0) >= _ORIGIN_PRECEDENCE.get(existing, 0):
        table[key] = origin
    values[RECIPE_BINDING_ORIGINS_KEY] = table
    return values


def binding_origin(runtime, section: str, name: str) -> str | None:
    """Return ``caller | stack_profile | body_default`` for an installed kernel binding."""
    if runtime is None:
        return None
    kernels = getattr(runtime, "kernels", None) or {}
    section_map = kernels.get(section) or {}
    if name not in section_map:
        return None
    meta = dict(getattr(runtime, "metadata", ()) or ())
    table = meta.get(RECIPE_BINDING_ORIGINS_KEY)
    if isinstance(table, dict):
        declared = table.get(f"{section}.{name}")
        if declared in ALLOWED_BINDING_ORIGINS:
            return str(declared)
    if section == "lowerings" and is_automatic_lowering(meta, name):
        return "body_default"
    return "caller"


def is_caller_pinned_binding(runtime, section: str, name: str) -> bool:
    """True only when ``section[name]`` was explicitly pinned by the caller."""
    return binding_origin(runtime, section, name) == "caller"


def with_automatic_lowering(metadata, selector):
    """Declare that a structural default, not the caller, installed ``selector``.

    The distinction is load-bearing: an automatically selected binding keeps
    analytical execution when a per-call geometry falls outside the source
    domain, while an explicitly requested binding fails during preflight.
    """
    values = dict(metadata or ())
    names = [str(name) for name in values.get(AUTOMATIC_LOWERINGS_KEY) or ()]
    if selector not in names:
        names.append(selector)
    values[AUTOMATIC_LOWERINGS_KEY] = tuple(names)
    return values


def is_automatic_lowering(metadata, selector):
    """Accept either a runtime metadata mapping or its parsed pair sequence."""
    return selector in tuple(dict(metadata or ()).get(AUTOMATIC_LOWERINGS_KEY) or ())


EXPLICIT_PRESERVED_REASON = "explicit_implementation_preserved"


def selected_note(family, binding, native_kind, reason):
    """One resolved native selection, as exported by every entry point."""
    return {"family": str(family), "binding": str(binding),
            "native_kind": str(native_kind), "reason": str(reason)}


def fallback_note(family, reason):
    """One family that retained analytical execution, with its exact cause."""
    return {"family": str(family), "fallback": "analytical", "reason": str(reason)}


def _record_note(record):
    family = str(record.get("family") or "unknown")
    reason = str(record.get("reason") or "")
    installed = str(record.get("selection") or "") == "default_installed"
    if installed or reason == EXPLICIT_PRESERVED_REASON:
        binding = (record.get("binding") or record.get("caller")
                   or record.get("implementation") or family)
        return selected_note(family, binding,
                             record.get("native_kind") or family, reason)
    return fallback_note(family, reason or "unspecified")


def operator_selection_summary(*, arch_family, phase, notes):
    """Deterministic projection of resolved selections and fallbacks."""
    selected, degraded = [], []
    for note in notes or ():
        if "fallback" in note:
            degraded.append(dict(note))
        elif "binding" in note:
            selected.append(dict(note))
    return {
        "arch_family": str(arch_family),
        "phase": str(phase),
        "selected_native": selected,
        "degraded_fallbacks": degraded,
        "selection_warnings": [
            f"{note['family']}->{note['fallback']}:{note['reason']}" for note in degraded
        ],
    }


def records_summary(*, arch_family, phase, records):
    """Project resolved implementation records onto the shared summary schema."""
    return operator_selection_summary(
        arch_family=arch_family, phase=phase,
        notes=[_record_note(record) for record in records or ()])


def combine_summaries(summaries):
    """Union phase-scoped per-iteration summaries into one deterministic record."""
    items = [summary for summary in (summaries or ()) if summary]
    if not items:
        return {}
    arch = ""
    phases = []
    selected, degraded = [], []
    seen_selected, seen_degraded = set(), set()
    for summary in items:
        arch = arch or str(summary.get("arch_family") or "")
        phase = str(summary.get("phase") or "")
        if phase and phase not in phases:
            phases.append(phase)
        for note in summary.get("selected_native", ()):
            key = (note.get("family"), note.get("binding"))
            if key not in seen_selected:
                seen_selected.add(key)
                selected.append(dict(note))
        for note in summary.get("degraded_fallbacks", ()):
            key = (note.get("family"), note.get("reason"))
            if key not in seen_degraded:
                seen_degraded.add(key)
                degraded.append(dict(note))
    return {
        "arch_family": arch,
        "phase": "+".join(phases) if phases else "unknown",
        "selected_native": selected,
        "degraded_fallbacks": degraded,
        "selection_warnings": [
            f"{note['family']}->{note['fallback']}:{note['reason']}" for note in degraded
        ],
    }


def merge_selection_summaries(base, extra):
    """Union two phase-scoped summaries without dropping either record."""
    if not base:
        return dict(extra)
    if not extra:
        return dict(base)
    merged = dict(extra)
    seen = {(item.get("family"), item.get("reason"))
            for item in merged.get("degraded_fallbacks", ())}
    degraded = list(merged.get("degraded_fallbacks", ()))
    for item in base.get("degraded_fallbacks", ()):
        key = (item.get("family"), item.get("reason"))
        if key not in seen:
            seen.add(key)
            degraded.append(dict(item))
    merged["degraded_fallbacks"] = degraded
    merged["selection_warnings"] = [
        f"{item['family']}->{item['fallback']}:{item['reason']}" for item in degraded]
    return merged


def is_policy(raw):
    return isinstance(raw, dict) and "cases" in raw


def alternatives(raw, selector):
    """Enumerate all declared recipe providers without executing their planners."""
    if not is_policy(raw):
        yield raw
        return
    if selector in _STATIC_SELECTORS and not all(
            isinstance(case, dict) and set(case.get("domain") or ()) == {"model_scopes"}
            for case in (raw.get("cases") or ())):
        raise ModelingError(f"implementation_policy_requires_static_model_preparation:{selector}")
    if set(raw) - {"cases", "otherwise"} or not raw["cases"]:
        raise ModelingError(f"implementation_policy_requires_cases:{selector}")
    for case in raw["cases"]:
        if set(case) != {"domain", "binding"}:
            raise ModelingError(f"implementation_case_requires_domain_and_binding:{selector}")
        _validate_domain(case["domain"], selector)
        if is_policy(case["binding"]) or case["binding"] == "error":
            raise ModelingError(f"implementation_case_requires_leaf_binding:{selector}")
        yield case["binding"]
    otherwise = raw.get("otherwise", "error")
    if is_policy(otherwise):
        raise ModelingError(f"implementation_case_requires_leaf_binding:{selector}")
    if otherwise != "error":
        yield otherwise


def _validate_domain(domain, selector):
    if not domain or set(domain) - (_GEOMETRY | _CATEGORIES.keys()):
        raise ModelingError(f"implementation_domain_unknown_or_empty:{selector}")
    for key, value in domain.items():
        if key in _CATEGORIES:
            allowed = _CATEGORIES[key]
            if (not isinstance(value, (tuple, list)) or not value
                    or set(value) - allowed or len(set(value)) != len(value)):
                raise ModelingError(f"implementation_domain_invalid:{selector}:{key}")
        elif (not isinstance(value, dict) or not value or set(value) - {"min", "max"}
              or any(type(bound) is not int or bound < 0 for bound in value.values())
              or value.get("min", 0) > value.get("max", float("inf"))):
            raise ModelingError(f"implementation_domain_invalid:{selector}:{key}")


def prepared_selection(bindings, selector):
    """Reserve optional prepared weights once, independently of active calls.

    The supported callers retain their original weights for analytical/prefill
    work. Keeping the prepared copy resident is not selecting its device call.
    """
    from .implementations import select

    raw = bindings.get(selector)
    if not is_policy(raw):
        return select(bindings, selector)
    if all(isinstance(case, dict) and set(case.get("domain") or ()) == {"model_scopes"}
           for case in (raw.get("cases") or ())):
        for case in raw["cases"]:
            if "target" in (case["domain"].get("model_scopes") or ()):
                binding = case["binding"]
                return None if binding == "analytical" else select({selector: binding}, selector)
        otherwise = raw.get("otherwise", "analytical")
        return (None if otherwise == "analytical" or otherwise == "error"
                else select({selector: otherwise}, selector))
    choices = [select({selector: binding}, selector)
               for binding in alternatives(raw, selector) if binding != "analytical"]
    if choices and any((choice.name, choice.options) != (choices[0].name, choices[0].options)
                       for choice in choices[1:]):
        raise ModelingError(f"implementation_prepared_inventory_must_be_shared:{selector}")
    return choices[0] if choices else None


def _matches(domain, geometry):
    for key, limits in domain.items():
        if key == "phases":
            # Exact phase composition makes a mixed declaration disjoint from
            # pure prefill/decode, even when one side has just a single row.
            if set(limits) != set(geometry[key]):
                return False
        elif key in {"replay_scopes", "model_scopes"}:
            if geometry[key.removesuffix("s")] not in limits:
                return False
        elif not limits.get("min", 0) <= geometry[key] <= limits.get("max", float("inf")):
            return False
    return True


def resolve(runtime, views, *, padded_tokens, replay_scope, global_views=None,
            model_scope="target"):
    """Select against this model call, before PCP/SP row partitioning.

    Draft calls use their actual query rows and independent phase composition.
    An idle expert participant uses the active cohort's phase composition but
    retains zero local row counts.
    """
    from .implementations import plan_record, select

    bindings = runtime.kernels.get("lowerings", {})
    if not any(is_policy(raw) for raw in bindings.values()):
        return runtime, ()
    geometry = {
        "phases": sorted({view.phase for view in (views or global_views or ())}),
        "valid_tokens": sum(view.q_len for view in views),
        "padded_tokens": padded_tokens,
        "sequences": len(views),
        "min_query_tokens": min((view.q_len for view in views), default=0),
        "max_query_tokens": max((view.q_len for view in views), default=0),
        "max_context_tokens": max((view.context for view in views), default=0),
        # Dense paged-MLA source kernels apply their split-core tile only from a
        # declared page-aligned context; the minimum keeps that a domain match
        # instead of a planner exception on short contexts.
        "min_context_tokens": min((view.context for view in views), default=0),
        "replay_scope": replay_scope,
        "model_scope": model_scope,
    }
    resolved, records = dict(bindings), []
    for selector, raw in bindings.items():
        if not is_policy(raw):
            continue
        # Validate every declared leaf, including currently inactive domains.
        for binding in alternatives(raw, selector):
            if binding != "analytical":
                select({selector: binding}, selector)
        matches = [(index, case) for index, case in enumerate(raw["cases"])
                   if _matches(case["domain"], geometry)]
        if len(matches) > 1:
            raise ModelingError(f"implementation_domains_overlap:{selector}")
        index, case = matches[0] if matches else ("otherwise", {})
        binding = case.get("binding", raw.get("otherwise", "error"))
        if binding == "error":
            raise ModelingError(f"implementation_domain_unmatched:{selector}:{geometry}")
        # Candidate inventory and the selected physical plan are separate
        # records.  An unselected candidate never contributes execution work.
        candidates = []
        for candidate_index, candidate in enumerate(raw["cases"]):
            candidate_binding = candidate["binding"]
            candidate_implementation = (
                "analytical" if candidate_binding == "analytical"
                else select({selector: candidate_binding}, selector).name)
            candidates.append({
                "case": candidate_index,
                "binding": candidate_binding,
                "implementation": candidate_implementation,
                "domain": candidate["domain"],
                "matches": bool(_matches(candidate["domain"], geometry)),
                "basis": "declared_candidate",
            })
        otherwise = raw.get("otherwise", "error")
        if otherwise != "error":
            candidates.append({
                "case": "otherwise", "binding": otherwise,
                "implementation": ("analytical" if otherwise == "analytical"
                                   else select({selector: otherwise}, selector).name),
                "domain": None, "matches": not matches,
                "basis": "declared_candidate",
            })
        if binding == "analytical":
            resolved.pop(selector)
            implementation = "analytical"
            selected_plan = plan_record(None)
        else:
            resolved[selector] = binding
            selected = select({selector: binding}, selector)
            implementation = selected.name
            selected_plan = plan_record(selected)
        records.append({
            "selector": selector, "case": index, "model_scope": model_scope,
            "implementation": implementation,
            "coverage": ("analytical_semantic_work" if binding == "analytical"
                         else "declared_implementation_domain"),
            "domain": case.get("domain", {}), "geometry": geometry,
            "basis": "selected_software_plan",
            "candidates": tuple(candidates),
            "selected": {"case": index,
                         "binding": binding,
                         "implementation": implementation,
                         "domain": case.get("domain", {})},
            "plan": selected_plan,
        })
    return replace(runtime, kernels={**runtime.kernels, "lowerings": resolved}), tuple(records)


@dataclass(frozen=True)
class LoweringRegion:
    """One target or draft call with its already resolved implementation set."""

    ops: tuple
    runtime: object
    source_dtype: str
    local_tokens: int
    selections: tuple = ()


def capture_state(ops, graph, runtime, replay_scope):
    """Annotate with GraphPolicy once and return the reusable state mapping."""
    annotated = graph.apply(tuple(ops), replay_scope, runtime)
    return annotated, _capture_values(annotated)


def _capture_values(ops):
    return {op.op_id: (dict(op.metadata).get("graph_captured"),
                       dict(op.metadata).get("graph_segment"))
            for op in ops}


def apply_capture_state(ops, state):
    """Materialize the complete partition on the actual DAG before native admission."""
    result = []
    for op in ops:
        value = state.get(op.op_id)
        if value is None or value[0] is None:
            result.append(op)
            continue
        metadata = dict(op.metadata)
        metadata["graph_captured"] = bool(value[0])
        metadata["graph_segment"] = value[1]
        result.append(replace(op, metadata=tuple(metadata.items())))
    return tuple(result)


def apply_capture_inheritance(ops, state, source_ops):
    """Public handoff for body/prepared entries: inherit pre-fusion partition."""
    return _inherit_capture(ops, state, source_ops)


def _inherit_capture(ops, state, original):
    """Keep the pre-fusion captured/eager decision for replacements.

    Only segment ordinals are recomputed; a fused node's new family is never
    used to expand the capture domain.
    """
    inherited = dict(state)
    generated = {}
    for op in ops:
        metadata = dict(op.metadata)
        value = inherited.get(op.op_id)
        if value is None:
            value = (metadata.get("graph_captured"),
                     metadata.get("graph_segment"))
        if value[0] is None:
            members = metadata.get("fusion_semantic_ids")
            if members:
                member_values = [state[member] for member in members
                                 if member in state]
                if member_values and all(item == member_values[0]
                                         for item in member_values):
                    value = member_values[0]
        # No suffix/name guessing: generated quantizer/representation nodes
        # must be present in the complete state returned by the lowering path.
        if value[0] is None:
            dep_values = [state[dep] for dep in op.deps if dep in state]
            if dep_values and all(item == dep_values[0] for item in dep_values):
                value = dep_values[0]
        if value[0] is not None:
            generated[op.op_id] = value
    result, segment, last_stage, last_captured = [], 0, None, False
    for op in ops:
        captured, _ = generated.get(op.op_id, (None, None))
        if captured is None:
            result.append(op)
            continue
        if op.stage != last_stage or bool(captured) != last_captured:
            segment += 1
        metadata = dict(op.metadata)
        metadata["graph_captured"] = bool(captured)
        metadata["graph_segment"] = segment
        result.append(replace(op, metadata=tuple(metadata.items())))
        last_stage, last_captured = op.stage, bool(captured)
    return tuple(result)


def lower_regions(regions, *, graph, replay_scope, runtime, prefix=(),
                  resource_fusion=True, public_outputs=()):
    """Insert representations, then partition the actual prefix+region DAG once."""
    from .transforms import insert_representations, lower_implementations
    from .native_fusion import lower_native_fusions, _ordered

    prefix_ops = tuple(prefix)
    inserted_regions = []
    for region in regions:
        inserted, preactivated = insert_representations(
            region.ops, region.runtime, region.source_dtype)
        inserted_regions.append((inserted, preactivated))
    actual = tuple(prefix_ops) + tuple(
        op for ops, _ in inserted_regions for op in ops)
    annotated_actual = graph.apply(actual, replay_scope, runtime)
    state = _capture_values(annotated_actual)
    annotated_iter = iter(annotated_actual)
    annotated_prefix = tuple(islice(annotated_iter, len(prefix_ops)))
    annotated_regions = [tuple(islice(annotated_iter, len(ops)))
                         for ops, _ in inserted_regions]
    lowered = []
    for region, (_, preactivated), annotated in zip(
            regions, inserted_regions, annotated_regions):
        selected = {op.op_id: state[op.op_id] for op in annotated
                    if op.op_id in state}
        lowered.append(lower_implementations(
            annotated, region.runtime, region.source_dtype,
            resource_fusion=False, local_tokens=region.local_tokens,
            replay_scope=replay_scope, capture_state=selected,
            skip_passes=("activation_quantization",),
            preactivated=preactivated))
        state.update(selected)
    lowered = [apply_capture_state(ops, state) for ops in lowered]
    transformed = (*annotated_prefix, *(op for ops in lowered for op in ops))
    result = list(annotated_prefix)
    if resource_fusion:
        # A fusion group is admitted inside one region.  Prefix and distinct
        # regions are separate calls; a declared group crossing that boundary
        # is rejected rather than joined after the partition.
        for region, ops in zip(regions, lowered):
            result.extend(lower_native_fusions(
                ops, region.runtime, local_tokens=region.local_tokens,
                public_outputs=public_outputs))
        positions = {op.op_id: index for index, op in enumerate(transformed)}
        def position(op):
            members = dict(op.metadata).get("fusion_semantic_ids", (op.op_id,))
            return min(positions[member] for member in members if member in positions)
        result = _ordered(sorted(result, key=position))
    else:
        for ops in lowered:
            result.extend(ops)
    if replay_scope != "none":
        result = apply_capture_inheritance(tuple(result), state, annotated_actual)
    selections = selection_delivery(
        tuple(result), tuple(record for region in regions
                            for record in region.selections),
        public_outputs=public_outputs)
    return annotate(result, selections), selections


def selection_delivery(ops, records, public_outputs=()):
    """Attach the actual selected occurrence/program boundary for T12/T04.

    This is a delivery record, not a second plan: values and capture are read
    from the already lowered ops and the selected plan is copied verbatim.
    """
    from .implementations import produced_values
    result = []
    for record in records:
        selected = record.get("selected") or {}
        implementation = selected.get("implementation")
        plan = record.get("plan") or {}
        recipes = set(plan.get("recipes") or ())
        matched = tuple(
            op for op in ops
            if (implementation and dict(op.metadata).get("implementation") == implementation)
            or (op.recipe and op.recipe in recipes))
        outputs = []
        for op in matched:
            try:
                values = produced_values(op)
            except ModelingError:
                values = ()
            for value in values:
                outputs.append({
                    "producer": value.get("producer") or op.op_id,
                    "output": value.get("output"),
                    "dtype": value.get("dtype"),
                    "role": value.get("role"),
                    "logical_shape": value.get("logical_shape"),
                    "storage_shape": value.get("storage_shape"),
                    "materialized": value.get("materialized"),
                    "alias_of": value.get("alias_of"),
                    "write_bytes": value.get("write_bytes"),
                    "state_version": value.get("state_version"),
                    "binding": value.get("binding"),
                })
        executions = tuple({
            "producer": op.op_id,
            "family": op.family,
            "recipe": op.recipe,
            "deps": tuple(op.deps),
            "stream": op.stream,
            "graph_captured": dict(op.metadata).get("graph_captured"),
            "graph_segment": dict(op.metadata).get("graph_segment"),
            "returns": tuple(op.returns),
        } for op in matched)
        result.append({
            **record,
            "delivery": {
                "selector": record.get("selector"),
                "selected": selected,
                "program": plan,
                "producer_outputs": tuple(outputs),
                "executions": executions,
                "public_roots": tuple(public_outputs),
            },
        })
    return tuple(result)


def annotate(ops, records):
    """Retain the decision on one graph root without copying it to every layer."""
    if not records or not ops:
        return tuple(ops)
    first, *rest = ops
    return (replace(first, metadata=first.metadata + (("implementation_selections", records),)), *rest)


def graph_selections(ops):
    return tuple(record for op in ops
                 for record in dict(op.metadata).get("implementation_selections", ()))
