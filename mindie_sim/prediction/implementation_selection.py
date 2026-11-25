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
from .program.implementation_registry import (
    _CATEGORIES,
    _GEOMETRY,
    _STATIC_SELECTORS,
    _validate_domain,
    alternatives,
    is_policy,
)


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
    replay_scope: str | None = None


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
    result, segment, last_key = [], 0, None
    for op in ops:
        captured, orig_segment = generated.get(op.op_id, (None, None))
        if captured is None:
            result.append(op)
            continue
        key = (op.stage, bool(captured), orig_segment)
        if key != last_key:
            segment += 1
        metadata = dict(op.metadata)
        metadata["graph_captured"] = bool(captured)
        metadata["graph_segment"] = segment
        result.append(replace(op, metadata=tuple(metadata.items())))
        last_key = key
    return tuple(result)


def _mark_eager_region_ops(ops):
    """Explicitly tag eager-overridden region ops so capture inheritance never flips them captured."""
    result = []
    for op in ops:
        meta = dict(op.metadata)
        meta["graph_captured"] = False
        meta["graph_segment"] = None
        result.append(replace(op, metadata=tuple(meta.items())))
    return tuple(result)


def _offset_segments(ops, offset: int) -> tuple[tuple, int]:
    if not ops:
        return (), offset
    result = []
    max_seg = offset
    for op in ops:
        meta = dict(op.metadata)
        seg = meta.get("graph_segment")
        if seg is not None:
            shifted = int(seg) + offset
            meta["graph_segment"] = shifted
            max_seg = max(max_seg, shifted)
            result.append(replace(op, metadata=tuple(meta.items())))
        else:
            result.append(op)
    return tuple(result), max_seg


_DEFAULT_SPLITTING_OPS = frozenset({
    "attention", "attention_qk", "attention_pv", "attention_softmax", "attention_merge",
})


def _capture_policy_key(scope: str, runtime) -> tuple[str, frozenset[str] | None]:
    raw_split = getattr(runtime, "splitting_ops", None)
    if raw_split is None and runtime is not None:
        raw_split = dict(getattr(runtime, "metadata", ()) or ()).get("splitting_ops")
    return (str(scope), None if raw_split is None else frozenset(raw_split))


def _resolve_region_capture(
    lowered_ops, pre_selected, selected, *, reg_scope, runtime, region_base, next_segment,
):
    """Assign region-local capture state to every lowered/generated op before joining regions."""
    if not lowered_ops or all(op.op_id in pre_selected for op in lowered_ops):
        return next_segment
    raw_split = getattr(runtime, "splitting_ops", None)
    split = set(raw_split) if raw_split is not None else _DEFAULT_SPLITTING_OPS
    entries: list[tuple[object, bool, int | None]] = []
    for op in lowered_ops:
        if op.op_id in pre_selected:
            cap, seg = pre_selected[op.op_id]
            entries.append((op, bool(cap), seg))
            continue
        meta = dict(op.metadata)
        members = meta.get("fusion_semantic_ids")
        if members:
            m_vals = [pre_selected[m] for m in members if m in pre_selected]
            if m_vals and all(item == m_vals[0] for item in m_vals):
                entries.append((op, bool(m_vals[0][0]), m_vals[0][1]))
                continue
        if meta.get("graph_captured") is not None and meta.get("graph_segment") is not None:
            entries.append((op, bool(meta["graph_captured"]), int(meta["graph_segment"])))
            continue
        if op.op_id in selected and selected[op.op_id][0] is not None and selected[op.op_id][1] is not None:
            cap, seg = selected[op.op_id]
            entries.append((op, bool(cap), int(seg) + region_base))
            continue
        cap = (reg_scope == "full") or (op.family not in split and op.recipe not in split)
        entries.append((op, bool(cap), None))
    # Fill unresolved segments strictly within this region's contiguous (stage, captured) runs.
    idx = 0
    while idx < len(entries):
        stage, cap = entries[idx][0].stage, entries[idx][1]
        end = idx + 1
        while end < len(entries) and entries[end][0].stage == stage and entries[end][1] == cap:
            end += 1
        known = [entries[j][2] for j in range(idx, end) if entries[j][2] is not None]
        if known:
            fallback_seg = known[0]
        else:
            fallback_seg = next_segment
            next_segment += 1
        for j in range(idx, end):
            op_j, cap_j, seg_j = entries[j]
            selected[op_j.op_id] = (cap_j, fallback_seg if seg_j is None else seg_j)
        idx = end
    return next_segment


def lower_regions(regions, *, graph, replay_scope, runtime, prefix=(),
                  resource_fusion=True, public_outputs=()):
    """Insert representations, then partition the actual prefix+region DAG per region scope."""
    from .transforms import insert_representations, lower_implementations
    from mindie_sim.prediction.operators.hyper_connections.native_fusion import lower_native_fusions, _ordered

    prefix_ops = tuple(prefix)
    inserted_regions = []
    for region in regions:
        inserted, preactivated = insert_representations(
            region.ops, region.runtime, region.source_dtype)
        inserted_regions.append((inserted, preactivated))
    region_scopes = [
        (region.replay_scope if region.replay_scope is not None else replay_scope)
        for region in regions
    ]
    base_key = _capture_policy_key(replay_scope, runtime)
    region_keys = [
        _capture_policy_key(reg_scope, region.runtime)
        for region, reg_scope in zip(regions, region_scopes)
    ]
    heterogeneous_scopes = any(key != base_key for key in region_keys)
    any_captured = replay_scope != "none" or any(sc != "none" for sc in region_scopes)
    if not heterogeneous_scopes:
        actual = tuple(prefix_ops) + tuple(
            op for ops, _ in inserted_regions for op in ops)
        annotated_actual = graph.apply(actual, replay_scope, runtime)
        state = _capture_values(annotated_actual)
        annotated_iter = iter(annotated_actual)
        annotated_prefix = tuple(islice(annotated_iter, len(prefix_ops)))
        annotated_regions = [tuple(islice(annotated_iter, len(ops)))
                             for ops, _ in inserted_regions]
    else:
        join_prefix_with_first = (
            bool(prefix_ops)
            and bool(regions)
            and replay_scope != "none"
            and region_keys[0] == base_key
        )
        annotated_prefix = (
            () if join_prefix_with_first
            else (graph.apply(prefix_ops, replay_scope, runtime) if prefix_ops else ())
        )
        seg_offset = max(
            (int(dict(op.metadata).get("graph_segment") or 0) for op in annotated_prefix),
            default=0,
        )
        annotated_regions = [()] * len(regions)
        idx = 0
        while idx < len(regions):
            reg_scope = region_scopes[idx]
            reg_key = region_keys[idx]
            if reg_scope == "none":
                ops_i = inserted_regions[idx][0]
                annotated_regions[idx] = _mark_eager_region_ops(ops_i) if any_captured else tuple(ops_i)
                idx += 1
                continue
            group_end = idx + 1
            while group_end < len(regions) and region_keys[group_end] == reg_key:
                group_end += 1
            lead_prefix = prefix_ops if (idx == 0 and join_prefix_with_first) else ()
            group_ops = lead_prefix + tuple(
                op for j in range(idx, group_end) for op in inserted_regions[j][0]
            )
            ann_group = graph.apply(group_ops, reg_scope, regions[idx].runtime) if group_ops else ()
            ann_group, seg_offset = _offset_segments(ann_group, seg_offset)
            g_iter = iter(ann_group)
            if lead_prefix:
                annotated_prefix = tuple(islice(g_iter, len(lead_prefix)))
            for j in range(idx, group_end):
                annotated_regions[j] = tuple(islice(g_iter, len(inserted_regions[j][0])))
            idx = group_end
        annotated_actual = annotated_prefix + tuple(op for ann in annotated_regions for op in ann)
        state = _capture_values(annotated_actual)
    next_segment = max(
        (int(seg) for _cap, seg in state.values() if seg is not None),
        default=0,
    ) + 1
    lowered = []
    for region, (_, preactivated), annotated, reg_scope in zip(
            regions, inserted_regions, annotated_regions, region_scopes):
        selected = {op.op_id: state[op.op_id] for op in annotated
                    if op.op_id in state}
        pre_selected = dict(selected)
        region_base = min(
            (int(seg) for _cap, seg in pre_selected.values() if seg is not None),
            default=1,
        ) - 1
        lowered_ops = lower_implementations(
            annotated, region.runtime, region.source_dtype,
            resource_fusion=False, local_tokens=region.local_tokens,
            replay_scope=reg_scope, capture_state=selected,
            skip_passes=("activation_quantization",),
            preactivated=preactivated)
        if reg_scope == "none" and any_captured:
            lowered_ops = _mark_eager_region_ops(lowered_ops)
            for op in lowered_ops:
                selected[op.op_id] = (False, None)
        elif reg_scope != "none":
            next_segment = _resolve_region_capture(
                lowered_ops, pre_selected, selected,
                reg_scope=reg_scope, runtime=region.runtime,
                region_base=region_base, next_segment=next_segment,
            )
        lowered.append(lowered_ops)
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
    if any_captured:
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
