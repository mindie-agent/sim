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


_GEOMETRY = {"valid_tokens", "padded_tokens", "sequences", "max_query_tokens",
             "max_context_tokens"}
_STATIC_SELECTORS = {"mla_prolog", "mx_matrix"}
_CATEGORIES = {"phases": {"prefill", "decode"},
               "replay_scopes": {"none", "full", "piecewise"},
               "model_scopes": {"target", "draft"}}


def is_policy(raw):
    return isinstance(raw, dict) and "cases" in raw


def alternatives(raw, selector):
    """Enumerate all declared recipe providers without executing their planners."""
    if not is_policy(raw):
        yield raw
        return
    if selector in _STATIC_SELECTORS:
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
    from .implementations import select

    bindings = runtime.kernels.get("lowerings", {})
    if not any(is_policy(raw) for raw in bindings.values()):
        return runtime, ()
    geometry = {
        "phases": sorted({view.phase for view in (views or global_views or ())}),
        "valid_tokens": sum(view.q_len for view in views),
        "padded_tokens": padded_tokens,
        "sequences": len(views),
        "max_query_tokens": max((view.q_len for view in views), default=0),
        "max_context_tokens": max((view.context for view in views), default=0),
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
        if binding == "analytical":
            resolved.pop(selector)
            implementation = "analytical"
        else:
            resolved[selector] = binding
            implementation = select({selector: binding}, selector).name
        records.append({"selector": selector, "case": index, "model_scope": model_scope,
                        "implementation": implementation,
                        "coverage": ("analytical_semantic_work" if binding == "analytical"
                                     else "declared_implementation_domain"),
                        "domain": case.get("domain", {}), "geometry": geometry})
    return replace(runtime, kernels={**runtime.kernels, "lowerings": resolved}), tuple(records)


@dataclass(frozen=True)
class LoweringRegion:
    """One target or draft call with its already resolved implementation set."""

    ops: tuple
    runtime: object
    source_dtype: str
    local_tokens: int
    selections: tuple = ()


def lower_regions(regions, *, graph, replay_scope, runtime, prefix=(), resource_fusion=True):
    """Keep transforms inside their selected model call and capture the whole DAG.

    Draft sample/target commit barriers retain the edges between regions. Graph
    segments are assigned once across their ordered concatenation, before native
    fusion checks that every member belongs to one captured segment.
    """
    from .transforms import lower_implementations
    from .native_fusion import lower_native_fusions, _ordered

    lowered = [lower_implementations(region.ops, region.runtime, region.source_dtype,
                                    resource_fusion=False, local_tokens=region.local_tokens)
               for region in regions]
    captured_ops = graph.apply((*prefix, *(op for ops in lowered for op in ops)),
                               replay_scope, runtime)
    captured = iter(captured_ops)
    result = list(islice(captured, len(prefix)))
    fusion_groups = []
    for region, ops in zip(regions, lowered):
        scoped = tuple(islice(captured, len(ops)))
        if not resource_fusion:
            result.extend(scoped)
            continue
        # An unscoped fusion declaration requires its roles in the complete
        # selected graph, not in every MTP step. Equal declarations share that
        # domain even when other lowering decisions differ between calls.
        kernels = region.runtime.kernels
        binding = kernels.get("lowerings", {}).get("native_fusion")
        configuration = {key: value for key, value in kernels.items() if key != "lowerings"}
        group = next((group for group in fusion_groups
                      if group[0] == (binding, configuration)), None)
        if group is None:
            group = [(binding, configuration), region.runtime, [], 0]
            fusion_groups.append(group)
        group[2].extend(scoped)
        group[3] += region.local_tokens
    if resource_fusion:
        for _, selected_runtime, ops, tokens in fusion_groups:
            result.extend(lower_native_fusions(ops, selected_runtime, local_tokens=tokens))
        positions = {op.op_id: index for index, op in enumerate(captured_ops)}
        def position(op):
            members = dict(op.metadata).get("fusion_semantic_ids", (op.op_id,))
            return min(positions[member] for member in members)
        result = _ordered(sorted(result, key=position))
    selections = tuple(record for region in regions for record in region.selections)
    return annotate(result, selections), selections


def annotate(ops, records):
    """Retain the decision on one graph root without copying it to every layer."""
    if not records or not ops:
        return tuple(ops)
    first, *rest = ops
    return (replace(first, metadata=first.metadata + (("implementation_selections", records),)), *rest)


def graph_selections(ops):
    return tuple(record for op in ops
                 for record in dict(op.metadata).get("implementation_selections", ()))
