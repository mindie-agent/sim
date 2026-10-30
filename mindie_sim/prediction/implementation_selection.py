"""Explicit phase and batch geometry domains for implementation bindings.

[Role] Resolve declared alternatives before constructing a target graph.
[Invariants] Domain matching precedes lowering. A planner failure never selects
  another implementation. Mixed batches have both phases; valid rows and graph
  padding are distinct. Analytical work requires an explicit declaration.
[Timeouts & Locks] O(configured cases + active requests); no history or I/O.
[Side-Effect Policy] Returns an immutable runtime replacement and audit records.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError


_GEOMETRY = {"valid_tokens", "padded_tokens", "sequences", "max_query_tokens",
             "max_context_tokens"}
_STATIC_SELECTORS = {"mla_prolog", "mx_matrix"}


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
    if not domain or set(domain) - (_GEOMETRY | {"phases", "replay_scopes"}):
        raise ModelingError(f"implementation_domain_unknown_or_empty:{selector}")
    for key, value in domain.items():
        if key in {"phases", "replay_scopes"}:
            allowed = {"prefill", "decode"} if key == "phases" else {"none", "full", "piecewise"}
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
        elif key == "replay_scopes":
            if geometry["replay_scope"] not in limits:
                return False
        elif not limits.get("min", 0) <= geometry[key] <= limits.get("max", float("inf")):
            return False
    return True


def resolve(runtime, views, *, padded_tokens, replay_scope, global_views=None):
    """Select against this rank's target batch, before PCP/SP row partitioning.

    Draft subgraphs inherit the target step's policy. An idle expert participant
    uses the active cohort's phase composition but retains zero local row counts.
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
        records.append({"selector": selector, "case": index,
                        "implementation": implementation,
                        "coverage": ("analytical_semantic_work" if binding == "analytical"
                                     else "declared_implementation_domain"),
                        "domain": case.get("domain", {}), "geometry": geometry})
    return replace(runtime, kernels={**runtime.kernels, "lowerings": resolved}), tuple(records)


def annotate(ops, records):
    """Retain the decision on one graph root without copying it to every layer."""
    if not records or not ops:
        return tuple(ops)
    first, *rest = ops
    return (replace(first, metadata=first.metadata + (("implementation_selections", records),)), *rest)


def graph_selections(ops):
    return tuple(record for op in ops
                 for record in dict(op.metadata).get("implementation_selections", ()))
