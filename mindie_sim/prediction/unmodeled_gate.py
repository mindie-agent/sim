"""Structural timing gate for genuinely unmodeled attention cores.

[Role] Keep structure export available while refusing to fabricate timing for a
  layer whose attention core has no physical model.
[Invariants] Only ``attention`` in ``unmodeled``/``projection_only``, an
  executed ``unmodeled_attention``/``unmodeled_attention_core`` recipe, or an
  operator that declares ``blocks_timing`` blocks a timing entry point. Known
  GQA/MLA/KDA/GDN backbones that merely carry informational
  ``unmodeled_features`` notes keep executing. Every refusal carries a
  ``mindie-unmodeled-structure-hint/1`` payload.
[Timeouts & Locks] Pure record inspection; no I/O, Device or Engine work.
[Side-Effect Policy] Raises a value exception or returns immutable hints.
"""
from __future__ import annotations

from collections.abc import Mapping

from mindie_sim.observed_contracts.vocabulary import build_unmodeled_onboarding_hint
from mindie_sim.public.errors import UnmodeledCapabilityError

UNMODELED_ATTENTION_KINDS = frozenset({"unmodeled", "projection_only"})
UNMODELED_ATTENTION_RECIPES = frozenset({
    "unmodeled_attention", "unmodeled_attention_core",
})
ONBOARDING_SCHEMA = "mindie-unmodeled-structure-hint/1"
HINT_KEYS = "unmodeled_structure_onboarding_hints"
TIMING_ENTRIES = (
    "predict", "operator_costs", "graph_costs", "execution_prepare",
    "evaluate_target_efficiency", "analyze_network_efficiency",
    "solve_sla_sizing", "sweep_topologies", "sweep_engine",
)
_AGENT_DIRECTIVE = (
    "implement the physical model inside mindie-sim/catalog + "
    "mindie_sim/prediction per the attached onboarding hint; structure export "
    "stays available, but no timing, throughput or efficiency number is "
    "fabricated for an unmodeled core"
)


def _model_of(case_like):
    """Return the declared model record from a Case, raw mapping or ModelSpec."""
    if case_like is None:
        return None
    if isinstance(case_like, Mapping):
        model = case_like.get("model")
        if isinstance(model, Mapping):
            return model
        return case_like if "layer_plans" in case_like else None
    model = getattr(case_like, "model", None)
    return model if model is not None else case_like


def _iter_plans(model):
    plans = getattr(model, "layer_plans", None)
    overrides = {}
    if plans is None and isinstance(model, Mapping):
        plans = model.get("layer_plans")
        raw_overrides = model.get("layer_overrides")
        if isinstance(raw_overrides, Mapping):
            overrides = {str(key): dict(value) for key, value in raw_overrides.items()
                         if isinstance(value, Mapping)}
    if plans is None and overrides:
        count = int(model.get("layers", len(overrides)) or len(overrides))
        plans = tuple({"index": index} for index in range(max(1, count)))
    for plan in plans or ():
        if isinstance(plan, Mapping):
            index = plan.get("index")
            index = -1 if index is None else int(index)
            override = overrides.get(str(index), {})
            yield (index,
                   str(override.get("attention", plan.get("attention", ""))),
                   str(override.get("architecture", plan.get("architecture", ""))))
        else:
            index = int(getattr(plan, "index", -1))
            override = overrides.get(str(index), {})
            yield (index,
                   str(override.get("attention", getattr(plan, "attention", ""))),
                   str(override.get("architecture",
                                    getattr(plan, "architecture", ""))))


def _metadata(op):
    raw = getattr(op, "metadata", None)
    if raw is None and isinstance(op, Mapping):
        raw = op.get("metadata")
    return dict(raw or ())


def op_blocks_timing(op) -> bool:
    """True only for an executed operator that would fabricate attention time."""
    recipe = str(getattr(op, "recipe", "") or (op.get("recipe", "")
                                              if isinstance(op, Mapping) else ""))
    if recipe in UNMODELED_ATTENTION_RECIPES:
        return True
    return bool(_metadata(op).get("blocks_timing"))


def _blocker_record(index, architecture, attention, recipes=(), op_ids=()):
    return {
        "layer_index": int(index),
        "architecture": str(architecture or attention or "unknown"),
        "attention": str(attention),
        "recipes": tuple(sorted(set(recipes))),
        "op_ids": tuple(sorted(set(op_ids))),
    }


def collect_blockers(case_like=None, ops=()) -> tuple[dict, ...]:
    """Declared core holes in the model declaration and the executed graph.

    One layer contributes one blocker: an executed marker merges into the
    declaration record for the same architecture instead of double-reporting.
    """
    found: dict[tuple, dict] = {}
    model = _model_of(case_like)
    for index, attention, architecture in _iter_plans(model):
        if attention not in UNMODELED_ATTENTION_KINDS:
            continue
        key = (index, architecture)
        if key in found:
            continue
        found[key] = _blocker_record(index, architecture, attention)
    for op in ops or ():
        if not op_blocks_timing(op):
            continue
        metadata = _metadata(op)
        raw_index = getattr(op, "layer", None)
        if raw_index is None and isinstance(op, Mapping):
            raw_index = op.get("layer")
        if raw_index is None:
            raw_index = metadata.get("layer")
        index = -1 if raw_index is None else int(raw_index)
        architecture = str(metadata.get("unmodeled_component")
                           or metadata.get("architecture") or "unmodeled")
        attention = str(metadata.get("attention_kind") or "unmodeled")
        key = (index, architecture)
        if key not in found and index < 0:
            same_core = next((item for item in found if item[1] == architecture), None)
            if same_core is not None:
                key = same_core
        record = found.get(key)
        if record is None:
            record = found[key] = _blocker_record(index, architecture, attention)
        record["recipes"] = tuple(sorted(set(record["recipes"]) | {
            str(getattr(op, "recipe", ""))}))
        op_id = str(getattr(op, "op_id", "") or "")
        if op_id:
            record["op_ids"] = tuple(sorted(set(record["op_ids"]) | {op_id}))
    return tuple(found[key] for key in sorted(found))


def _hint_for(blocker) -> dict:
    return build_unmodeled_onboarding_hint(
        observed_kernel_name=f"{blocker['architecture']}_attention",
        observed_op_type=blocker["attention"],
        observed_abi_summary={
            "layer_index": blocker["layer_index"],
            "architecture": blocker["architecture"],
            "attention_kind": blocker["attention"],
            "unmodeled_recipes": list(blocker["recipes"]),
            "op_ids": list(blocker["op_ids"]),
            "charging_policy": "no_timing_fabricated_for_unmodeled_core",
        },
    )


def structure_hints(ops=(), case_like=None) -> tuple[dict, ...]:
    """Deterministic hints for a partial structure payload (never raises)."""
    return tuple(_hint_for(blocker) for blocker in collect_blockers(case_like, ops))


def _runtime_enable_dbo(case_like) -> bool:
    if case_like is None:
        return False
    if isinstance(case_like, Mapping):
        rt = case_like.get("runtime")
        return bool(rt.get("enable_dbo", False)) if isinstance(rt, Mapping) else False
    rt = getattr(case_like, "runtime", None)
    return bool(getattr(rt, "enable_dbo", False)) if rt is not None else False


def assert_timing_supported(case_like=None, ops=(), *, entry: str) -> None:
    """Refuse to price a graph whose attention core or DBO mode has no physical model."""
    if _runtime_enable_dbo(case_like):
        raise UnmodeledCapabilityError(
            capability="dbo",
            entry=entry,
            code="unmodeled_in_op_dbo_pending_stage3",
            message=(
                f"{entry} refused to fabricate timing for enable_dbo=True: "
                "whole-graph DBO splitting has been removed and in-operator "
                "DBO is scheduled for operators/grouped_moe & operators/mc2."
            ),
            affected_results=_affected_results(entry),
            minimal_replay_input={"enable_dbo": True},
        )
    blockers = collect_blockers(case_like, ops)
    if not blockers:
        return
    hints = tuple(_hint_for(blocker) for blocker in blockers)
    first = blockers[0]
    detail = (f"layer {first['layer_index']} architecture "
              f"{first['architecture']!r} declares attention "
              f"{first['attention']!r}")
    if first["recipes"]:
        detail += f" and executes {', '.join(first['recipes'])}"
    message = (
        f"{entry} refused to fabricate timing: {detail}. Action: "
        f"{_AGENT_DIRECTIVE} (onboarding schema {ONBOARDING_SCHEMA}).")
    raise UnmodeledCapabilityError(
        capability="attention",
        entry=entry,
        code=f"unmodeled_attention_core:{first['architecture']}",
        message=message,
        blockers=blockers,
        hints=hints,
        affected_results=_affected_results(entry),
        minimal_replay_input={"attention": first["attention"],
                              "architecture": first["architecture"],
                              "layer_index": first["layer_index"]},
    )


def _affected_results(entry: str) -> tuple[str, ...]:
    if entry in {"predict", "execution_prepare"}:
        return ("latency", "throughput", "metrics")
    if entry in {"operator_costs", "graph_costs"}:
        return ("latency", "operator_timeline")
    if entry in {"evaluate_target_efficiency", "analyze_network_efficiency"}:
        return ("mbu", "mfu", "efficiency", "roofline")
    if entry == "solve_sla_sizing":
        return ("sla_concurrency", "throughput")
    if entry in {"sweep_topologies", "sweep_engine"}:
        return ("topology_ranking", "throughput", "latency")
    return ("latency",)


def assert_report_supported(report, *, entry: str) -> None:
    """Inspect a precomputed operator/timeline report before reusing it."""
    if not isinstance(report, Mapping):
        return
    rows = list(report.get("operators") or ()) + list(report.get("timeline") or ())
    ops = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        nested = row.get("operator")
        if isinstance(nested, Mapping):
            ops.append(nested)
        elif "recipe" in row or "family" in row:
            # A flat execution-timeline row carries the executed recipe
            # directly; it is the same occurrence contract as a nested row.
            ops.append(row)
    assert_timing_supported(None, tuple(ops), entry=entry)


__all__ = [
    "HINT_KEYS", "ONBOARDING_SCHEMA", "TIMING_ENTRIES",
    "UNMODELED_ATTENTION_KINDS", "UNMODELED_ATTENTION_RECIPES",
    "assert_report_supported", "assert_timing_supported", "collect_blockers",
    "op_blocks_timing", "structure_hints",
]
