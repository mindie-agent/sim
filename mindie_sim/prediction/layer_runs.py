"""Contiguous homogeneous layer-run identification and reuse programs.

[Role]
  Identify maximal contiguous runs of structurally identical body layers and
  bind one canonical layer template to its original operator occurrences.  A
  run never re-implements graph execution: reusable occurrences stay ordinary
  ``Program`` objects owned by the caller's Engine, and a ``LayerRunProgram``
  only sequences them under one explicit binding plan.

[Invariants]
  * A run key includes the ordered physical operator keys, internal dependency
    and stream topology, entry/exit ports, state ownership/effects, recipe and
    capture boundaries, complete collective cohort and stage placement.
  * Mandatory cuts are never folded across: dense<->MoE, GDN/KDA<->DSA/MLA/GQA,
    cache/indexer/compressor ownership changes, PLE or attention-residual bank
    geometry changes, PP stage boundaries, draft/MTP boundaries, quantization
    changes and capture/submission contract changes.
  * Template reuse executes every occurrence through the Engine; this module
    never multiplies a duration.  Time skipping requires an Engine-certified
    repeated frontier and is refused otherwise.
  * ``compact`` reports keep their per-operator/per-layer identities; the run
    summary is an additional projection, not a replacement.

[Timeouts & Locks]
  Pure bounded inspection of already-lowered operators and optional plans.

[Side-Effect Policy]
  No I/O, no Engine mutation and no global state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from mindie_sim.schema.common import ModelingError


ZERO = 0
ONE = 1
MIN_RUN_LENGTH = 2


def _metadata(op: Any) -> Mapping[str, Any]:
    return dict(getattr(op, "metadata", ()) or ())


def _plan_template(plan: Any) -> tuple:
    if plan is None:
        return ()
    # Occurrence index and absolute layer ids are identity, not geometry.
    fields = (
        "attention", "ffn", "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim",
        "qk_rope_head_dim", "v_head_dim", "num_heads", "num_kv_heads",
        "head_dim", "window", "compress_ratio", "indexer", "indexer_type",
        "num_experts", "top_k", "moe_intermediate_size",
        "num_shared_experts", "shared_intermediate_size", "intermediate_size",
        "router_input_size", "router_mode", "qk_norm", "q_output_gate_kind",
        "q_gate_projection", "k_eq_v", "residual_block_size", "engram",
        "hyper_connections", "hc_mult", "scope", "cache_kind",
        "compressor_kind", "compressor_width", "native_layer_id",
        "index_topk", "index_heads", "index_dim", "index_input",
    )
    template = tuple(getattr(plan, field, None) for field in fields)
    # Ownership is relative to the occurrence: a self-owner is one pattern,
    # a cross-layer owner is a distinct one.  Absolute ids are normalized to
    # their offset from the layer so corresponding runs compare equal.
    index = int(getattr(plan, "index", ZERO))
    ownership = []
    for field in ("kv_owner_layer_id", "index_k_owner_layer_id",
                  "topk_owner_layer_id", "candidate_source_layer_id",
                  "index_source_layer_id"):
        owner = int(getattr(plan, field, -ONE) or -ONE)
        ownership.append(-ONE if owner < ZERO else owner - index)
    return template + tuple(ownership)


def _op_run_key(op: Any, plan: Any) -> tuple:
    from .physical_key import physical_key
    # physical_key already normalizes occurrence references relative to this
    # operator, so identical layers produce the same ordered key sequence.
    return (
        int(getattr(op, "stage", ZERO)),
        str(getattr(op, "stream", "") or ""),
        str(getattr(op, "dtype", "") or ""),
        _plan_template(plan),
        bool(_metadata(op).get("graph_captured")),
        str(_metadata(op).get("graph_segment", "")),
        tuple(sorted((str(key), repr(value)) for key, value in _metadata(op).items()
                     if key in {"group_axis", "collective", "local_tp_width",
                                "capture_bucket", "recipe", "kernel"})),
        str(physical_key(op)),
    )


def layer_templates(ops: Sequence[Any],
                    plans: Sequence[Any] | None = None) -> tuple[tuple, ...]:
    """Ordered per-layer run keys, preserving declared operator order."""
    by_layer: dict[int, list[Any]] = {}
    order: list[int] = []
    for op in ops:
        layer = int(getattr(op, "layer", -ONE))
        if layer < ZERO:
            continue
        if layer not in by_layer:
            by_layer[layer] = []
            order.append(layer)
        by_layer[layer].append(op)
    plans = tuple(plans or ())
    plan_for = {int(getattr(plan, "index", index)): plan
                for index, plan in enumerate(plans)}
    return tuple(tuple(_op_run_key(op, plan_for.get(layer))
                       for op in by_layer[layer]) for layer in order)


def _run_cut(previous: tuple, current: tuple, plan_previous: Any,
             plan_current: Any) -> str | None:
    if previous != current:
        return "operator_template_changed"
    if plan_previous is None or plan_current is None:
        return None
    if getattr(plan_previous, "ffn", None) != getattr(plan_current, "ffn", None):
        return "dense_moe_boundary"
    family = {getattr(plan_previous, "attention", ""),
              getattr(plan_current, "attention", "")}
    if family & {"gdn", "kda"} and family - {"gdn", "kda"}:
        return "linear_full_attention_boundary"
    if _plan_template(plan_previous) != _plan_template(plan_current):
        return "layer_plan_changed"
    return None


@dataclass(frozen=True)
class LayerRun:
    """One contiguous run of identical layer templates."""

    start: int
    count: int
    key: tuple
    layers: tuple[int, ...]

    @property
    def stop(self) -> int:
        return self.start + self.count


def homogeneous_layer_runs(ops: Sequence[Any],
                           plans: Sequence[Any] | None = None) -> tuple[LayerRun, ...]:
    """Identify maximal contiguous homogeneous runs over lowered body layers."""
    templates = layer_templates(ops, plans)
    if not templates:
        return ()
    layers = sorted({int(getattr(op, "layer", -ONE)) for op in ops
                     if int(getattr(op, "layer", -ONE)) >= ZERO})
    plans = tuple(plans or ())
    plan_for = {int(getattr(plan, "index", index)): plan
                for index, plan in enumerate(plans)}
    runs: list[LayerRun] = []
    start, key = ZERO, templates[ZERO]
    for position in range(ONE, len(templates)):
        cut = _run_cut(templates[position - ONE], templates[position],
                       plan_for.get(layers[position - ONE]),
                       plan_for.get(layers[position]))
        if cut is not None:
            runs.append(LayerRun(start, position - start, key,
                                 tuple(layers[start:position])))
            start, key = position, templates[position]
    runs.append(LayerRun(start, len(templates) - start, key, tuple(layers[start:])))
    return tuple(run for run in runs if run.count >= MIN_RUN_LENGTH)


@dataclass(frozen=True)
class LayerRunProgram:
    """Sequential execution of one run's caller-prepared occurrence programs.

    Template reuse always executes every occurrence.  A caller that wants time
    skipping must first certify a repeated Engine frontier; otherwise
    construction refuses instead of inventing ``duration x count``.
    """

    run: LayerRun
    occurrences: tuple[Any, ...]
    skip_authorized: bool = False

    def __post_init__(self) -> None:
        if len(self.occurrences) != self.run.count:
            raise ModelingError(
                "layer_run_occurrence_count_mismatch:"
                f"{len(self.occurrences)}!={self.run.count}")
        if self.skip_authorized:
            raise ModelingError("layer_run_time_skip_requires_engine_certification")

    def start(self, engine: Any, name: str, ready: Any, *, owner: str = "") -> Any:
        event = ready
        for position, program in enumerate(self.occurrences):
            event = program.start(engine, f"{name}:l{position}", event, owner=owner)
        return event


def run_summary(runs: Sequence[LayerRun]) -> dict[str, Any]:
    """Structural projection of identified runs; not a second duration."""
    return {
        "runs": [
            {"start": run.start, "count": run.count, "layers": list(run.layers)}
            for run in runs
        ],
        "folded_layers": sum(run.count for run in runs),
        "templates": len({run.key for run in runs}),
        "basis": "structural_layer_template_reuse_not_duration_multiplication",
    }


def bind_run_programs(run: LayerRun,
                      lower: Callable[[int], Any]) -> LayerRunProgram:
    """Lower each original occurrence through the caller's Program factory."""
    return LayerRunProgram(
        run=run, occurrences=tuple(lower(layer) for layer in run.layers))
