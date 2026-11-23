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

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from mindie_sim.schema.common import ModelingError


MIN_RUN_LENGTH = 2
DEFAULT_COHORT_CACHE_LIMIT = 1024
NS_PER_MS = 1_000_000.0


def _metadata(op: Any) -> Mapping[str, Any]:
    return dict(getattr(op, "metadata", ()) or ())


#: Geometry fields that define a reusable layer template.  Occurrence index and
#: absolute layer ids are identity, not geometry, and are handled separately.
TEMPLATE_FIELDS = (
    "attention", "ffn", "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim",
    "qk_rope_head_dim", "v_head_dim", "num_heads", "num_kv_heads",
    "head_dim", "window", "compress_ratio", "indexer", "indexer_type",
    "num_experts", "top_k", "moe_intermediate_size",
    "num_shared_experts", "shared_intermediate_size", "intermediate_size",
    "router_input_size", "router_mode", "qk_norm", "q_output_gate_kind",
    "q_gate_projection", "k_eq_v", "residual_block_size", "engram",
    "hyper_connections", "hc_mult", "hc_lowrank", "hc_sinkhorn_iters",
    "scope", "cache_kind", "compressor_kind", "compressor_width",
    "index_topk", "index_heads", "index_dim", "index_input", "index_kpool",
)

#: Cross-layer owners.  Each is stored as a labelled delta, never absolute.
OWNERSHIP_FIELDS = (
    "kv_owner_layer_id", "index_k_owner_layer_id", "topk_owner_layer_id",
    "candidate_source_layer_id", "index_source_layer_id",
)

ABSENT = ("absent",)
LAYER_DELTA = "layer_delta"
NATIVE_LAYER_DELTA = "native_layer_delta"
SEGMENT_DELTA = "seg_delta"
SEGMENT_TAG = "seg_tag"


def _relative(value: Any, index: int) -> tuple:
    """Labelled layer delta, distinguishing ``absent`` from ``owner == 0``."""
    if value is None:
        return ABSENT
    resolved = int(value)
    if resolved < 0:
        return ABSENT
    return (LAYER_DELTA, resolved - int(index))


def _plan_template(plan: Any) -> tuple:
    if plan is None:
        return ()
    template = tuple(getattr(plan, field, None) for field in TEMPLATE_FIELDS)
    # Ownership and native ids are relative to the occurrence: a self-owner is
    # one pattern, a cross-layer owner is a distinct one.  ``or -1`` collapsed a
    # genuine owner of layer 0 into "no owner", so the sentinel is an explicit
    # labelled tuple instead.
    index = int(getattr(plan, "index", 0))
    ownership = tuple(_relative(getattr(plan, field, None), index)
                      for field in OWNERSHIP_FIELDS)
    native = getattr(plan, "native_layer_id", None)
    native_relative = (ABSENT if native is None
                       else (NATIVE_LAYER_DELTA, int(native) - index))
    return template + ownership + (native_relative,)


def _segment_signature(op: Any, base_segment: int | None) -> tuple:
    """Intra-layer capture-segment offset, never the absolute capture ordinal.

    Piecewise ACLGraph numbers capture segments over the whole unrolled graph,
    so two structurally identical layers hold different absolute ordinals while
    their intra-layer capture split is the same.  The relative delta is the
    physical identity; a non-numeric tag is preserved verbatim.
    """
    raw = _metadata(op).get("graph_segment")
    if raw is None or raw == "":
        return (ABSENT,)
    if isinstance(raw, int) and not isinstance(raw, bool):
        anchor = raw if base_segment is None else base_segment
        return (SEGMENT_DELTA, int(raw) - int(anchor))
    text = str(raw)
    if text.isdigit():
        value = int(text)
        anchor = value if base_segment is None else base_segment
        return (SEGMENT_DELTA, value - int(anchor))
    return (SEGMENT_TAG, text)


def _first_layer_segment(ops: Sequence[Any]) -> int | None:
    """The first numeric capture ordinal declared inside one layer."""
    for op in ops:
        raw = _metadata(op).get("graph_segment")
        if isinstance(raw, int) and not isinstance(raw, bool):
            return int(raw)
        if isinstance(raw, str) and raw.isdigit():
            return int(raw)
    return None


def _op_run_key(op: Any, plan: Any, base_segment: int | None = None) -> tuple:
    from mindie_sim.prediction.program.physical_key import physical_key
    meta = _metadata(op)
    abi = getattr(op, "precision_abi", None)
    sched = getattr(op, "owner_schedule", None)
    attnres_view = meta.get("attn_res_view")
    attnres_sig = ()
    if isinstance(attnres_view, Mapping):
        attnres_sig = (
            int(attnres_view.get("valid_blocks", 0) or 0),
            int(attnres_view.get("block_write_idx", -1)),
            bool(attnres_view.get("fuse_add", False)),
            bool(attnres_view.get("output_norm", False)),
            bool(attnres_view.get("mix", True)),
            bool(attnres_view.get("save_materialized", False)),
            int(attnres_view.get("bank_capacity", 0) or 0),
        )
    # physical_key already normalizes occurrence references relative to this
    # operator, so identical layers produce the same ordered key sequence.
    return (
        int(getattr(op, "stage", 0)),
        str(getattr(op, "stream", "") or ""),
        str(getattr(op, "dtype", "") or ""),
        _plan_template(plan),
        bool(meta.get("graph_captured")),
        _segment_signature(op, base_segment),
        abi.abi_signature if abi is not None else "",
        (sched.tiling_strategy, sched.active_owners, sched.work_tiles) if sched is not None else (),
        attnres_sig,
        tuple(sorted((str(key), repr(value)) for key, value in meta.items()
                     if key in {"group_axis", "collective", "local_tp_width",
                                "capture_bucket", "recipe", "kernel",
                                "valid_blocks", "residual_block_id",
                                "block_write_idx", "linear_method",
                                "precision_abi_signature"})),
        physical_key(op),
    )


def layer_templates(ops: Sequence[Any],
                    plans: Sequence[Any] | None = None) -> tuple[tuple, ...]:
    """Ordered per-layer run keys, preserving declared operator order."""
    by_layer: dict[int, list[Any]] = {}
    order: list[int] = []
    for op in ops:
        layer = int(getattr(op, "layer", -1))
        if layer < 0:
            continue
        if layer not in by_layer:
            by_layer[layer] = []
            order.append(layer)
        by_layer[layer].append(op)
    plans = tuple(plans or ())
    plan_for = {int(getattr(plan, "index", index)): plan
                for index, plan in enumerate(plans)}
    return tuple(tuple(_op_run_key(op, plan_for.get(layer),
                                   _first_layer_segment(by_layer[layer]))
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
    layers = list(dict.fromkeys(
        int(getattr(op, "layer", -1)) for op in ops
        if int(getattr(op, "layer", -1)) >= 0
    ))
    plans = tuple(plans or ())
    plan_for = {int(getattr(plan, "index", index)): plan
                for index, plan in enumerate(plans)}
    runs: list[LayerRun] = []
    start, key = 0, templates[0]
    for position in range(1, len(templates)):
        cut = _run_cut(templates[position - 1], templates[position],
                       plan_for.get(layers[position - 1]),
                       plan_for.get(layers[position]))
        if cut is not None:
            runs.append(LayerRun(start, position - start, key,
                                 tuple(layers[start:position])))
            start, key = position, templates[position]
    runs.append(LayerRun(start, len(templates) - start, key, tuple(layers[start:])))
    return tuple(run for run in runs if run.count >= MIN_RUN_LENGTH)


@dataclass(frozen=True)
class ExecutionStateSnapshot:
    """Full live-Engine state fingerprint across a layer transition boundary."""

    l2_residency: tuple[tuple[str, int], ...] = ()
    l2_dirty: tuple[str, ...] = ()
    stream_tail_offsets_ns: tuple[tuple[str, int], ...] = ()
    bank_state: tuple[tuple[str, Any], ...] = ()
    owner_queues: tuple[tuple[str, int], ...] = ()
    queue_credits: tuple[tuple[str, int], ...] = ()
    state_versions: tuple[tuple[str, Any], ...] = ()
    active_allocations: tuple[tuple[str, int], ...] = ()

    @classmethod
    def capture(
        cls,
        engine: Any,
        *,
        stream_tails_ms: Mapping[str, float] | None = None,
        bank_state: Mapping[str, Any] | Sequence[tuple[str, Any]] = (),
        owner_queues: Mapping[str, int] | Sequence[tuple[str, int]] = (),
        queue_credits: Mapping[str, int] | Sequence[tuple[str, int]] = (),
    ) -> "ExecutionStateSnapshot":
        now_ms = float(getattr(engine, "now", getattr(engine, "now_ms", 0.0)) or 0.0)
        now_ns = int(round(now_ms * NS_PER_MS))
        cache_domains = getattr(engine, "cache_domains", None) or {}
        l2_res = tuple(sorted(
            (str(k), int(len(v)) if hasattr(v, "__len__") else int(bool(v)))
            for k, v in (cache_domains.items() if isinstance(cache_domains, Mapping) else ())
        ))
        dirty_raw = getattr(engine, "dirty_resource_components", None) or ()
        l2_dirty = tuple(sorted(str(x) for x in dirty_raw))
        stream_offsets = tuple(sorted(
            (str(s), max(0, int(round(float(t) * NS_PER_MS)) - now_ns))
            for s, t in (stream_tails_ms or {}).items()
        ))
        bank_items = tuple(sorted(
            (str(k), v) for k, v in (bank_state.items() if isinstance(bank_state, Mapping) else bank_state)
        ))
        oq_items = tuple(sorted(
            (str(k), int(v)) for k, v in (owner_queues.items() if isinstance(owner_queues, Mapping) else owner_queues)
        ))
        qc_source = queue_credits
        if not qc_source:
            fifos = getattr(engine, "host_fifos", None) or {}
            if isinstance(fifos, Mapping):
                qc_source = tuple(
                    (str(k), int(getattr(f, "available_credits", getattr(f, "credits", 0)) or 0))
                    for k, f in fifos.items()
                )
        qc_items = tuple(sorted(
            (str(k), int(v)) for k, v in (qc_source.items() if isinstance(qc_source, Mapping) else qc_source)
        ))
        versions_raw = getattr(engine, "state_versions", None) or {}
        versions = tuple(sorted(
            (str(k), v) for k, v in (versions_raw.items() if isinstance(versions_raw, Mapping) else ())
        ))
        allocs_raw = getattr(engine, "allocated", None) or {}
        allocs = tuple(sorted(
            (str(k), int(v)) for k, v in (allocs_raw.items() if isinstance(allocs_raw, Mapping) else ())
            if int(v) > 0
        ))
        return cls(
            l2_residency=l2_res,
            l2_dirty=l2_dirty,
            stream_tail_offsets_ns=stream_offsets,
            bank_state=bank_items,
            owner_queues=oq_items,
            queue_credits=qc_items,
            state_versions=versions,
            active_allocations=allocs,
        )


@dataclass(frozen=True)
class TransitionEquivalenceProof:
    """Mechanical proof that repeating a layer transition preserves full Engine state."""

    before_state: ExecutionStateSnapshot
    after_state: ExecutionStateSnapshot
    cycle_delta_ns: int
    correspondence: tuple[tuple[Any, Any], ...] = ()
    verified_by: str = "engine_frontier_equivalence"

    def __post_init__(self) -> None:
        if self.before_state != self.after_state:
            raise ModelingError("layer_run_state_equivalence_failed")
        if isinstance(self.cycle_delta_ns, bool) or not isinstance(self.cycle_delta_ns, int) or self.cycle_delta_ns < 0:
            raise ModelingError("layer_run_cycle_delta_invalid")

    @property
    def is_valid(self) -> bool:
        return self.before_state == self.after_state and self.cycle_delta_ns >= 0


class CompiledCohortProgramCache:
    """Bounded LRU cache compiling structural cohort programs once per execution context."""

    def __init__(self, max_entries: int = DEFAULT_COHORT_CACHE_LIMIT) -> None:
        self.max_entries = max(1, int(max_entries))
        self._store: OrderedDict[Any, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get_or_compile(self, key: Any, compiler: Callable[[], Any]) -> Any:
        if key in self._store:
            self.hits += 1
            self._store.move_to_end(key)
            return self._store[key]
        self.misses += 1
        compiled = compiler()
        self._store[key] = compiled
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
            self.evictions += 1
        return compiled

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


@dataclass(frozen=True)
class LayerRunProgram:
    """Sequential execution of one run's caller-prepared occurrence programs.

    Template reuse always executes every occurrence on the live Engine.  Time
    skipping requires an explicit ``TransitionEquivalenceProof`` certifying
    full execution-state equivalence (L2 residency/dirty, stream tails, bank
    state, owner queues, queue credits, state versions, and allocations);
    otherwise construction refuses instead of inventing ``duration x count``.
    """

    run: LayerRun
    occurrences: tuple[Any, ...]
    skip_authorized: bool = False
    certification: TransitionEquivalenceProof | None = None

    def __post_init__(self) -> None:
        if len(self.occurrences) != self.run.count:
            raise ModelingError(
                "layer_run_occurrence_count_mismatch:"
                f"{len(self.occurrences)}!={self.run.count}")
        if self.skip_authorized and (self.certification is None or not self.certification.is_valid):
            raise ModelingError("layer_run_time_skip_requires_engine_certification")

    def start(self, engine: Any, name: str, ready: Any, *, owner: str = "") -> Any:
        event = ready
        for position, program in enumerate(self.occurrences):
            event = program.start(engine, f"{name}:l{position}", event, owner=owner)
        return event


@dataclass(frozen=True)
class CohortOccurrenceBinding:
    """One native layer or tail occurrence bound in original execution order."""

    occurrence_index: int
    native_layer_id: int
    template_key: Any
    context_key: Any
    is_final_tail: bool = False


def replay_ordered_cohort(
    bindings: Sequence[CohortOccurrenceBinding],
    *,
    compile_template: Callable[[CohortOccurrenceBinding], Any],
    bind_occurrence: Callable[[Any, CohortOccurrenceBinding], Any] | None = None,
    cache: CompiledCohortProgramCache | None = None,
) -> tuple[Any, ...]:
    """Compile each distinct structural cohort once and emit real ordered occurrences.

    Preserves exact original execution order ``0..L-1 + final`` without
    mutating layer indices or multiplying representative durations.
    """
    active_cache = cache if cache is not None else CompiledCohortProgramCache()
    programs: list[Any] = []
    for binding in bindings:
        cache_key = (binding.template_key, binding.context_key)
        compiled = active_cache.get_or_compile(
            cache_key, lambda b=binding: compile_template(b),
        )
        bound = bind_occurrence(compiled, binding) if bind_occurrence is not None else compiled
        programs.append(bound)
    return tuple(programs)


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
