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
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass, replace
import re
from typing import Any, Callable, Mapping, Sequence

from mindie_sim.schema.common import ModelingError


MIN_RUN_LENGTH = 2
DEFAULT_COHORT_CACHE_LIMIT = 1024
NS_PER_MS = 1_000_000.0
COHORT_MIN_BODY_LAYERS = 6
COHORT_CROSS_LAYER_OVERLAP_EPS_MS = 1e-9
_NON_BODY_OP_PREFIXES = ("draft", "final.", "lm_head", "vocab", "embed", "sample", "spec_")
_NON_BODY_FAMILIES = frozenset({"speculative_acceptance", "embedding", "lm_head", "vocab_logits"})
_MAIN_COMPUTE_STREAMS = frozenset({"", "main", "compute"})
_QUANT_LAYER_TOKEN_RE = re.compile(r"((?:layers|layer|blocks|h)[._])(\d+)([._]|$)")
_SEQ_DRAFT_LAYER_RE = re.compile(r"^(draft(?:\d+|_prefill)\.s\d+)\.l(\d+)\.")
_COHORT_POSITION_FIELDS = frozenset({
    "index",
    "native_layer_id",
    "kv_owner_layer_id",
    "index_k_owner_layer_id",
    "topk_owner_layer_id",
    "candidate_source_layer_id",
    "index_source_layer_id",
})


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


@dataclass(frozen=True)
class LayerCohortSpec:
    """One structural cohort of native body layers preserving exact ``0..L-1`` indices."""

    cohort_index: int
    stage: int
    representative_layer: int
    layer_indices: tuple[int, ...]
    representative_plan: Any
    signature: tuple[Any, ...]

    @property
    def count(self) -> int:
        return len(self.layer_indices)


@dataclass(frozen=True)
class LayerCohortPlan:
    """Canonical non-destructive structural cohort plan over native layers ``0..L-1``."""

    cohorts: tuple[LayerCohortSpec, ...]
    cohort_by_layer: Mapping[int, int]
    stage_counts: tuple[int, ...]
    total_target_layers: int
    total_draft_layers: int

    def layer_weights_by_rep(self) -> dict[int, int]:
        return {c.representative_layer: c.count for c in self.cohorts}


def _stage_counts_for_case(case: Any) -> tuple[int, ...]:
    runtime = getattr(case, "runtime", None)
    parallel = getattr(runtime, "parallel", None)
    pp = max(1, int(getattr(parallel, "pp", 1) or 1))
    plans = tuple(getattr(getattr(case, "model", None), "layer_plans", ()) or ())
    n_layers = len(plans)
    part = tuple(int(v) for v in (getattr(runtime, "pp_layer_partition", ()) or ()))
    if len(part) == pp and sum(part) == n_layers:
        return part
    base, rem = divmod(n_layers, pp)
    return tuple(base + (1 if s < rem else 0) for s in range(pp))


def _freeze_quant_sig(val: Any) -> Any:
    if isinstance(val, Mapping):
        return tuple(sorted((str(k), _freeze_quant_sig(v)) for k, v in val.items()))
    if isinstance(val, (list, tuple)):
        return tuple(_freeze_quant_sig(v) for v in val)
    return repr(val)


def _lookup_layer_override_sig(
    overrides: Any, layer_idx: int, native_layer_id: int,
) -> Any | None:
    if not isinstance(overrides, Mapping):
        return None
    layers_map = overrides.get("layers") if isinstance(overrides.get("layers"), Mapping) else overrides
    for key in (str(layer_idx), layer_idx):
        if key in layers_map and layers_map[key] is not None:
            return _freeze_quant_sig(layers_map[key])
    native_map = overrides.get("native_layers")
    if isinstance(native_map, Mapping) and native_layer_id >= 0:
        for key in (str(native_layer_id), native_layer_id):
            if key in native_map and native_map[key] is not None:
                return _freeze_quant_sig(native_map[key])
    return None


def _layer_quant_signature(
    quant: Any,
    layer_idx: int,
    *,
    native_layer_id: int = -1,
    model_metadata: Any = None,
) -> tuple[Any, ...]:
    """Per-layer quant exclusion/override signature normalized to the layer token."""
    target = str(int(layer_idx))
    exclusions: list[str] = []
    overrides: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    lqo_entries: list[Any] = []
    if quant is not None:
        for raw in getattr(quant, "exclusions", ()) or ():
            tok = str(raw)
            match = _QUANT_LAYER_TOKEN_RE.search(tok)
            if match is not None and match.group(2) == target:
                exclusions.append(_QUANT_LAYER_TOKEN_RE.sub(r"\1*\3", tok))
        raw_ov = getattr(quant, "category_overrides", ()) or ()
        items = raw_ov.items() if isinstance(raw_ov, Mapping) else raw_ov
        for key, val in items:
            tok = str(key)
            match = _QUANT_LAYER_TOKEN_RE.search(tok)
            if match is not None and match.group(2) == target:
                norm = _QUANT_LAYER_TOKEN_RE.sub(r"\1*\3", tok)
                if isinstance(val, Mapping):
                    ov_sig = tuple(sorted((str(k), repr(v)) for k, v in val.items()))
                else:
                    ov_sig = (("value", repr(val)),)
                overrides.append((norm, ov_sig))
        if hasattr(quant, "layer_quant_overrides_dict") and getattr(quant, "layer_quant_overrides", ()):
            q_lqo = _lookup_layer_override_sig(quant.layer_quant_overrides_dict(), int(layer_idx), int(native_layer_id))
            if q_lqo is not None:
                lqo_entries.append(q_lqo)
    if model_metadata:
        meta_dict = dict(model_metadata) if not isinstance(model_metadata, Mapping) else model_metadata
        m_lqo = _lookup_layer_override_sig(meta_dict.get("layer_quant_overrides"), int(layer_idx), int(native_layer_id))
        if m_lqo is not None and m_lqo not in lqo_entries:
            lqo_entries.append(m_lqo)
    if not exclusions and not overrides and not lqo_entries:
        return ()
    if not lqo_entries:
        return (tuple(sorted(set(exclusions))), tuple(sorted(set(overrides))))
    return (tuple(sorted(set(exclusions))), tuple(sorted(set(overrides))), tuple(lqo_entries))


def _cohort_layer_signature(
    plan: Any,
    stage: int,
    *,
    cand_producers: frozenset[int],
    quant: Any,
    model_metadata: Any = None,
) -> tuple[Any, ...]:
    idx = int(getattr(plan, "index", 0))
    native_id = int(getattr(plan, "native_layer_id", -1))
    cand_src = int(getattr(plan, "candidate_source_layer_id", -1))
    idx_src = int(getattr(plan, "index_source_layer_id", -1))
    kv_own = int(getattr(plan, "kv_owner_layer_id", -1))
    idx_own = int(getattr(plan, "index_k_owner_layer_id", -1))
    topk_own = int(getattr(plan, "topk_owner_layer_id", -1))
    if is_dataclass(plan):
        field_sig = tuple(
            (f.name, getattr(plan, f.name))
            for f in dataclass_fields(plan)
            if f.name not in _COHORT_POSITION_FIELDS
        )
    else:
        field_sig = tuple((f, getattr(plan, f, None)) for f in TEMPLATE_FIELDS)
    return (
        int(stage),
        field_sig,
        (kv_own >= 0, kv_own == idx),
        (idx_own >= 0, idx_own == idx),
        (topk_own >= 0, topk_own == idx),
        idx in cand_producers,
        cand_src >= 0,
        (idx_src >= 0, idx_src == idx),
        _layer_quant_signature(quant, idx, native_layer_id=native_id, model_metadata=model_metadata),
    )


def discover_layer_cohorts(case: Any) -> LayerCohortPlan:
    """Group native body layers into structural cohorts without mutating ``model.layers``."""
    model = getattr(case, "model", None)
    plans = tuple(getattr(model, "layer_plans", ()) or ())
    stage_counts = _stage_counts_for_case(case)
    layer_stage = [s for s, cnt in enumerate(stage_counts) for _ in range(cnt)]
    cand_producers = frozenset(
        int(getattr(lp, "candidate_source_layer_id", -1))
        for lp in plans
        if int(getattr(lp, "candidate_source_layer_id", -1)) >= 0
    )
    quant = getattr(model, "quant", None)
    model_metadata = getattr(model, "metadata", None)
    groups: OrderedDict[tuple[Any, ...], list[Any]] = OrderedDict()
    for lp in plans:
        idx = int(lp.index)
        st = layer_stage[idx] if idx < len(layer_stage) else 0
        sig = _cohort_layer_signature(
            lp, st, cand_producers=cand_producers, quant=quant, model_metadata=model_metadata,
        )
        groups.setdefault(sig, []).append((st, lp))

    cohorts: list[LayerCohortSpec] = []
    cohort_by_layer: dict[int, int] = {}
    for c_idx, (sig, entries) in enumerate(groups.items()):
        stage = int(entries[0][0])
        rep_plan = entries[0][1]
        indices = tuple(int(lp.index) for _, lp in entries)
        cohorts.append(LayerCohortSpec(
            cohort_index=c_idx,
            stage=stage,
            representative_layer=indices[0],
            layer_indices=indices,
            representative_plan=rep_plan,
            signature=sig,
        ))
        for full_idx in indices:
            cohort_by_layer[full_idx] = c_idx

    aux = getattr(getattr(case, "model", None), "auxiliary", None)
    dspark = getattr(aux, "dspark", None) if aux is not None else None
    mtp_spec = getattr(aux, "mtp", None) if aux is not None else None
    draft_spec = dspark or mtp_spec
    draft_layers = max(0, int(getattr(draft_spec, "layers", 0) or 0)) if draft_spec is not None else 0
    return LayerCohortPlan(
        cohorts=tuple(cohorts),
        cohort_by_layer=cohort_by_layer,
        stage_counts=stage_counts,
        total_target_layers=len(plans),
        total_draft_layers=draft_layers,
    )


def build_native_cohort_bindings(
    case: Any,
    *,
    include_final_tail: bool = True,
) -> tuple[CohortOccurrenceBinding, ...]:
    """Emit ordered ``0..L-1 (+ final)`` cohort bindings preserving exact depth geometry."""
    plan = discover_layer_cohorts(case)
    plans = tuple(getattr(getattr(case, "model", None), "layer_plans", ()) or ())
    bindings: list[CohortOccurrenceBinding] = []
    for pos, lp in enumerate(plans):
        idx = int(lp.index)
        c_idx = plan.cohort_by_layer[idx]
        block = int(getattr(lp, "residual_block_size", 0) or 0)
        depth_key = (
            ("attn_res_valid", (idx + block - 1) // block, idx % block == 0)
            if block > 0 else ("plain",)
        )
        bindings.append(CohortOccurrenceBinding(
            occurrence_index=pos,
            native_layer_id=idx,
            template_key=("cohort", c_idx),
            context_key=depth_key,
            is_final_tail=False,
        ))
    if include_final_tail:
        bindings.append(CohortOccurrenceBinding(
            occurrence_index=len(bindings),
            native_layer_id=len(plans),
            template_key=("final_tail",),
            context_key=("final", len(plans)),
            is_final_tail=True,
        ))
    return tuple(bindings)


def _is_target_body_op(op: Any) -> bool:
    if int(getattr(op, "layer", -1)) < 0:
        return False
    oid = str(getattr(op, "op_id", "") or "")
    if oid.startswith(_NON_BODY_OP_PREFIXES) or "recover_ssm" in oid:
        return False
    return str(getattr(op, "family", "") or "") not in _NON_BODY_FAMILIES


def _is_diff_op_safe_to_price_isolated(op: Any) -> bool:
    if float(getattr(op, "link_bytes", 0.0) or 0.0) > 0.0:
        return False
    if str(getattr(op, "stream", "") or "").lower() not in _MAIN_COMPUTE_STREAMS:
        return False
    meta = dict(getattr(op, "metadata", ()) or ())
    return "group_axis" not in meta


def _body_layer_key(op: Any) -> tuple[str, int] | None:
    if _is_target_body_op(op):
        return ("target", int(op.layer))
    if int(getattr(op, "layer", -1)) >= 0 and str(getattr(op, "family", "") or "") not in _NON_BODY_FAMILIES:
        oid = str(getattr(op, "op_id", "") or "")
        match = _SEQ_DRAFT_LAYER_RE.match(oid)
        if match is not None and "recover_ssm" not in oid:
            return (f"draft:{match.group(1)}", int(match.group(2)))
    return None


def _plan_native_cohort_reduction(
    eff_case: Any,
    r0_ops: Sequence[Any],
    cohort_plan: LayerCohortPlan,
) -> tuple[
    dict[str, tuple[str, int]],
    dict[tuple[str, int], list[Any]],
    set[tuple[str, int]],
    dict[tuple[str, int], tuple[str, int]],
    dict[str, str],
    dict[tuple[str, int], dict[int, tuple[tuple[Any, ...], tuple[Any, ...]]]],
    dict[tuple[Any, ...], Any],
] | None:
    from mindie_sim.prediction.program.physical_key import physical_key

    pp = max(1, int(getattr(eff_case.runtime.parallel, "pp", 1) or 1))
    microbatches = max(1, int(getattr(eff_case.runtime, "pipeline_microbatches", 1) or 1))
    if pp > 1 and microbatches > 1:
        return None
    if len(getattr(eff_case.model, "layer_plans", ()) or ()) < COHORT_MIN_BODY_LAYERS:
        return None

    body_key_by_op: dict[str, tuple[str, int]] = {}
    by_layer: dict[tuple[str, int], list[Any]] = {}
    stage_layers: dict[tuple[str, int], list[tuple[str, int]]] = {}
    draft_layers_by_scope: dict[str, list[tuple[str, int]]] = {}
    for op in r0_ops:
        bkey = _body_layer_key(op)
        if bkey is None:
            continue
        body_key_by_op[op.op_id] = bkey
        by_layer.setdefault(bkey, []).append(op)
        st_i = int(getattr(op, "stage", 0) or 0)
        st_list = stage_layers.setdefault((bkey[0], st_i), [])
        if bkey not in st_list:
            st_list.append(bkey)
        if bkey[0] != "target":
            d_list = draft_layers_by_scope.setdefault(bkey[0], [])
            if bkey not in d_list:
                d_list.append(bkey)
    target_count = sum(1 for sc, _ in by_layer if sc == "target")
    if target_count < COHORT_MIN_BODY_LAYERS:
        return None

    rep_layers: set[tuple[str, int]] = set()
    if pp > 1:
        for keys in stage_layers.values():
            if keys:
                rep_layers.add(min(keys))
                rep_layers.add(max(keys))
    skip_map: dict[tuple[str, int], tuple[str, int]] = {}
    op_rep_map: dict[str, str] = {}
    layer_diffs: dict[tuple[str, int], dict[int, tuple[tuple[Any, ...], tuple[Any, ...]]]] = {}
    diff_ops_by_key: dict[tuple[Any, ...], Any] = {}

    cohort_groups: list[tuple[tuple[str, int], tuple[tuple[str, int], ...]]] = [
        (("target", int(c.representative_layer)), tuple(("target", int(i)) for i in c.layer_indices))
        for c in cohort_plan.cohorts
    ]
    for _d_scope, d_keys in draft_layers_by_scope.items():
        if len(d_keys) >= MIN_RUN_LENGTH:
            by_skel: OrderedDict[tuple[Any, ...], list[tuple[str, int]]] = OrderedDict()
            for dk in sorted(d_keys):
                skel = tuple((op.family, op.recipe, op.stream) for op in by_layer.get(dk, ()))
                by_skel.setdefault(skel, []).append(dk)
            for grp in by_skel.values():
                if len(grp) >= MIN_RUN_LENGTH:
                    cohort_groups.append((grp[0], tuple(grp)))
                else:
                    rep_layers.update(grp)
        else:
            rep_layers.update(d_keys)

    for rep_key, member_keys in cohort_groups:
        rep_ops = by_layer.get(rep_key, [])
        if not rep_ops:
            continue
        rep_layers.add(rep_key)
        rep_skel = tuple((op.family, op.recipe, op.stream) for op in rep_ops)
        rep_pkeys = tuple((int(op.stage), physical_key(op)) for op in rep_ops)
        for l_key in member_keys:
            if l_key == rep_key or l_key in rep_layers:
                continue
            l_ops = by_layer.get(l_key, [])
            l_skel = tuple((op.family, op.recipe, op.stream) for op in l_ops)
            if l_skel != rep_skel:
                rep_layers.add(l_key)
                continue
            safe = True
            cand_diffs: dict[int, tuple[tuple[Any, ...], tuple[Any, ...]]] = {}
            for pos, (op_r, pkey_r, op_l) in enumerate(zip(rep_ops, rep_pkeys, l_ops)):
                pkey_l = (int(op_l.stage), physical_key(op_l))
                if pkey_l != pkey_r:
                    if not _is_diff_op_safe_to_price_isolated(op_l) or not _is_diff_op_safe_to_price_isolated(op_r):
                        safe = False
                        break
                    cand_diffs[pos] = (pkey_r, pkey_l)
            if not safe:
                rep_layers.add(l_key)
                continue
            skip_map[l_key] = rep_key
            if cand_diffs:
                layer_diffs[l_key] = cand_diffs
                for pos, (pkey_r, pkey_l) in cand_diffs.items():
                    diff_ops_by_key.setdefault(pkey_r, rep_ops[pos])
                    diff_ops_by_key.setdefault(pkey_l, l_ops[pos])
            for op_r, op_l in zip(rep_ops, l_ops):
                op_rep_map[op_l.op_id] = op_r.op_id
    if not skip_map:
        return None
    return body_key_by_op, by_layer, rep_layers, skip_map, op_rep_map, layer_diffs, diff_ops_by_key


def _reduce_replica_graphs(
    graphs: Mapping[int, Sequence[Any]],
    body_key_by_op: Mapping[str, tuple[str, int]],
    rep_layers: set[tuple[str, int]],
    external_keys: set[str],
) -> dict[int, tuple[Any, ...]] | None:
    reduced: dict[int, tuple[Any, ...]] = {}
    for replica, r_ops in graphs.items():
        kept_set = {
            op.op_id for op in r_ops
            if (bkey := body_key_by_op.get(op.op_id)) is None or (bkey in rep_layers)
        }
        prev_by_stream: dict[str, str] = {}
        last_kept: str | None = None
        redirect: dict[str, str | None] = {}
        for op in r_ops:
            st = str(getattr(op, "stream", "") or "main")
            if op.op_id in kept_set:
                prev_by_stream[st] = op.op_id
                last_kept = op.op_id
            else:
                redirect[op.op_id] = prev_by_stream.get(st) or last_kept
        red_ops: list[Any] = []
        seen_so_far: set[str] = set()
        for op in r_ops:
            if op.op_id not in kept_set:
                continue
            new_deps: list[str] = []
            for dep in op.deps:
                if dep in kept_set:
                    new_deps.append(dep)
                elif dep in external_keys:
                    new_deps.append(dep)
                else:
                    target = redirect.get(dep)
                    if target is None or target == op.op_id or target not in seen_so_far:
                        return None
                    new_deps.append(target)
            dedup_deps = tuple(dict.fromkeys(new_deps))
            red_ops.append(replace(op, deps=dedup_deps) if dedup_deps != op.deps else op)
            seen_so_far.add(op.op_id)
        reduced[int(replica)] = tuple(red_ops)
    return reduced


def _has_cross_layer_overlap(
    tl_by_layer: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    kept_layers_ordered: Sequence[tuple[str, int]],
) -> bool:
    prev_end_by_scope_stage: dict[tuple[str, int], float] = {}
    for l_key in kept_layers_ordered:
        rows = tl_by_layer.get(l_key, ())
        if not rows:
            continue
        stage = int(rows[0].get("stage", 0) or 0)
        l_start = min(float(r["ready_ms"]) for r in rows)
        l_end = max(float(r["complete_ms"]) for r in rows)
        ss = (l_key[0], stage)
        prev_end = prev_end_by_scope_stage.get(ss)
        if prev_end is not None and l_start + COHORT_CROSS_LAYER_OVERLAP_EPS_MS < prev_end:
            return True
        prev_end_by_scope_stage[ss] = l_end
    return False


def _price_depth_varying_ops(
    eff_case: Any,
    diff_ops_by_key: Mapping[tuple[Any, ...], Any],
    *,
    evaluation: Any,
    recipes: Any,
) -> dict[tuple[Any, ...], float]:
    from mindie_sim.prediction.execution.operator_replay import operator_costs
    from mindie_sim.prediction.operator_analysis import physical_id
    from mindie_sim.prediction.program import RecipeBook

    if not diff_ops_by_key:
        return {}
    pid_cache: dict[tuple[Any, ...], float] | None = None
    if evaluation is not None:
        pid_cache = getattr(evaluation, "_cohort_pkey_dur_cache", None)
        if pid_cache is None:
            pid_cache = {}
            try:
                setattr(evaluation, "_cohort_pkey_dur_cache", pid_cache)
            except (AttributeError, TypeError):
                pid_cache = None

    pkey_dur_ms: dict[tuple[Any, ...], float] = {}
    missing_ops: list[Any] = []
    pid_to_pkeys: dict[tuple[int, str], list[tuple[Any, ...]]] = {}
    for pkey, op in diff_ops_by_key.items():
        if pid_cache is not None and pkey in pid_cache:
            pkey_dur_ms[pkey] = pid_cache[pkey]
        else:
            missing_ops.append(op)
            pid_to_pkeys.setdefault((int(op.stage), physical_id(op)), []).append(pkey)
    if missing_ops:
        book = recipes or RecipeBook.from_mapping(eff_case.runtime.kernels)
        oc = operator_costs(
            eff_case, missing_ops,
            recipes=book, evaluation=evaluation, replica=0,
        )
        for row in oc["operators"]:
            row_stage = int(dict(row.get("operator") or {}).get("stage", 0) or 0)
            row_pid = str(row["physical_id"])
            val = float(row["duration_ms"])
            for pkey in pid_to_pkeys.get((row_stage, row_pid), ()):
                pkey_dur_ms[pkey] = val
                if pid_cache is not None:
                    pid_cache[pkey] = val
    return pkey_dur_ms


def _emit_skipped_layer_rows(
    lyr: int,
    rep_rows: Sequence[Mapping[str, Any]],
    rep_ops: Sequence[Any],
    lyr_ops: Sequence[Any],
    diffs: Mapping[int, tuple[tuple[Any, ...], tuple[Any, ...]]],
    pid_dur_ms: Mapping[tuple[Any, ...], float],
    *,
    layer_start_ms: float,
) -> tuple[list[dict[str, Any]], float]:
    rep_pos_by_id = {op.op_id: pos for pos, op in enumerate(rep_ops)}
    rep_base = min(float(r["ready_ms"]) for r in rep_rows)
    rep_span = max(float(r["complete_ms"]) for r in rep_rows) - rep_base
    emitted: list[dict[str, Any]] = []
    intra_shift = 0.0
    for row in rep_rows:
        pos = rep_pos_by_id.get(str(row["op_id"]))
        if pos is None or pos >= len(lyr_ops):
            continue
        op_l = lyr_ops[pos]
        rel_ready = float(row["ready_ms"]) - rep_base
        elapsed = float(row["elapsed_ms"])
        diff_pair = diffs.get(pos)
        if diff_pair is not None:
            pid_r, pid_l = diff_pair
            new_elapsed = float(pid_dur_ms[pid_l])
            delta = new_elapsed - float(pid_dur_ms[pid_r])
            eff_ready = layer_start_ms + rel_ready + intra_shift
            eff_end = eff_ready + new_elapsed
            intra_shift += delta
            elapsed = new_elapsed
        else:
            eff_ready = layer_start_ms + rel_ready + intra_shift
            eff_end = eff_ready + elapsed
        clone = dict(row)
        clone.update({
            "id": f"r0:{op_l.op_id}",
            "op_id": op_l.op_id,
            "layer": lyr,
            "predecessors": [f"r0:{d}" for d in op_l.deps],
            "ready_ms": eff_ready,
            "complete_ms": eff_end,
            "elapsed_ms": elapsed,
        })
        emitted.append(clone)
    return emitted, rep_span + intra_shift


def _expand_cohort_timeline(
    red_timeline: Sequence[Mapping[str, Any]],
    body_key_by_op: Mapping[str, tuple[str, int]],
    by_layer: Mapping[tuple[str, int], Sequence[Any]],
    skip_map: Mapping[tuple[str, int], tuple[str, int]],
    layer_diffs: Mapping[tuple[str, int], Mapping[int, tuple[tuple[Any, ...], tuple[Any, ...]]]],
    pid_dur_ms: Mapping[tuple[Any, ...], float],
) -> list[dict[str, Any]]:
    tl_by_layer: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in red_timeline:
        bkey = body_key_by_op.get(str(row.get("op_id", "")))
        if bkey is not None:
            tl_by_layer.setdefault(bkey, []).append(row)

    skipped_by_scope: dict[str, list[tuple[str, int]]] = {}
    for l_key in sorted(skip_map):
        skipped_by_scope.setdefault(l_key[0], []).append(l_key)
    cursors: dict[str, int] = {sc: 0 for sc in skipped_by_scope}
    expanded: list[dict[str, Any]] = []
    shift_ms = 0.0
    last_end_by_scope: dict[str, float] = {}

    def _flush_scope(scope_name: str, upper_layer: int | None) -> None:
        nonlocal shift_ms
        sk_list = skipped_by_scope.get(scope_name)
        if not sk_list:
            return
        cur = cursors[scope_name]
        while cur < len(sk_list):
            l_key = sk_list[cur]
            if upper_layer is not None and l_key[1] >= upper_layer:
                break
            rep_key = skip_map[l_key]
            rep_rows = tl_by_layer.get(rep_key, ())
            if rep_rows:
                rows_l, span_l = _emit_skipped_layer_rows(
                    l_key[1], rep_rows, by_layer[rep_key], by_layer[l_key],
                    layer_diffs.get(l_key, {}), pid_dur_ms,
                    layer_start_ms=last_end_by_scope.get(scope_name, 0.0),
                )
                expanded.extend(rows_l)
                shift_ms += span_l
                last_end_by_scope[scope_name] = last_end_by_scope.get(scope_name, 0.0) + span_l
            cur += 1
        cursors[scope_name] = cur

    active_scope: str | None = None
    for row in red_timeline:
        bkey = body_key_by_op.get(str(row.get("op_id", "")))
        if bkey is not None:
            if active_scope is not None and active_scope != bkey[0]:
                _flush_scope(active_scope, None)
            active_scope = bkey[0]
            _flush_scope(active_scope, bkey[1])
        elif active_scope is not None:
            _flush_scope(active_scope, None)
            active_scope = None
        shifted = dict(row)
        r_ms = float(row["ready_ms"]) + shift_ms
        c_ms = float(row["complete_ms"]) + shift_ms
        shifted["ready_ms"] = r_ms
        shifted["complete_ms"] = c_ms
        expanded.append(shifted)
        if bkey is not None and c_ms > last_end_by_scope.get(bkey[0], 0.0):
            last_end_by_scope[bkey[0]] = c_ms
    if active_scope is not None:
        _flush_scope(active_scope, None)
    return expanded


def execute_native_cohort_graph(
    eff_case: Any,
    graphs: Mapping[int, Sequence[Any]],
    *,
    cohort_plan: LayerCohortPlan | None = None,
    recipes: Any = None,
    external_inputs: Mapping[str, float] | None = None,
    include_submission: bool = False,
    isolated: bool = False,
    trace_limit: int = 0,
    compact: bool = False,
    evaluation: Any = None,
    fast_span_only: bool = False,
    stream_plans: Any = None,
    request_views: Any = None,
    padded_tokens: int = 0,
    sequence_layout: str = "",
    timeline_replicas: Sequence[int] | None = (0,),
) -> tuple[dict[str, Any], dict[int, tuple[Any, ...]] | None, dict[str, str]]:
    """Execute native ``0..L-1 + final`` graphs via exact cohort reduction on ONE Engine.

    Never mutates ``model.layers`` or ``layer_plans``.  Falls back cleanly to
    literal full-depth ``graph_costs`` whenever cross-layer overlap, PP > 1,
    isolated-pricing requests, or non-standard cross-layer dependencies occur.
    """
    from mindie_sim.prediction.graph_analysis import graph_costs
    from mindie_sim.prediction.operator_analysis import TargetGraphSet

    def _literal_run() -> tuple[dict[str, Any], None, dict[str, str]]:
        rep = graph_costs(
            eff_case, graphs, recipes=recipes, external_inputs=external_inputs,
            include_submission=include_submission, isolated=isolated,
            trace_limit=trace_limit, compact=compact, evaluation=evaluation,
            fast_span_only=fast_span_only, stream_plans=stream_plans,
            request_views=request_views, padded_tokens=padded_tokens,
            sequence_layout=sequence_layout, timeline_replicas=timeline_replicas,
        )
        return rep, None, {}

    if isolated or compact or 0 not in graphs:
        return _literal_run()
    plan = cohort_plan if isinstance(cohort_plan, LayerCohortPlan) else discover_layer_cohorts(eff_case)
    planned = _plan_native_cohort_reduction(eff_case, graphs[0], plan)
    if planned is None:
        return _literal_run()
    body_key_by_op, by_layer, rep_layers, skip_map, op_rep_map, layer_diffs, diff_ops_by_key = planned
    ext_keys = set(external_inputs or ())
    reduced_map = _reduce_replica_graphs(graphs, body_key_by_op, rep_layers, ext_keys)
    if reduced_map is None:
        return _literal_run()

    reduced_arg: Any = (
        TargetGraphSet(
            reduced_map,
            kernels=getattr(graphs, "effective_kernels", None),
            selection=getattr(graphs, "default_selection", None),
        )
        if isinstance(graphs, TargetGraphSet) else reduced_map
    )
    if fast_span_only:
        red_report = graph_costs(
            eff_case, reduced_arg, recipes=recipes, external_inputs=external_inputs,
            include_submission=include_submission, isolated=False,
            trace_limit=trace_limit, compact=False, evaluation=evaluation,
            fast_span_only=True, stream_plans=stream_plans,
            request_views=request_views, padded_tokens=padded_tokens,
            sequence_layout=sequence_layout, timeline_replicas=None,
            layer_span_body_ids=body_key_by_op,
        )
        bounds = red_report.get("layer_bounds") or {}
        prev_by_scope_stage: dict[tuple[str, int], float] = {}
        for l_key in sorted(rep_layers):
            entry = bounds.get(l_key)
            if entry is None:
                continue
            st_i, lo_ms, hi_ms = entry
            ss = (l_key[0], st_i)
            prev_end = prev_by_scope_stage.get(ss)
            if prev_end is not None and lo_ms + COHORT_CROSS_LAYER_OVERLAP_EPS_MS < prev_end:
                return _literal_run()
            prev_by_scope_stage[ss] = hi_ms
        pid_dur_ms = _price_depth_varying_ops(
            eff_case, diff_ops_by_key, evaluation=evaluation, recipes=recipes,
        )
        total_ms = float(red_report["duration_ms"])
        for l_key, rep_key in skip_map.items():
            entry = bounds.get(rep_key)
            rep_span = (float(entry[2]) - float(entry[1])) if entry is not None else 0.0
            delta = sum(
                float(pid_dur_ms[pid_l]) - float(pid_dur_ms[pid_r])
                for pid_r, pid_l in layer_diffs.get(l_key, {}).values()
            )
            total_ms += rep_span + delta
        out = dict(red_report)
        out.pop("layer_bounds", None)
        out["duration_ms"] = total_ms
        out["complete_ms"] = float(red_report.get("input_ready_ms", 0.0) or 0.0) + total_ms
        out["fast_span_only"] = True
        return out, reduced_map, op_rep_map

    red_report = graph_costs(
        eff_case, reduced_arg, recipes=recipes, external_inputs=external_inputs,
        include_submission=include_submission, isolated=False,
        trace_limit=trace_limit, compact=False, evaluation=evaluation,
        fast_span_only=False, stream_plans=stream_plans,
        request_views=request_views, padded_tokens=padded_tokens,
        sequence_layout=sequence_layout, timeline_replicas=(0,),
    )
    red_timeline = red_report.get("timeline") or ()
    tl_by_layer: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in red_timeline:
        bkey = body_key_by_op.get(str(row.get("op_id", "")))
        if bkey is not None:
            tl_by_layer.setdefault(bkey, []).append(row)
    if _has_cross_layer_overlap(tl_by_layer, sorted(rep_layers)):
        return _literal_run()

    pid_dur_ms = _price_depth_varying_ops(
        eff_case, diff_ops_by_key, evaluation=evaluation, recipes=recipes,
    )
    total_ms = float(red_report["duration_ms"])
    for l_key, rep_key in skip_map.items():
        rows_rep = tl_by_layer.get(rep_key, ())
        rep_span = (
            max(float(r["complete_ms"]) for r in rows_rep) - min(float(r["ready_ms"]) for r in rows_rep)
            if rows_rep else 0.0
        )
        delta = sum(
            float(pid_dur_ms[pid_l]) - float(pid_dur_ms[pid_r])
            for pid_r, pid_l in layer_diffs.get(l_key, {}).values()
        )
        total_ms += rep_span + delta

    out = dict(red_report)
    out["duration_ms"] = total_ms
    out["complete_ms"] = float(red_report.get("input_ready_ms", 0.0) or 0.0) + total_ms
    expanded = _expand_cohort_timeline(
        red_timeline, body_key_by_op, by_layer, skip_map, layer_diffs, pid_dur_ms,
    )
    out["timeline"] = expanded
    out["operator_occurrences"] = sum(
        1 for r in expanded if str(r.get("family", "")) != "barrier"
    )
    return out, reduced_map, op_rep_map
