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
COHORT_MIN_STEADY_GROUP_LENGTH = 3
DEFAULT_COHORT_CACHE_LIMIT = 1024
NS_PER_MS = 1_000_000.0
COHORT_MIN_BODY_LAYERS = 6
COHORT_CROSS_LAYER_OVERLAP_EPS_MS = 1e-9
_CROSS_DP_GROUP_AXES = frozenset({"ep", "local_tp", "cp", "dcp_dp", "dcp_a2a"})
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
    draft_layers = 0
    if aux is not None:
        aux_fields = (
            tuple(f.name for f in dataclass_fields(aux))
            if is_dataclass(aux) else ("dspark", "mtp", "eagle3")
        )
        for field_name in aux_fields:
            spec = getattr(aux, field_name, None)
            if spec is not None:
                layers_val = getattr(spec, "layers", None)
                if isinstance(layers_val, int) and not isinstance(layers_val, bool) and layers_val > 0:
                    draft_layers += int(layers_val)
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
    *,
    reuse_preceding_warmup: bool = True,
) -> tuple[
    dict[str, tuple[str, int]],
    dict[tuple[str, int], list[Any]],
    set[tuple[str, int]],
    dict[tuple[str, int], tuple[str, int]],
    dict[str, str],
    dict[tuple[str, int], dict[int, tuple[tuple[Any, ...], tuple[Any, ...]]]],
    dict[tuple[Any, ...], Any],
    frozenset[tuple[str, int]],
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

    stage_first: dict[tuple[str, int], tuple[str, int]] = {
        sk: min(keys) for sk, keys in stage_layers.items() if keys
    }
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
    reused_warmup_reps: set[tuple[str, int]] = set()

    cohort_groups: list[tuple[tuple[str, int], tuple[tuple[str, int], ...]]] = []
    for c in cohort_plan.cohorts:
        t_skel: OrderedDict[tuple[Any, ...], list[tuple[str, int]]] = OrderedDict()
        for i in c.layer_indices:
            lk = ("target", int(i))
            ops_l = by_layer.get(lk, ())
            if not ops_l:
                continue
            stage_i = int(getattr(ops_l[0], "stage", 0) or 0)
            skel = (stage_i, tuple((op.family, op.recipe, op.stream) for op in ops_l))
            t_skel.setdefault(skel, []).append(lk)
        for (stage_i, _), grp in t_skel.items():
            if (reuse_preceding_warmup and len(grp) >= MIN_RUN_LENGTH
                    and grp[0] != stage_first.get((grp[0][0], stage_i))):
                cohort_groups.append((grp[0], tuple(grp)))
                if len(grp) >= COHORT_MIN_STEADY_GROUP_LENGTH:
                    reused_warmup_reps.add(grp[0])
            elif len(grp) >= COHORT_MIN_STEADY_GROUP_LENGTH:
                rep_layers.add(grp[0])
                cohort_groups.append((grp[1], tuple(grp[1:])))
            else:
                rep_layers.update(grp)
    for _d_scope, d_keys in draft_layers_by_scope.items():
        if len(d_keys) >= MIN_RUN_LENGTH:
            by_skel: OrderedDict[tuple[Any, ...], list[tuple[str, int]]] = OrderedDict()
            for dk in sorted(d_keys):
                ops_d = by_layer.get(dk, ())
                if not ops_d:
                    continue
                stage_d = int(getattr(ops_d[0], "stage", 0) or 0)
                skel = (stage_d, tuple((op.family, op.recipe, op.stream) for op in ops_d))
                by_skel.setdefault(skel, []).append(dk)
            for (stage_d, _), grp in by_skel.items():
                if (reuse_preceding_warmup and len(grp) >= MIN_RUN_LENGTH
                        and grp[0] != stage_first.get((grp[0][0], stage_d))):
                    cohort_groups.append((grp[0], tuple(grp)))
                    if len(grp) >= COHORT_MIN_STEADY_GROUP_LENGTH:
                        reused_warmup_reps.add(grp[0])
                elif len(grp) >= COHORT_MIN_STEADY_GROUP_LENGTH:
                    rep_layers.add(grp[0])
                    cohort_groups.append((grp[1], tuple(grp[1:])))
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
    active_reused = frozenset(reused_warmup_reps & set(skip_map.values()))
    return body_key_by_op, by_layer, rep_layers, skip_map, op_rep_map, layer_diffs, diff_ops_by_key, active_reused


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
    pkey_dur_ms: dict[tuple[Any, ...], float] = {}
    missing_ops: list[Any] = []
    pid_to_pkeys: dict[tuple[int, str], list[tuple[Any, ...]]] = {}
    for pkey, op in diff_ops_by_key.items():
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
    return pkey_dur_ms


def _price_depth_varying_op_reports(
    eff_case: Any,
    diff_ops_by_key: Mapping[tuple[Any, ...], Any],
    replica_ids: Sequence[int],
    *,
    evaluation: Any,
    recipes: Any,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Price each unique depth-varying articulation op via ``graph_costs`` across active replicas."""
    from mindie_sim.prediction.graph_analysis import graph_costs

    if not diff_ops_by_key:
        return {}
    reps = tuple(sorted(int(r) for r in (replica_ids or (0,))))
    r0 = reps[0]
    reports: dict[tuple[Any, ...], dict[str, Any]] = {}
    for pkey, op in diff_ops_by_key.items():
        solo = replace(op, deps=())
        solo_graphs = {r: (solo,) for r in reps}
        rep = graph_costs(
            eff_case, solo_graphs, recipes=recipes, evaluation=evaluation,
            isolated=False, fast_span_only=False, timeline_replicas=(r0,),
        )
        ex = rep.get("execution") or {}
        recipes_map = {str(k): int(v) for k, v in (ex.get("recipes") or {}).items()}
        iso_rep_count = max(1, sum(recipes_map.values()))
        basis = rep.get("exposed_breakdown_basis") or {}
        reports[pkey] = {
            "duration_ms": float(rep["duration_ms"]),
            "serviced": {str(k): float(v) for k, v in (ex.get("serviced") or {}).items() if float(v)},
            "busy_ms": {str(k): float(v) for k, v in (ex.get("resource_busy_ms") or {}).items() if float(v)},
            "work": {str(k): float(v) for k, v in (ex.get("work") or {}).items() if float(v)},
            "recipes": recipes_map,
            "completed_tasks": int(ex.get("completed_tasks", 0) or 0),
            "iso_rep_count": iso_rep_count,
            "exposed_breakdown_ms": {
                str(k): float(v) for k, v in (rep.get("exposed_breakdown_ms") or {}).items()
            },
            "exposed_available": bool(basis.get("available")),
        }
    return reports


def _build_cohort_probe_sinks(
    by_layer: Mapping[tuple[str, int], Sequence[Any]],
    probe_order: Sequence[tuple[tuple[str, int], tuple[str, int]]],
) -> dict[tuple[str, int], frozenset[str]]:
    """Return effective sink op IDs for each probed layer (avoids per-op C11 callback churn)."""
    sinks: dict[tuple[str, int], frozenset[str]] = {}
    for pair in probe_order:
        for lk in pair:
            if lk is None or lk in sinks:
                continue
            ops_l = by_layer.get(lk, ())
            if not ops_l:
                continue
            _, _, _, eff_sinks = _intra_layer_structure(ops_l)
            sinks[lk] = frozenset(ops_l[pos].op_id for pos in eff_sinks)
    return sinks


def _verify_full_cohort_diff_ops(
    red_report: Mapping[str, Any],
    tl_by_layer: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    by_layer: Mapping[tuple[str, int], Sequence[Any]],
    skip_map: Mapping[tuple[str, int], tuple[str, int]],
    layer_diffs: Mapping[tuple[str, int], Mapping[int, tuple[tuple[Any, ...], tuple[Any, ...]]]],
    pid_reports: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> bool:
    """Verify every depth-varying op matches isolated timing on all replicas and has zero overlap."""
    eps = COHORT_CROSS_LAYER_OVERLAP_EPS_MS
    rep_tracked = red_report.get("replica_tracked_op_elapsed_ms")
    tracked_elapsed = red_report.get("tracked_op_elapsed_ms") or {}
    rep_ids = sorted((red_report.get("replica_layer_bounds") or {0: {}}).keys())
    need_exposed = bool((red_report.get("exposed_breakdown_basis") or {}).get("available"))
    checked_rep_pos: set[tuple[tuple[str, int], int]] = set()
    for l_key, diffs in layer_diffs.items():
        rep_key = skip_map[l_key]
        rep_ops = by_layer[rep_key]
        rep_rows = tl_by_layer.get(rep_key, ())
        rows_by_id = {str(r["op_id"]): r for r in rep_rows}
        for pos, (pid_r, pid_l) in diffs.items():
            info_r = pid_reports.get(pid_r)
            info_l = pid_reports.get(pid_l)
            if info_r is None or info_l is None:
                return False
            if need_exposed and (not info_r.get("exposed_available") or not info_l.get("exposed_available")):
                return False
            if (rep_key, pos) in checked_rep_pos:
                continue
            checked_rep_pos.add((rep_key, pos))
            oid = rep_ops[pos].op_id
            dur_r = float(info_r["duration_ms"])
            for r_i in rep_ids:
                obs_ms = (
                    rep_tracked.get((r_i, oid)) if rep_tracked is not None
                    else (tracked_elapsed.get(oid) if r_i == 0 else None)
                )
                if obs_ms is None or abs(float(obs_ms) - dur_r) > eps:
                    return False
            row_r = rows_by_id.get(oid)
            if row_r is None or abs(float(row_r["elapsed_ms"]) - dur_r) > eps:
                return False
            r_lo, r_hi = float(row_r["ready_ms"]), float(row_r["complete_ms"])
            for other in rep_rows:
                if str(other["op_id"]) == oid:
                    continue
                if float(other["ready_ms"]) + eps < r_hi and float(other["complete_ms"]) > r_lo + eps:
                    return False
    return True


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
    orig_ops_by_id: Mapping[str, Any] | None = None,
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
    stream_tail: dict[tuple[int, int, str], str] = {}

    def _record_row(row_item: dict[str, Any]) -> None:
        rep_i = int(row_item.get("replica", 0) or 0)
        st_i = int(row_item.get("stage", 0) or 0)
        stream_s = str(row_item.get("stream", "") or "main")
        skey = (rep_i, st_i, stream_s)
        orig_op = orig_ops_by_id.get(str(row_item.get("op_id", ""))) if orig_ops_by_id else None
        if orig_op is not None:
            preds = [f"r{rep_i}:{d}" for d in orig_op.deps]
            prev_s = stream_tail.get(skey)
            if prev_s is not None:
                preds.append(prev_s)
            row_item["predecessors"] = list(dict.fromkeys(preds))
        stream_tail[skey] = str(row_item["id"])
        expanded.append(row_item)

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
                for row_l in rows_l:
                    _record_row(row_l)
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
        _record_row(shifted)
        if bkey is not None and c_ms > last_end_by_scope.get(bkey[0], 0.0):
            last_end_by_scope[bkey[0]] = c_ms
    if active_scope is not None:
        _flush_scope(active_scope, None)
    return expanded


@dataclass(frozen=True)
class CohortStructuralProof:
    """Formal soundness certificate authorizing native cohort layer skipping."""

    proven: bool
    reason: str
    verified_layers: tuple[tuple[str, int], ...] = ()
    l2_fingerprint_before: str = ""
    l2_fingerprint_after: str = ""


def _intra_layer_structure(
    layer_ops: Sequence[Any],
) -> tuple[dict[str, int], tuple[tuple[int, ...], ...], tuple[int, ...], tuple[int, ...]]:
    """Return ``(pos_by_id, explicit_intra_deps, effective_entries, effective_sinks)``.

    Explicit ``op.deps`` are returned intact so any DAG mutation fails isomorphism.
    Effective entries and sinks additionally account for same-stream FIFO
    serialization (``runner.py:stream_tail``) for non-grouped ops on the same
    ``(stage, stream or "main")``.
    """
    pos_by_id = {op.op_id: pos for pos, op in enumerate(layer_ops)}
    intra_deps: list[tuple[int, ...]] = []
    eff_has_pred = [False] * len(layer_ops)
    eff_has_succ = [False] * len(layer_ops)
    last_on_stream: dict[tuple[int, str], int] = {}
    for pos, op in enumerate(layer_ops):
        deps = tuple(sorted({pos_by_id[d] for d in op.deps if d in pos_by_id}))
        intra_deps.append(deps)
        if deps:
            eff_has_pred[pos] = True
            for d_pos in deps:
                eff_has_succ[d_pos] = True
        if "device_kernel_group" not in dict(getattr(op, "metadata", ()) or ()):
            skey = (int(getattr(op, "stage", 0) or 0), str(getattr(op, "stream", "") or "main"))
            prev_pos = last_on_stream.get(skey)
            if prev_pos is not None:
                eff_has_pred[pos] = True
                eff_has_succ[prev_pos] = True
            last_on_stream[skey] = pos
    entries = tuple(pos for pos, has_p in enumerate(eff_has_pred) if not has_p)
    sinks = tuple(pos for pos, has_s in enumerate(eff_has_succ) if not has_s)
    return pos_by_id, tuple(intra_deps), entries, sinks


def _is_serial_articulation_point(
    intra_deps: Sequence[tuple[int, ...]], target_pos: int,
) -> bool:
    """True iff every other op in the layer is a strict ancestor or descendant of ``target_pos``."""
    n = len(intra_deps)
    children: list[list[int]] = [[] for _ in range(n)]
    for pos, deps in enumerate(intra_deps):
        for d in deps:
            children[d].append(pos)
    ancestors: set[int] = set()
    stack = list(intra_deps[target_pos])
    while stack:
        cur = stack.pop()
        if cur not in ancestors:
            ancestors.add(cur)
            stack.extend(intra_deps[cur])
    descendants: set[int] = set()
    stack = list(children[target_pos])
    while stack:
        cur = stack.pop()
        if cur not in descendants:
            descendants.add(cur)
            stack.extend(children[cur])
    return (len(ancestors) + len(descendants) + 1) == n


def _layer_op_side_effects(op: Any) -> tuple[Any, ...]:
    return (
        op.family,
        op.recipe,
        op.stream,
        int(getattr(op, "stage", 0) or 0),
        float(getattr(op, "link_bytes", 0.0) or 0.0),
        tuple(getattr(op, "allocations", ()) or ()),
        tuple(getattr(op, "state_effects", ()) or ()),
    )


def _check_layer_pair_equivalence(
    rep_ops: Sequence[Any],
    l_ops: Sequence[Any],
    diff_positions: Mapping[int, Any],
    physical_key_fn: Callable[[Any], Any],
) -> tuple[bool, str, tuple[tuple[int, ...], ...], tuple[int, ...], tuple[int, ...]]:
    if not rep_ops or len(l_ops) != len(rep_ops):
        return False, "layer_op_count_mismatch", (), (), ()
    _, r_intra, r_entries, r_sinks = _intra_layer_structure(rep_ops)
    _, l_intra, l_entries, l_sinks = _intra_layer_structure(l_ops)
    if l_intra != r_intra or l_entries != r_entries or l_sinks != r_sinks:
        return False, "intra_layer_dag_mismatch", (), (), ()
    for pos, (op_r, op_l) in enumerate(zip(rep_ops, l_ops)):
        if _layer_op_side_effects(op_l) != _layer_op_side_effects(op_r):
            return False, "layer_op_side_effect_mismatch", (), (), ()
        if pos in diff_positions:
            if not _is_serial_articulation_point(l_intra, pos):
                return False, "diff_op_not_serial_articulation", (), (), ()
            exp_pair = diff_positions[pos]
            if (isinstance(exp_pair, tuple) and len(exp_pair) == 2
                    and (
                        (int(op_r.stage), physical_key_fn(op_r)) != exp_pair[0]
                        or (int(op_l.stage), physical_key_fn(op_l)) != exp_pair[1]
                    )):
                return False, "replica_diff_op_physical_key_mismatch", (), (), ()
        elif physical_key_fn(op_l) != physical_key_fn(op_r):
            return False, "replica_physical_key_mismatch", (), (), ()
    return True, "ok", l_intra, l_entries, l_sinks


def _build_scope_sync_chain(
    r_by_layer: Mapping[tuple[str, int], Sequence[Any]],
    struct_cache: Mapping[tuple[str, int], tuple[Any, Any, tuple[int, ...], tuple[int, ...]]],
) -> tuple[
    dict[str, list[tuple[str, int]]],
    dict[tuple[str, int], tuple[str, int] | None],
    dict[tuple[str, int], tuple[str, int] | None],
    dict[tuple[str, int], int],
    dict[tuple[str, int], int],
]:
    """Build per-scope layer ordering and prefix sums of unbroken same-stage synchronous cuts."""
    scope_layers: dict[str, list[tuple[str, int]]] = {}
    for bkey in sorted(r_by_layer):
        scope_layers.setdefault(bkey[0], []).append(bkey)
    prev_layer: dict[tuple[str, int], tuple[str, int] | None] = {}
    next_layer: dict[tuple[str, int], tuple[str, int] | None] = {}
    pos_in_scope: dict[tuple[str, int], int] = {}
    bad_cut_prefix: dict[tuple[str, int], int] = {}
    for ordered in scope_layers.values():
        bad_so_far = 0
        for idx, bkey in enumerate(ordered):
            pos_in_scope[bkey] = idx
            prev_key = ordered[idx - 1] if idx > 0 else None
            prev_layer[bkey] = prev_key
            next_layer[bkey] = ordered[idx + 1] if idx + 1 < len(ordered) else None
            if prev_key is not None:
                prev_ops = r_by_layer[prev_key]
                cur_ops = r_by_layer[bkey]
                same_stage = int(getattr(prev_ops[0], "stage", 0) or 0) == int(getattr(cur_ops[0], "stage", 0) or 0)
                prev_sinks = {prev_ops[p].op_id for p in struct_cache[prev_key][3]}
                cur_entries = struct_cache[bkey][2]
                cut_ok = bool(
                    same_stage and prev_sinks and cur_entries
                    and all(prev_sinks.issubset(cur_ops[p].deps) for p in cur_entries)
                )
                if not cut_ok:
                    bad_so_far += 1
            bad_cut_prefix[bkey] = bad_so_far
    return scope_layers, prev_layer, next_layer, pos_in_scope, bad_cut_prefix


def _post_body_last_sink_ancestors(
    r_ops: Sequence[Any],
    op_to_layer: Mapping[str, tuple[str, int]],
    all_last_sinks: set[str],
) -> dict[str, frozenset[str]]:
    """Compute transitive last-layer sink ancestors for non-body ops via non-body edges."""
    non_body = {op.op_id: op for op in r_ops if op.op_id not in op_to_layer}
    memo: dict[str, frozenset[str]] = {}
    visiting: set[str] = set()

    def _anc(oid: str) -> frozenset[str]:
        cached = memo.get(oid)
        if cached is not None:
            return cached
        if oid in visiting:
            return frozenset()
        visiting.add(oid)
        op = non_body.get(oid)
        found: set[str] = set()
        if op is not None:
            for dep in op.deps:
                if dep in all_last_sinks:
                    found.add(dep)
                elif dep in non_body:
                    found.update(_anc(dep))
        visiting.discard(oid)
        res = frozenset(found)
        memo[oid] = res
        return res

    for oid in non_body:
        _anc(oid)
    return memo


def _check_replica_boundary_cuts(
    r_ops: Sequence[Any],
    body_key_by_op: Mapping[str, tuple[str, int]],
    skip_map: Mapping[tuple[str, int], tuple[str, int]],
    layer_diffs: Mapping[tuple[str, int], Mapping[int, Any]],
    external_inputs: Mapping[str, float] | None,
    physical_key_fn: Callable[[Any], Any],
) -> CohortStructuralProof:
    r_by_layer: dict[tuple[str, int], list[Any]] = {}
    op_to_layer: dict[str, tuple[str, int]] = {}
    for op in r_ops:
        bkey = body_key_by_op.get(op.op_id)
        if bkey is not None:
            r_by_layer.setdefault(bkey, []).append(op)
            op_to_layer[op.op_id] = bkey
    struct_cache = {bkey: _intra_layer_structure(ops_list) for bkey, ops_list in r_by_layer.items()}
    scope_layers, prev_layer, next_layer, pos_in_scope, bad_prefix = _build_scope_sync_chain(
        r_by_layer, struct_cache,
    )
    last_sinks_by_scope = {
        scope: frozenset(r_by_layer[ordered[-1]][p].op_id for p in struct_cache[ordered[-1]][3])
        for scope, ordered in scope_layers.items()
    }
    post_ancestors = _post_body_last_sink_ancestors(
        r_ops, op_to_layer, set().union(*last_sinks_by_scope.values()) if last_sinks_by_scope else set(),
    )

    def _dominated_earlier(src_key: tuple[str, int] | None, dst_key: tuple[str, int]) -> bool:
        if src_key is None or src_key[0] != dst_key[0]:
            return False
        return (
            pos_in_scope.get(src_key, 0) < pos_in_scope.get(dst_key, 0)
            and bad_prefix.get(dst_key, -1) == bad_prefix.get(src_key, -2)
        )

    ext_map = {str(k): float(v) for k, v in (external_inputs or {}).items()}
    for op in r_ops:
        if any(ext_map.get(d, 0.0) > 0.0 for d in op.deps):
            return CohortStructuralProof(False, "nonzero_external_input_release")
    consumers_by_layer: dict[tuple[str, int], list[tuple[Any, tuple[str, int] | None]]] = {}
    for op in r_ops:
        src_bkey = op_to_layer.get(op.op_id)
        for dep in op.deps:
            dep_bkey = op_to_layer.get(dep)
            if dep_bkey is not None and dep_bkey != src_bkey:
                consumers_by_layer.setdefault(dep_bkey, []).append((op, src_bkey))

    for l_key, rep_key in sorted(skip_map.items()):
        rep_ops = r_by_layer.get(rep_key, ())
        l_ops = r_by_layer.get(l_key, ())
        ok, reason, _, l_entries, l_sinks = _check_layer_pair_equivalence(
            rep_ops, l_ops, layer_diffs.get(l_key, {}), physical_key_fn,
        )
        if not ok:
            return CohortStructuralProof(False, reason)
        prev_key = prev_layer.get(l_key)
        if prev_key is None or prev_key not in r_by_layer:
            return CohortStructuralProof(False, "skipped_layer_missing_predecessor")
        prev_ops = r_by_layer[prev_key]
        if int(getattr(prev_ops[0], "stage", 0) or 0) != int(getattr(l_ops[0], "stage", 0) or 0):
            return CohortStructuralProof(False, "skipped_layer_crosses_stage")
        prev_sink_ids = {prev_ops[p].op_id for p in struct_cache[prev_key][3]}
        l_pos = struct_cache[l_key][0]
        for op in l_ops:
            ext_deps = [d for d in op.deps if d not in l_pos]
            if any(
                op_to_layer.get(d) != prev_key and not _dominated_earlier(op_to_layer.get(d), l_key)
                for d in ext_deps
            ):
                return CohortStructuralProof(False, "non_adjacent_cross_layer_dependency")
        if not prev_sink_ids or any(not prev_sink_ids.issubset(l_ops[p].deps) for p in l_entries):
            return CohortStructuralProof(False, "layer_boundary_not_synchronous_cut")
        l_sink_ids = {l_ops[p].op_id for p in l_sinks}
        nxt_key = next_layer.get(l_key)
        last_key = scope_layers[l_key[0]][-1]
        last_sink_ids = last_sinks_by_scope[l_key[0]]
        for cons_op, cons_bkey in consumers_by_layer.get(l_key, ()):
            if nxt_key is not None and cons_bkey != nxt_key:
                dom_body = cons_bkey is not None and _dominated_earlier(l_key, cons_bkey)
                dom_post = (
                    cons_bkey is None
                    and (l_key == last_key or _dominated_earlier(l_key, last_key))
                    and bool(last_sink_ids)
                    and last_sink_ids.issubset(post_ancestors.get(cons_op.op_id, ()))
                )
                if not (dom_body or dom_post):
                    return CohortStructuralProof(False, "skipped_layer_skip_edge_consumer")
            if nxt_key is None and cons_bkey is not None:
                return CohortStructuralProof(False, "skipped_layer_cross_scope_consumer")
        if nxt_key is not None and nxt_key in r_by_layer:
            nxt_ops = r_by_layer[nxt_key]
            nxt_entries = struct_cache[nxt_key][2]
            if not l_sink_ids or any(not l_sink_ids.issubset(nxt_ops[p].deps) for p in nxt_entries):
                return CohortStructuralProof(False, "next_layer_not_synchronous_cut")

    for rep_key in sorted(set(skip_map.values())):
        rep_ops = r_by_layer.get(rep_key, ())
        rep_pos, _, rep_entries, _ = struct_cache[rep_key]
        prev_rep = prev_layer.get(rep_key)
        if prev_rep is not None and prev_rep in r_by_layer:
            prev_ops = r_by_layer[prev_rep]
            prev_sink_ids = {prev_ops[p].op_id for p in struct_cache[prev_rep][3]}
            for op in rep_ops:
                ext_deps = [d for d in op.deps if d not in rep_pos]
                if any(
                    op_to_layer.get(d) != prev_rep and not _dominated_earlier(op_to_layer.get(d), rep_key)
                    for d in ext_deps
                ):
                    return CohortStructuralProof(False, "rep_layer_non_adjacent_dependency")
            if not prev_sink_ids or any(not prev_sink_ids.issubset(rep_ops[p].deps) for p in rep_entries):
                return CohortStructuralProof(False, "rep_layer_not_synchronous_cut")
        else:
            entry_ext_times = [ext_map[d] for pos in rep_entries for d in rep_ops[pos].deps if d in ext_map]
            max_entry_t = max(entry_ext_times, default=0.0)
            for pos, op in enumerate(rep_ops):
                if pos not in rep_entries and any(d in ext_map and ext_map[d] > max_entry_t for d in op.deps):
                    return CohortStructuralProof(False, "rep_layer_late_external_trigger")

    branch_proof = _check_non_body_branch_safety(
        r_ops, op_to_layer, r_by_layer, struct_cache, skip_map,
        last_sinks_by_scope, post_ancestors,
    )
    if branch_proof is not None:
        return branch_proof

    return CohortStructuralProof(True, "structural_verified", tuple(sorted(skip_map)))


def _check_non_body_branch_safety(
    r_ops: Sequence[Any],
    op_to_layer: Mapping[str, tuple[str, int]],
    r_by_layer: Mapping[tuple[str, int], Sequence[Any]],
    struct_cache: Mapping[tuple[str, int], tuple[Any, ...]],
    skip_map: Mapping[tuple[str, int], tuple[str, int]],
    last_sinks_by_scope: Mapping[str, frozenset[str]],
    post_ancestors: Mapping[str, frozenset[str]],
) -> CohortStructuralProof | None:
    """Reject non-body ops that fork before skipped layers or cut inside a skipped terminal layer."""
    max_skipped_by_scope: dict[str, tuple[str, int]] = {}
    for s_lk in skip_map:
        cur_max = max_skipped_by_scope.get(s_lk[0])
        if cur_max is None or s_lk > cur_max:
            max_skipped_by_scope[s_lk[0]] = s_lk
    for op in r_ops:
        if op.op_id in op_to_layer:
            continue
        deps_by_scope: dict[str, list[tuple[str, int]]] = {}
        for dep in op.deps:
            dep_lk = op_to_layer.get(dep)
            if dep_lk is not None:
                deps_by_scope.setdefault(dep_lk[0], []).append(dep_lk)
        if not deps_by_scope:
            continue
        op_anc = post_ancestors.get(op.op_id, frozenset())
        for sc, lks in deps_by_scope.items():
            max_skip = max_skipped_by_scope.get(sc)
            if max_skip is None:
                continue
            latest_lk = max(lks)
            last_sinks = last_sinks_by_scope.get(sc, frozenset())
            if latest_lk < max_skip:
                if not (last_sinks and last_sinks.issubset(op_anc)):
                    return CohortStructuralProof(False, "non_body_forks_before_skipped_layer")
            elif latest_lk in skip_map:
                latest_ops = r_by_layer.get(latest_lk, ())
                latest_sinks = frozenset(latest_ops[p].op_id for p in struct_cache[latest_lk][3])
                if not latest_sinks or not latest_sinks.issubset(set(op.deps) | set(op_anc)):
                    return CohortStructuralProof(False, "non_body_not_synchronous_cut")
    return None


def _prove_structural_equivalence(
    graphs: Mapping[int, Sequence[Any]],
    body_key_by_op: Mapping[str, tuple[str, int]],
    skip_map: Mapping[tuple[str, int], tuple[str, int]],
    layer_diffs: Mapping[tuple[str, int], Mapping[int, Any]],
    *,
    include_submission: bool,
    external_inputs: Mapping[str, float] | None,
) -> CohortStructuralProof:
    if include_submission:
        return CohortStructuralProof(False, "host_submission_pipelining_active")
    from mindie_sim.prediction.program.physical_key import physical_key

    for r_ops in graphs.values():
        proof = _check_replica_boundary_cuts(
            r_ops, body_key_by_op, skip_map, layer_diffs, external_inputs, physical_key,
        )
        if not proof.proven:
            return proof
    return CohortStructuralProof(True, "structural_verified", tuple(sorted(skip_map)))


def _has_non_body_overlap(
    non_body_intervals: Sequence[tuple[float, float]],
    rep_intervals: Sequence[tuple[float, float]],
) -> bool:
    eps = COHORT_CROSS_LAYER_OVERLAP_EPS_MS
    for nb_lo, nb_hi in non_body_intervals:
        for rep_lo, rep_hi in rep_intervals:
            if nb_lo + eps < rep_hi and nb_hi > rep_lo + eps:
                return True
    return False


def _build_cohort_probe_order(
    rep_layers: set[tuple[str, int]],
    skip_map: Mapping[tuple[str, int], tuple[str, int]],
) -> tuple[tuple[tuple[str, int], tuple[str, int]], ...] | None:
    """Pair each steady representative layer with its immediately preceding kept layer."""
    kept_by_scope: dict[str, list[tuple[str, int]]] = {}
    for lk in sorted(rep_layers):
        kept_by_scope.setdefault(lk[0], []).append(lk)
    order: list[tuple[tuple[str, int], tuple[str, int]]] = []
    for rep_key in sorted(set(skip_map.values())):
        scope_kept = kept_by_scope.get(rep_key[0], ())
        try:
            idx = scope_kept.index(rep_key)
        except ValueError:
            return None
        if idx == 0:
            return None
        order.append((rep_key, scope_kept[idx - 1]))
    return tuple(order)


def _layer_has_cross_dp_collective(ops: Sequence[Any]) -> bool:
    for op in ops:
        axis = dict(getattr(op, "metadata", ()) or ()).get("group_axis")
        if axis in _CROSS_DP_GROUP_AXES:
            return True
    return False


def _verify_cohort_steady_deltas(
    red_report: Mapping[str, Any],
    by_layer: Mapping[tuple[str, int], Sequence[Any]],
    skip_map: Mapping[tuple[str, int], tuple[str, int]],
    *,
    require_aligned: bool,
) -> Mapping[Any, Mapping[str, Any]] | None:
    """Verify per-representative L2 residency invariance (and alignment when needed)."""
    rep_targets = sorted(set(skip_map.values()))
    deltas = red_report.get("cohort_layer_deltas")
    if deltas:
        for rk in rep_targets:
            d_info = deltas.get(rk)
            if not d_info or not d_info.get("l2_steady"):
                return None
            if require_aligned and not d_info.get("aligned"):
                return None
            if not d_info.get("aligned") and _layer_has_cross_dp_collective(by_layer.get(rk, ())):
                return None
        return deltas
    fp_before, fp_after = red_report.get("l2_residency_fingerprints") or ("", "")
    if not fp_before or fp_before != fp_after:
        return None
    if require_aligned and red_report.get("resources"):
        return None
    return {}


def _verify_replica_bounds_and_expand(
    red_report: Mapping[str, Any],
    rep_layers: set[tuple[str, int]],
    skip_map: Mapping[tuple[str, int], tuple[str, int]],
    probe_order: Sequence[tuple[tuple[str, int], tuple[str, int]]],
    layer_diffs: Mapping[tuple[str, int], Mapping[int, tuple[tuple[Any, ...], tuple[Any, ...]]]],
    pid_dur_ms: Mapping[tuple[Any, ...], float],
) -> tuple[float, float] | None:
    """Validate non-overlap per replica and return expanded ``(total_ms, end_ms)``."""
    rep_bounds_by_rep = red_report.get("replica_layer_bounds")
    if not rep_bounds_by_rep:
        single = red_report.get("layer_bounds")
        if not single:
            return None
        rep_bounds_by_rep = {0: single}
    non_body_by_rep = red_report.get("replica_non_body_bounds")
    if non_body_by_rep is None:
        non_body_by_rep = {0: red_report.get("non_body_bounds") or ()}
    input_ready_ms = float(red_report.get("input_ready_ms", 0.0) or 0.0)
    default_complete = float(red_report.get("complete_ms", input_ready_ms + float(red_report["duration_ms"])))
    rep_complete_by_rep = red_report.get("replica_complete_ms") or {
        r: default_complete for r in rep_bounds_by_rep
    }
    eps = COHORT_CROSS_LAYER_OVERLAP_EPS_MS
    rep_targets = set(skip_map.values())
    probe_prev = dict(probe_order)
    expanded_ends: list[float] = []
    for rep_i, bounds in rep_bounds_by_rep.items():
        prev_by_scope_stage: dict[tuple[str, int], float] = {}
        rep_intervals: list[tuple[float, float]] = []
        for l_key in sorted(rep_layers):
            entry = bounds.get(l_key)
            if entry is None:
                return None
            st_i, lo_ms, hi_ms = entry
            ss = (l_key[0], st_i)
            prev_end = prev_by_scope_stage.get(ss)
            if prev_end is not None and lo_ms + eps < prev_end:
                return None
            prev_by_scope_stage[ss] = hi_ms
            if l_key in rep_targets:
                prev_k = probe_prev.get(l_key)
                if prev_k is not None:
                    prev_entry = bounds.get(prev_k)
                    if prev_entry is None or abs(float(lo_ms) - float(prev_entry[2])) > eps:
                        return None
                rep_intervals.append((float(lo_ms), float(hi_ms)))
        if _has_non_body_overlap(non_body_by_rep.get(rep_i, ()), rep_intervals):
            return None
        end_r = float(rep_complete_by_rep.get(rep_i, default_complete))
        for l_key, rep_key in skip_map.items():
            entry = bounds.get(rep_key)
            if entry is None:
                return None
            delta = sum(
                float(pid_dur_ms[pid_l]) - float(pid_dur_ms[pid_r])
                for pid_r, pid_l in layer_diffs.get(l_key, {}).values()
            )
            end_r += (float(entry[2]) - float(entry[1])) + delta
        expanded_ends.append(end_r)
    if not expanded_ends:
        return None
    end_ms = max(expanded_ends)
    return end_ms - input_ready_ms, end_ms


def _strip_internal_cohort_fields(out: dict[str, Any]) -> None:
    for key in (
        "layer_bounds",
        "replica_layer_bounds",
        "non_body_bounds",
        "replica_non_body_bounds",
        "replica_complete_ms",
        "l2_residency_fingerprints",
        "tracked_op_elapsed_ms",
        "replica_tracked_op_elapsed_ms",
        "cohort_layer_deltas",
    ):
        out.pop(key, None)


def _assemble_fast_cohort_report(
    eff_case: Any,
    red_report: Mapping[str, Any],
    by_layer: Mapping[tuple[str, int], Sequence[Any]],
    rep_layers: set[tuple[str, int]],
    skip_map: Mapping[tuple[str, int], tuple[str, int]],
    probe_order: Sequence[tuple[tuple[str, int], tuple[str, int]]],
    layer_diffs: Mapping[tuple[str, int], Mapping[int, tuple[tuple[Any, ...], tuple[Any, ...]]]],
    diff_ops_by_key: Mapping[tuple[Any, ...], Any],
    *,
    evaluation: Any,
    recipes: Any,
) -> dict[str, Any] | None:
    deltas = _verify_cohort_steady_deltas(
        red_report, by_layer, skip_map, require_aligned=False,
    )
    if deltas is None:
        return None
    pid_dur_ms = (
        _price_depth_varying_ops(
            eff_case, diff_ops_by_key, evaluation=evaluation, recipes=recipes,
        )
        if layer_diffs else {}
    )
    if layer_diffs:
        rep_tracked = red_report.get("replica_tracked_op_elapsed_ms")
        tracked_elapsed = red_report.get("tracked_op_elapsed_ms") or {}
        rep_ids = sorted((red_report.get("replica_layer_bounds") or {0: {}}).keys())
        for l_key, diffs in layer_diffs.items():
            rep_ops = by_layer[skip_map[l_key]]
            for pos, (pid_r, _pid_l) in diffs.items():
                oid = rep_ops[pos].op_id
                for r_i in rep_ids:
                    obs_ms = (
                        rep_tracked.get((r_i, oid)) if rep_tracked is not None
                        else (tracked_elapsed.get(oid) if r_i == 0 else None)
                    )
                    if (obs_ms is None
                            or abs(float(obs_ms) - float(pid_dur_ms[pid_r]))
                            > COHORT_CROSS_LAYER_OVERLAP_EPS_MS):
                        return None
    expanded_span = _verify_replica_bounds_and_expand(
        red_report, rep_layers, skip_map, probe_order, layer_diffs, pid_dur_ms,
    )
    if expanded_span is None:
        return None
    total_ms, end_ms = expanded_span
    out = dict(red_report)
    _strip_internal_cohort_fields(out)
    out["duration_ms"] = total_ms
    out["complete_ms"] = end_ms
    out["fast_span_only"] = True
    return out


def _accumulate_diff_counter_deltas(
    deltas: Mapping[Any, Mapping[str, Any]],
    skip_map: Mapping[tuple[str, int], tuple[str, int]],
    layer_diffs: Mapping[tuple[str, int], Mapping[int, tuple[tuple[Any, ...], tuple[Any, ...]]]],
    pid_reports: Mapping[tuple[Any, ...], Mapping[str, Any]],
    add_serviced: dict[str, float],
    add_busy: dict[str, float],
    add_work: dict[str, float],
    add_recipes: dict[str, int],
    add_exposed: dict[str, float],
) -> tuple[bool, int]:
    """Accumulate exact isolated diff-op counter adjustments across skipped layers."""
    extra_tasks = 0
    for l_key, diffs in layer_diffs.items():
        d_info = deltas.get(skip_map[l_key])
        if not d_info:
            return False, 0
        active_reps = max(1, int(d_info.get("active_rep_count", 1) or 1))
        for pid_r, pid_l in diffs.values():
            ir = pid_reports.get(pid_r)
            il = pid_reports.get(pid_l)
            if ir is None or il is None:
                return False, 0
            r_scale = float(active_reps) / float(max(1, int(ir.get("iso_rep_count", 1) or 1)))
            l_scale = float(active_reps) / float(max(1, int(il.get("iso_rep_count", 1) or 1)))
            for k in set(ir["serviced"]) | set(il["serviced"]):
                add_serviced[k] = add_serviced.get(k, 0.0) + float(il["serviced"].get(k, 0.0)) - float(ir["serviced"].get(k, 0.0))
            for k in set(ir["busy_ms"]) | set(il["busy_ms"]):
                add_busy[k] = add_busy.get(k, 0.0) + float(il["busy_ms"].get(k, 0.0)) - float(ir["busy_ms"].get(k, 0.0))
            for k in set(ir["work"]) | set(il["work"]):
                add_work[k] = add_work.get(k, 0.0) + l_scale * float(il["work"].get(k, 0.0)) - r_scale * float(ir["work"].get(k, 0.0))
            for k in set(ir["recipes"]) | set(il["recipes"]):
                add_recipes[k] = add_recipes.get(k, 0) + int(round(
                    l_scale * int(il["recipes"].get(k, 0)) - r_scale * int(ir["recipes"].get(k, 0))
                ))
            extra_tasks += int(round(
                l_scale * int(il.get("completed_tasks", 0)) - r_scale * int(ir.get("completed_tasks", 0))
            ))
            for cat in set(ir["exposed_breakdown_ms"]) | set(il["exposed_breakdown_ms"]):
                add_exposed[cat] = (
                    add_exposed.get(cat, 0.0)
                    + float(il["exposed_breakdown_ms"].get(cat, 0.0))
                    - float(ir["exposed_breakdown_ms"].get(cat, 0.0))
                )
    return True, extra_tasks


def _rebuild_full_cohort_counters(
    out: dict[str, Any],
    deltas: Mapping[Any, Mapping[str, Any]],
    skip_map: Mapping[tuple[str, int], tuple[str, int]],
    total_ms: float,
    end_ms: float,
    *,
    layer_diffs: Mapping[tuple[str, int], Mapping[int, tuple[tuple[Any, ...], tuple[Any, ...]]]] = (),  # type: ignore[assignment]
    pid_reports: Mapping[tuple[Any, ...], Mapping[str, Any]] = (),  # type: ignore[assignment]
) -> bool:
    """Scale resources, execution counters and exposed breakdown by skipped cohort counts."""
    skip_counts: dict[tuple[str, int], int] = {}
    for rep_key in skip_map.values():
        skip_counts[rep_key] = skip_counts.get(rep_key, 0) + 1
    add_serviced: dict[str, float] = {}
    add_busy: dict[str, float] = {}
    add_work: dict[str, float] = {}
    add_recipes: dict[str, int] = {}
    add_exposed: dict[str, float] = {}
    add_tasks = 0
    for rk, cnt in skip_counts.items():
        d_info = deltas.get(rk)
        if not d_info:
            return False
        for k, v in (d_info.get("serviced") or {}).items():
            add_serviced[k] = add_serviced.get(k, 0.0) + cnt * float(v)
        for k, v in (d_info.get("busy_ms") or {}).items():
            add_busy[k] = add_busy.get(k, 0.0) + cnt * float(v)
        for k, v in (d_info.get("work") or {}).items():
            add_work[k] = add_work.get(k, 0.0) + cnt * float(v)
        for k, v in (d_info.get("recipes") or {}).items():
            add_recipes[k] = add_recipes.get(k, 0) + cnt * int(v)
        add_tasks += cnt * int(d_info.get("completed_tasks", 0) or 0)
    if layer_diffs:
        ok_diff, extra_tasks = _accumulate_diff_counter_deltas(
            deltas, skip_map, layer_diffs, pid_reports,
            add_serviced, add_busy, add_work, add_recipes, add_exposed,
        )
        if not ok_diff:
            return False
        add_tasks += extra_tasks

    if isinstance(out.get("resources"), list):
        existing_res = {str(row["resource"]) for row in out["resources"]}
        if any(abs(v) > COHORT_CROSS_LAYER_OVERLAP_EPS_MS and k not in existing_res and "stream_quota@" not in k for k, v in add_serviced.items()):
            return False
        new_res: list[dict[str, Any]] = []
        for row in out["resources"]:
            rname = str(row["resource"])
            work_val = float(row["work"]) + add_serviced.get(rname, 0.0)
            rate_val = float(row["capacity_per_ms"])
            busy_val = float(row["busy_ms"]) + add_busy.get(rname, 0.0)
            new_row = dict(row)
            new_row["work"] = work_val
            new_row["capacity_bound_ms"] = work_val / rate_val if rate_val > 0.0 else float(row["capacity_bound_ms"])
            new_row["busy_ms"] = busy_val
            new_res.append(new_row)
        new_res.sort(key=lambda r: float(r["capacity_bound_ms"]), reverse=True)
        out["resources"] = new_res

    if isinstance(out.get("execution"), Mapping):
        ex = dict(out["execution"])
        ex["time_ms"] = end_ms
        if isinstance(ex.get("serviced"), Mapping):
            sv = dict(ex["serviced"])
            for k, dv in add_serviced.items():
                if k in sv:
                    sv[k] = float(sv[k]) + dv
            ex["serviced"] = sv
        if isinstance(ex.get("resource_busy_ms"), Mapping):
            bm = dict(ex["resource_busy_ms"])
            for k, dv in add_busy.items():
                if k in bm:
                    bm[k] = float(bm[k]) + dv
            ex["resource_busy_ms"] = bm
        if "completed_tasks" in ex:
            ex["completed_tasks"] = int(ex["completed_tasks"]) + add_tasks
        if isinstance(ex.get("work"), Mapping):
            wk = dict(ex["work"])
            for k, dv in add_work.items():
                wk[k] = float(wk.get(k, 0.0)) + dv
            ex["work"] = wk
        if isinstance(ex.get("recipes"), Mapping):
            rc = dict(ex["recipes"])
            for k, dv in add_recipes.items():
                rc[k] = int(rc.get(k, 0)) + dv
            ex["recipes"] = rc
        out["execution"] = ex

    basis = out.get("exposed_breakdown_basis")
    if isinstance(basis, Mapping) and basis.get("available"):
        if any(not (deltas.get(rk) or {}).get("exposed_available") for rk in skip_counts):
            return False
        eb_ms = dict(out.get("exposed_breakdown_ms") or {})
        new_basis = dict(basis)
        cw = dict(new_basis.get("causal_wait_ms") or {})
        for rk, cnt in skip_counts.items():
            d_info = deltas[rk]
            for cat, val in (d_info.get("exposed_breakdown_ms") or {}).items():
                eb_ms[cat] = float(eb_ms.get(cat, 0.0)) + cnt * float(val)
            d_basis = d_info.get("exposed_breakdown_basis") or {}
            if "operator_intervals" in new_basis and "operator_intervals" in d_basis:
                new_basis["operator_intervals"] = int(new_basis["operator_intervals"]) + cnt * int(d_basis["operator_intervals"])
            if "realized_edges" in new_basis and "realized_edges" in d_basis:
                new_basis["realized_edges"] = int(new_basis["realized_edges"]) + cnt * int(d_basis["realized_edges"])
            d_cw = d_basis.get("causal_wait_ms") or {}
            for wk_k in ("communication_rendezvous", "control_dispatch"):
                if wk_k in cw and wk_k in d_cw:
                    cw[wk_k] = float(cw[wk_k]) + cnt * float(d_cw[wk_k])
        for cat, dv in add_exposed.items():
            eb_ms[cat] = float(eb_ms.get(cat, 0.0)) + dv
        new_basis["duration_ms"] = total_ms
        if cw:
            new_basis["causal_wait_ms"] = cw
        out["exposed_breakdown_ms"] = eb_ms
        out["exposed_breakdown_pct"] = {
            cat: (100.0 * float(val) / total_ms if total_ms > 0.0 else 0.0)
            for cat, val in eb_ms.items()
        }
        out["exposed_breakdown_basis"] = new_basis
    return True


def _assemble_full_cohort_report(
    eff_case: Any,
    red_report: Mapping[str, Any],
    body_key_by_op: Mapping[str, tuple[str, int]],
    by_layer: Mapping[tuple[str, int], Sequence[Any]],
    rep_layers: set[tuple[str, int]],
    skip_map: Mapping[tuple[str, int], tuple[str, int]],
    probe_order: Sequence[tuple[tuple[str, int], tuple[str, int]]],
    layer_diffs: Mapping[tuple[str, int], Mapping[int, tuple[tuple[Any, ...], tuple[Any, ...]]]],
    diff_ops_by_key: Mapping[tuple[Any, ...], Any],
    orig_r0_ops: Sequence[Any],
    *,
    evaluation: Any,
    recipes: Any,
) -> dict[str, Any] | None:
    deltas = _verify_cohort_steady_deltas(
        red_report, by_layer, skip_map, require_aligned=True,
    )
    if deltas is None:
        return None
    red_timeline = red_report.get("timeline") or ()
    tl_by_layer: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    non_body_intervals: list[tuple[float, float]] = []
    for row in red_timeline:
        bkey = body_key_by_op.get(str(row.get("op_id", "")))
        if bkey is not None:
            tl_by_layer.setdefault(bkey, []).append(row)
        else:
            non_body_intervals.append((float(row["ready_ms"]), float(row["complete_ms"])))
    if _has_cross_layer_overlap(tl_by_layer, sorted(rep_layers)):
        return None
    rep_intervals = [
        (
            min(float(r["ready_ms"]) for r in tl_by_layer[rk]),
            max(float(r["complete_ms"]) for r in tl_by_layer[rk]),
        )
        for rk in sorted(set(skip_map.values()))
        if tl_by_layer.get(rk)
    ]
    if _has_non_body_overlap(non_body_intervals, rep_intervals):
        return None
    pid_reports: dict[tuple[Any, ...], dict[str, Any]] = {}
    pid_dur_ms: dict[tuple[Any, ...], float] = {}
    if layer_diffs:
        if not deltas:
            return None
        replica_ids = tuple(sorted(int(r) for r in (red_report.get("replicas") or (0,))))
        pid_reports = _price_depth_varying_op_reports(
            eff_case, diff_ops_by_key, replica_ids, evaluation=evaluation, recipes=recipes,
        )
        if not _verify_full_cohort_diff_ops(
            red_report, tl_by_layer, by_layer, skip_map, layer_diffs, pid_reports,
        ):
            return None
        pid_dur_ms = {k: float(v["duration_ms"]) for k, v in pid_reports.items()}
    if red_report.get("replica_layer_bounds"):
        expanded_span = _verify_replica_bounds_and_expand(
            red_report, rep_layers, skip_map, probe_order, layer_diffs, pid_dur_ms,
        )
        if expanded_span is None:
            return None
        total_ms, end_ms = expanded_span
    else:
        total_ms = float(red_report["duration_ms"])
        for l_key, rep_key in skip_map.items():
            rows_rep = tl_by_layer.get(rep_key, ())
            if not rows_rep:
                return None
            delta = sum(
                float(pid_dur_ms[pid_l]) - float(pid_dur_ms[pid_r])
                for pid_r, pid_l in layer_diffs.get(l_key, {}).values()
            )
            total_ms += (
                max(float(r["complete_ms"]) for r in rows_rep)
                - min(float(r["ready_ms"]) for r in rows_rep)
            ) + delta
        end_ms = float(red_report.get("input_ready_ms", 0.0) or 0.0) + total_ms
    out = dict(red_report)
    if deltas and not _rebuild_full_cohort_counters(
        out, deltas, skip_map, total_ms, end_ms,
        layer_diffs=layer_diffs, pid_reports=pid_reports,
    ):
        return None
    _strip_internal_cohort_fields(out)
    out["duration_ms"] = total_ms
    out["complete_ms"] = end_ms
    orig_ops_by_id = {op.op_id: op for op in orig_r0_ops}
    expanded = _expand_cohort_timeline(
        red_timeline, body_key_by_op, by_layer, skip_map, layer_diffs, pid_dur_ms,
        orig_ops_by_id=orig_ops_by_id,
    )
    out["timeline"] = expanded
    n_reps = max(1, len(red_report.get("replicas") or (0,)))
    skipped_non_barrier = sum(
        1 for l_key in skip_map for op in by_layer[l_key] if op.family != "barrier"
    )
    out["operator_occurrences"] = int(red_report.get("operator_occurrences", 0)) + n_reps * skipped_non_barrier
    return out


def _execute_native_cohort_attempt(
    eff_case: Any,
    graphs: Mapping[int, Sequence[Any]],
    plan: LayerCohortPlan,
    *,
    reuse_preceding_warmup: bool,
    recipes: Any,
    external_inputs: Mapping[str, float] | None,
    include_submission: bool,
    trace_limit: int,
    evaluation: Any,
    fast_span_only: bool,
    stream_plans: Any,
    request_views: Any,
    padded_tokens: int,
    sequence_layout: str,
) -> tuple[dict[str, Any] | None, dict[int, tuple[Any, ...]] | None, dict[str, str], bool]:
    from mindie_sim.prediction.graph_analysis import graph_costs
    from mindie_sim.prediction.operator_analysis import TargetGraphSet

    planned = _plan_native_cohort_reduction(
        eff_case, graphs[0], plan, reuse_preceding_warmup=reuse_preceding_warmup,
    )
    if planned is None:
        return None, None, {}, False
    (
        body_key_by_op, by_layer, rep_layers, skip_map, op_rep_map,
        layer_diffs, diff_ops_by_key, active_reused,
    ) = planned
    probe_order = _build_cohort_probe_order(rep_layers, skip_map)
    if probe_order is None:
        return None, None, {}, False
    struct_proof = _prove_structural_equivalence(
        graphs, body_key_by_op, skip_map, layer_diffs,
        include_submission=include_submission, external_inputs=external_inputs,
    )
    if not struct_proof.proven:
        return None, None, {}, False
    reduced_map = _reduce_replica_graphs(
        graphs, body_key_by_op, rep_layers, set(external_inputs or ()),
    )
    if reduced_map is None:
        return None, None, {}, False
    reduced_arg: Any = (
        TargetGraphSet(
            reduced_map,
            kernels=getattr(graphs, "effective_kernels", None),
            selection=getattr(graphs, "default_selection", None),
        )
        if isinstance(graphs, TargetGraphSet) else reduced_map
    )
    probe_sinks = _build_cohort_probe_sinks(by_layer, probe_order)
    track_ids = tuple(sorted({
        by_layer[skip_map[l_key]][pos].op_id
        for l_key, diffs in layer_diffs.items()
        for pos in diffs
    }))
    red_report = graph_costs(
        eff_case, reduced_arg, recipes=recipes, external_inputs=external_inputs,
        include_submission=include_submission, isolated=False,
        trace_limit=trace_limit, compact=False, evaluation=evaluation,
        fast_span_only=fast_span_only, stream_plans=stream_plans,
        request_views=request_views, padded_tokens=padded_tokens,
        sequence_layout=sequence_layout,
        timeline_replicas=None if fast_span_only else (0,),
        layer_span_body_ids=body_key_by_op, fast_track_op_ids=track_ids,
        cohort_probe_order=probe_order, cohort_probe_sinks=probe_sinks,
    )
    if fast_span_only:
        assembled = _assemble_fast_cohort_report(
            eff_case, red_report, by_layer, rep_layers, skip_map, probe_order,
            layer_diffs, diff_ops_by_key, evaluation=evaluation, recipes=recipes,
        )
    else:
        assembled = _assemble_full_cohort_report(
            eff_case, red_report, body_key_by_op, by_layer, rep_layers,
            skip_map, probe_order, layer_diffs, diff_ops_by_key, graphs[0],
            evaluation=evaluation, recipes=recipes,
        )
    if assembled is not None:
        return assembled, reduced_map, op_rep_map, False
    deltas = red_report.get("cohort_layer_deltas")
    l2_warmup_miss = bool(
        reuse_preceding_warmup and active_reused and isinstance(deltas, Mapping) and deltas
        and any(rk in deltas and not deltas[rk].get("l2_steady") for rk in active_reused)
    )
    return None, None, {}, l2_warmup_miss


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
    """Execute native ``0..L-1 + final`` graphs via certified cohort reduction on ONE Engine.

    Never mutates ``model.layers`` or ``layer_plans``.  Falls back cleanly to
    literal full-depth ``graph_costs`` whenever ``TransitionEquivalenceProof``
    cannot certify exact intra-layer DAG isomorphism, synchronous layer cuts,
    invariant L2 residency fingerprints, and zero cross-layer overlap.
    """
    from mindie_sim.prediction.graph_analysis import graph_costs

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
    attempt_kw = {
        "recipes": recipes, "external_inputs": external_inputs,
        "include_submission": include_submission, "trace_limit": trace_limit,
        "evaluation": evaluation, "fast_span_only": fast_span_only,
        "stream_plans": stream_plans, "request_views": request_views,
        "padded_tokens": padded_tokens, "sequence_layout": sequence_layout,
    }
    assembled, reduced_map, op_rep_map, l2_warmup_miss = _execute_native_cohort_attempt(
        eff_case, graphs, plan, reuse_preceding_warmup=True, **attempt_kw,
    )
    if assembled is not None:
        return assembled, reduced_map, op_rep_map
    if l2_warmup_miss:
        assembled, reduced_map, op_rep_map, _ = _execute_native_cohort_attempt(
            eff_case, graphs, plan, reuse_preceding_warmup=False, **attempt_kw,
        )
        if assembled is not None:
            return assembled, reduced_map, op_rep_map
    return _literal_run()
