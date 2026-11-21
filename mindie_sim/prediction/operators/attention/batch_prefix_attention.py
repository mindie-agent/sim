"""Generic shared-KV batch attention tile traversal.

[Role]
  A declared batch-key-tile attention recipe that reuses physical KV ranges
  across participating queries while preserving each request's QK/PV FLOPs,
  causal masks, running-softmax state, sparse selections and spill work.

[Invariants]
  * Only complete immutable physical KV ranges are shared; speculative/private
    rows remain request-specific.
  * Sparse TopK selections are shared only over their actual selected union;
    without selected-row identity the conservative read bound is retained.
  * Causal per-tile stage order is ``query_load -> key_tile_fill -> qk ->
    softmax -> pv -> output_store``; multi-tile schedules overlap bounded slots.

[Timeouts & Locks]
  Pure program construction; no I/O or locks.

[Side-Effect Policy]
  Returns immutable program records.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

from mindie_sim.prediction.prefix_reads import (
    CacheAccess, apply_prefix_read_plan, plan_prefix_reads,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.runtime_features import canonical_prefix_cache_key


TILE_SLOTS = 2
KEY_TILE_BYTES = 262144.0
MAX_KEY_TILES = 32
RECIPE_NAME = "batch_prefix_attention"
MATRIX_ROLES = ("cube", "matrix", "mte1", "fixpipe")
STORE_ROLES = ("mte3",)
VECTOR_ROLES = ("vector",)


@dataclass(frozen=True)
class SparsePrefixFill:
    """Cold external fill bytes of a declared sparse shared-prefix selection.

    ``selected`` is one query row's selected key count.  A shared prefix key
    denotes one immutable physical region, so every request's shared selection
    collapses onto the union bounded by the region itself, while each request
    keeps its own private selected rows.  The result is the external HBM volume;
    the remaining gathers are L2 hits and stay on the core copy pipes.
    """

    shared_selected: int
    shared_unique: int
    private_rows: int
    external_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared_selected_per_row": self.shared_selected,
            "shared_unique_rows": self.shared_unique,
            "private_rows": self.private_rows,
            "external_rows": self.external_rows,
        }


def nested_prefix_union(extent_counts):
    """Largest distinct subset of nested [0, extent) pools with given draws."""
    used = 0
    for extent, count in sorted(extent_counts):
        used = min(extent, used + count)
    return used


def selected_prefix_count(selected, extent, visible, population):
    if not selected or not extent:
        return 0
    if selected == visible:
        return extent
    estimated = (selected * extent + population // 2) // max(1, population)
    return max(selected - (visible - extent), min(estimated, selected, extent))


def prefix_members(shape, sequences):
    """Named immutable prefixes shared by at least two live requests."""
    from collections import Counter
    raw = shape.get("prefix_tokens") or 0
    extents = raw if isinstance(raw, (list, tuple)) else (raw,) * len(sequences)
    names = shape.get("prefix_cache_keys")
    shared = shape.get("prefix_cache_mode") == "shared"
    members = []
    for i, (query, context) in enumerate(sequences):
        extent = min(context, max(0, int(extents[i]))) if query and shared else 0
        key = (str(names[i] or "") if names and i < len(names) else
               "" if names else "__implicit__")
        members.append((key, extent))
    counts = Counter(key for key, extent in members if key and extent)
    return tuple((key, extent if counts[key] > 1 else 0) for key, extent in members)


def sparse_prefix_fill(*, rows: int, requests: int, selected: int,
                       prefix_tokens: int, context: int) -> SparsePrefixFill:
    """Bound the first-touch external rows of a sparse shared-prefix batch."""
    if min(rows, requests, selected) < 0 or context <= 0 or prefix_tokens < 0:
        raise ModelingError("sparse_prefix_fill_requires_declared_geometry")
    if requests > rows:
        raise ModelingError("sparse_prefix_fill_requests_exceed_query_rows")
    shared = min(prefix_tokens, context)
    shared_selected = selected_prefix_count(selected, shared, context, context)
    shared_unique = nested_prefix_union(((shared, requests * shared_selected),))
    private_rows = requests * (selected - shared_selected)
    return SparsePrefixFill(shared_selected, shared_unique, private_rows,
                            shared_unique + private_rows)


def _matches_role(key: str, roles: tuple[str, ...], mapped: set[str]) -> bool:
    return key in mapped or any(key == role or key.startswith(role + "@") for role in roles)


def _scale_work(work: dict[str, float], factor: float) -> dict[str, float]:
    return {k: v * factor for k, v in work.items() if v * factor > 0.0}


@dataclass(frozen=True)
class BatchPrefixAttention:
    """Shared key/value tile traversal with causal QK -> softmax -> PV order."""

    algorithm = RECIPE_NAME
    key_tiles: int | None = None

    def lower(self, context):
        from mindie_sim.prediction.program import KernelProgram, PipelineProgram, Stage

        metadata = dict(getattr(context.op, "metadata", ()) or {})
        if metadata.get("prefix_read_reuse_applied") is not True:
            raise ModelingError("batch_prefix_attention_requires_applied_read_plan")
        if not metadata.get("selected_row_identity") and metadata.get("sparse_selection_unknown"):
            raise ModelingError("batch_prefix_attention_sparse_rows_unknown")
        names = dict(getattr(context, "resource_names", {}) or {})
        mat_set = {names.get(r, r) for r in MATRIX_ROLES}
        vec_set = {names.get(r, r) for r in VECTOR_ROLES}
        store_set = {names.get(r, r) for r in STORE_ROLES}
        matrix_work, vector_work, store_work, fill_work = {}, {}, {}, {}
        for name, val in context.work.items():
            if not val:
                continue
            fval = float(val)
            if _matches_role(name, MATRIX_ROLES, mat_set):
                matrix_work[name] = fval
            elif _matches_role(name, VECTOR_ROLES, vec_set):
                vector_work[name] = fval
            elif _matches_role(name, STORE_ROLES, store_set):
                store_work[name] = fval
            else:
                fill_work[name] = fval
        out_b = max(0.0, float(getattr(context.op, "mte3_bytes", 0.0) or 0.0))
        hbm_total = max(0.0, float(getattr(context.op, "hbm_bytes", 0.0) or 0.0))
        kv_hbm_b = float(metadata.get("prefix_read_hbm_bytes", 0.0) or 0.0)
        q_in_b = (
            max(0.0, hbm_total - kv_hbm_b - out_b)
            if kv_hbm_b > 0.0 and hbm_total >= kv_hbm_b + out_b
            else out_b
        )
        hbm_set = {names.get("hbm", "hbm")}
        q_load_work: dict[str, float] = {}
        out_store_work = dict(store_work)
        key_fill_work: dict[str, float] = {}
        for name, val in fill_work.items():
            is_hbm = name in hbm_set or name == "hbm" or name.startswith("memory@")
            if is_hbm and val > (q_in_b + out_b) and (q_in_b + out_b) > 0.0:
                if q_in_b > 0.0:
                    q_load_work[name] = q_in_b
                if out_b > 0.0:
                    out_store_work[name] = out_store_work.get(name, 0.0) + out_b
                key_fill_work[name] = val - q_in_b - out_b
            elif not is_hbm and val > q_in_b and q_in_b > 0.0:
                q_load_work[name] = q_in_b
                key_fill_work[name] = val - q_in_b
            else:
                key_fill_work[name] = val
        shape = tuple(getattr(context.op, "semantic_shape", ()) or ())
        qk_share = (
            float(shape[3]) / max(1.0, float(shape[3] + shape[4]))
            if len(shape) >= 5 else 0.5
        )
        qk_work = _scale_work(matrix_work, qk_share)
        pv_work = {k: v - qk_work.get(k, 0.0) for k, v in matrix_work.items()
                   if v - qk_work.get(k, 0.0) > 0.0}
        hbm_plan_b = float(metadata.get("prefix_read_hbm_bytes", 0.0) or 0.0)
        tiles = (
            max(1, int(self.key_tiles)) if self.key_tiles is not None
            else min(MAX_KEY_TILES, max(1, math.ceil(hbm_plan_b / KEY_TILE_BYTES)))
            if hbm_plan_b > KEY_TILE_BYTES else 1
        )
        prefix_stages = (Stage("setup", latency=context.setup_ms),) if context.setup_ms else ()
        q_after = ("setup",) if context.setup_ms else ()
        has_q = bool(q_load_work)
        lead_stages = prefix_stages + (
            (Stage("query_load", q_load_work, after=q_after),) if has_q else ()
        )
        tile_dep = ("query_load",) if has_q else q_after
        if tiles <= 1:
            body = (
                Stage("key_tile_fill", key_fill_work, after=tile_dep),
                Stage("qk", qk_work, after=("key_tile_fill",)),
                Stage("softmax", vector_work, after=("qk",)),
                Stage("pv", pv_work, after=("softmax",)),
                Stage("output_store", out_store_work, after=("pv",)),
            )
            return KernelProgram(lead_stages + body, minimum_ms=context.minimum_ms)
        inv = 1 / float(tiles)
        t_fill = _scale_work(key_fill_work, inv)
        t_qk = _scale_work(qk_work, inv)
        t_soft = _scale_work(vector_work, inv)
        t_pv = _scale_work(pv_work, inv)

        def make_tile(_idx: int):
            return (
                Stage("key_tile_fill", t_fill),
                Stage("qk", t_qk, after=("key_tile_fill",)),
                Stage("softmax", t_soft, after=("qk",)),
                Stage("pv", t_pv, after=("softmax",), previous=("pv",)),
            )

        pipe = PipelineProgram(tiles, slots=TILE_SLOTS, tile=make_tile)
        stages = lead_stages + (
            Stage("key_tiles", after=tile_dep, program=pipe),
            Stage("output_store", out_store_work, after=("key_tiles",)),
        )
        return KernelProgram(stages, minimum_ms=context.minimum_ms)


RECIPE = BatchPrefixAttention()


def select(op: Any, runtime: Any) -> str | None:
    """Return the registered recipe name for an eligible shared-KV scan."""
    metadata = dict(getattr(op, "metadata", ()) or {})
    if getattr(runtime, "prefix_cache_mode", "private") != "shared":
        return None
    if getattr(runtime, "prefix_read_mode", "auto") == "per_sequence":
        return None
    if metadata.get("prefix_read_reuse_applied") is not True:
        return None
    if metadata.get("read_reuse_applied") is False:
        return None
    return RECIPE_NAME


def _view_query_tokens(view: Any) -> int:
    ranges = getattr(view, "query_ranges", None)
    if ranges:
        return max(1, sum(
            int(end) - int(start) for start, end in ranges if int(end) > int(start)))
    return max(1, int(getattr(view, "q_len", 1) or 1))


def apply_attention_prefix_reads(
    op: Any, *, views: Sequence[Any], sweeps_by_view: Sequence[int],
    shared_sweeps_by_view: Sequence[int], row_bytes: float, layer_index: int,
    runtime: Any, kernel_reuse: str = "tile", allow_recipe_rebind: bool = True,
    kv_hbm_bytes: float | None = None,
    kv_mte2_bytes: float | None = None,
    unique_prefix_rows_by_view: Sequence[int] | None = None,
    unique_rows_by_view: Sequence[int] | None = None,
    shared_intervals_by_view: Sequence[tuple[tuple[int, int], ...]] | None = None,
    geometry: Any = None,
) -> Any:
    """Replace the precharged KV components with their physical union."""
    if runtime is None or getattr(runtime, "prefix_cache_mode", "private") != "shared":
        return op
    prefix_caps = (
        tuple(int(value) for value in unique_prefix_rows_by_view)
        if unique_prefix_rows_by_view is not None else None
    )
    unique_caps = (
        tuple(int(value) for value in unique_rows_by_view)
        if unique_rows_by_view is not None else None
    )
    interval_caps = (
        tuple(tuple((int(lo), int(hi)) for lo, hi in item)
              for item in shared_intervals_by_view)
        if shared_intervals_by_view is not None else None
    )
    accesses = []
    for idx, (view, sweeps, shared) in enumerate(
        zip(views, sweeps_by_view, shared_sweeps_by_view)
    ):
        if int(sweeps) <= 0:
            continue
        q_tokens = _view_query_tokens(view)
        raw_context = getattr(view, "context", None)
        context_tokens = (
            max(0, int(raw_context))
            if raw_context is not None
            else max(0, int(sweeps) - q_tokens)
        )
        physical_prefix = min(int(shared), int(sweeps))
        if prefix_caps is not None:
            physical_prefix = min(physical_prefix, max(0, prefix_caps[idx]))
        elif getattr(view, "prefix_tokens", None) is not None:
            physical_prefix = min(physical_prefix, max(0, int(view.prefix_tokens)))
        accesses.append(CacheAccess(
            request_id=str(view.request_id),
            context_tokens=context_tokens,
            q_len=q_tokens,
            prefix_tokens=max(0, physical_prefix),
            prefix_key=canonical_prefix_cache_key(
                "shared", getattr(view, "prefix_cache_key", "")),
            row_bytes=float(row_bytes),
            layer=int(layer_index),
            component="kv",
            logical_rows=int(sweeps),
            unique_rows=0 if unique_caps is None else max(0, unique_caps[idx]),
            window_tokens=0,
            shared_intervals=() if interval_caps is None else interval_caps[idx],
        ))
    read_mode = str(getattr(runtime, "prefix_read_mode", "auto") or "auto")
    plan = plan_prefix_reads(
        accesses, mode=read_mode, kernel_reuse=kernel_reuse, geometry=geometry
    )
    updated = apply_prefix_read_plan(
        op, plan, hbm_component=kv_hbm_bytes, mte2_component=kv_mte2_bytes)
    if allow_recipe_rebind and select(updated, runtime) == RECIPE_NAME:
        updated = replace(updated, recipe=RECIPE_NAME)
    return updated


def annotate_sparse_prefix_unknown(op: Any, runtime: Any) -> Any:
    """Record why dynamic sparse TopK skips key-read deduplication without row IDs."""
    if runtime is None or getattr(runtime, "prefix_cache_mode", "private") != "shared":
        return op
    metadata = dict(getattr(op, "metadata", ()) or ())
    metadata.update({
        "sparse_selection_unknown": True,
        "prefix_read_reuse_applied": False,
        "prefix_read_reason": "sparse_selected_rows_unknown",
    })
    return replace(
        op, metadata=tuple(sorted(metadata.items(), key=lambda pair: pair[0])))


def apply_sparse_prefix_reads(op, *, views, row_bytes, runtime, hierarchy,
                              kv_hbm_bytes, kv_mte2_bytes):
    """Bound unknown selections by their physical shared/private KV union."""
    runtime_meta = dict(getattr(runtime, "metadata", ()) or ())
    raw_eager = bool(runtime_meta.get("raw_eager")
                     or runtime_meta.get("execution_policy", {}).get("raw_eager"))
    if (getattr(runtime, "prefix_cache_mode", "private") != "shared"
            or getattr(runtime, "prefix_read_mode", "auto") == "per_sequence"
            or raw_eager):
        return annotate_sparse_prefix_unknown(op, runtime)
    shared, private = {}, 0
    for view in views:
        if view.q_len <= 0 or view.padding:
            continue
        prefix = min(max(0, int(view.prefix_tokens)), int(view.context))
        key = canonical_prefix_cache_key("shared", view.prefix_cache_key)
        shared[key] = max(shared.get(key, 0), prefix)
        private += max(0, int(view.kv_len) - prefix)
    # With no row IDs the complete candidate plane is a conservative union
    # bound. Never assume each query selects the same TopK rows.
    working_set = min(float(kv_mte2_bytes),
                      (sum(shared.values()) + private) * row_bytes)
    retained = min(working_set, float(hierarchy.effective_l2_bytes))
    spill = (max(0.0, 1.0 - retained / working_set)
             if working_set else 0.0)
    external = min(float(kv_hbm_bytes), working_set
                   + max(0.0, kv_mte2_bytes - working_set) * spill)
    metadata = dict(op.metadata)
    metadata.update(
        sparse_selection_unknown=True,
        prefix_read_reuse_applied=external < kv_hbm_bytes,
        prefix_read_reason="shared_sparse_candidate_working_set_with_l2_spill",
        prefix_read_hbm_bytes=external,
        prefix_read_l2_core_bytes=float(kv_mte2_bytes),
        prefix_cache_reads={"working_set_bytes": working_set,
                            "l2_capacity_bytes": float(hierarchy.effective_l2_bytes),
                            "spill_fraction": spill})
    traffic = dict(metadata.get("attention_traffic_plan", {}))
    if traffic:
        traffic.update(hbm_read_bytes=external,
                       l2_hit_bytes=max(0.0, kv_mte2_bytes - external),
                       working_set_bytes=working_set,
                       residency_basis=metadata["prefix_read_reason"])
        metadata["attention_traffic_plan"] = traffic
    return replace(op, hbm_bytes=op.hbm_bytes - kv_hbm_bytes + external,
                   metadata=tuple(metadata.items()))
