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


ZERO = 0
ONE = 1
TWO = 2
THREE = 3
FOUR = 4
FIVE = 5
FIRST = 0
ZERO_FLOAT = 0.0
ONE_FLOAT = 1.0
HALF = 0.5
TILE_SLOTS = 2
KEY_TILE_BYTES = 262144.0
MAX_KEY_TILES = 32
RECIPE_NAME = "batch_prefix_attention"
MATRIX_ROLES = ("cube", "matrix", "mte1", "fixpipe")
STORE_ROLES = ("mte3",)
VECTOR_ROLES = ("vector",)


def _matches_role(key: str, roles: tuple[str, ...], mapped: set[str]) -> bool:
    return key in mapped or any(key == role or key.startswith(role + "@") for role in roles)


def _scale_work(work: dict[str, float], factor: float) -> dict[str, float]:
    return {k: v * factor for k, v in work.items() if v * factor > ZERO_FLOAT}


@dataclass(frozen=True)
class BatchPrefixAttention:
    """Shared key/value tile traversal with causal QK -> softmax -> PV order."""

    algorithm = RECIPE_NAME
    key_tiles: int | None = None

    def lower(self, context):
        from .program import KernelProgram, PipelineProgram, Stage

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
        q_out_b = max(ZERO_FLOAT, float(getattr(context.op, "mte3_bytes", ZERO_FLOAT) or ZERO_FLOAT))
        q_load_work: dict[str, float] = {}
        out_store_work = dict(store_work)
        key_fill_work: dict[str, float] = {}
        for name, val in fill_work.items():
            if val > TWO * q_out_b and q_out_b > ZERO_FLOAT:
                q_load_work[name] = q_out_b
                out_store_work[name] = out_store_work.get(name, ZERO_FLOAT) + q_out_b
                key_fill_work[name] = val - TWO * q_out_b
            else:
                key_fill_work[name] = val
        shape = tuple(getattr(context.op, "semantic_shape", ()) or ())
        qk_share = (
            float(shape[THREE]) / max(ONE_FLOAT, float(shape[THREE] + shape[FOUR]))
            if len(shape) >= FIVE else HALF
        )
        qk_work = _scale_work(matrix_work, qk_share)
        pv_work = {k: v - qk_work.get(k, ZERO_FLOAT) for k, v in matrix_work.items()
                   if v - qk_work.get(k, ZERO_FLOAT) > ZERO_FLOAT}
        hbm_plan_b = float(metadata.get("prefix_read_hbm_bytes", ZERO_FLOAT) or ZERO_FLOAT)
        tiles = (
            max(ONE, int(self.key_tiles)) if self.key_tiles is not None
            else min(MAX_KEY_TILES, max(ONE, math.ceil(hbm_plan_b / KEY_TILE_BYTES)))
            if hbm_plan_b > KEY_TILE_BYTES else ONE
        )
        prefix_stages = (Stage("setup", latency=context.setup_ms),) if context.setup_ms else ()
        q_after = ("setup",) if context.setup_ms else ()
        has_q = bool(q_load_work)
        lead_stages = prefix_stages + (
            (Stage("query_load", q_load_work, after=q_after),) if has_q else ()
        )
        tile_dep = ("query_load",) if has_q else q_after
        if tiles <= ONE:
            body = (
                Stage("key_tile_fill", key_fill_work, after=tile_dep),
                Stage("qk", qk_work, after=("key_tile_fill",)),
                Stage("softmax", vector_work, after=("qk",)),
                Stage("pv", pv_work, after=("softmax",)),
                Stage("output_store", out_store_work, after=("pv",)),
            )
            return KernelProgram(lead_stages + body, minimum_ms=context.minimum_ms)
        inv = ONE / float(tiles)
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


def apply_attention_prefix_reads(
    op: Any, *, views: Sequence[Any], sweeps_by_view: Sequence[int],
    shared_sweeps_by_view: Sequence[int], row_bytes: float, layer_index: int,
    runtime: Any, kernel_reuse: str = "tile", allow_recipe_rebind: bool = True,
    kv_hbm_bytes: float | None = None,
    kv_mte2_bytes: float | None = None,
) -> Any:
    """Replace the precharged KV components with their physical union."""
    if runtime is None or getattr(runtime, "prefix_cache_mode", "private") != "shared":
        return op
    accesses = [
        CacheAccess(
            request_id=str(view.request_id),
            context_tokens=max(ZERO, int(sweeps) - ONE),
            q_len=ONE,
            prefix_tokens=max(ZERO, min(int(shared), int(sweeps))),
            prefix_key=canonical_prefix_cache_key(
                "shared", getattr(view, "prefix_cache_key", "")),
            row_bytes=float(row_bytes),
            layer=int(layer_index),
            component="kv",
            logical_rows=int(sweeps),
        )
        for view, sweeps, shared in zip(views, sweeps_by_view, shared_sweeps_by_view)
        if int(sweeps) > ZERO
    ]
    read_mode = str(getattr(runtime, "prefix_read_mode", "auto") or "auto")
    plan = plan_prefix_reads(accesses, mode=read_mode, kernel_reuse=kernel_reuse)
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
        op, metadata=tuple(sorted(metadata.items(), key=lambda pair: pair[FIRST])))
