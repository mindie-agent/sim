"""Staged analytic GQA prefill on the common finite resource engine.

[Role] Lower causal prefill into owner-local query tiles and bounded KV streams.
[Invariants] A task retains its query tile and online max, sum and numerator.
  QK and PV price the actual KV rectangle through the causal diagonal.
  Logical pair counts remain a separate geometry ledger. Resident buffers
  use declared pools and each task writes its output once.
[Timeouts & Locks] Tasks use lazy owner strides and finite key-tile windows.
[Side-Effect Policy] Plans submit only to the caller's virtual engine.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from functools import lru_cache
import math
from typing import Mapping

from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.mechanism import (
    Geometry, OperatorPlan, PlannedRecipe, Work,
)
from mindie_sim.prediction.ownership import read_role, unit_group
from .attention_kv_access import KVCacheAccess, kv_access
from .attention_prefill_geometry import (
    IMPLEMENTATION, NATIVE_S2_PRELOAD, _DTYPE_ALIASES, _FP32_BYTES,
    _PREFILL_PHASES, _SOURCE_LARGE_HEAD_THRESHOLD, _SOURCE_QUERY_TILE_DEFAULT,
    _SOURCE_QUERY_TILE_LARGE_HEAD, _Spec, _Task, _ceil_div, _dtype,
    _explicit_reconstruct, _iter_tasks, _native_tasks, _nonnegative_int,
    _positive_int, _report, _task_at, _task_at_for, _task_pairs, _task_work,
    _time_major_tasks, _total_work, logical_causal_pairs, source_query_tile,
)
from mindie_sim.prediction.attention_prefill_issue import (
    source_c2_path, source_compatible_alternatives, source_issue_kind,
    source_issue_operand,
)
from mindie_sim.prediction.attention_prefill_nofd import _PrefillProgram, owner_s2_plan
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, RecipeContext, Stage
from mindie_sim.prediction.refill_steady import compact_periodic_pipeline
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

RECIPE_ID = IMPLEMENTATION

# Mechanism references pinned at ops-transformer HEAD.  They support the
# declared head-local policy; they are not evidence of a captured dispatch.
SOURCE_REFERENCES = (
    ("cann/ops-transformer", "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
     "attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp",
     "748-864,1095-1178",
     "Q heads and outer query blocks are independent tiling axes"),
    ("cann/ops-transformer", "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
     "attention/fused_infer_attention_score/op_host/arch22/"
     "fused_infer_attention_score_tiling_v3.cpp", "1403-1408",
     "arch22 RouteToFia rejects TND without rope for headDim 256"),
    ("cann/ops-transformer", "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
     "attention/fused_infer_attention_score/op_host/flash_attention_infer_tiling.h",
     "220-253,373-399",
     "FAInfer also uses one query head per task when query length exceeds 128"),
    ("cann/ops-transformer", "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
     "attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp",
     "6415-6443",
     "BF16 CVDiff starts Souter at 64 for aligned head size >=200 and at 128 "
     "for head size >=32; the query-tile panel is selected by head dimension"),
    ("cann/ops-transformer", "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
     "attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp",
     "161-167,2886-2894",
     "PP_BLOCK_BUFFER_SIZE=128*128 bounds the resident query/key panel; "
     "mUbd=floor(PP_BLOCK_BUFFER_SIZE/embeddingSizeAligned/128)*128"),
)


def source_softmax_outer(head_dim: int) -> int:
    """Arch35 ``softmaxSOuterFactor`` for the selected V4 FIA branch.

    ``FusedInferAttentionScoreTilingImpl::AdjustSinnerAndSouter`` starts at
    ``SOUTER_64`` and sets ``softmaxSOuterFactor=SOUTER_32`` only when
    ``vHeadDim > 128`` (and the prefill S1 is not 1).  This is the vector
    chunk, not the cube S1 basic block: ``sOuterSize = SOuterFactor*CV_RATIO``
    is 128 for the default V4 branch.
    """
    dim = _positive_int(head_dim, "head_dim")
    return 32 if dim > 128 else 64


def arch35_query_tile(head_dim: int) -> int:
    """Arch35 V4 cube S1 basic block (``SOuterFactor*CV_RATIO``).

    ``AdjustSinnerAndSouter`` defaults SOuter to 64; ``CV_RATIO`` is 2, so
    one cube task owns 128 query rows.  D256 non-quant GQA is excluded from
    the registered templates and uses this V4 branch; D128 uses the same
    128-row cube block in the fallback branch.
    """
    _positive_int(head_dim, "head_dim")
    return 128


@dataclass(frozen=True)
class StagedGQAPrefill:
    """Explicit prefill tiling policy.

    ``query_tile`` bounds query rows per task; ``None`` selects the native
    CVDiff start from the declared head dimension.  ``key_tile`` bounds the
    actual K/V rows in one key-tile step.  ``slots`` is the finite K/V pipeline
    window.  Explicit integers remain available for counterfactual schedules.
    """

    query_tile: int | None = None
    key_tile: int = 128
    # Generic K/V double buffer for non-native callers.  Native FIA with no
    # explicit reconstruct flag keeps preload 3.  That is not a template pick.
    slots: int = 2
    # SOFTMAX_BUFFER_NUM=3 is scratch capacity for softmax Exp/BMM2, not
    # concurrent tile depth.
    softmax_slots: int = 3

    def __post_init__(self):
        if self.query_tile is not None:
            _positive_int(self.query_tile, "query_tile")
        _positive_int(self.key_tile, "key_tile")
        _positive_int(self.slots, "slots")
        _positive_int(self.softmax_slots, "softmax_slots")

    @classmethod
    def arch35_noquant_candidate(cls, **overrides):
        """Explicit counterfactual with the source S2 preload already used natively."""
        options = {"slots": 3, "query_tile": 128, "key_tile": 128}
        options.update(overrides)
        return cls(**options)

    def spec(self, op: OpSpec) -> "_Spec":
        return _spec(op, self)

    def work(self, op: OpSpec, *, include_mte1: bool = False,
             read: str = "mte2", vread: str | None = None,
             write: str | None = "mte3", operand: str | None = "mte1") -> Work:
        """Physical work, including the executed causal rectangle."""
        return _total_work(self.spec(op), include_mte1=include_mte1,
                           read=read, vread=vread, write=write, operand=operand)

    def report(self, op: OpSpec, *, resource_roles=None,
               cores: int = 32) -> dict:
        """Geometry ledger separating logical pairs from executed rectangles.

        ``cores`` enables the source S2 ownership/FD ledger.  FD bytes are a
        workspace diagnostic until the split-range execution is bound; they
        are not silently added to the cyclic task work.
        """
        spec = self.spec(op)
        report = _report(spec)
        if spec.native_fia:
            gqa = dict(op.metadata).get("gqa_prefill")
            fields = gqa if isinstance(gqa, Mapping) else {}
            report["source_compatible"] = source_compatible_alternatives(
                spec.head_dim,
                tuple(query for query, _context in spec.requests),
                query_dtype=spec.query_dtype,
                kv_dtype=spec.kv_dtype,
                out_dtype=fields.get("out_dtype", spec.query_dtype),
                layout=fields.get("input_layout", "TND"),
                quant=fields.get("quant", "NoQuant"),
                mask=fields.get("mask", True),
                rope=fields.get("rope", False),
                kv_paged=fields.get("kv_paged", False),
                kv_prefix=fields.get("kv_prefix", False),
            )
        plan = owner_s2_plan(spec, cores)
        rows = min(spec.query_tile, max((q for q, _ in spec.requests), default=0))
        partial = plan["fd_splits"] * rows * spec.head_dim * _FP32_BYTES
        lse = plan["fd_splits"] * rows * 2 * _FP32_BYTES
        report.update({
            "owner_s2_loads": plan["loads"],
            "owner_s2_total": plan["total"],
            "fd_splits": plan["fd_splits"],
            "fd_fragments": plan["fd_fragments"],
            "fd_partial_write_bytes": partial,
            "fd_partial_read_bytes": partial,
            "fd_lse_bytes": lse,
            "fd_reduce_ops": plan["fd_splits"] * rows * (spec.head_dim + 2),
            "fd_service_basis": "source_owner_split_diagnostic",
        })
        if resource_roles is not None:
            roles = set(resource_roles)
            report["unmodeled_transfer_roles"] = [
                role for role, present in (
                    ("matrix_input_copy", bool(roles & {"mte2", "mte2_aic"})),
                    ("matrix_operand_copy", "mte1" in roles),
                    ("output_store", "mte3" in roles),
                ) if not present]
        return report

    def owner_plan(self, op: OpSpec, cores: int = 32) -> dict:
        """Source S2-tile ownership and FD split count for this geometry."""
        return owner_s2_plan(self.spec(op), cores)

    def build(self, op: OpSpec, geometry: Geometry | None = None) -> OperatorPlan:
        spec = self.spec(op)
        plan = None
        cores = 32
        if geometry is None:
            read, vread, write, operand = "mte2", "mte2", "mte3", "mte1"
            mte1 = False
        else:
            read = read_role(geometry.resource_units)
            vread = read_role(geometry.resource_units, aiv=0)
            write = "mte3" if "mte3" in geometry.resource_units else None
            operand = "mte1" if "mte1" in geometry.resource_units else None
            mte1 = operand is not None
            cores = max(1, int(geometry.compute_units))
            if spec.kv_cache.needs_conversion and operand is None:
                raise ModelingError(
                    "attention_prefill_conversion_requires_mte1_operand_copy")
            if spec.kv_cache.needs_conversion and write is None:
                raise ModelingError(
                    "attention_prefill_conversion_requires_mte3_result_copy")
        if spec.native_fia and geometry is not None:
            from .attention_prefill_nofd import owner_s2_plan
            plan = owner_s2_plan(spec, cores)
        work = _total_work(
            spec, include_mte1=mte1, read=read, vread=vread,
            write=write, operand=operand, plan=plan)
        metadata = {"kv_cache_access": spec.kv_cache.report()}
        if plan is not None:
            metadata["prefill_plan"] = {
                "source": "owner_s2_plan",
                "cores": cores,
                "total": plan["total"],
                "loads": tuple(plan["loads"]),
                "fd_splits": plan["fd_splits"],
                "fd_fragments": 2 * plan["fd_fragments"],
                "split_tasks": plan["fd_fragments"],
                "issue_kind": spec.issue_kind,
            }
        return OperatorPlan(
            work, lambda context: _PrefillProgram(context, self, spec),
            metadata=metadata)

    def lower(self, context: RecipeContext) -> KernelProgram:
        spec = self.spec(context.op)
        return KernelProgram(
            (
                Stage("setup", latency=context.setup_ms),
                Stage("body", after=("setup",),
                      program=_PrefillProgram(context, self, spec)),
            ),
            minimum_ms=context.minimum_ms,
        )


def _spec(op: OpSpec, recipe: StagedGQAPrefill) -> _Spec:
    if getattr(op, "phase", None) not in _PREFILL_PHASES:
        raise ModelingError("attention_prefill_only_supports_prefill_phases")
    metadata = dict(getattr(op, "metadata", ()))
    raw = metadata.get("gqa_prefill")
    if not isinstance(raw, Mapping):
        raise ModelingError("attention_prefill_gqa_prefill_metadata_required")
    raw_requests = raw.get("requests")
    if not isinstance(raw_requests, (list, tuple)):
        raise ModelingError("attention_prefill_requests_must_be_sequence")
    requests = []
    for item in raw_requests:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ModelingError("attention_prefill_request_must_be_query_context_pair")
        requests.append((_nonnegative_int(item[0], "query_len"),
                         _nonnegative_int(item[1], "context")))
    requests = tuple(requests)
    heads = _positive_int(raw.get("query_heads"), "query_heads")
    kv_heads = _positive_int(raw.get("kv_heads"), "kv_heads")
    dim = _positive_int(raw.get("head_dim"), "head_dim")
    if heads % kv_heads:
        raise ModelingError("attention_prefill_query_heads_not_divisible_by_kv_heads")
    query_dtype = _dtype(raw.get("query_dtype"), "query_dtype")
    kv_dtype = _dtype(raw.get("kv_dtype"), "kv_dtype")
    if raw.get("causal") is not True:
        raise ModelingError("attention_prefill_causal_true_required")
    if raw.get("window", 0) != 0:
        raise ModelingError("attention_prefill_windowed_prefill_not_selected")
    if raw.get("cp", 1) != 1:
        raise ModelingError("attention_prefill_context_parallel_not_selected")
    # The ordinary FIA descriptor binds the actual selected primitive
    # (FusedInferAttentionScore).  Generic GQA prefill callers retain the
    # CVDiff query-tile rule; the selected A5 FIA V4 branch uses the arch35
    # cube S1 basic block (SOuter64*CV_RATIO).
    native_fia = any(
        str(key) == "native_operator" and str(value) == "FusedInferAttentionScore"
        for key, value in metadata.items())
    query_tile = (recipe.query_tile if recipe.query_tile is not None
                  else (arch35_query_tile(dim) if native_fia
                        else source_query_tile(dim)))

    ends, tile_counts, total = [], [], 0
    for query, _context in requests:
        tiles = _ceil_div(query, query_tile)
        tile_counts.append(tiles)
        total += heads * tiles
        ends.append(total)
    softmax_outer = (source_softmax_outer(dim) if native_fia else 8)
    # Preload 3 stays on the unselected path.  issue_kind is set only from an
    # explicit reconstruct flag and is never inferred from the head dimension.
    s2_preload = NATIVE_S2_PRELOAD if native_fia else recipe.slots
    reconstruct = _explicit_reconstruct(metadata)
    issue_kind = source_issue_kind(reconstruct, dim) if native_fia else None
    return _Spec(requests, heads, kv_heads, dim, query_dtype, kv_dtype,
                 query_tile, recipe.key_tile, recipe.slots, s2_preload,
                 recipe.softmax_slots, softmax_outer, native_fia,
                 reconstruct, issue_kind, tuple(ends), tuple(tile_counts),
                 kv_access(op, query_dtype=query_dtype, kv_dtype=kv_dtype,
                           head_dim=dim))


def build(op: OpSpec, geometry: Geometry | None = None) -> OperatorPlan:
    """Independent verification binding for the default recipe options."""
    return StagedGQAPrefill().build(op, geometry)


# ---------------------------------------------------------------------------
# Placement, buffers and executable programs.
# ---------------------------------------------------------------------------


def _placement(context: RecipeContext, core: int, *roles: str) -> dict:
    present = tuple(role for role in roles if role in context.resource_names)
    return context.owned(core, context.core_count, present) if present else {}


def _read_role(context: RecipeContext, *, vector: bool = False) -> str:
    return read_role(context.resource_names, aiv=0 if vector else None)


def _write_role(context: RecipeContext) -> str | None:
    return "mte3" if "mte3" in context.resource_names else None


def _operand_role(context: RecipeContext) -> str | None:
    return "mte1" if "mte1" in context.resource_names else None


def _allocation(context: RecipeContext, core: int, pool: str, name: str,
                size: float) -> Allocation | None:
    if pool not in context.pool_names or size <= 0:
        return None
    capacity = context.pool_capacities.get(pool)
    if capacity is not None and size > capacity:
        raise ModelingError(
            f"attention_prefill_buffer_exceeds_pool:{pool}:{name}:{size}>{capacity}")
    units = unit_group(core, context.core_count,
                       context.pool_units.get(pool, 1))
    return Allocation(name, context.pool_names[pool], float(size), units)


def _allocations(items) -> tuple[Allocation, ...]:
    return tuple(item for item in items if item is not None)


def _task_program(context: RecipeContext, recipe: StagedGQAPrefill,
                  spec: _Spec, task: _Task, core: int,
                  *, partial: bool = False) -> KernelProgram:
    dim, rows = spec.head_dim, task.rows
    q_bytes = rows * dim * spec.query_bytes
    read = _read_role(context)
    write = _write_role(context)
    operand = _operand_role(context)
    cache = spec.kv_cache
    if cache.needs_conversion and not operand:
        raise ModelingError(
            "attention_prefill_conversion_requires_mte1_operand_copy")
    copy_in = (read,)
    vector_copy_in = (_read_role(context, vector=True),)
    copy_out = (write,) if write else ()
    operand_roles = (operand,) if operand else ()
    cube, vector = ("cube",), ("vector",)

    q_l1 = _allocation(context, core, "l1", "q_l1", q_bytes)
    # Arch35 streams Q from L1 through the two shared 32KiB L0A microtile
    # buffers; full Q is not resident in L0A.
    q_l0a = _allocation(context, core, "l0a", "q_l0a", min(q_bytes, 32 * 1024))
    ub_state = _allocation(context, core, "ub", "online_state",
                           rows * (dim * _FP32_BYTES + 2 * _FP32_BYTES))
    persistent = _allocations((q_l1, q_l0a, ub_state))
    # Q remains resident while PV consumes a probability operand.  A larger
    # declared tile cannot make progress if these two mandatory buffers do not
    # fit together, even when each allocation individually fits.
    outer = min(rows, spec.softmax_outer)
    softmax_window = min(recipe.softmax_slots, task.key_tiles)
    l0a_required = 64 * 1024  # two shared 32 KiB arch35 L0A microtile buffers
    preload = spec.s2_preload
    ub_required = rows * (dim * _FP32_BYTES + 2 * _FP32_BYTES) \
        + softmax_window * outer * min(recipe.key_tile, task.horizon) * _FP32_BYTES
    max_rows = min(recipe.key_tile, task.horizon)
    if cache.needs_conversion:
        # conversion needs packed input + converted output + scales in UB,
        # and the converted operand in L1 before the MTE1 L1->L0B copy.
        slots = max(1, min(recipe.slots, spec.s2_preload, task.key_tiles))
        per_slot = max_rows * (
            cache.key_payload_row + cache.value_payload_row
            + cache.key_scale_row + cache.value_scale_row)
        ub_conv = max(
            max_rows * (cache.key_payload_row + cache.key_operand_row
                        + cache.key_scale_row),
            max_rows * (cache.value_payload_row + cache.value_operand_row
                        + cache.value_scale_row))
        ub_required = (rows * (dim * _FP32_BYTES + 2 * _FP32_BYTES)
                       + softmax_window * outer * max_rows * _FP32_BYTES
                       + slots * per_slot + ub_conv)
        operand_row = max(cache.key_operand_row, cache.value_operand_row)
        l1_required = q_bytes + slots * max_rows * operand_row
        l0b_required = slots * max_rows * operand_row
    else:
        l1_required = q_bytes + max_rows * (
            cache.key_payload_row + cache.value_payload_row)
        l0b_required = 0
    required_pools = [("l0a", l0a_required), ("ub", ub_required),
                      ("l1", l1_required)]
    if cache.needs_conversion:
        required_pools.append(("l0b", l0b_required))
    for pool, required in required_pools:
        capacity = context.pool_capacities.get(pool)
        if pool in context.pool_names and capacity is not None and required > capacity:
            raise ModelingError(
                f"attention_prefill_resident_working_set_exceeds_pool:{pool}:{required}>{capacity}")
    q_work = {"hbm": q_bytes, read: q_bytes}
    stages = [
        Stage("q_load", context.bind(Work(resources=q_work)),
              acquire=persistent,
              placement={**_placement(context, core, *copy_in),
                         **_placement(context, core, *operand_roles)}),
    ]
    page_meta = page_scale = 0
    page_scale_ub = None
    if cache.paged:
        page_start = task.key_start * recipe.key_tile
        page_end = min((task.key_start + task.key_tiles) * recipe.key_tile,
                       task.horizon)
        pages = cache.pages(page_start, page_end)
        page_meta = pages * cache.page_meta_bytes
        if cache.scale_scope == "page":
            page_scale = pages * cache.page_scale_bytes
    if page_meta or page_scale:
        page_work = {"hbm": page_meta + page_scale}
        if page_meta:
            page_work[read] = page_work.get(read, 0) + page_meta
        if page_scale:
            vector_read = _read_role(context, vector=True)
            page_work[vector_read] = page_work.get(vector_read, 0) + page_scale
            page_scale_ub = _allocation(
                context, core, "ub", "page_scale_ub", page_scale)
        stages.append(Stage(
            "page_access", context.bind(Work(resources=page_work)),
            acquire=_allocations((page_scale_ub,)), after=("q_load",),
            placement={**_placement(context, core, *copy_in),
                       **_placement(context, core, *vector_copy_in)}))
    key_ready = "page_access" if (page_meta or page_scale) else "q_load"
    max_tile = min(recipe.key_tile, max(1, task.horizon))

    def tile(index: int):
        start = (task.key_start + index) * recipe.key_tile
        valid = min(recipe.key_tile, task.horizon - start)
        return tile_template(valid, index % preload)

    # All full key panels of an owner use the same work and buffer slots.
    # Only the final partial panel differs. Reuse bound stage descriptions;
    # PipelineProgram still creates independent events and lifetimes per panel.
    @lru_cache(maxsize=preload + 1)
    def tile_template(valid: int, slot: int):
        rect = rows * valid
        key_needs = cache.key_needs_conversion
        value_needs = cache.value_needs_conversion
        key_payload = valid * cache.key_payload_row
        value_payload = valid * cache.value_payload_row
        key_operand = valid * cache.key_operand_row
        value_operand = valid * cache.value_operand_row
        key_scale = (valid * cache.key_scale_row
                     if cache.scale_scope == "row" else 0)
        value_scale = (valid * cache.value_scale_row
                       if cache.scale_scope == "row" else 0)
        suffix = f"s{slot}"
        k_l1 = (_allocation(context, core, "l1", f"k_l1_{suffix}", key_payload)
                if not key_needs else None)
        k_l0b = (_allocation(
            context, core, "l0b", f"k_l0b_{suffix}",
            min(key_payload, 32 * 1024)) if not key_needs else None)
        # CVDiff softmax/PV cuts the probability panel to
        # softmaxSOuterFactor rows; it does not materialize the full Souter
        # row set in L0A/UB at once.
        scores = _allocation(context, core, "l0c", f"scores_{suffix}",
                             outer * max_tile * _FP32_BYTES)
        probs = _allocation(context, core, "ub", f"probs_{suffix}",
                            outer * max_tile * _FP32_BYTES)
        p_l0a = _allocation(context, core, "l0a", f"p_l0a_{suffix}",
                           min(outer * valid * spec.query_bytes, 32 * 1024))
        v_l1 = (_allocation(context, core, "l1", f"v_l1_{suffix}", value_payload)
                if not value_needs else None)
        v_l0b = (_allocation(
            context, core, "l0b", f"v_l0b_{suffix}",
            min(value_payload, 32 * 1024)) if not value_needs else None)
        pv_acc = _allocation(context, core, "l0c", f"pv_acc_{suffix}",
                             rows * dim * _FP32_BYTES)
        score_acquire = _allocations((scores,))
        prob_acquire = _allocations((probs,))
        pv_acquire = _allocations((pv_acc,))
        score_release = _allocations((scores,))
        prob_release = _allocations((probs,))

        qk = Work(matrix={spec.query_dtype: 2 * dim * rect},
                  resources=({"mte1": q_bytes} if operand else {}))
        softmax = Work(resources={"vector": 7 * rect + rows})
        pv = Work(matrix={spec.query_dtype: 2 * dim * rect})
        online = Work(resources={"vector": rows * (3 * dim + 9)})
        dequant = cache.dequant_ops_per_element
        emitted = []

        # K: packed payload -> UB, vector conversion -> UB, MTE3 -> L1,
        # MTE1 -> L0B, QK.  Non-converting sides keep the direct path.
        key_needs = cache.key_needs_conversion
        value_needs = cache.value_needs_conversion
        if key_needs:
            k_payload_ub = _allocation(
                context, core, "ub", f"k_payload_{suffix}", key_payload)
            k_conv_ub = _allocation(
                context, core, "ub", f"k_conv_{suffix}", key_operand)
            k_post_l1 = _allocation(
                context, core, "l1", f"k_post_{suffix}", key_operand)
            k_l0b = _allocation(
                context, core, "l0b", f"k_l0b_{suffix}", key_operand)
            k_scale_ub = (_allocation(
                context, core, "ub", f"k_scale_{suffix}", key_scale)
                if key_scale else None)
            emitted.append(Stage(
                "k_load", context.bind(Work(resources={
                    "hbm": key_payload,
                    _read_role(context, vector=True): key_payload})),
                acquire=_allocations((k_payload_ub,)),
                placement=_placement(context, core, *vector_copy_in)))
            after = ("k_load",)
            if key_scale:
                emitted.append(Stage(
                    "k_scale_load", context.bind(Work(resources={
                        "hbm": key_scale,
                        _read_role(context, vector=True): key_scale})),
                    after=("k_load",), acquire=_allocations((k_scale_ub,)),
                    placement=_placement(context, core, *vector_copy_in)))
                after = ("k_scale_load",)
            emitted.append(Stage(
                "dequant_k", context.bind(Work(resources={
                    "vector": cache.key_conversion_ops(valid)})),
                after=after, acquire=_allocations((k_conv_ub,)),
                free=_allocations((k_payload_ub, k_scale_ub)),
                placement=_placement(context, core, *vector)))
            emitted.append(Stage(
                "conv_store_k", context.bind(Work(resources={write: key_operand})),
                after=("dequant_k",), acquire=_allocations((k_post_l1,)),
                free=_allocations((k_conv_ub,)),
                placement=_placement(context, core, *copy_out)))
            emitted.append(Stage(
                "operand_k", context.bind(Work(resources={operand: key_operand})),
                after=("conv_store_k",), acquire=_allocations((k_l0b,)),
                free=_allocations((k_post_l1,)),
                placement=_placement(context, core, *operand_roles)))
            k_ready = "operand_k"
            k_free = _allocations((k_l0b,))
        else:
            k_load = {"hbm": key_payload, read: key_payload}
            if operand:
                k_load["mte1"] = key_payload
            emitted.append(Stage(
                "k_load", context.bind(Work(resources=k_load)),
                acquire=_allocations((k_l1, k_l0b)),
                placement={**_placement(context, core, *copy_in),
                           **_placement(context, core, *operand_roles)}))
            k_ready = "k_load"
            k_free = _allocations((k_l1, k_l0b))
        emitted.append(Stage(
            "qk", context.bind(qk), after=(k_ready,),
            acquire=score_acquire, free=k_free,
            placement={**_placement(context, core, *cube),
                       **_placement(context, core, *operand_roles)}))
        emitted.append(Stage(
            "softmax", context.bind(softmax), after=("qk",),
            acquire=prob_acquire, free=score_release,
            placement=_placement(context, core, *vector)))

        # V mirrors K; PV consumes the materialized V operand and probability.
        if value_needs:
            v_payload_ub = _allocation(
                context, core, "ub", f"v_payload_{suffix}", value_payload)
            v_conv_ub = _allocation(
                context, core, "ub", f"v_conv_{suffix}", value_operand)
            v_post_l1 = _allocation(
                context, core, "l1", f"v_post_{suffix}", value_operand)
            v_l0b = _allocation(
                context, core, "l0b", f"v_l0b_{suffix}", value_operand)
            v_scale_ub = (_allocation(
                context, core, "ub", f"v_scale_{suffix}", value_scale)
                if value_scale else None)
            emitted.append(Stage(
                "v_load", context.bind(Work(resources={
                    "hbm": value_payload,
                    _read_role(context, vector=True): value_payload})),
                after=("softmax",), acquire=_allocations((v_payload_ub,)),
                placement=_placement(context, core, *vector_copy_in)))
            after = ("v_load",)
            if value_scale:
                emitted.append(Stage(
                    "v_scale_load", context.bind(Work(resources={
                        "hbm": value_scale,
                        _read_role(context, vector=True): value_scale})),
                    after=("v_load",), acquire=_allocations((v_scale_ub,)),
                    placement=_placement(context, core, *vector_copy_in)))
                after = ("v_scale_load",)
            emitted.append(Stage(
                "dequant_v", context.bind(Work(resources={
                    "vector": cache.value_conversion_ops(valid)})),
                after=after, acquire=_allocations((v_conv_ub,)),
                free=_allocations((v_payload_ub, v_scale_ub)),
                placement=_placement(context, core, *vector)))
            emitted.append(Stage(
                "conv_store_v", context.bind(Work(resources={write: value_operand})),
                after=("dequant_v",), acquire=_allocations((v_post_l1,)),
                free=_allocations((v_conv_ub,)),
                placement=_placement(context, core, *copy_out)))
            emitted.append(Stage(
                "operand_v", context.bind(Work(resources={operand: value_operand})),
                after=("conv_store_v",), acquire=_allocations((v_l0b,)),
                free=_allocations((v_post_l1,)),
                placement=_placement(context, core, *operand_roles)))
            v_ready = "operand_v"
            v_free = _allocations((v_l0b,))
        else:
            v_load = {"hbm": value_payload, read: value_payload}
            if operand:
                v_load["mte1"] = value_payload
            emitted.append(Stage(
                "v_load", context.bind(Work(resources=v_load)),
                after=("softmax",), acquire=_allocations((v_l1, v_l0b)),
                placement={**_placement(context, core, *copy_in),
                           **_placement(context, core, *operand_roles)}))
            v_ready = "v_load"
            v_free = _allocations((v_l1, v_l0b))
        emitted.extend((
            Stage("probability_operand",
                  context.bind(Work(resources=(
                      {"mte1": rect * spec.query_bytes} if operand else {}))),
                  after=("softmax",), acquire=_allocations((p_l0a,)),
                  placement=_placement(context, core, *operand_roles)),
            Stage("pv", context.bind(pv),
                  after=(v_ready, "probability_operand"),
                  acquire=pv_acquire,
                  free=_allocations((v_l0b, p_l0a)) + tuple(pv_acquire)
                  + (_allocations((v_l1,)) if not value_needs else ()),
                  placement=_placement(context, core, *cube)),
            Stage("online_update", context.bind(online), after=("pv",),
                  free=prob_release, previous=("online_update",),
                  placement=_placement(context, core, *vector)),
        ))
        return tuple(emitted)

    use_compact = len(set(spec.requests)) <= 1
    # Simplified init is the first three taskIds and the drain is three extra
    # gS1 steps on the last BN.  GQA withholds C2 until PRELOAD_N-1.  Three
    # covers both without selecting a kernel.
    fill = NATIVE_S2_PRELOAD if spec.native_fia else 1
    stages.append(Stage(
        "key_tiles", after=(key_ready,),
        program=(
            compact_periodic_pipeline(
                context, task.key_tiles, preload, tile,
                head=fill, tail=fill, resident=persistent)
            if use_compact
            else PipelineProgram(task.key_tiles, preload, tile)
        ),
    ))
    if page_scale_ub is not None:
        stages.append(Stage("page_scale_release", after=("key_tiles",),
                            free=_allocations((page_scale_ub,))))
    out_bytes = rows * dim * spec.query_bytes
    if partial:
        # A split S2 owner publishes an FP32 accumulator plus LSE partial;
        # the FD merge below consumes both and writes the final BF16 output.
        part = rows * dim * _FP32_BYTES
        lse = rows * 2 * _FP32_BYTES
        stages.append(Stage(
            "final_store", after=("key_tiles",),
            work=context.bind(Work(resources={
                "hbm": part + lse,
                **(({write: part + lse}) if write else {}),
                "vector": rows * (dim + 2),
            })),
            free=persistent,
            placement={**_placement(context, core, *copy_out),
                       **_placement(context, core, *vector)},
        ))
    else:
        stages.append(Stage(
            "final_store", after=("key_tiles",),
            work=context.bind(Work(resources={
                "hbm": out_bytes,
                **(({write: out_bytes}) if write else {}),
                "vector": rows * (dim + 1),
            })),
            free=persistent,
            placement={**_placement(context, core, *copy_out),
                       **_placement(context, core, *vector)},
        ))
    return KernelProgram(tuple(stages))


def planned_prefill_recipe(**options) -> PlannedRecipe:
    """Preserve all prefill recipe options for the T10 factory."""
    return PlannedRecipe(StagedGQAPrefill(**options).build)


__all__ = [
    "IMPLEMENTATION",
    "RECIPE_ID",
    "SOURCE_REFERENCES",
    "StagedGQAPrefill",
    "build",
    "logical_causal_pairs",
    "planned_prefill_recipe",
]
