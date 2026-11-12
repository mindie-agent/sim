"""Staged analytic GQA prefill on the common finite resource engine.

[Role]
  Lower the declared ``metadata['gqa_prefill']`` geometry into one task per
  ``(request, query_head, query_token_tile)``.  A task keeps its query tile and
  online softmax state resident, streams bounded K/V key tiles for that query
  head, and writes its output once.  Tasks are cycled over the actual physical
  owners.

[Invariants]
  * Q, K/V and output traffic follow real request lengths.  A task reloads K/V
    per query head; the flattened decode GQA reuse mapping is not applied.
  * QK and PV each execute the full ``query_rows * actual_kv_rows`` rectangle,
    including the causal-masked diagonal.  Logical valid pairs are reported but
    never substituted for that physical execution.
  * Online max/sum/numerator persist across key tiles.  Scores are ready before
    PV, and the previous accumulator is complete before the next update.
  * On-chip buffers use advertised pools and are released by their owner; they
    are never charged to HBM.  Finished tasks, programs and buffers are released.

[Timeouts & Locks]
  Planning is pure arithmetic.  Tasks are generated lazily by owner stride and
  key tiles by bounded pipeline window; no all-pairs or all-task list is kept.

[Side-Effect Policy]
  No device runtimes, measurements, files or external I/O.  Programs submit only
  to the caller's virtual ``Engine``.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from functools import lru_cache
import math
from typing import Mapping

from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.ownership import read_role, unit_group
from mindie_sim.prediction.attention_prefill_issue import (
    source_c2_path, source_compatible_alternatives, source_issue_kind,
    source_issue_operand,
)
from mindie_sim.prediction.attention_prefill_nofd import _PrefillProgram, owner_s2_plan
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, RecipeContext, Stage
from mindie_sim.prediction.refill_steady import compact_periodic_pipeline
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

IMPLEMENTATION = "staged_gqa_prefill"
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

_DTYPE_ALIASES = {
    "bf16": "bf16", "bfloat16": "bf16",
    "fp16": "fp16", "float16": "fp16",
    "fp32": "fp32", "float32": "fp32",
}
_FP32_BYTES = 4
_PREFILL_PHASES = {"prefill", "draft_prefill"}
# Source CVDiff Souter start (prompt_flash_attention_tiling.cpp 6415-6443).
# The query panel is chosen by the aligned head dimension, not by a request
# length or a measured duration.  Explicit recipe options still override it.
_SOURCE_QUERY_TILE_LARGE_HEAD = 64      # alignedHeadSize >= 200
_SOURCE_QUERY_TILE_DEFAULT = 128        # 32 <= alignedHeadSize < 200
_SOURCE_LARGE_HEAD_THRESHOLD = 200


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


def source_query_tile(head_dim: int) -> int:
    """CVDiff Souter start for the declared head dimension (source tiling)."""
    dim = _positive_int(head_dim, "head_dim")
    return (_SOURCE_QUERY_TILE_LARGE_HEAD
            if dim >= _SOURCE_LARGE_HEAD_THRESHOLD
            else _SOURCE_QUERY_TILE_DEFAULT)


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"attention_prefill_{name}_must_be_positive_int")
    return value


def _nonnegative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingError(f"attention_prefill_{name}_must_be_nonnegative_int")
    return value


def _dtype(value, name: str) -> str:
    if not isinstance(value, str):
        raise ModelingError(f"attention_prefill_{name}_must_be_dtype")
    try:
        return _DTYPE_ALIASES[value.strip().lower()]
    except KeyError:
        raise ModelingError(f"attention_prefill_unsupported_{name}:{value}") from None


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def logical_causal_pairs(rows: int, context: int, first_token: int = 0) -> int:
    """Valid causal pairs for contiguous query rows starting at ``first_token``."""
    return rows * (context + first_token + 1) + rows * (rows - 1) // 2


# Unselected native path.  GQA's resident set is PRELOAD_N=3.  Simplified
# runInfo[4] is four task descriptors, not four K/V panels; that cursor runs
# only when issue_kind is explicit.  Depth 3 here is not a 589824 B refusal.
NATIVE_S2_PRELOAD = 3


def _explicit_reconstruct(metadata: Mapping):
    """Top-level flag only.  Absence does not select Simplified or GQA."""
    if "reconstruct" not in metadata:
        return None
    value = metadata["reconstruct"]
    if value is not True and value is not False:
        raise ModelingError("attention_prefill_reconstruct_must_be_bool_or_absent")
    return value


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

    def work(self, op: OpSpec, *, include_mte1: bool = False) -> Work:
        """Physical work, including the executed causal rectangle."""
        return _total_work(self.spec(op), include_mte1=include_mte1)

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
        mte1 = geometry is not None and "mte1" in geometry.resource_units
        return OperatorPlan(
            self.work(op, include_mte1=mte1),
            lambda context: _PrefillProgram(context, self, spec),
        )

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


@dataclass(frozen=True)
class _Task:
    request: int
    query_head: int
    kv_head: int
    row_start: int
    row_end: int
    token_start: int
    token_end: int
    horizon: int
    key_tiles: int
    key_start: int = 0
    heads: int = 1

    @property
    def rows(self) -> int:
        return self.row_end - self.row_start


@dataclass(frozen=True)
class _Spec:
    requests: tuple[tuple[int, int], ...]
    query_heads: int
    kv_heads: int
    head_dim: int
    query_dtype: str
    kv_dtype: str
    query_tile: int
    key_tile: int
    slots: int
    s2_preload: int
    softmax_slots: int
    softmax_outer: int
    native_fia: bool
    reconstruct: bool | None
    issue_kind: str | None
    request_ends: tuple[int, ...]
    request_tile_counts: tuple[int, ...]

    @property
    def group(self) -> int:
        return self.query_heads // self.kv_heads

    @property
    def query_bytes(self) -> int:
        return int(dtype_bytes(self.query_dtype))

    @property
    def kv_bytes(self) -> int:
        return int(dtype_bytes(self.kv_dtype))

    @property
    def task_count(self) -> int:
        return self.request_ends[-1] if self.request_ends else 0


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
                 reconstruct, issue_kind, tuple(ends), tuple(tile_counts))


def _task_at(spec: _Spec, index: int) -> _Task:
    """Resolve one global cyclic task index without materializing task rows."""
    if index < 0 or index >= spec.task_count:
        raise IndexError("attention_prefill_task_index_out_of_range")
    request = bisect_right(spec.request_ends, index)
    base = spec.request_ends[request - 1] if request else 0
    local = index - base
    tiles = spec.request_tile_counts[request]
    query_head, tile = divmod(local, tiles)
    query, context = spec.requests[request]
    token_start = tile * spec.query_tile
    token_end = min(query, token_start + spec.query_tile)
    horizon = context + token_end
    return _Task(request, query_head, query_head // spec.group,
                 token_start, token_end, token_start, token_end, horizon,
                 _ceil_div(horizon, spec.key_tile), 0, 1)


def _iter_tasks(spec: _Spec):
    if spec.native_fia:
        for _index, task in _native_tasks(spec):
            yield task
        return
    for index in range(spec.task_count):
        yield _task_at(spec, index)


def _native_tasks(spec: _Spec):
    """Native TND S1G M tiles: time x G flattened into 128-row tiles.

    A tile covers ``128/G`` query tokens and all G query heads that share one
    KV head.  K/V is loaded once for the tile, not once per query head.
    """
    index = 0
    width = spec.group
    for request, (query, context) in enumerate(spec.requests):
        total = query * width
        for kv_head in range(spec.kv_heads):
            for m in range(_ceil_div(total, 128)):
                row_start = m * 128
                row_end = min(total, row_start + 128)
                token_start = row_start // width
                token_end = _ceil_div(row_end, width)
                horizon = context + token_end
                yield index, _Task(
                    request, kv_head * width, kv_head, row_start, row_end,
                    token_start, token_end, horizon,
                    _ceil_div(horizon, spec.key_tile), 0, width)
                index += 1


def _task_at_for(spec: _Spec, index: int) -> _Task:
    if spec.native_fia:
        for position, (_idx, task) in enumerate(_native_tasks(spec)):
            if position == index:
                return task
        raise IndexError("attention_prefill_native_task_index_out_of_range")
    return _task_at(spec, index)


def _time_major_tasks(spec: _Spec):
    """S1G flattening order: outer query tile, inner query head.

    The registered TND GQA tiler walks ``[sequence, group]`` time-major, not
    the head-major order used by the cyclic executor.  Owner costs are S2
    tiles in this order.
    """
    if spec.native_fia:
        for index, task in _native_tasks(spec):
            yield index, task
        return
    for request, (query, _context) in enumerate(spec.requests):
        base = spec.request_ends[request - 1] if request else 0
        tiles = spec.request_tile_counts[request]
        for tile in range(tiles):
            for head in range(spec.query_heads):
                index = base + head * tiles + tile
                yield index, _task_at(spec, index)


def _task_pairs(task: _Task, context: int) -> tuple[int, int]:
    """Return (logical valid pairs, physical full-rectangle pairs)."""
    heads = max(1, task.heads)
    rows = task.rows // heads
    logical = heads * logical_causal_pairs(rows, context, task.token_start)
    return logical, task.rows * task.horizon


def _total_work(spec: _Spec, *, include_mte1: bool) -> Work:
    reads = writes = vector = mte1 = 0
    flops = 0
    for task in _iter_tasks(spec):
        _query, context = spec.requests[task.request]
        rows, dim = task.rows, spec.head_dim
        q_bytes = rows * dim * spec.query_bytes
        out_bytes = rows * dim * spec.query_bytes
        reads += q_bytes
        writes += out_bytes
        if include_mte1:
            # Q GM->L1 is paid once, but every S2 loop enters MatmulK/Base
            # and reloads the resident Q tile L1->L0A through the shared DB.
            mte1 += q_bytes * task.key_tiles
        vector += rows * (dim + 1)
        for tile in range(task.key_tiles):
            valid = min(spec.key_tile, task.horizon - tile * spec.key_tile)
            kv_bytes = valid * dim * spec.kv_bytes
            rect = rows * valid
            reads += 2 * kv_bytes
            if include_mte1:
                mte1 += 2 * kv_bytes + rect * spec.query_bytes
            flops += 4 * dim * rect
            vector += 7 * rect + rows * (3 * dim + 10)
        # Logical causal pairs are intentionally not used for service demand.
    resources = {
        "hbm": reads + writes,
        "mte2": reads,
        "mte3": writes,
        "vector": vector,
    }
    if include_mte1:
        resources["mte1"] = mte1
    return Work(matrix={spec.query_dtype: flops}, resources=resources)


def _report(spec: _Spec) -> dict:
    tasks = logical = physical = kv_read = q_read = out_write = 0
    q_l1_l0a = 0
    key_loads = 0
    for task in _iter_tasks(spec):
        _query, context = spec.requests[task.request]
        tasks += 1
        task_logical, task_physical = _task_pairs(task, context)
        logical += task_logical
        physical += task_physical
        key_loads += task.key_tiles
        task_q = task.rows * spec.head_dim * spec.query_bytes
        q_read += task_q
        q_l1_l0a += task_q * task.key_tiles
        out_write += task.rows * spec.head_dim * spec.query_bytes
        kv_read += task.horizon * 2 * spec.head_dim * spec.kv_bytes
    if spec.native_fia:
        # M tiles cut [time, group] and tails can split a head row; logical
        # causal pairs are still per original query head and KV group.
        logical = sum(spec.query_heads * logical_causal_pairs(query, context)
                      for query, context in spec.requests)
    return {
        "implementation": IMPLEMENTATION,
        "tasks": tasks,
        "key_tile_loads": key_loads,
        "logical_pairs": logical,
        "executed_rectangle_pairs": physical,
        "causal_masked_pairs": physical - logical,
        "q_bytes": q_read,
        "q_l1_l0a_bytes": q_l1_l0a,
        "kv_bytes": kv_read,
        "output_bytes": out_write,
        "query_tile": spec.query_tile,
        "key_tile": spec.key_tile,
        "slots": spec.slots,
        "s2_preload": spec.s2_preload,
        "reconstruct": spec.reconstruct,
        "issue_kind": spec.issue_kind,
        "source_state": "pinned_source_mechanism_not_captured_dispatch",
    }


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
    copy_in = (read,)
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
    l1_required = q_bytes + min(recipe.key_tile, task.horizon) * dim * spec.kv_bytes
    for pool, required in (("l0a", l0a_required), ("ub", ub_required),
                           ("l1", l1_required)):
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
        kv_bytes = valid * dim * spec.kv_bytes
        suffix = f"s{slot}"
        k_l1 = _allocation(context, core, "l1", f"k_l1_{suffix}", kv_bytes)
        k_l0b = _allocation(context, core, "l0b", f"k_l0b_{suffix}",
                            min(kv_bytes, 32 * 1024))
        # CVDiff softmax/PV cuts the probability panel to
        # softmaxSOuterFactor rows; it does not materialize the full Souter
        # row set in L0A/UB at once.
        scores = _allocation(context, core, "l0c", f"scores_{suffix}",
                             outer * max_tile * _FP32_BYTES)
        probs = _allocation(context, core, "ub", f"probs_{suffix}",
                            outer * max_tile * _FP32_BYTES)
        p_l0a = _allocation(context, core, "l0a", f"p_l0a_{suffix}",
                           min(outer * valid * spec.query_bytes, 32 * 1024))
        v_l1 = _allocation(context, core, "l1", f"v_l1_{suffix}", kv_bytes)
        v_l0b = _allocation(context, core, "l0b", f"v_l0b_{suffix}",
                            min(kv_bytes, 32 * 1024))
        pv_acc = _allocation(context, core, "l0c", f"pv_acc_{suffix}",
                             rows * dim * _FP32_BYTES)
        k_acquire = _allocations((k_l1, k_l0b))
        v_acquire = _allocations((v_l1, v_l0b))
        pv_acquire = _allocations((pv_acc,))
        score_acquire = _allocations((scores,))
        prob_acquire = _allocations((probs,))
        k_release = _allocations((k_l1, k_l0b))
        score_release = _allocations((scores,))
        v_release = _allocations((v_l1, v_l0b))
        prob_release = _allocations((probs,))

        k_load = {"hbm": kv_bytes, read: kv_bytes}
        if operand:
            k_load["mte1"] = kv_bytes
        qk_work = {"mte1": q_bytes} if operand else {}
        qk = Work(matrix={spec.query_dtype: 2 * dim * rect},
                  resources=qk_work)
        softmax = Work(resources={"vector": 7 * rect + rows})
        v_load = {"hbm": kv_bytes, read: kv_bytes}
        if operand:
            v_load["mte1"] = kv_bytes
        pv = Work(matrix={spec.query_dtype: 2 * dim * rect})
        online = Work(resources={"vector": rows * (3 * dim + 9)})
        return (
            Stage("k_load", context.bind(Work(resources=k_load)),
                  acquire=k_acquire,
                  placement={**_placement(context, core, *copy_in),
                             **_placement(context, core, *operand_roles)}),
            Stage("qk", context.bind(qk), after=("k_load",),
                  acquire=score_acquire, free=k_release,
                  placement={**_placement(context, core, *cube),
                             **_placement(context, core, *operand_roles)}),
            Stage("softmax", context.bind(softmax), after=("qk",),
                  acquire=prob_acquire, free=score_release,
                  placement=_placement(context, core, *vector)),
            Stage("v_load", context.bind(Work(resources=v_load)),
                  after=("softmax",), acquire=v_acquire,
                  placement={**_placement(context, core, *copy_in),
                             **_placement(context, core, *operand_roles)}),
            Stage("probability_operand",
                  context.bind(Work(resources=(
                      {"mte1": rect * spec.query_bytes} if operand else {}))),
                  after=("softmax",), acquire=_allocations((p_l0a,)),
                  placement=_placement(context, core, *operand_roles)),
            Stage("pv", context.bind(pv),
                  after=("v_load", "probability_operand"),
                  acquire=pv_acquire,
                  free=v_release + _allocations((p_l0a,)),
                  placement=_placement(context, core, *cube)),
            Stage("online_update", context.bind(online), after=("pv",),
                  free=prob_release + pv_acquire, previous=("online_update",),
                  placement=_placement(context, core, *vector)),
        )

    use_compact = len(set(spec.requests)) <= 1
    # Simplified init is the first three taskIds and the drain is three extra
    # gS1 steps on the last BN.  GQA withholds C2 until PRELOAD_N-1.  Three
    # covers both without selecting a kernel.
    fill = NATIVE_S2_PRELOAD if spec.native_fia else 1
    stages.append(Stage(
        "key_tiles", after=("q_load",),
        program=(
            compact_periodic_pipeline(
                context, task.key_tiles, preload, tile,
                head=fill, tail=fill, resident=persistent)
            if use_compact
            else PipelineProgram(task.key_tiles, preload, tile)
        ),
    ))
    out_bytes = rows * dim * spec.query_bytes
    if partial:
        # A split S2 owner publishes an FP32 accumulator plus LSE partial;
        # the FD merge below consumes both and writes the final BF16 output.
        part = rows * dim * _FP32_BYTES
        lse = rows * 2 * _FP32_BYTES
        stages.append(Stage(
            "final_store", after=("key_tiles",),
            work=context.bind(Work(resources={
                "hbm": part + lse, "mte3": part + lse,
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
                "hbm": out_bytes, "mte3": out_bytes, "vector": rows * (dim + 1),
            })),
            free=persistent,
            placement={**_placement(context, core, *copy_out),
                       **_placement(context, core, *vector)},
        ))
    return KernelProgram(tuple(stages))


__all__ = [
    "IMPLEMENTATION",
    "RECIPE_ID",
    "SOURCE_REFERENCES",
    "StagedGQAPrefill",
    "build",
    "logical_causal_pairs",
]
