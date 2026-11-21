"""Tiled FlashAttention and paged-attention physical recipe.

[Role] Partition Q load, KV refill, QK, online softmax, PV and final store into
  finite phases over a bounded query/head tile pipeline.
[Invariants] QK and PV issue the declared Cube work; softmax and normalization
  issue Vector work; KV refills remain private/L2 service and HBM misses are
  charged once by the traffic planner. No aggregate envelope stage is emitted.
[Timeouts & Locks] Bounded tile count and phase count; no I/O or locks.
[Side-Effect Policy] Submits only to the caller's virtual Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING

from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.operators.matrix.refill_steady import compact_periodic_pipeline
from mindie_sim.prediction.ownership import read_role, unit_group
from mindie_sim.schema.common import ModelingError

from mindie_sim.prediction.physical_recipe_base import (
    IO_HBM_CATEGORY, Phase, build_plan, io_weight, tiled_phased_program,
)
from mindie_sim.prediction.program import (
    KernelProgram, PipelinePhase, PipelineProgram, PipelineTemplate,
    PipelineTemplateProgram, RecipeContext, Stage,
)

from .common import AttentionContract

if TYPE_CHECKING:
    from .attention_decode import StagedGQADecode, _Contribution, _Output, _Spec

Q_LOAD_SHARE = 0.25
KV_LOAD_SHARE = 0.75
SOFTMAX_SHARE = 0.35
NORMALIZE_SHARE = 0.65
PIPELINE_SLOTS = 2
ATTENTION_TILE_CAP = 64
ATTENTION_SHAPE_WITH_VALUE_DIM = 5


def _matrix_phase_weights(op) -> tuple[int, int]:
    """The phase allocator normalizes the widths against their sum."""
    contract = dict(op.metadata).get("attention_contract")
    if isinstance(contract, AttentionContract):
        return contract.head_dim, contract.v_head_dim
    if contract is not None:
        return contract["head_dim"], contract["v_head_dim"]
    if len(op.local_shape) == ATTENTION_SHAPE_WITH_VALUE_DIM:
        qk_dim, v_dim = op.local_shape[-2:]
        return qk_dim, v_dim
    # Ordinary GQA uses the same width for both matrix products.
    return 1, 1


@dataclass(frozen=True)
class TiledFlashAndPagedAttentionRecipe:
    """Class A: attention Q/KV microtiles with coupled score/PV work."""

    tile_cap: int = ATTENTION_TILE_CAP
    slots: int = PIPELINE_SLOTS
    traffic_rewrite_capable: bool = True

    def build(self, op, geometry):
        return build_plan(self, op, geometry)

    def lower(self, context):
        read_share = io_weight(context, "hbm")
        write_share = io_weight(context, "hbm_write")
        qk_weight, pv_weight = _matrix_phase_weights(context.op)
        phases = (
            Phase("q_load", {IO_HBM_CATEGORY: Q_LOAD_SHARE * read_share,
                             "mte2": Q_LOAD_SHARE}),
            Phase("kv_load", {IO_HBM_CATEGORY: KV_LOAD_SHARE * read_share,
                              "mte2": KV_LOAD_SHARE}),
            Phase("qk", {"cube": qk_weight}, after=("q_load", "kv_load")),
            Phase("online_softmax", {"vector": SOFTMAX_SHARE}, after=("qk",)),
            Phase("pv", {"cube": pv_weight}, after=("online_softmax",)),
            Phase("normalize_store", {
                "vector": NORMALIZE_SHARE,
                IO_HBM_CATEGORY: write_share,
                "mte3": 1.0,
            }, after=("pv",)),
        )
        return tiled_phased_program(
            context, phases, slots=self.slots, cap=self.tile_cap)

    def lower_single(self, context) -> KernelProgram:
        """Diagnostic single-tile lowering retaining the same phase DAG."""
        qk_weight, pv_weight = _matrix_phase_weights(context.op)
        return tiled_phased_program(context, (
            Phase("q_load", {IO_HBM_CATEGORY: 1.0, "mte2": 1.0}),
            Phase("kv_load", {}),
            Phase("qk", {"cube": qk_weight}, after=("q_load", "kv_load")),
            Phase("online_softmax", {"vector": SOFTMAX_SHARE}, after=("qk",)),
            Phase("pv", {"cube": pv_weight}, after=("online_softmax",)),
            Phase("normalize_store", {"vector": NORMALIZE_SHARE,
                                      "mte3": 1.0}, after=("pv",)),
        ), slots=self.slots, cap=max(1, context.core_count))


TiledAttentionRecipe = TiledFlashAndPagedAttentionRecipe


DECODE_SOFTMAX_OPS_PER_PAIR = 5
DECODE_OUTPUT_UPDATE_OPS_PER_VALUE = 3
DECODE_ONLINE_STATE_OPS = 9
TEMPLATE_CACHE_SLOTS_FACTOR = 4

_FP32_BYTES = 4


def _partial_row_bytes(dim: int) -> int:
    """Per output row: FP32 partial accumulator plus max and sum scalars."""
    return dim * _FP32_BYTES + 2 * _FP32_BYTES


def _rows_per_token(output: _Output, group: int, token: int) -> int:
    return max(0, min(output.row_end, (token + 1) * group)
               - max(output.row_start, token * group))


def _placement(context: RecipeContext, core: int, *roles: str) -> dict:
    present = tuple(role for role in roles if role in context.resource_names)
    return context.owned(core, context.core_count, present) if present else {}


def _read_role(context: RecipeContext, *, vector=False) -> str:
    return read_role(context.resource_names, aiv=0 if vector else None)


def _write_role(context: RecipeContext) -> str | None:
    return "mte3" if "mte3" in context.resource_names else None


def _operand_role(context: RecipeContext) -> str | None:
    return "mte1" if "mte1" in context.resource_names else None


def _allocation(context: RecipeContext, core: int, pool: str, name: str,
                size: float) -> Allocation | None:
    if pool not in context.pool_names:
        return None
    if size <= 0:
        return None
    capacity = context.pool_capacities.get(pool)
    if capacity is not None and size > capacity:
        raise ModelingError(
            f"attention_decode_buffer_exceeds_pool:{pool}:{name}:{size}>{capacity}")
    units = unit_group(core, context.core_count,
                       context.pool_units.get(pool, 1))
    return Allocation(name, context.pool_names[pool], float(size), units)


def _allocations(items) -> tuple[Allocation, ...]:
    return tuple(item for item in items if item is not None)


def _tile_counts(output: _Output, context: int, group: int,
                 tile_start: int, tile_end: int) -> tuple[int, int]:
    """Return (valid query/key pairs, active query rows) for one key tile."""
    pairs = row_count = 0
    for token in range(output.token_first, output.token_last + 1):
        rows = _rows_per_token(output, group, token)
        if not rows:
            continue
        valid = max(0, min(tile_end, context + token + 1) - tile_start)
        if valid:
            pairs += rows * valid
            row_count += rows
    return pairs, row_count


@dataclass(frozen=True)
class _AttentionPipelineTemplate(PipelineTemplate):
    """Attention slot buffers are tile-scoped in the finite phase table.

    The detailed producer already names slot buffers ``..._s0/_s1`` and the
    Engine scopes them per tile.  Marking them ``panel`` preserves that
    identity for both residue stages without inferring ownership from buffer
    names.
    """

    @property
    def allocation_bindings(self):
        rows = []
        seen = set()
        stages = []
        for phase in self.phases:
            stages.extend(phase.stages)
            for group in phase.residue_stages:
                stages.extend(group)
        for stage in stages:
            for allocation, action in (
                    *((item, "acquire") for item in stage.acquire),
                    *((item, "free") for item in stage.free)):
                key = (allocation.name, action)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "name": allocation.name, "scope": "panel",
                    "owner": self.owner, "ordinal": 0,
                    "pool": allocation.pool, "size": allocation.size,
                    "units": tuple(allocation.units),
                    "action": action,
                })
        return tuple(rows)


def _fold_interior_phase(phase: PipelinePhase) -> PipelinePhase:
    """Return one interior key-tile phase unchanged.

    Summing repeated work into one task per stage would collapse the slot
    window's overlap into a single stream. Keep each phase intact so elapsed
    service and buffer lifetimes follow the declared pipeline schedule.
    """
    return phase


def _attention_pipeline_template(tile, key_of, count, *, rows, key_tile,
                                 slots, core):
    """Group the real key-tile stream into finite head/full/tail phases.

    ``full`` carries one Stage set per slot residue; its repeated body is
    selected by absolute tile index, so a full phase never replays slot0
    acquire/free on slot1.  Head and partial tail stages remain explicit.
    """
    if count <= 0:
        return None
    slots = max(1, int(slots))
    keys = [tuple(key_of(index)) for index in range(count)]

    tail_start = count
    while tail_start > 0 and keys[tail_start - 1][0] < key_tile:
        tail_start -= 1
    if tail_start == 0:
        return None

    full_start = 0
    periodic = False
    while full_start < tail_start:
        if tail_start - full_start >= slots and all(
                keys[index] == keys[index + slots]
                for index in range(full_start, tail_start - slots)):
            periodic = True
            break
        full_start += 1
    if not periodic:
        return None
    full_end = tail_start
    full_count = full_end - full_start
    if full_count < slots or full_count % slots:
        return None
    base_stages = tuple(tile(full_start))
    for offset in range(1, slots):
        other = tuple(tile(full_start + offset))
        if len(other) != len(base_stages):
            return None
        for left, right in zip(base_stages, other):
            if (left.name, left.work, left.caps, left.coupled,
                    left.latency, left.after, left.previous,
                    left.placement) != (
                    right.name, right.work, right.caps, right.coupled,
                    right.latency, right.after, right.previous,
                    right.placement):
                # The consumer selects phase.stages, so every residue must
                # carry the same physical work and dependency ordering.
                return None

    phases = []

    def add_run(kind, start, end):
        phases.append(PipelinePhase(
            kind, tuple(tile(start)), end - start,
            key=keys[start], end=end))

    index = 0
    while index < full_start:
        end = index + 1
        while end < full_start and keys[end] == keys[index]:
            end += 1
        add_run("head", index, end)
        index = end
    if full_count:
        phases.append(PipelinePhase(
            "full", tuple(tile(full_start)), full_count,
            key=(*keys[full_start][:3], None),
            residue_stages=tuple(
                tuple(tile(full_start + ((offset - full_start) % slots)))
                for offset in range(slots)),
            end=full_end))
    index = full_end
    while index < count:
        end = index + 1
        while end < count and keys[end] == keys[index]:
            end += 1
        add_run("tail", index, end)
        index = end
    if sum(phase.count for phase in phases) != count:
        raise ModelingError("attention_decode_phase_count_mismatch")
    phases = [_fold_interior_phase(phase) for phase in phases]
    return _AttentionPipelineTemplate(core, slots, tuple(phases))


def _producer_program(context: RecipeContext, recipe: StagedGQADecode,
                      spec: _Spec, output: _Output,
                      contribution: _Contribution, core: int, *,
                      peer_counts: tuple[int, ...] | None = None) -> KernelProgram:
    _query, context_len = spec.requests[output.request]
    dim, rows = spec.head_dim, output.rows
    q_bytes = rows * dim * spec.query_bytes
    read = _read_role(context)
    write = _write_role(context)
    operand = _operand_role(context)
    vector = ("vector",)
    copy_in = (read,)
    vector_copy_in = (_read_role(context, vector=True),)
    copy_out = (write,) if write else ()
    operand_role = (operand,) if operand else ()

    q_l1 = _allocation(context, core, "l1", "q_l1", q_bytes)
    q_l0a = _allocation(context, core, "l0a", "q_l0a", q_bytes)
    ub_state = _allocation(context, core, "ub", "online_state",
                           rows * _partial_row_bytes(dim))
    persistent = _allocations((q_l1, q_l0a, ub_state))

    q_work = {"hbm": q_bytes, read: q_bytes}
    if operand:
        q_work["mte1"] = q_bytes
    q_place = {**_placement(context, core, *copy_in),
               **_placement(context, core, *operand_role)}
    stages = [
        Stage("q_load", context.bind(Work(resources=q_work)),
              acquire=persistent, placement=q_place),
    ]
    cache = spec.kv_cache
    if cache.needs_conversion and not operand:
        raise ModelingError("attention_decode_conversion_requires_mte1_operand_copy")
    page_meta = page_scale = 0
    page_scale_ub = None
    if cache.paged:
        page_start = contribution.tile_start * recipe.key_tile
        page_end = min((contribution.tile_start + contribution.tile_count)
                       * recipe.key_tile, output.horizon)
        pages = cache.pages(page_start, page_end)
        page_meta = pages * cache.page_meta_bytes
        if cache.scale_scope == "page":
            page_scale = pages * cache.page_scale_bytes
    if page_meta or page_scale:
        page_work = {"hbm": page_meta + page_scale}
        if page_meta:
            page_work[read] = page_work.get(read, 0) + page_meta
        if page_scale:
            vector_read, = vector_copy_in
            page_work[vector_read] = page_work.get(vector_read, 0) + page_scale
            page_scale_ub = _allocation(
                context, core, "ub", "page_scale_ub", page_scale)
        stages.append(Stage(
            "page_access", context.bind(Work(resources=page_work)),
            acquire=_allocations((page_scale_ub,)), after=("q_load",),
            placement={**_placement(context, core, *copy_in),
                       **_placement(context, core, *vector_copy_in)}))
    key_ready = "page_access" if (page_meta or page_scale) else "q_load"

    max_tile = min(recipe.key_tile, max(1, output.horizon))

    causal_start = max(
        0,
        (context_len + output.token_first + 1) // recipe.key_tile - contribution.tile_start,
    )
    tail_count = max(2, contribution.tile_count - causal_start)

    def key_of(index: int):
        start = (contribution.tile_start + index) * recipe.key_tile
        end = min(start + recipe.key_tile, output.horizon)
        valid = max(0, end - start)
        pairs, active_rows = _tile_counts(
            output, context_len, spec.group, start, end)
        return valid, pairs, active_rows, index % recipe.slots

    def tile(index: int):
        valid, pairs, active_rows, slot = key_of(index)
        return template(valid, pairs, active_rows, slot)

    @lru_cache(maxsize=TEMPLATE_CACHE_SLOTS_FACTOR * recipe.slots + tail_count)
    def template(valid: int, pairs: int, active_rows: int, slot: int):
        return _decode_tile_stages(
            valid, pairs, active_rows, slot, context, core, spec, cache, rows, dim, max_tile,
            read, write, operand, vector_copy_in)

    contributions = len(output.contributions)
    use_compact = len(set(spec.requests)) <= 1 and contributions == 1
    # A head, the actual causal tail, and two complete slot periods are the
    # structural minimum. The compactor verifies convergence before folding.
    min_periodic_tiles = 1 + tail_count + 2 * max(2, recipe.slots)
    if use_compact and contribution.tile_count >= min_periodic_tiles:
        peer_tails = (
            tuple(
                max(
                    PIPELINE_SLOTS,
                    c.tile_count - max(
                        0,
                        (spec.requests[out.request][1] + out.token_first + 1) // recipe.key_tile
                        - c.tile_start,
                    ),
                )
                for out in spec.outputs for c in out.contributions if c.tile_count > 0
            )
            if peer_counts is not None else None
        )
        key_program = compact_periodic_pipeline(
            context, contribution.tile_count, recipe.slots, tile,
            head=1, tail=tail_count, threshold=min_periodic_tiles - 1,
            resident=persistent, peer_counts=peer_counts, peer_tails=peer_tails)
    else:
        finite_template = _attention_pipeline_template(
            tile, key_of, contribution.tile_count,
            rows=rows, key_tile=recipe.key_tile, slots=recipe.slots, core=core)
        key_program = (PipelineTemplateProgram(finite_template)
                       if finite_template is not None else PipelineProgram(
                           contribution.tile_count, recipe.slots, tile))
    stages.append(Stage(
        "key_tiles", after=(key_ready,), program=key_program,
    ))

    if page_scale_ub is not None:
        stages.append(Stage("page_scale_release", after=("key_tiles",),
                            free=_allocations((page_scale_ub,))))
    if contributions == 1:
        out_bytes = rows * dim * spec.query_bytes
        final_resources = {"hbm": out_bytes, "vector": rows * (dim + 1)}
        if write:
            final_resources[write] = out_bytes
        final = Work(resources=final_resources)
        stages.append(Stage(
            "final_store", after=("key_tiles",),
            work=context.bind(final), free=persistent,
            placement={**_placement(context, core, *copy_out),
                       **_placement(context, core, *vector)},
        ))
    else:
        partial_bytes = rows * _partial_row_bytes(dim)
        stages.append(Stage(
            "partial_store", after=("key_tiles",),
            work=context.bind(Work(resources={
                "hbm": partial_bytes,
                **(({write: partial_bytes}) if write else {}),
            })),
            free=persistent,
            placement=_placement(context, core, *copy_out),
        ))
    return KernelProgram(tuple(stages))


def _append_decode_key_load(
    stages_local, context, core, cache, valid, suffix, key_needs, key_payload, key_operand,
    key_scale, read, write, operand, vector_copy_in,
):
    copy_in = (read,)
    copy_out = (write,) if write else ()
    operand_role = (operand,) if operand else ()
    vector = ("vector",)
    # K: packed -> UB, vector conversion -> UB, MTE3 -> L1, MTE1 -> L0B.
    if key_needs:
        k_payload_ub = _allocation(context, core, "ub", f"k_payload_{suffix}",
                                   key_payload)
        k_conv_ub = _allocation(context, core, "ub", f"k_conv_{suffix}",
                                key_operand)
        k_post_l1 = _allocation(context, core, "l1", f"k_post_{suffix}",
                                key_operand)
        k_l0b = _allocation(context, core, "l0b", f"k_l0b_{suffix}",
                            key_operand)
        k_scale_ub = (_allocation(context, core, "ub", f"k_scale_{suffix}",
                                  key_scale) if key_scale else None)
        stages_local.append(Stage(
            "k_load", context.bind(Work(resources={
                "hbm": key_payload,
                _read_role(context, vector=True): key_payload})),
            acquire=_allocations((k_payload_ub,)),
            placement=_placement(context, core, *vector_copy_in)))
        k_after = ("k_load",)
        if key_scale:
            stages_local.append(Stage(
                "k_scale_load", context.bind(Work(resources={
                    "hbm": key_scale,
                    _read_role(context, vector=True): key_scale})),
                after=("k_load",), acquire=_allocations((k_scale_ub,)),
                placement=_placement(context, core, *vector_copy_in)))
            k_after = ("k_scale_load",)
        stages_local.append(Stage(
            "dequant_k", context.bind(Work(resources={
                "vector": cache.key_conversion_ops(valid)})),
            after=k_after, acquire=_allocations((k_conv_ub,)),
            free=_allocations((k_payload_ub, k_scale_ub)),
            placement=_placement(context, core, *vector)))
        stages_local.append(Stage(
            "conv_store_k", context.bind(Work(resources={write: key_operand})),
            after=("dequant_k",), acquire=_allocations((k_post_l1,)),
            free=_allocations((k_conv_ub,)),
            placement=_placement(context, core, *copy_out)))
        stages_local.append(Stage(
            "operand_k", context.bind(Work(resources={operand: key_operand})),
            after=("conv_store_k",), acquire=_allocations((k_l0b,)),
            free=_allocations((k_post_l1,)),
            placement=_placement(context, core, *operand_role)))
        k_ready = "operand_k"
        k_free = _allocations((k_l0b,))
    else:
        k_l1 = _allocation(context, core, "l1", f"k_l1_{suffix}", key_payload)
        k_l0b = _allocation(context, core, "l0b", f"k_l0b_{suffix}", key_payload)
        k_load_work = {"hbm": key_payload, read: key_payload}
        if operand:
            k_load_work[operand] = key_payload
        stages_local.append(Stage(
            "k_load", context.bind(Work(resources=k_load_work)),
            acquire=_allocations((k_l1, k_l0b)),
            placement={**_placement(context, core, *copy_in),
                       **_placement(context, core, *operand_role)}))
        k_ready = "k_load"
        k_free = _allocations((k_l1, k_l0b))
    return k_ready, k_free


def _decode_tile_stages(
    valid, pairs, active_rows, slot, context, core, spec, cache, rows, dim, max_tile, read,
    write, operand, vector_copy_in,
):
    copy_in = (read,)
    copy_out = (write,) if write else ()
    operand_role = (operand,) if operand else ()
    cube, vector = ("cube",), ("vector",)

    # Full tiles repeat immutable work and slot-local buffer descriptors.
    # Retain at most a full/tail pair per live slot, within this producer;
    # every tile still submits its original finite stages and dependencies.
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
    scores = _allocation(context, core, "l0c", f"scores_{suffix}",
                         rows * max_tile * _FP32_BYTES)
    probs = _allocation(context, core, "ub", f"probs_{suffix}",
                        rows * max_tile * _FP32_BYTES)
    p_l0a = _allocation(context, core, "l0a", f"p_l0a_{suffix}",
                       rows * valid * spec.query_bytes)
    pv_acc = _allocation(context, core, "l0c", f"pv_acc_{suffix}",
                         rows * dim * _FP32_BYTES)
    score_buffer = _allocations((scores,))
    prob_buffer = _allocations((probs,))
    pv_acc_buffer = _allocations((pv_acc,))

    qk = Work(
        matrix={spec.query_dtype: 2 * dim * pairs},
        resources={"vector": 2 * pairs},
    )
    softmax = Work(resources={"vector": DECODE_SOFTMAX_OPS_PER_PAIR * pairs + active_rows})
    pv = Work(matrix={spec.query_dtype: 2 * dim * pairs})
    # Combine the tile-local numerator with the running numerator using
    # their maxima: two scalings plus one add per output element.
    online = Work(resources={"vector": active_rows * (DECODE_OUTPUT_UPDATE_OPS_PER_VALUE * dim + DECODE_ONLINE_STATE_OPS)})
    stages_local = []

    k_ready, k_free = _append_decode_key_load(
        stages_local, context, core, cache, valid, suffix, key_needs, key_payload,
        key_operand, key_scale, read, write, operand, vector_copy_in)
    stages_local.append(Stage(
        "qk", context.bind(qk), after=(k_ready,),
        acquire=score_buffer, free=k_free, placement={
            **_placement(context, core, *cube),
            **_placement(context, core, *vector)}))
    stages_local.append(Stage(
        "softmax", context.bind(softmax), after=("qk",),
        acquire=prob_buffer, free=score_buffer,
        placement=_placement(context, core, *vector)))

    # V mirrors K; PV consumes the materialized V operand and probability.
    if value_needs:
        v_payload_ub = _allocation(context, core, "ub", f"v_payload_{suffix}",
                                   value_payload)
        v_conv_ub = _allocation(context, core, "ub", f"v_conv_{suffix}",
                                value_operand)
        v_post_l1 = _allocation(context, core, "l1", f"v_post_{suffix}",
                                value_operand)
        v_l0b = _allocation(context, core, "l0b", f"v_l0b_{suffix}",
                            value_operand)
        v_scale_ub = (_allocation(context, core, "ub", f"v_scale_{suffix}",
                                  value_scale) if value_scale else None)
        stages_local.append(Stage(
            "v_load", context.bind(Work(resources={
                "hbm": value_payload,
                _read_role(context, vector=True): value_payload})),
            after=("softmax",),
            acquire=_allocations((v_payload_ub,)) + pv_acc_buffer,
            placement=_placement(context, core, *vector_copy_in)))
        v_after = ("v_load",)
        if value_scale:
            stages_local.append(Stage(
                "v_scale_load", context.bind(Work(resources={
                    "hbm": value_scale,
                    _read_role(context, vector=True): value_scale})),
                after=("v_load",), acquire=_allocations((v_scale_ub,)),
                placement=_placement(context, core, *vector_copy_in)))
            v_after = ("v_scale_load",)
        stages_local.append(Stage(
            "dequant_v", context.bind(Work(resources={
                "vector": cache.value_conversion_ops(valid)})),
            after=v_after, acquire=_allocations((v_conv_ub,)),
            free=_allocations((v_payload_ub, v_scale_ub)),
            placement=_placement(context, core, *vector)))
        stages_local.append(Stage(
            "conv_store_v", context.bind(Work(resources={write: value_operand})),
            after=("dequant_v",), acquire=_allocations((v_post_l1,)),
            free=_allocations((v_conv_ub,)),
            placement=_placement(context, core, *copy_out)))
        stages_local.append(Stage(
            "operand_v", context.bind(Work(resources={operand: value_operand})),
            after=("conv_store_v",), acquire=_allocations((v_l0b,)),
            free=_allocations((v_post_l1,)),
            placement=_placement(context, core, *operand_role)))
        v_ready = "operand_v"
        v_free = _allocations((v_l0b,))
    else:
        v_l1 = _allocation(context, core, "l1", f"v_l1_{suffix}", value_payload)
        v_l0b = _allocation(context, core, "l0b", f"v_l0b_{suffix}", value_payload)
        v_load_work = {"hbm": value_payload, read: value_payload}
        if operand:
            v_load_work[operand] = value_payload
        stages_local.append(Stage(
            "v_load", context.bind(Work(resources=v_load_work)),
            after=("softmax",),
            acquire=_allocations((v_l1, v_l0b)) + pv_acc_buffer,
            placement={**_placement(context, core, *copy_in),
                       **_placement(context, core, *operand_role)}))
        v_ready = "v_load"
        v_free = _allocations((v_l1, v_l0b))
    stages_local.extend((
        Stage("probability_operand", context.bind(Work(resources=(
                  {operand: pairs * spec.query_bytes} if operand else {}))),
              after=("softmax",), acquire=_allocations((p_l0a,)),
              placement=_placement(context, core, *operand_role)),
        Stage("pv", context.bind(pv), after=(v_ready, "probability_operand"),
              free=v_free + _allocations((p_l0a,)) + pv_acc_buffer,
              placement=_placement(context, core, *cube)),
        Stage("online_update", context.bind(online), after=("pv",),
              free=prob_buffer, previous=("online_update",),
              placement=_placement(context, core, *vector)),
    ))
    return tuple(stages_local)
