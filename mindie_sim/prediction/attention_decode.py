"""Staged analytic GQA decode on the common finite resource engine.

[Role]
  Lower the declared ``metadata['gqa_decode']`` geometry into finite
  ``(batch, kv_head, query_tile, kv_split)`` owners.  Each owner loads a
  resident query tile, streams bounded key/value tiles through
  QK -> softmax -> PV -> online accumulation, and stores a partial or final
  output.  Split partials are merged by a separate, barrier-gated program.

[Invariants]
  * Real request lengths and the causal triangular tail determine work; no
    padded cache capacity or query-head multiplier is applied to KV.
  * One key tile is loaded once per contributing split and query tile.  K/V
    traffic is never multiplied by the GQA query-head group.
  * QK and PV together are ``4 * pairs * query_heads * head_dim`` FLOPs.
  * A split partial is materialized only when at least two split producers
    contribute to one output.  One contributing split writes the final output
    directly.  The final output is written exactly once.
  * On-chip operands, scores, probabilities and accumulators use advertised
    pools when present and are released by the owning task.  They are never
    added to HBM traffic.
  * Owners are physical placements; their task streams cannot borrow idle units.

[Timeouts & Locks]
  Planning and lowering are pure arithmetic.  Key tiles are generated lazily
  through a bounded pipeline window; no per-tile list is materialized.

[Side-Effect Policy]
  No device runtimes, measurements, files or external I/O.  Programs submit
  only to the caller's virtual ``Engine``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
from typing import Callable, Mapping

from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.mechanism import (
    Geometry, OperatorPlan, PlannedRecipe, Work,
)
from mindie_sim.prediction.ownership import read_role, unit_group
from .attention_kv_access import KVCacheAccess, kv_access, paged_fia_plan
from mindie_sim.prediction.program import (
    KernelProgram, PipelinePhase, PipelineProgram, PipelineTemplate,
    PipelineTemplateProgram, RecipeContext, Stage,
)
from mindie_sim.prediction.refill_steady import compact_periodic_pipeline
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

from .attention_decode_helpers import (
    _DTYPE_ALIASES, _FP32_BYTES, _AttentionPipelineTemplate, _Contribution,
    _MergeStream, _Output, _Spec, _allocation, _allocations,
    _attention_pipeline_template, _ceil_div, _dtype, _fold_interior_phase,
    _merge_program, _nonnegative_int, _operand_role, _output_pairs,
    _output_row_tiles, _partial_row_bytes, _placement, _positive_int,
    _read_role, _rows_per_token, _spec, _split_ranges, _tile_counts,
    _total_work, _write_role, causal_pairs,
)

IMPLEMENTATION = "staged_gqa_decode"
RECIPE_ID = IMPLEMENTATION

# Mechanism-only references.  These pin the source features being represented;
# they are not a claim of native instruction, compiler or dispatch fidelity.
SOURCE_REFERENCES = (
    "repository://refs/ops-transformer/attention/"
    "fused_infer_attention_score/op_kernel/arch35/fia_kernel_noquant_gqa.h",
    "repository://refs/ops-transformer/attention/"
    "fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_gqa.h",
)


@dataclass(frozen=True)
class StagedGQADecode:
    """Finite analytic GQA decode recipe.

    ``key_tile`` is the K/V row tile.  ``query_tile`` bounds the flattened
    ``(token, query_head_in_group)`` rows in one owner task.  ``kv_splits`` is
    an explicit requested partition count, reduced to real key tiles.
    ``slots`` is the bounded key/value pipeline window.
    Optional ``target_cores`` increases partitions to fill the declared owner
    budget after batch/head/query parallelism. It is an analytic mapping;
    captured dispatch should continue to declare its explicit split count.
    ``kv_mapping='fai_flash_decode'`` instead uses the pinned FAI host budget
    across query boundaries and a global producer/merge barrier. Device stages
    remain analytical; selecting host ownership does not select a native kernel.
    ``fai_host`` also selects the regular speculative-query host path when the
    source occupancy threshold disables flash splitting.
    """

    key_tile: int = 128
    query_tile: int = 64
    kv_splits: int = 1
    slots: int = 2
    target_cores: int | None = None
    kv_mapping: str = "uniform"

    def __post_init__(self):
        _positive_int(self.key_tile, "key_tile")
        _positive_int(self.query_tile, "query_tile")
        _positive_int(self.kv_splits, "kv_splits")
        _positive_int(self.slots, "slots")
        if self.target_cores is not None:
            _positive_int(self.target_cores, "target_cores")
        if self.kv_mapping not in {"uniform", "fai_flash_decode", "fai_host"}:
            raise ModelingError("attention_decode_unknown_kv_mapping")
        if self.kv_mapping in {"fai_flash_decode", "fai_host"} and self.kv_splits != 1:
            raise ModelingError("fai_decode_mapping_owns_splits:remove_kv_splits")

    def spec(self, op: OpSpec) -> "_Spec":
        return _spec(op, self)

    def work(self, op: OpSpec, *, include_mte1: bool = False,
             read: str = "mte2", vread: str | None = None,
             write: str | None = "mte3", operand: str | None = "mte1") -> Work:
        """Physical work of the lowered mechanism, independent of service rates.

        ``read``/``vread`` are the actual producer and vector read pipes.  The
        default generic pair keeps the historical verification ledger; a
        bound geometry supplies the split AIC/AIV pipe names.
        """
        return _total_work(self.spec(op), include_mte1=include_mte1,
                           read=read, vread=vread, write=write, operand=operand)

    def kv_access(self, op: OpSpec) -> KVCacheAccess:
        """Declared KV storage/page/scale access for this operator geometry."""
        return self.spec(op).kv_cache

    def native_plan(self, op: OpSpec) -> dict | None:
        """Typed ordinary paged-FIA ports and geometry for the actual op."""
        spec = self.spec(op)
        cache = spec.kv_cache
        if not cache.paged or cache.key_format is None or cache.value_format is None:
            return None
        plan = paged_fia_plan(
            requests=spec.requests,
            query_lengths=tuple(query for query, _ in spec.requests),
            kv_lengths=tuple(context + query
                             for query, context in spec.requests),
            page_size=cache.page_size,
            query_heads=spec.query_heads,
            kv_heads=spec.kv_heads,
            head_dim=spec.head_dim,
            query_dtype=spec.query_dtype,
            output_dtype=spec.query_dtype,
            key_format=cache.key_format,
            value_format=cache.value_format,
            phase=getattr(op, "phase", "decode"),
            mask="causal",
            return_lse=False,
            output_prepare=None,
        )
        outputs = []
        full_tiles = tail_tiles = 0
        for output in spec.outputs:
            for contribution in output.contributions:
                last_start = (contribution.tile_start
                              + contribution.tile_count - 1) * spec.key_tile
                last_valid = min(spec.key_tile,
                                 output.horizon - last_start)
                for tile in range(contribution.tile_count):
                    start = (contribution.tile_start + tile) * spec.key_tile
                    valid = min(spec.key_tile, output.horizon - start)
                    if valid < spec.key_tile:
                        tail_tiles += 1
                    else:
                        full_tiles += 1
                outputs.append({
                    "request": output.request, "kv_head": output.kv_head,
                    "tiles": contribution.tile_count,
                    "tile_start": contribution.tile_start,
                    "last_valid": last_valid,
                    "tail": last_valid < spec.key_tile,
                })
        plan["strategy"] = {
            "key_tile": spec.key_tile,
            "query_tile": spec.query_tile,
            "kv_splits": spec.kv_splits,
            "slots": spec.slots,
            "key_tiles": full_tiles + tail_tiles,
            "full_tiles": full_tiles,
            "tail_tiles": tail_tiles,
            "outputs": outputs,
            "transactions": {
                "page_reads": plan["work"]["pages"],
                "key_page_reads": plan["work"]["key_page_reads"],
                "value_page_reads": plan["work"]["value_page_reads"],
                "static_kv_read": True,
                "state_writes": 0,
            },
            "compute": {"qk": "fia", "softmax": "online",
                        "pv": "fia", "causal": True},
        }
        plan["cache_access"] = cache.report()
        plan["operator"] = {"op_id": op.op_id, "stage": op.stage,
                            "layer": op.layer, "phase": op.phase}
        return plan

    def build(self, op: OpSpec, geometry: Geometry | None = None) -> OperatorPlan:
        """Verification binding using this recipe's options and work ledger."""
        spec = self.spec(op)
        if geometry is None:
            read, vread, write, operand = "mte2", "mte2", "mte3", "mte1"
            mte1 = False
        else:
            read = read_role(geometry.resource_units)
            vread = read_role(geometry.resource_units, aiv=0)
            write = "mte3" if "mte3" in geometry.resource_units else None
            operand = "mte1" if "mte1" in geometry.resource_units else None
            mte1 = operand is not None
            if spec.kv_cache.needs_conversion and operand is None:
                raise ModelingError(
                    "attention_decode_conversion_requires_mte1_operand_copy")
            if spec.kv_cache.needs_conversion and write is None:
                raise ModelingError(
                    "attention_decode_conversion_requires_mte3_result_copy")
        metadata = {"kv_cache_access": spec.kv_cache.report()}
        native = self.native_plan(op)
        if native is not None:
            metadata["native_paged_fia"] = native
        return OperatorPlan(
            self.work(op, include_mte1=mte1, read=read, vread=vread,
                      write=write, operand=operand),
            lambda context: _GQADecodeProgram(context, self, spec),
            metadata=metadata,
        )

    def lower(self, context: RecipeContext) -> KernelProgram:
        """Lower one kernel setup and the finite staged body."""
        spec = self.spec(context.op)
        return KernelProgram(
            (
                Stage("setup", latency=context.setup_ms),
                Stage("body", after=("setup",),
                      program=_GQADecodeProgram(context, self, spec)),
            ),
            minimum_ms=context.minimum_ms,
        )


def build(op: OpSpec, geometry: Geometry | None = None) -> OperatorPlan:
    """Independent verification binding for the default recipe options."""
    return StagedGQADecode().build(op, geometry)


# ---------------------------------------------------------------------------
# Executable programs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProducerTaskProgram:
    context: RecipeContext
    recipe: StagedGQADecode
    spec: _Spec
    output: _Output
    contribution: _Contribution
    core: int
    signal: Callable[[], None] | None = None

    def start(self, engine, name, ready, *, owner=""):
        program = _producer_program(
            self.context, self.recipe, self.spec, self.output,
            self.contribution, self.core,
        )
        end = program.start(engine, name, ready, owner=owner)
        if self.signal is None:
            return end
        return engine.submit(
            name + ":split_produced", after=(end,),
            on_complete=lambda _: self.signal(), owner=owner,
        )


def _producer_program(context: RecipeContext, recipe: StagedGQADecode,
                      spec: _Spec, output: _Output,
                      contribution: _Contribution, core: int) -> KernelProgram:
    _query, context_len = spec.requests[output.request]
    dim, rows = spec.head_dim, output.rows
    q_bytes = rows * dim * spec.query_bytes
    read = _read_role(context)
    write = _write_role(context)
    operand = _operand_role(context)
    cube = ("cube",)
    vector = ("vector",)
    copy_in = (read,)
    vector_copy_in = (_read_role(context, vector=True),)
    copy_out = (write,) if write else ()
    operand_role = (operand,) if operand else ()

    q_l1 = _allocation(context, core, "l1", "q_l1", q_bytes)
    q_l0a = _allocation(context, core, "l0a", "q_l0a", q_bytes)
    ub_state = _allocation(context, core, "ub", "online_state",
                           rows * (dim * _FP32_BYTES + 2 * _FP32_BYTES))
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
            vector_read = _read_role(context, vector=True)
            page_work[vector_read] = page_work.get(vector_read, 0) + page_scale
        if page_scale:
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

    @lru_cache(maxsize=4 * recipe.slots + tail_count)
    def template(valid: int, pairs: int, active_rows: int, slot: int):
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
        score_acquire = _allocations((scores,))
        prob_acquire = _allocations((probs,))
        pv_acc_acquire = _allocations((pv_acc,))
        score_release = _allocations((scores,))
        prob_release = _allocations((probs,))

        qk = Work(
            matrix={spec.query_dtype: 2 * dim * pairs},
            resources={"vector": 2 * pairs},
        )
        softmax = Work(resources={"vector": 5 * pairs + active_rows})
        pv = Work(matrix={spec.query_dtype: 2 * dim * pairs})
        # Combine the tile-local numerator with the running numerator using
        # their maxima: two scalings plus one add per output element.
        online = Work(resources={"vector": active_rows * (3 * dim + 9)})
        stages_local = []

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
        stages_local.append(Stage(
            "qk", context.bind(qk), after=(k_ready,),
            acquire=score_acquire, free=k_free, placement={
                **_placement(context, core, *cube),
                **_placement(context, core, *vector)}))
        stages_local.append(Stage(
            "softmax", context.bind(softmax), after=("qk",),
            acquire=prob_acquire, free=score_release,
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
                acquire=_allocations((v_payload_ub,)) + tuple(pv_acc_acquire),
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
                acquire=_allocations((v_l1, v_l0b)) + tuple(pv_acc_acquire),
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
                  free=v_free + _allocations((p_l0a,)) + tuple(pv_acc_acquire),
                  placement=_placement(context, core, *cube)),
            Stage("online_update", context.bind(online), after=("pv",),
                  free=prob_release, previous=("online_update",),
                  placement=_placement(context, core, *vector)),
        ))
        return tuple(stages_local)

    contributions = len(output.contributions)
    use_compact = len(set(spec.requests)) <= 1 and contributions == 1
    # A head, the actual causal tail, and two complete slot periods are the
    # structural minimum. The compactor verifies convergence before folding.
    min_periodic_tiles = 1 + tail_count + 2 * recipe.slots
    if use_compact and contribution.tile_count >= min_periodic_tiles:
        key_program = compact_periodic_pipeline(
            context, contribution.tile_count, recipe.slots, tile,
            head=1, tail=tail_count, threshold=min_periodic_tiles - 1,
            resident=persistent)
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


@dataclass(frozen=True)
class _GQADecodeProgram:
    context: RecipeContext
    recipe: StagedGQADecode
    spec: _Spec

    def start(self, engine, name, ready, *, owner=""):
        outputs = self.spec.outputs
        barriers, remaining = {}, {}
        for index, output in enumerate(outputs):
            if len(output.contributions) > 1:
                barriers[index] = engine.future(
                    f"{name}:split_barrier:{index}")
                remaining[index] = len(output.contributions)

        def signal(index):
            remaining[index] -= 1
            if remaining[index] == 0:
                engine.resolve(barriers[index])

        tasks = tuple(
            (output_index, contribution)
            for output_index, output in enumerate(outputs)
            for contribution in output.contributions
        )
        cores = self.context.core_count
        streams = [[] for _ in range(cores)]
        for index, (output_index, contribution) in enumerate(tasks):
            core = index % cores if contribution.owner is None else contribution.owner
            if core >= cores:
                raise ModelingError("attention_decode_mapping_exceeds_hardware_cores")
            streams[core].append((output_index, contribution))
        producer_events = []
        for core, stream in enumerate(streams):
            if not stream:
                continue
            count = len(stream)

            def tile(index: int, core: int = core, stream=tuple(stream)):
                output_index, contribution = stream[index]
                callback = ((lambda index=output_index: signal(index))
                            if output_index in barriers else None)
                return (Stage("task", program=_ProducerTaskProgram(
                    self.context, self.recipe, self.spec,
                    outputs[output_index], contribution, core, callback,
                )),)

            producer_events.append(PipelineProgram(count, 1, tile).start(
                engine, f"{name}:owner{core}:producer", ready,
                owner=f"{owner}:gqa{core}" if owner else f"gqa{core}",
            ))

        merges = tuple((index, output) for index, output in enumerate(outputs)
                       if len(output.contributions) > 1)
        merge_events = []
        merge_ready = (engine.submit(name + ":all_producers", after=tuple(producer_events),
                                     owner=owner)
                       if self.recipe.kv_mapping in {"fai_flash_decode", "fai_host"} else ready)
        for core in range(min(len(merges), cores)):
            stream = _MergeStream(
                self.context, self.recipe, self.spec,
                merges[core::cores], barriers, core,
            )
            merge_events.append(stream.start(
                engine, f"{name}:owner{core}:merge", merge_ready,
                owner=f"{owner}:gqa_merge{core}" if owner else f"gqa_merge{core}",
            ))
        return engine.submit(
            name + ":complete", after=tuple(producer_events + merge_events),
            owner=owner,
        )


def planned_recipe(**options) -> PlannedRecipe:
    """Return a planned recipe with every declared option preserved.

    The recipe factory registers this callable as ``staged_gqa_decode``.
    ``PlannedRecipe`` applies the work ledger and metadata once through
    ``prepare`` on the caller's engine.
    """
    return PlannedRecipe(StagedGQADecode(**options).build)


__all__ = [
    "IMPLEMENTATION",
    "KVCacheAccess",
    "RECIPE_ID",
    "SOURCE_REFERENCES",
    "StagedGQADecode",
    "build",
    "causal_pairs",
    "planned_recipe",
]
