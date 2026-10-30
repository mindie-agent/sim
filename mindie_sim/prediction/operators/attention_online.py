"""Selected vLLM pointer attention, with typed source work and a dataflow graph.

[Role] Model the one-sequence 2D pointer branch, including CC100 large heads.
[Invariants] Source tensor operations, logical memory operands, matrix work and
  compiler instruction issue are distinct. Every non-matrix operation requires
  an explicit service rate. A CTA owns one physical compute unit; HBM is shared.
  Reductions retain their source input cardinality, never invented shuffle trees.
[Timeouts & Locks] Pure arithmetic. Closed-form totals and lazy CTA/tile programs
  keep memory proportional to active work, not sequence length.
[Side-Effect Policy] No device contexts, observations, files or external effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.program import ParallelProgram, PipelineProgram, Stage
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

IMPLEMENTATION_ID = "vllm_triton_unified_attention_2d_pointer"
RECIPE_ID = "triton_unified_attention_2d"
SELECTED_PATH = "2D pointer; no KV quantization, sinks, bias, window or prefix"
SOURCE_PATH = (
    "/Users/maoxx241/code/vllm-v0.25.1/vllm/v1/attention/ops/"
    "triton_unified_attention.py"
)
HELPERS_PATH = SOURCE_PATH.replace("triton_unified_attention", "triton_attention_helpers")
SOURCE_SHA256 = "a653bd1e1c200be4189804956f680bb462ca80e582b418d279b5afc6ff855311"
HELPERS_SHA256 = "8c730611e7b3c5fb7579ec7846d56a2ab7e348ce06b39136da22072ecc363c95"
SOURCE_CHECKOUT = "752a3a504485790a2e8491cacbb35c137339ad34"
SOURCE_STATE = "current_checkout_clean_not_historical_capture_image"
DEFAULT_COMPUTE_CAPABILITY = (8, 9)
DEFAULT_PAGE_BLOCK_SIZE = 16  # vllm/config/cache.py:44-47, layout input.

# This mapping is a synthetic-control interface, not a rate or hardware default.
SERVICE_ROLES = dict.fromkeys((
    "fp32_add", "fp32_sub", "fp32_mul", "fp32_max", "fp32_div", "fp32_exp",
    "fp32_compare", "fp32_select", "int32_compare", "predicate_and",
    "cvt_fp32_bf16", "cvt_fp32_fp16",
    "source_tl_reduce_max_elements", "source_tl_reduce_sum_elements",
), "vector")


def _pow2(value):
    return 1 << (value - 1).bit_length()


def _floor_sum(n, m, a, b):
    """Sum floor((a*i+b)/m) without enumerating query blocks."""
    result = 0
    while True:
        if a >= m:
            result += (n - 1) * n * (a // m) // 2
            a %= m
        if b >= m:
            result += n * (b // m)
            b %= m
        y = a * n + b
        if y < m:
            return result
        n, b, m, a = y // m, y % m, a, m


@dataclass(frozen=True)
class Schedule:
    query: int
    context: int
    heads: int
    key_heads: int
    head_dim: int
    dtype: str
    causal: bool
    compute_capability: tuple[int, int]
    page_block_size: int
    requested_block: int | None

    @property
    def ratio(self):
        return self.heads // self.key_heads

    @property
    def tuned(self):
        # Wrapper 945-955 checks the capability family, not just CC100 exactly.
        return (self.head_dim == 256 and self.query > 1 and self.ratio <= 16
                and self.compute_capability[0] == 10)

    @property
    def bm(self):
        return 32 if self.tuned else (16 if self.ratio <= 16 else _pow2(self.ratio))

    @property
    def bq(self):
        return self.bm // self.ratio

    @property
    def row_span(self):
        return (self.bm - 1) // self.ratio + 1

    @property
    def tile(self):
        return 128 if self.tuned else (self.requested_block or 32)

    @property
    def dp(self):
        return _pow2(self.head_dim)

    @property
    def kv(self):
        return self.context + self.query

    @property
    def grid_qblocks(self):
        return self.query // self.bq + 1

    @property
    def active_qblocks(self):
        return (self.query + self.bq - 1) // self.bq

    def horizon(self, qblock):
        if not self.causal:
            return self.kv
        return min(self.context + qblock * self.bq + self.row_span, self.kv)

    def tiles(self, qblock):
        return (self.horizon(qblock) + self.tile - 1) // self.tile

    def valid_rows(self, qblock):
        # Non-power-of-two GQA can overlap a query row in adjacent CTAs.
        return min(self.bm, max(0, self.query - qblock * self.bq) * self.ratio)

    @property
    def tile_count(self):
        n = self.active_qblocks - 1
        if not self.causal:
            return self.key_heads * self.active_qblocks * self.tiles(0)
        return self.key_heads * (
            _floor_sum(n, self.tile, self.bq,
                       self.context + self.row_span + self.tile - 1)
            + self.tiles(n))

    @property
    def valid_keys(self):
        n = self.active_qblocks - 1
        if not self.causal:
            return self.key_heads * self.active_qblocks * self.kv
        return self.key_heads * (
            n * self.context + self.bq * n * (n - 1) // 2
            + n * self.row_span + self.horizon(n))

    @property
    def query_rows(self):
        n = self.active_qblocks - 1
        return self.key_heads * (n * self.bm + self.valid_rows(n))

    @property
    def local_state_bytes(self):
        # Only a lower bound, not compiler register/shared occupancy.
        return self.bm * self.dp * 6 + 8 * self.bm

    @property
    def conversion(self):
        return f"cvt_fp32_{self.dtype}"

    def params(self):
        return {
            "query": self.query, "context": self.context, "heads": self.heads,
            "key_heads": self.key_heads, "head_dim": self.head_dim,
            "dtype": self.dtype, "causal": self.causal,
            "compute_capability": list(self.compute_capability),
            "page_block_size": self.page_block_size, "block": self.requested_block,
        }

    def counts(self):
        grid = self.grid_qblocks * self.key_heads
        active = self.active_qblocks * self.key_heads
        q_bytes = self.query_rows * self.head_dim * 2
        kv_bytes = 2 * self.valid_keys * self.head_dim * 2
        # Helpers45-106: five source loads over three distinct int32 addresses.
        # CSE, repeated-address transaction merging and caches remain separate.
        metadata_bytes = grid * 5 * 4
        table_bytes = 4 * self.tile * self.tile_count
        pairs = self.bm * self.tile * self.tile_count
        semantic_pairs = self.heads * (
            self.query * self.context + self.query * (self.query + 1) // 2
            if self.causal else self.query * self.kv)
        return {
            "grid_q_blocks": self.grid_qblocks, "active_q_blocks": self.active_qblocks,
            "grid_programs": grid, "active_programs": active,
            "early_return_programs": grid - active,
            "executed_tiles": self.tile_count, "sum_valid_keys": self.valid_keys,
            "first_qblock_tiles": self.tiles(0),
            "last_qblock_tiles": self.tiles(self.active_qblocks - 1),
            "last_tile_valid_keys": (
                (self.horizon(self.active_qblocks - 1) - 1) % self.tile + 1),
            "executed_row_slots": active * self.bm,
            "query_row_slots": self.query_rows,
            "query_row_slot_overlap": self.query_rows - self.query * self.heads,
            "executed_pair_slots": pairs, "semantic_pairs": semantic_pairs,
            "semantic_dense_flops": 4 * self.head_dim * semantic_pairs,
            "matrix_flops_qk": 2 * self.dp * pairs,
            "matrix_flops_pv": 2 * self.dp * pairs,
            "metadata_read_bytes": metadata_bytes,
            "metadata_total_loads": grid * 5, "metadata_total_addresses": grid * 3,
            "active_metadata_bytes": active * 20,
            "early_return_metadata_bytes": (grid - active) * 20,
            "q_read_bytes": q_bytes, "output_write_bytes": q_bytes,
            "kv_read_bytes": kv_bytes, "block_table_read_bytes": table_bytes,
            "local_state_bytes": self.local_state_bytes,
        }

    def dispatch(self):
        if self.tuned:
            tile_source = "source_wrapper_cc100_large_head_128"
        elif self.requested_block in (None, 32):
            tile_source = "source_wrapper_prefill_default_32"
        else:
            tile_source = "explicit_manual_kernel_not_wrapper_tile"
        return {
            "implementation": IMPLEMENTATION_ID, "recipe": RECIPE_ID,
            "selected_path": SELECTED_PATH, "source_kernel": "kernel_unified_attention",
            "source_sha256": SOURCE_SHA256, "helpers_sha256": HELPERS_SHA256,
            "path": "2d_pointer", "is_3d": False, "use_td": False, "use_td_qo": False,
            "kv_quant_mode": "NONE", "num_seqs": 1, "dtype": self.dtype,
            "causal": self.causal, "use_per_seq_causal": False,
            "use_sinks": False, "use_alibi": False, "use_qq_bias": False,
            "use_softcap": False, "sliding_window": 0, "chunk_lookback": -1,
            "use_mm_prefix": False, "use_rswa": False,
            "compute_capability": list(self.compute_capability),
            "tuned_large_head": self.tuned,
            "wrapper_branch": "cc100_large_head_prefill" if self.tuned else "default_prefill",
            "block_m": self.bm, "block_q": self.bq, "row_span": self.row_span,
            "num_queries_per_kv": self.ratio, "head_dim": self.head_dim,
            "head_dim_padded": self.dp, "tile_size": self.tile,
            "requested_block": self.requested_block, "tile_size_source": tile_source,
            "num_warps": 8 if self.tuned else None,
            "num_stages": 2 if self.tuned else None,
            "page_block_size": self.page_block_size,
            "grid": [self.grid_qblocks, self.key_heads],
            "softmax_state": "fp32_M_L_accumulator",
            "cta_occupancy": "declared_one_cta_per_unit_not_compiled_occupancy",
            "tile_occupancy": "declared_one_tile_in_flight",
        }


def _schedule(params: Mapping):
    required = {"query", "context", "heads", "key_heads", "head_dim", "dtype", "causal"}
    optional = {"compute_capability", "page_block_size", "kv_page_block_size", "block"}
    if required - params.keys() or params.keys() - required - optional:
        raise ModelingError("attention_online_source_parameters_required")
    values = {key: int(params[key]) for key in
              ("query", "context", "heads", "key_heads", "head_dim")}
    if (values["context"] < 0 or any(values[k] < 1 for k in values if k != "context")
            or any(values[k] != params[k] for k in values)):
        raise ModelingError("attention_online_invalid_geometry")
    if values["heads"] % values["key_heads"]:
        raise ModelingError("attention_online_heads_not_divisible_by_key_heads")
    dtype = {"float16": "fp16", "bfloat16": "bf16"}.get(params["dtype"], params["dtype"])
    if dtype not in {"bf16", "fp16"} or type(params["causal"]) is not bool:
        raise ModelingError("attention_online_selected_dtype_or_causal_branch_required")
    cc = params.get("compute_capability", DEFAULT_COMPUTE_CAPABILITY)
    if isinstance(cc, int):
        cc = (cc // 10, cc % 10)
    cc = tuple(cc)
    if len(cc) != 2 or any(type(v) is not int or v < 0 for v in cc):
        raise ModelingError("attention_online_invalid_compute_capability")
    page = params.get("page_block_size", params.get("kv_page_block_size", DEFAULT_PAGE_BLOCK_SIZE))
    block = params.get("block")
    if (type(page) is not int or page < 1 or
            (block is not None and (type(block) is not int or block < 1 or block & (block - 1)))):
        raise ModelingError("attention_online_invalid_page_or_tile_shape")
    return Schedule(**values, dtype=dtype, causal=params["causal"],
                    compute_capability=cc, page_block_size=page, requested_block=block)


def _tile_work(s):
    """Source tensor-operation inventory, before any target-specific lowering."""
    rows, scores, accum = s.bm, s.bm * s.tile, s.bm * s.dp
    flops = 2 * rows * s.tile * s.dp
    return {
        # helpers314-317 broadcasts the noncausal (1,T) comparison over rows.
        "mask": Work(resources={"int32_compare": scores if s.causal else s.tile,
                                "predicate_and": rows + scores}),
        "qk": Work(matrix={s.dtype: flops}),
        "score_scale": Work(resources={"fp32_mul": scores}),
        "score_mask": Work(resources={"fp32_select": scores}),
        "max_reduce": Work(resources={"source_tl_reduce_max_elements": scores}),
        "max_update": Work(resources={"fp32_max": rows}),
        "max_compare": Work(resources={"fp32_compare": rows}),
        "max_clamp": Work(resources={"fp32_select": rows}),
        "alpha_sub": Work(resources={"fp32_sub": rows}),
        "alpha_exp": Work(resources={"fp32_exp": rows}),
        "p_sub": Work(resources={"fp32_sub": scores}),
        "p_exp": Work(resources={"fp32_exp": scores}),
        "p_sum": Work(resources={"source_tl_reduce_sum_elements": scores}),
        "l_scale": Work(resources={"fp32_mul": rows}),
        "l_update": Work(resources={"fp32_add": rows}),
        "rescale": Work(resources={"fp32_mul": accum}),
        "p_cast": Work(resources={s.conversion: scores}),
        "pv": Work(matrix={s.dtype: flops}),
    }


_TILE_AFTER = {
    "mask": (), "qk": ("k_load",), "score_scale": ("qk",),
    "score_mask": ("score_scale", "mask"), "max_reduce": ("score_mask",),
    "max_update": ("max_reduce",), "max_compare": ("max_update",),
    "max_clamp": ("max_compare",), "alpha_sub": ("max_clamp",),
    "alpha_exp": ("alpha_sub",), "p_sub": ("max_clamp",), "p_exp": ("p_sub",),
    "p_sum": ("p_exp",), "l_scale": ("alpha_exp",),
    "l_update": ("l_scale", "p_sum"), "rescale": ("alpha_exp",),
    "p_cast": ("p_exp",), "pv": ("p_cast", "rescale", "v_load"),
}


def _output_work(s):
    return {
        "output_divide": Work(resources={"fp32_div": s.bm * s.dp}),
        "output_cast": Work(resources={s.conversion: s.bm * s.dp}),
    }


def _work(s):
    c = s.counts()
    reads = sum(c[k] for k in ("metadata_read_bytes", "q_read_bytes",
                               "kv_read_bytes", "block_table_read_bytes"))
    result = Work(resources={"hbm": reads + c["output_write_bytes"],
                             "mte2": reads, "mte3": c["output_write_bytes"]})
    for work in _tile_work(s).values():
        result += work.scaled(s.tile_count)
    for work in _output_work(s).values():
        result += work.scaled(s.active_qblocks * s.key_heads)
    return result


def make_op(params: dict):
    s = _schedule(params)
    shape = (s.query, s.kv, s.heads, s.head_dim)
    metadata = {
        **s.counts(), **s.dispatch(), "dispatch": s.dispatch(),
        "source_params": s.params(), "source_path": SOURCE_PATH,
        "helpers_path": HELPERS_PATH, "source_checkout": SOURCE_CHECKOUT,
        "source_state": SOURCE_STATE, "source_kernel_launches": 1,
        "softmax_max_init": "-inf", "softmax_sum_init": 1.0,
        "source_work_kind": "tensor_operation_lanes_and_reduction_input_elements",
        "instruction_issue": "not_derived_from_compiled_triton",
        "unmodeled_source_work": (
            "address and index arithmetic", "preloop masks and register initialization",
            "compiler lowering, reduction trees and synchronization",
            "cache transactions, occupancy and software pipelining",
        ),
        "local_state_scope": (
            "Q_acc_M_L_lower_bound_in_declared_register_pool;"
            "unconstrained_without_register_pool;not_compiled_occupancy"),
        "memory_scope": "logical_source_operands_not_cacheline_or_HBM_transactions",
        "schedule_scope": "dependency_graph_one_tile_one_cta_per_owned_unit",
    }
    op = OpSpec(
        op_id="attention_online", family="attention", phase="operator", stage=0,
        layer=0, requests=(), semantic_shape=shape, local_shape=shape,
        padded_shape=(s.query, (s.kv + s.tile - 1) // s.tile * s.tile, s.heads, s.dp),
        launch_shape=(s.grid_qblocks, s.key_heads), recipe=RECIPE_ID, source_boundary="kernel", dtype=s.dtype,
        layout="contiguous", metadata=tuple(sorted(metadata.items())),
        notes=("Selected one-sequence 2D pointer branch.",
               "Typed source work requires explicit service rates.",
               "CPU structural controls do not establish hardware accuracy."),
    )
    return _work(s).apply(op)


def build(op: OpSpec, geometry: Geometry):
    s = _schedule(dict(op.metadata)["source_params"])
    if (op.recipe != RECIPE_ID or op.dtype != s.dtype or
            tuple(op.local_shape) != (s.query, s.kv, s.heads, s.head_dim)):
        raise ModelingError("attention_online_selected_source_contract_mismatch")
    counts = s.counts()
    tile_work = _tile_work(s)

    def program(context):
        owners = context.core_count
        workers = min(counts["grid_programs"], owners)
        pool = "register" if "register" in geometry.buffers else None

        def stage(name, work, core, after=(), **kwargs):
            roles = tuple(r for r in work.resources if r != "hbm")
            if work.matrix:
                roles += ("cube",)
            return Stage(name, context.bind(work), after=after,
                         placement=context.owned(core, owners, roles), **kwargs)

        def tiles(core, qblock):
            def tile(index):
                valid = min(s.tile, s.horizon(qblock) - index * s.tile)
                table = Work(resources={"hbm": 4 * s.tile, "mte2": 4 * s.tile})
                operand = Work(resources={"hbm": valid * s.head_dim * 2,
                                          "mte2": valid * s.head_dim * 2})
                stages = [stage("page_table", table, core),
                          stage("k_load", operand, core, ("page_table",)),
                          stage("v_load", operand, core, ("page_table",))]
                stages.extend(stage(name, work, core, _TILE_AFTER[name])
                              for name, work in tile_work.items())
                return tuple(stages)
            # Completion joins M/L/acc branches before the next iteration.
            return PipelineProgram(s.tiles(qblock), 1, tile)

        def cta(core, index):
            cta_id = core + index * workers
            qblock = cta_id % s.grid_qblocks
            stages = [stage("metadata", Work(resources={"hbm": 20, "mte2": 20}), core)]
            if qblock * s.bq >= s.query:
                return tuple(stages)
            allocation = ()
            if pool is not None:
                from mindie_sim.prediction.ownership import unit_group
                units = unit_group(core, owners, context.pool_units.get(pool, owners))
                allocation = (Allocation(
                    f"cta_{core}_{index}", context.pool_names[pool], s.local_state_bytes, units),)
            q_bytes = s.valid_rows(qblock) * s.head_dim * 2
            stages.append(stage(
                "q_load", Work(resources={"hbm": q_bytes, "mte2": q_bytes}),
                core, ("metadata",), acquire=allocation))
            stages.append(Stage("tiles", after=("q_load",), program=tiles(core, qblock)))
            previous = "tiles"
            for name, work in _output_work(s).items():
                stages.append(stage(name, work, core, (previous,)))
                previous = name
            stages.append(stage(
                "output", Work(resources={"hbm": q_bytes, "mte3": q_bytes}),
                core, (previous,), free=allocation))
            return tuple(stages)

        return ParallelProgram(tuple(
            PipelineProgram((counts["grid_programs"] - 1 - core) // workers + 1, 1,
                            lambda index, core=core: cta(core, index))
            for core in range(workers)))

    return OperatorPlan(_work(s), program)
