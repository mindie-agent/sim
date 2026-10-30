"""CANN KeyPool caller contract with analytical native-kernel envelopes.

[Role] Preserve BF16 K/gate, FP32 head weights, sentinel and rollback work.
[Invariants] Caller shapes and formats are source facts. Native score algebra,
  internal traffic and core scheduling remain explicit analytical assumptions.
  Capacity rows never become fictitious valid pools or attention candidates.
[Timeouts & Locks] O(requests) scalar counts; no I/O, contexts or locks.
[Side-Effect Policy] Constructs immutable metadata and caller-owned graph nodes.
"""
from dataclasses import replace
import math

from mindie_sim.catalog.index_pool import pool_selection
from mindie_sim.catalog.ops import compute_dtype_for
from mindie_sim.schema.common import ModelingError
from .mla_native import index_layer_norm
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .program import ResourceEnvelope
from .structure_groups import declare_group

IMPLEMENTATION = "vllm_ascend_cann_keypool_caller"
SOURCE = {
    "revision": "c9aebf229e999f54bedf5b0f13f8e3ffd9989e61",
    "path": "vllm_ascend/attention/indexer_kpool.py",
    "sha256": "7b185038d89cd46d245b4e2a8d768bea51e3d8c82bbe999305d97621a105d498",
    "lines": "624-629,648-705,727-861",
    "adapters": {
        "cann_keypool.py": "57348e3889486d75015a7cb67d3dde3cdd50f003156749887dd86d800d362ba6",
        "glm5_next_cann_keypool.py": "39ab2a610de3db8259b44a1196ae91c3c9eb83103254de8eafab89366d394595",
        "prepare_nonempty_keypool_inputs.py": "1864a2affc64dee1f033aeae7ce92d9159948ee3b6bb80b6fa81c739b9745184",
    },
    "scope": "native_caller_contract; installed_binary_and_core_schedule_unbound",
}


def recipes():
    return {name: PlannedRecipe(build) for name in ("cann_key_pool", "cann_pool_key_indexer")}


def build(op, geometry):
    work = Work(matrix={op.dtype: op.flops} if op.flops else {}, resources={
        "hbm": op.hbm_bytes, "mte2": op.mte2_bytes,
        "mte3": op.mte3_bytes, "vector": op.vector_ops})
    return OperatorPlan(work, ResourceEnvelope().lower, metadata={
        "schedule": "analytical_resource_envelope; native_core_schedule_unbound",
        "traffic": "logical_external_tensors; no_inferred_internal_spill_or_cache_reuse"})


def prepare_model(model, runtime, binding):
    options = binding.options
    pools = [p.index_kpool for p in model.layer_plans if p.index_kpool > 1]
    if not pools:
        raise ModelingError("native_keypool_requires_pooled_index_layers")
    capacity = options.get("tail_capacity", max(pools))
    columns = options.get("block_table_columns", math.ceil(runtime.max_model_len / 128))
    if (not isinstance(capacity, int) or capacity < max(pools)
            or not isinstance(columns, int) or columns < 1):
        raise ModelingError("native_keypool_requires_positive_table_and_tail_capacity_at_least_pool")
    metadata = {**dict(model.metadata), "native_keypool_caller": {
        "implementation": IMPLEMENTATION, "tail_capacity": capacity,
        "block_table_columns": columns, "source": SOURCE}}
    return replace(model, metadata=tuple(sorted(metadata.items())))


def _contract(b, views, plan, quant):
    if (not plan.index_kpool_compress or plan.index_kpool <= 1
            or plan.index_input != "hidden" or plan.compress_ratio or plan.qk_rope_head_dim
            or b.model.activation_dtype != "bf16" or quant.index_dtype != "bf16"
            or any(v.phase != "decode" or v.q_len < 1 for v in views)
            or compute_dtype_for(b.quant, "indexer.weights")[:2] != ("fp32", "fp32")):
        raise ModelingError("native_keypool_caller_requires_bf16_nope_decode_and_fp32_head_weights")
    return dict(b.model.metadata)["native_keypool_caller"]


def _common(**facts):
    return tuple({"implementation": IMPLEMENTATION, "source": SOURCE,
                  "implementation_basis": "caller_contract_with_analytical_resource_envelope",
                  "native_owner_schedule": "not_reconstructed", **facts}.items())


def _native(b, name, deps, shape, *, kernel, **work):
    metadata = work.pop("metadata", ()) + (("kernel", kernel), ("native_kind", kernel))
    result = b.work(name, deps, shape, metadata=metadata, **work)
    b.ops[-1] = replace(b.ops[-1], source_boundary="kernel")
    return result


def producer(b, *, source, views, plan, model, quant, padded_tokens, **options):
    declaration = _contract(b, views, plan, quant)
    count, width, pool, batch = padded_tokens, plan.index_dim, plan.index_kpool, len(views)
    geometry = tuple(pool_selection(v.q_len, v.context, plan.index_topk, pool,
                                   always_select_tail=plan.index_kpool_always_select_tail) for v in views)
    completed = sum(g.completed_pools for g in geometry)
    capacity = min(count, count // pool + batch)
    state_bytes = batch * pool * 2 * width * 4
    common = _common(valid_query_rows=sum(v.q_len for v in views), padded_query_rows=count,
                     pool_size=pool, completed_pools=completed, pooled_capacity=capacity,
                     query_lengths=tuple(v.q_len for v in views),
                     tail_capacity=declaration["tail_capacity"])
    start = len(b.ops)
    gather = b.work("cann_keypool_state_gather", (source,), (batch, pool, 2 * width),
        read=state_bytes + batch * pool * 8, write=state_bytes, family="copy", dtype="fp32",
        metadata=common + (("state_shape", (1 + 2 * batch, pool, 2 * width)),
                           ("state_input_pages", (1, batch + 1)),
                           ("temporary_state_bytes", (1 + 2 * batch) * pool * 2 * width * 4)))
    # The envelope counts logical external tensors and projection arithmetic.
    # It does not materialize K/Gate or infer a private CANN tiling schedule.
    pooled = _native(b, "cann_key_pool", (source, gather), (count, width, model.hidden_size),
        kernel="KeyPool", family="key_pool", dtype="bf16",
        flops=4 * count * width * model.hidden_size,
        vector=count * width * 8 + completed * width * (8 * pool - 2),
        read=(count * model.hidden_size + 2 * width * model.hidden_size) * 2
             + state_bytes + (2 * width + pool * width) * 4 + batch * 16 + (batch + 1) * 4,
        write=completed * width * 2 + state_bytes,
        metadata=common + (("projection_formats", {"hidden": "bf16", "wk": "bf16", "gate": "bf16"}),
            ("state_dtype", "fp32"), ("output_shape", (capacity, width)),
            ("valid_output_rows", completed), ("state_io_basis", "full_request_pool_reference_bound"),
            ("internal_traffic", "not_reconstructed")))
    tail_key = tail_gate = None
    if declaration["tail_capacity"] > pool:
        retained_width = min(declaration["tail_capacity"], max(v.q_len for v in views))
        rows = batch * retained_width
        gathered = b.work("cann_keypool_retained_hidden", (pooled, source), (rows, model.hidden_size),
            read=rows * (model.hidden_size * 2 + 8), write=rows * model.hidden_size * 2,
            family="copy", dtype="bf16", metadata=common)
        outputs = []
        for name in ("key", "gate"):
            projected = b.work("cann_keypool_rollback_" + name, (gathered,), (rows, width, model.hidden_size),
                flops=2 * rows * width * model.hidden_size,
                read=(rows * model.hidden_size + width * model.hidden_size) * 2,
                write=rows * width * 2, family="gemm", dtype="bf16", metadata=common)
            outputs.append(b.vector("cann_keypool_rollback_" + name + "_fp32", (projected,), rows * width, 1,
                family="cast", dtype="fp32", shape=(rows, width),
                read_bytes_override=rows * width * 2, write_bytes_override=rows * width * 4,
                metadata=common))
            if name == "key":
                outputs[-1] = index_layer_norm(b, source=outputs[-1], rows=rows, width=width,
                                               input_dtype="fp32", output_dtype="fp32")
        tail_key = outputs[0]
        tail_gate = outputs[1]
        valid_tail_rows = sum(min(v.q_len, retained_width) for v in views)
        tail_mode = "mtp_rollback_reprojection"
    else:
        rows = batch * pool
        valid_tail_rows = sum((v.context + v.q_len) % pool for v in views)
        tail_mode = "cann_state_tail"
    deps = (pooled, tail_key, tail_gate) if tail_key else (pooled,)
    stored = b.work("cann_keypool_cache_write", deps, (rows + capacity, width),
        read=valid_tail_rows * width * 8 + completed * width * 2 + (rows + capacity) * 8,
        write=valid_tail_rows * width * 8 + completed * width * 2,
        family="cache_store", dtype="bf16", metadata=common + (
            ("tail_mode", tail_mode), ("tail_launch_rows", rows), ("valid_tail_rows", valid_tail_rows),
            ("pooled_launch_rows", capacity), ("masked_pooled_rows", capacity - completed),
            ("tail_dtype", "fp32"), ("cache_visible", True)))
    declare_group(b.ops, key=b.prefix + ".key_pool", role="key_pool", family="pooled_indexer",
        members=tuple(op.op_id for op in b.ops[start:]), entries=(gather,), anchor=stored,
        view={"tokens": count, "valid_tokens": sum(v.q_len for v in views), "width": width,
              "pool_size": pool, "completed_pools": completed, "pooled_capacity": capacity,
              "tail_capacity": declaration["tail_capacity"], "cache_dtype": "bf16", "state_dtype": "fp32"},
        implementation={"native_kind": "KeyPool", "conditions": {
            "caller": IMPLEMENTATION, "native_core_schedule": "not_reconstructed"}},
        source=SOURCE, coverage={"kind": "native_caller_analytical_envelopes"})
    return stored, None


def selection(b, *, query, weights, cache, views, plan, quant, padded_tokens,
              score_mode="linear_weighted_query", **options):
    declaration = _contract(b, views, plan, quant)
    if score_mode not in {"linear_weighted_query", "per_head_relu"}:
        raise ModelingError("native_keypool_reference_score_mode_unknown")
    # The generic MLA composer creates cache producers first. This caller
    # computes Q/head weights before entering _forward_cann, then gathers
    # state and calls KeyPool. Restore that real enqueue/dependency order.
    group = b.prefix + ".key_pool"
    producer_ops = [op for op in b.ops if group in dict(op.metadata).get(
        "structure_subgraph_memberships", ())]
    producer_ids = {op.op_id for op in producer_ops}
    b.ops = [op for op in b.ops if op.op_id not in producer_ids]
    b.ops.extend(replace(op, deps=tuple(dict.fromkeys((*op.deps, query, weights))))
                 if op.recipe == "cann_keypool_state_gather" else op for op in producer_ops)
    tokens, batch, heads, width, pool = padded_tokens, len(views), plan.index_heads, plan.index_dim, plan.index_kpool
    valid = sum(v.q_len for v in views)
    geometry = tuple(pool_selection(v.q_len, v.context, plan.index_topk, pool,
                                   always_select_tail=plan.index_kpool_always_select_tail) for v in views)
    scanned = sum(g.scanned_pools for g in geometry)
    selected = sum(g.selected_pools for g in geometry)
    output_width, columns = plan.index_topk + pool - 1, declaration["block_table_columns"]
    # Graph padding belongs to the appended request: its start is the last
    # real cumulative end, while its new end is padded_tokens + 1.
    dummy_queries = tokens + 1 - valid
    safe_scans = scanned + sum(max(0, min(v.q_len, pool - 1 - v.context)) for v in views) + dummy_queries
    common = _common(valid_query_rows=valid, padded_query_rows=tokens, pool_size=pool,
        query_lengths=tuple(v.q_len for v in views), context_lengths=tuple(v.context for v in views),
        active_requests=batch, native_requests=batch + 1, native_query_rows=tokens + 1,
        sentinel_rows=1, dummy_request_queries=dummy_queries, scanned_pools=scanned,
        safe_reference_scanned_pools=safe_scans, selected_pools=selected,
        selected_token_rows=sum(g.selected_tokens for g in geometry),
        score_mode=score_mode, score_algebra_basis="declared_analytical_reference; native_CANN_algebra_unbound")
    start = len(b.ops)
    input_bytes = tokens * heads * (width + 1) * 2 + batch * (24 + columns * 4)
    padded_bytes = (tokens + 1) * heads * (width + 1) * 2 + (batch + 1) * (24 + columns * 4)
    prepared = b.work("cann_keypool_nonempty_inputs", (query, weights, cache), (tokens + 1, heads, width),
        read=input_bytes, write=padded_bytes, vector=batch + 1,
        family="copy", dtype="bf16", metadata=common + (
            ("block_table_shape", (batch + 1, columns)), ("key_lengths", "max(original,1); dummy=1"),
            ("copy_grid", max(math.ceil(max((tokens + 1) * heads * width,
                     (batch + 1) * columns) / 8192), math.ceil((batch + 1) / 16)))))
    flops = 2 * safe_scans * heads * width if score_mode == "per_head_relu" else 0
    vector = (2 * safe_scans * heads if flops else
              2 * (tokens + 1) * heads * width + 2 * safe_scans * width)
    vector += safe_scans * math.log2(max(2, plan.index_topk // pool)) + selected * pool * 2
    indices = _native(b, "cann_pool_key_indexer", (prepared, cache), (tokens + 1, heads, width),
        kernel="PoolKeyIndexer", family="indexer", dtype="bf16", flops=flops, vector=vector,
        read=safe_scans * width * 2 + padded_bytes,
        write=(tokens + 1) * output_width * 4, metadata=common + (
            ("output_shape", (tokens + 1, output_width)),
            ("topk_tokens", plan.index_topk), ("native_output", "expanded_token_indices"),
            ("internal_traffic", "not_reconstructed"), ("cache_reuse", "none_in_reference_envelope")))
    packed = b.work("index_topk", (indices,), (tokens, output_width),
        read=selected * pool * 4 + valid * 8 + tokens * 8,
        write=tokens * output_width * 4, vector=tokens * output_width,
        family="topk", dtype="int32", metadata=common + (
            ("selection_capacity", output_width), ("selection_output_dtype", "int32"),
            ("selection_layout", "native_token_indices_then_causal_tail; padded_rows_minus_one"),
            ("pool_id_expansion_in_adapter", False), ("causal_tail_rows", sum(g.causal_tail_tokens for g in geometry))))
    declare_group(b.ops, key=b.prefix + ".pool_key_indexer", role="pool_key_indexer", family="pooled_indexer",
        members=tuple(op.op_id for op in b.ops[start:]), entries=(prepared,), anchor=packed,
        view={"tokens": tokens, "valid_tokens": valid, "native_tokens": tokens + 1,
              "heads": heads, "width": width, "pool_size": pool, "output_width": output_width,
              "query_dtype": "bf16", "cache_dtype": "bf16", "score_mode": score_mode},
        implementation={"native_kind": "PoolKeyIndexer", "conditions": {
            "caller": IMPLEMENTATION, "native_core_schedule": "not_reconstructed"}},
        source=SOURCE, coverage={"kind": "native_caller_analytical_envelopes"})
    return packed
