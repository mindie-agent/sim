"""CANN KeyPool caller contract with analytical native-kernel envelopes.

[Role] Preserve BF16 K/gate, FP32 head weights, sentinel and rollback work, and
  bind the pooled indexer to its declared 32-AIC wave program.
[Invariants] Caller shapes and formats are source facts. Native score algebra,
  internal traffic and core scheduling remain explicit analytical
  declarations, never fitted durations. Capacity rows never become fictitious
  valid pools or attention candidates. Local-TP row shards move query rows and
  their sequence metadata together, and only the assembled index plane is
  visible after the group gather.
[Timeouts & Locks] O(requests) scalar counts; no I/O, contexts or locks.
[Side-Effect Policy] Constructs immutable metadata and caller-owned graph nodes.
"""
from dataclasses import replace
import math

from mindie_sim.catalog.index_pool import pool_selection
from mindie_sim.catalog.ops import comm_op, compute_dtype_for
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.prediction.operators.attention.batch_prefix_attention import apply_sparse_prefix_reads
from .index_pool import (
    _row_shard_value, _selection_for, native_shard_sizes, row_shard_tp, row_slice,
)
from mindie_sim.prediction.operators.mla.mla_native import index_layer_norm
from mindie_sim.prediction.mechanism import PlannedRecipe
from .sparse_indexer_pipeline import (
    FP32_ELEMENT_BYTES, INDEXER_WAVE_ROWS, KEYPOOL_LEDGER, POOL_LEDGER,
    KeyPoolRingRecipe, PoolKeyIndexerRecipe, indexer_waves, key_blocks,
    key_pool_work, pool_key_indexer_work,
)
from mindie_sim.prediction.structure_groups import declare_group


FP32_ELEMENT_BYTES = 4
POSITION_AND_GATE_BYTES = 8
SLOT_AND_POSITION_BYTES = 8

IMPLEMENTATION = "vllm_ascend_cann_keypool_caller"
FP8_INDEX_DTYPE = "fp8_e4m3"
FP8_INDEX_DTYPES = frozenset({"fp8", FP8_INDEX_DTYPE})
FP32_BYTES = int(dtype_bytes("fp32"))
FP8_BYTES = int(dtype_bytes("fp8_e4m3"))
INT32_BYTES = int(dtype_bytes("int32"))
INT64_BYTES = int(dtype_bytes("int64"))
INDEX_POOL_SCALES = 1
SUPPORTED_INDEX_DTYPES = frozenset({"bf16", *FP8_INDEX_DTYPES})
#: The native caller consumes FP32 scales (or none) for an FP8 pool row.
SUPPORTED_INDEX_SCALES = frozenset({"fp32", "none"})
#: Checkpoint categories whose native caller keeps BF16 wk/wgate projections.
POOL_PROJECTION_CATEGORIES = ("indexer.pool_projection", "indexer.pool_gate")
SUPPORTED_HEAD_WEIGHTS = frozenset({("bf16", "bf16"), ("fp32", "fp32")})
SUPPORTED_PHASES = frozenset({"prefill", "decode", "draft"})
POOL_KEY_RECIPE = "cann_pool_key_indexer"
KEY_POOL_RECIPE = "cann_key_pool"
ROW_SHARD_GATHER = "ring_allgather"
BLOCK_TABLE_ROW = 128
#: Per-request scalar metadata: prompt/context lengths and slot identities.
REQUEST_METADATA_BYTES = 24
#: Two BF16 projections (key and gate), each a multiply and an add per element.
POOL_PROJECTION_FLOPS = 4
#: Declared copy-tile element count for the non-empty input preparation grid.
COPY_TILE_ELEMENTS = 8192
SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
    "path": "vllm_ascend/attention/indexer_kpool.py",
    "sha256": "c7268fa24836f3ef33f9fb626ece9430c6c752a596e3bc07d230de5e9fb99e9b",
    "lines": "44-106,168-200,315-367,935-950,1112-1150",
    "adapters": {
        "cann_keypool.py": "5deb787e94c4b31f5246747778b2bbe3a9c69447b2ca9701c90b500c0da94696",
        "glm5_next_cann_keypool.py": "b02108fc0ac4e5ec77828abd2f69811511ee226ee90894500943d2175c2b4dd6",
        "prepare_nonempty_keypool_inputs.py": "1864a2affc64dee1f033aeae7ce92d9159948ee3b6bb80b6fa81c739b9745184",
    },
    "row_shard": {"path": "vllm_ascend/attention/indexer_kpool.py",
                  "scope": "configured_caller_and_cost_balanced_TP_exchange"},
    "scope": "native_caller_contract; installed_binary_and_core_schedule_unbound",
}


def recipes():
    """Both recipes dispatch to their own physical program, not one envelope."""
    return {
        KEY_POOL_RECIPE: PlannedRecipe(KeyPoolRingRecipe().build),
        POOL_KEY_RECIPE: PlannedRecipe(PoolKeyIndexerRecipe().build),
    }


def build(op, geometry):
    if op.recipe == KEY_POOL_RECIPE:
        return KeyPoolRingRecipe().build(op, geometry)
    return PoolKeyIndexerRecipe().build(op, geometry)


def prepare_model(model, runtime, binding):
    options = binding.options
    pools = [p.index_kpool for p in model.layer_plans if p.index_kpool > 1]
    if not pools:
        raise ModelingError("native_keypool_requires_pooled_index_layers")
    capacity = options.get("tail_capacity", max(pools))
    columns = options.get("block_table_columns",
                          math.ceil(runtime.max_model_len / BLOCK_TABLE_ROW))
    if (not isinstance(capacity, int) or capacity < max(pools)
            or not isinstance(columns, int) or columns < 1):
        raise ModelingError("native_keypool_requires_positive_table_and_tail_capacity_at_least_pool")
    metadata = {**dict(model.metadata), "native_keypool_caller": {
        "implementation": IMPLEMENTATION, "tail_capacity": capacity,
        "block_table_columns": columns,
        "block_table_columns_explicit": "block_table_columns" in options,
        "row_shard_tp": _explicit_width(options, runtime),
        "source": SOURCE}}
    # The analytic prefill fallback executes the checkpoint's BF16 wk/wgate
    # projections.  Bind those two categories to the resident native format so
    # prefill and the native decode caller read the same checkpoint weights;
    # an explicit caller override (including a parent role) still wins.
    native_format = str(model.quant.native_format or "bf16")
    explicit = dict(model.quant.category_overrides)

    def declared(category):
        names = category.split(".")
        return any(".".join(names[:index + 1]) in explicit
                   for index in range(len(names)))

    additions = tuple(
        (category, (("weight_format", native_format),
                    ("activation_format", native_format)))
        for category in POOL_PROJECTION_CATEGORIES
        if not declared(category))
    quant = (replace(model.quant, category_overrides=(
        *model.quant.category_overrides, *additions))
        if additions else model.quant)
    return replace(model, quant=quant, metadata=tuple(sorted(metadata.items())))


def _supported(views, plan, model, quant):
    """Native causal TND scorer with BF16 projections and the declared cache dtype."""
    from mindie_sim.catalog.quant import projection_quant
    # The builder's per-layer prepared quant is the operand contract; a
    # caller-supplied model-level preset may still be W8A8.
    resolved = projection_quant(quant, plan)
    weights = compute_dtype_for(resolved, "indexer.weights")[:2]
    return bool(
        plan.index_kpool_compress and plan.index_kpool > 1
        and plan.index_input == "hidden" and not plan.compress_ratio
        and not plan.qk_rope_head_dim
        and model.activation_dtype == "bf16"
        and resolved.index_dtype in SUPPORTED_INDEX_DTYPES
        and (resolved.index_dtype not in FP8_INDEX_DTYPES
             or resolved.index_scale_dtype in SUPPORTED_INDEX_SCALES)
        and all(v.phase in SUPPORTED_PHASES and v.q_len >= 1 for v in views)
        and weights in SUPPORTED_HEAD_WEIGHTS)


def supported(*, views, plan, model, quant):
    """Public preflight for the native caller's declared call domain."""
    return _supported(views, plan, model, quant)


def _contract(b, views, plan, quant):
    if not _supported(views, plan, b.model, b.quant):
        raise ModelingError("native_keypool_caller_requires_bf16_nope_decode_and_declared_head_weights")
    return dict(b.model.metadata)["native_keypool_caller"]


def _pooled_scale_bytes(quant):
    """Declared scale bytes of one FP8 pool row (FP32 or none)."""
    if quant.index_dtype not in FP8_INDEX_DTYPES:
        return 0
    if str(quant.index_scale_dtype) in {"", "none"}:
        return 0
    return INDEX_POOL_SCALES * FP32_BYTES


def _pooled_row_bytes(quant, width):
    """One quantized pool row: payload plus its declared group scale."""
    if quant.index_dtype in FP8_INDEX_DTYPES:
        return width * FP8_BYTES + _pooled_scale_bytes(quant)
    return width * int(dtype_bytes(quant.index_dtype))


def _common(**facts):
    return tuple({"implementation": IMPLEMENTATION, "source": SOURCE,
                  "implementation_basis": "caller_contract_with_analytical_resource_envelope",
                  "native_owner_schedule": "declared_wave_program", **facts}.items())


def _planned(b, name, deps, shape, work, *, kernel, family, dtype,
             metadata=(), recipe=None):
    """Bind one native occurrence to exactly the work its program consumes."""
    op_id = b.work(name, deps, shape, family=family, dtype=dtype, recipe=recipe,
                   metadata=metadata + (("kernel", kernel), ("native_kind", kernel)))
    b.ops[-1] = replace(work.apply(b.ops[-1]), source_boundary="kernel")
    return op_id


def _keypool_ledger(views, plan, model, count):
    geometry = tuple(pool_selection(
        v.q_len, v.context, plan.index_topk, plan.index_kpool,
        always_select_tail=plan.index_kpool_always_select_tail) for v in views)
    completed = sum(g.completed_pools for g in geometry)
    width = plan.index_dim
    return geometry, {
        "batch": len(views),
        "steps": max((v.q_len for v in views), default=1),
        "pool_size": plan.index_kpool,
        "width": width,
        "completed_pools": completed,
        "projection_flops": (float(POOL_PROJECTION_FLOPS) * count * width
                             * model.hidden_size),
        "hidden_bytes": float(count) * model.hidden_size * dtype_bytes(
            model.activation_dtype),
        "weight_bytes": 2.0 * width * model.hidden_size * dtype_bytes(
            model.activation_dtype),
    }


def producer(b, *, source, views, plan, model, quant, padded_tokens, **options):
    declaration = _contract(b, views, plan, quant)
    count, width, pool, batch = padded_tokens, plan.index_dim, plan.index_kpool, len(views)
    geometry, ledger = _keypool_ledger(views, plan, model, count)
    completed = ledger["completed_pools"]
    capacity = min(count, count // pool + batch)
    state_bytes = batch * pool * 2 * width * FP32_ELEMENT_BYTES
    common = _common(valid_query_rows=sum(v.q_len for v in views), padded_query_rows=count,
                     pool_size=pool, completed_pools=completed, pooled_capacity=capacity,
                     query_lengths=tuple(v.q_len for v in views),
                     tail_capacity=declaration["tail_capacity"])
    start = len(b.ops)
    gather = b.work("cann_keypool_state_gather", (source,), (batch, pool, 2 * width),
        read=state_bytes + batch * pool * POSITION_AND_GATE_BYTES, write=state_bytes, family="copy", dtype="fp32",
        metadata=common + (("state_shape", (1 + 2 * batch, pool, 2 * width)),
                           ("state_input_pages", (1, batch + 1)),
                           ("temporary_state_bytes", (1 + 2 * batch) * pool * 2 * width * FP32_ELEMENT_BYTES)))
    # The two-stage native program: BF16 wk/wgate on the AIC owners, a workspace
    # join, then one token-serial FP32 ring per sequence on the AIV owners.
    pooled = _planned(
        b, KEY_POOL_RECIPE, (source, gather), (count, width, model.hidden_size),
        key_pool_work(ledger), kernel="KeyPool", family="key_pool", dtype="bf16",
        recipe=KEY_POOL_RECIPE, metadata=common + (
            (KEYPOOL_LEDGER, ledger),
            ("projection_formats", {"hidden": "bf16", "wk": "bf16", "gate": "bf16"}),
            ("state_dtype", "fp32"), ("output_shape", (capacity, width)),
            ("valid_output_rows", completed),
            ("state_io_basis", "full_request_pool_reference_bound"),
            ("ring_steps", ledger["steps"]), ("ring_owners_per_step", 1),
            ("internal_traffic", "declared_aic_projection_and_aiv_ring")))
    tail_key = tail_gate = None
    if declaration["tail_capacity"] > pool:
        retained_width = min(declaration["tail_capacity"], max(v.q_len for v in views))
        rows = batch * retained_width
        gathered = b.work("cann_keypool_retained_hidden", (pooled, source), (rows, model.hidden_size),
            read=rows * (model.hidden_size * 2 + SLOT_AND_POSITION_BYTES), write=rows * model.hidden_size * 2,
            family="copy", dtype="bf16", metadata=common)
        outputs = []
        for name in ("key", "gate"):
            projected = b.work("cann_keypool_rollback_" + name, (gathered,), (rows, width, model.hidden_size),
                flops=2 * rows * width * model.hidden_size,
                read=(rows * model.hidden_size + width * model.hidden_size) * 2,
                write=rows * width * 2, family="gemm", dtype="bf16", metadata=common)
            if name == "key":
                # ``aclnnLayerNorm`` consumes the BF16 projected key directly
                # and keeps FP32 accumulation in UB; no external Cast pair.
                outputs.append(index_layer_norm(
                    b, source=projected, rows=rows, width=width, native=True))
            else:
                outputs.append(b.vector(
                    "cann_keypool_rollback_" + name + "_fp32", (projected,),
                    rows * width, 1, family="cast", dtype="fp32",
                    shape=(rows, width), read_bytes_override=rows * width * 2,
                    write_bytes_override=rows * width * FP32_ELEMENT_BYTES, metadata=common))
        tail_key = outputs[0]
        tail_gate = outputs[1]
        valid_tail_rows = sum(min(v.q_len, retained_width) for v in views)
        tail_mode = "mtp_rollback_reprojection"
    else:
        rows = batch * pool
        valid_tail_rows = sum((v.context + v.q_len) % pool for v in views)
        tail_mode = "cann_state_tail"
    deps = (pooled, tail_key, tail_gate) if tail_key else (pooled,)
    pooled_row = _pooled_row_bytes(quant, width)
    tail_row_bytes = 2 * width * FP32_BYTES
    stored = b.work("cann_keypool_cache_write", deps, (rows + capacity, width),
        read=valid_tail_rows * tail_row_bytes + completed * width * 2
             + (rows + capacity) * INT64_BYTES,
        write=valid_tail_rows * tail_row_bytes + completed * pooled_row,
        family="cache_store", dtype=quant.index_dtype, metadata=common + (
            ("tail_mode", tail_mode), ("tail_launch_rows", rows), ("valid_tail_rows", valid_tail_rows),
            ("pooled_launch_rows", capacity), ("masked_pooled_rows", capacity - completed),
            ("tail_dtype", "fp32"), ("cache_visible", True),
            ("pooled_row_bytes", pooled_row),
            ("pooled_payload_bytes", width * int(dtype_bytes(quant.index_dtype))),
            ("pooled_scale_bytes", _pooled_scale_bytes(quant)),
            ("pooled_scale_dtype", "fp32" if _pooled_scale_bytes(quant) else "none"),
            ("cache_dtype", quant.index_dtype)))
    declare_group(b.ops, key=b.prefix + ".key_pool", role="key_pool", family="pooled_indexer",
        members=tuple(op.op_id for op in b.ops[start:]), entries=(gather,), anchor=stored,
        view={"tokens": count, "valid_tokens": sum(v.q_len for v in views), "width": width,
              "pool_size": pool, "completed_pools": completed, "pooled_capacity": capacity,
              "tail_capacity": declaration["tail_capacity"],
              "cache_dtype": quant.index_dtype, "state_dtype": "fp32"},
        implementation={"native_kind": "KeyPool", "conditions": {
            "caller": IMPLEMENTATION,
            "native_core_schedule": "aic_projection_then_aiv_token_serial_ring"}},
        source=SOURCE, coverage={"kind": "native_caller_declared_wave_program"})
    return stored, None


def _candidate_pools(views, plan):
    """Pool rows the selection can scan: the populated request cache extent."""
    pool = max(1, int(plan.index_kpool))
    return max((max(1, (int(view.kv_len) if plan.noncausal_attention
                        else int(view.context) + int(view.q_len)) // pool)
                for view in views), default=1)


def _explicit_width(options, runtime):
    width = _row_shard_value(options)
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    return width if width is not None else _row_shard_value(metadata)


def _row_shard(views, plan, *, options, runtime):
    valid = tuple(v for v in views if not v.padding)
    rows = sum(v.q_len for v in valid)
    explicit = _explicit_width(options, runtime)
    if explicit is not None:
        width = row_shard_tp(options=options, runtime=runtime)
        q, r = divmod(rows, width)
        return tuple(q + (i < r) for i in range(width))
    parallel = getattr(runtime, "parallel", None)
    if (getattr(parallel, "pcp", 1) != 1 or getattr(parallel, "dcp", 1) != 1
            or (getattr(runtime, "graph_mode", "none") not in {"none", "off", "disabled", "eager"}
                and (runtime.graph_capture_mode == "full" or
                     _captured(views, runtime)))):
        return (rows,)
    return native_shard_sizes(valid, plan.index_kpool, getattr(parallel, "tp", 1))


def _captured(views, runtime):
    from mindie_sim.prediction.graph import GraphPolicy
    return GraphPolicy().select(views, runtime)[1] != "none"


def _pool_ledger(views, plan, quant, local_tokens, output_width, pool_row_bytes,
                 visited_pools):
    geometry = tuple(_selection_for(v, plan) for v in views)
    heads, width, pool = plan.index_heads, plan.index_dim, plan.index_kpool
    blocks = key_blocks(visited_pools)
    score_dtype = (
        FP8_INDEX_DTYPE if quant.index_dtype in FP8_INDEX_DTYPES else str(quant.index_dtype)
    )
    return geometry, {
        "query_rows": int(local_tokens),
        "score_dtype": score_dtype,
        "wave_rows": INDEXER_WAVE_ROWS,
        "waves": indexer_waves(local_tokens),
        "blocks": blocks,
        "heads": heads,
        "width": width,
        "pool_size": pool,
        "output_width": int(output_width),
        "pool_row_bytes": int(pool_row_bytes),
        "score_pairs": float(sum(g.scanned_pools for g in geometry) * heads),
        "wave_cache_bytes": float(blocks * 1024 * pool_row_bytes),
        "wave_descale_bytes": float(blocks * heads * 1024 * FP32_ELEMENT_BYTES),
        "weight_bytes": float(heads * width * int(dtype_bytes(score_dtype)) + heads * FP32_BYTES),
        "output_bytes": float(local_tokens) * output_width * INT32_BYTES,
    }


def _emit_row_shard_gather(
    b, packed, sizes, divisor, forced, valid_rows, output_width, selection_ids, selection_rows,
):
    row_bytes = output_width * INT32_BYTES
    packet = max(sizes) * row_bytes
    assembled = sum(sizes) * row_bytes
    scratch = b.work("index_exchange_stage", (packed,), (max(sizes), output_width),
                     read=valid_rows * row_bytes, write=packet, family="copy")
    wire = b.add(comm_op(
        op_id=b.prefix + ".index_kpool_row_shard_gather", deps=(scratch,),
        family="allgather", phase=b.common["phase"], recipe=ROW_SHARD_GATHER,
        stage=b.common["stage"], layer=b.common["layer"], requests=b.common["requests"],
        link_bytes=packet * (divisor - 1), shape=(divisor * max(sizes), output_width),
        metadata=tuple(sorted({
            "group_axis": "local_tp" if forced else "tp", "steps": divisor - 1,
            "local_tp_width": divisor, "row_shard_tp": divisor,
            "local_rows": valid_rows, "shard_sizes": sizes,
            "local_payload_bytes": valid_rows * row_bytes,
            "peer_shard_bytes": packet * (divisor - 1),
            "assembled_bytes": assembled, "assembled_shape": (sum(sizes), output_width),
            "padded_exchange_bytes": packet * divisor, "output_dtype": "int32",
            "gather_position": "before_sfa_consumes_index_topk",
        }.items()))))
    gathered = b.work("index_exchange_restore", (wire,), (sum(sizes), output_width),
                      read=assembled, write=assembled, family="copy", dtype="int32",
                      metadata=(
                          ("selected_request_ids", selection_ids),
                          ("selected_tokens_by_request", selection_rows),
                          ("selected_token_rows", sum(selection_rows))))
    return scratch, wire, gathered


def _restore_producer_order(b, prepared, query, weights):
    """Keep KeyPool producer ops (and PCP index gather) before ``prepared``."""
    group = b.prefix + ".key_pool"
    producer_ops = [op for op in b.ops if group in dict(op.metadata).get(
        "structure_subgraph_memberships", ())]
    producer_ids = {op.op_id for op in producer_ops}
    pcp_gathers = [
        op for op in b.ops
        if op.op_id not in producer_ids and op.recipe == "pcp_index_gather"
        and any(dep in producer_ids for dep in op.deps)
    ]
    moved_ids = producer_ids | {op.op_id for op in pcp_gathers}
    remaining = [op for op in b.ops if op.op_id not in moved_ids]
    rewired_producers = [
        replace(op, deps=tuple(dict.fromkeys((*op.deps, query, weights))))
        if op.recipe == "cann_keypool_state_gather" else op
        for op in producer_ops
    ]
    insert_at = next(
        (idx for idx, op in enumerate(remaining) if op.op_id == prepared),
        len(remaining),
    )
    b.ops = remaining[:insert_at] + rewired_producers + pcp_gathers + remaining[insert_at:]


def selection(b, *, query, weights, cache, views, plan, quant, padded_tokens,
              score_mode="linear_weighted_query", hierarchy=None, runtime=None,
              shard_sizes=None, **options):
    declaration = _contract(b, views, plan, quant)
    if score_mode not in {"linear_weighted_query", "per_head_relu"}:
        raise ModelingError("native_keypool_reference_score_mode_unknown")
    tokens, heads, width, pool = padded_tokens, plan.index_heads, plan.index_dim, plan.index_kpool
    pool_topk = plan.index_topk // pool
    forced = _explicit_width(options, runtime) is not None
    if not forced:
        plan = replace(plan, index_kpool_always_select_tail=True)
    output_width = plan.index_topk + (pool - 1 if plan.index_kpool_always_select_tail else 0)
    output_columns = options.get("output_buffer_columns", output_width)
    if type(output_columns) is not int or output_columns < output_width:
        raise ModelingError("native_keypool_output_buffer_too_narrow")
    live_context = max((v.context + v.q_len for v in views), default=0)
    columns = (declaration["block_table_columns"]
               if declaration.get("block_table_columns_explicit", True)
               else max(declaration["block_table_columns"],
                        math.ceil(live_context / BLOCK_TABLE_ROW)))
    pool_row_bytes = _pooled_row_bytes(quant, width)
    sizes = shard_sizes or _row_shard(views, plan, options=options, runtime=runtime)
    divisor = len(sizes)
    rank = options.get("row_shard_rank", 0)
    if type(rank) is not int or not 0 <= rank < divisor:
        raise ModelingError("index_pool_row_shard_rank_out_of_range")
    valid_views = tuple(v for v in views if not v.padding)
    begin = sum(sizes[:rank])
    owned_views = row_slice(valid_views, begin, begin + sizes[rank])
    valid_rows = sizes[rank]
    local_tokens = valid_rows if divisor > 1 else tokens
    # The captured graph appends one sentinel request, so the native call owns
    # one dummy query row beyond the padded partition.
    native_rows = local_tokens + 1
    visited_pools = _candidate_pools(owned_views, plan)
    geometry, ledger = _pool_ledger(owned_views, plan, quant, native_rows,
                                    output_width, pool_row_bytes, visited_pools)
    # The row-shard gather assembles the complete selection plane, so its
    # declared per-request rows cover every request, not only the local badge.
    global_geometry = tuple(_selection_for(v, plan) for v in valid_views)
    selection_ids = tuple(v.request_id for v in valid_views)
    selection_rows = tuple(g.selected_tokens for g in global_geometry)
    scanned = sum(g.scanned_pools for g in geometry)
    selected = sum(g.selected_pools for g in geometry)
    common = _common(valid_query_rows=valid_rows,
        padded_query_rows=local_tokens, native_query_rows=native_rows,
        pool_size=pool, query_lengths=tuple(v.q_len for v in owned_views),
        context_lengths=tuple(v.context for v in owned_views),
        active_requests=len(owned_views), native_requests=len(owned_views) + 1,
        sentinel_rows=1,
        # Padding and the sentinel row never consume a valid query row, so a
        # narrow local shard cannot report a negative dummy count.
        dummy_request_queries=max(0, native_rows - valid_rows),
        local_tp_shard=divisor,
        scanned_pools=scanned, selected_pools=selected,
        safe_reference_scanned_pools=scanned,
        selected_token_rows=sum(g.selected_tokens for g in geometry),
        selected_request_ids=tuple(v.request_id for v in owned_views),
        selected_tokens_by_request=tuple(g.selected_tokens for g in geometry),
        blocks=ledger["blocks"], waves=ledger["waves"],
        score_mode=score_mode, score_algebra_basis="declared_analytical_reference; native_CANN_algebra_unbound")
    input_bytes = (local_tokens * heads * (width + 1) * 2
                   + len(owned_views) * (REQUEST_METADATA_BYTES
                                         + columns * INT32_BYTES))
    prepared = b.work("cann_keypool_nonempty_inputs", (query, weights, cache),
        (native_rows, heads, width),
        read=input_bytes, write=input_bytes, vector=len(owned_views) + 1,
        family="copy", dtype="bf16", metadata=common + (
            ("wave_rows", INDEXER_WAVE_ROWS),
            ("stage_role", "nonempty_inputs"),
            ("block_table_shape", (len(owned_views) + 1, columns)),
            ("tile_grid", max(math.ceil((local_tokens * heads * width) / COPY_TILE_ELEMENTS), 1))))
    indices = _planned(
        b, POOL_KEY_RECIPE, (prepared, cache), (native_rows, heads, width),
        pool_key_indexer_work(ledger), kernel="PoolKeyIndexer",
        family="indexer", dtype=ledger["score_dtype"], recipe=POOL_KEY_RECIPE,
        metadata=common + (
            (POOL_LEDGER, ledger),
            ("output_shape", (native_rows, output_width)),
            ("topk_tokens", plan.index_topk), ("native_output", "expanded_token_indices"),
            ("pool_row_bytes", pool_row_bytes), ("score_dtype", ledger["score_dtype"]),
            ("score_arithmetic", "cube_dot_product_per_query_head"),
            ("cache_reuse", "batch_shared_prefix_l2_deduplicated"),
            ("internal_traffic", "declared_wave_stream_and_score_tile")))
    if hierarchy is not None and runtime is not None:
        by_id = {op.op_id: op for op in b.ops}
        scored = by_id[indices]
        # The physical candidate plane is the wave-deduplicated cache the
        # recipe already priced; a raw scanned-pool row total can exceed the
        # operator's own HBM bytes and drive the shared-prefix delta negative.
        kv_bytes = float(ledger["waves"] * ledger["wave_cache_bytes"])
        b.ops[b.ops.index(scored)] = apply_sparse_prefix_reads(
            scored, views=owned_views,
            row_bytes=float(pool_row_bytes) / float(max(1, pool)),
            runtime=runtime, hierarchy=hierarchy, kv_hbm_bytes=kv_bytes,
            kv_mte2_bytes=kv_bytes)
    # The generic MLA composer creates cache producers first. This caller
    # computes Q/head weights before entering the producer, then gathers state
    # and calls KeyPool. Restore that real enqueue/dependency order while
    # keeping producer ops (and any PCP index gather consuming them) strictly
    # before ``cann_keypool_nonempty_inputs`` so the DAG stays topological.
    _restore_producer_order(b, prepared, query, weights)
    packed = b.work("index_topk", (indices,), (local_tokens, output_width),
        read=selected * pool * INT32_BYTES + (valid_rows + local_tokens) * INT64_BYTES,
        write=local_tokens * output_columns * INT32_BYTES, vector=local_tokens * output_columns,
        family="elementwise", dtype="int32", metadata=common + (
            ("native_kind", "cann_keypool_pack"),
            ("kernel", "cann_keypool_pack"),
            ("selection_capacity", output_width), ("selection_output_dtype", "int32"),
            ("selection_layout", "native_token_indices_then_causal_tail; padded_rows_minus_one"),
            ("pool_id_expansion_in_adapter", False),
            ("causal_tail_rows", sum(g.causal_tail_tokens for g in geometry))))
    gathered = packed
    if divisor > 1:
        scratch, wire, gathered = _emit_row_shard_gather(
            b, packed, sizes, divisor, forced, valid_rows, output_width,
            selection_ids, selection_rows)
    group_ops = (prepared, indices, packed)
    if divisor > 1:
        group_ops += (scratch, wire, gathered)
    # The producer group is re-appended after `start` while restoring enqueue
    # order, so this group names its own members instead of slicing the list.
    declare_group(b.ops, key=b.prefix + ".pool_key_indexer", role="pool_key_indexer",
        family="pooled_indexer",
        members=group_ops,
        entries=(prepared,), anchor=gathered,
        view={"tokens": local_tokens, "valid_tokens": valid_rows,
              "native_tokens": native_rows,
              "global_tokens": tokens, "heads": heads, "width": width, "pool_size": pool,
              "pool_topk": pool_topk, "output_width": output_width,
              "query_dtype": "bf16", "cache_dtype": quant.index_dtype,
              "score_mode": score_mode, "score_dtype": quant.index_dtype,
              "pool_row_bytes": pool_row_bytes, "row_shard_tp": divisor},
        implementation={"native_kind": "PoolKeyIndexer", "conditions": {
            "caller": IMPLEMENTATION,
            "native_core_schedule": "32_aic_wave_quant_mode_0_score_topk_expand",
            "row_shard": ("ascend_indexer_kpool_row_shard" if divisor > 1
                          else "replicated_query_plane")}},
        source=SOURCE, coverage={"kind": "native_caller_declared_wave_program"})
    return gathered
