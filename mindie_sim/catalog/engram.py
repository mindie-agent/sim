"""Engram lookup, exchange, gate and optional host-staging path.

[Role]
  Lower V4.1 Engram layers as explicit query construction, EDP ID exchange,
  local device/UVA host gather-dequant, EDP row all-to-all, TP all-gather,
  WKV projection, optional QuaRot inverse rotation, and fused gate+residual.

[Invariants]
  * ``(max_ngram-1)*n_heads`` lookups per table/token (24 for V4.1); no dedup
    is applied by default.
  * Lookup heads shard uniformly across ``tp * edp`` ranks; non-divisible
    topologies raise ``catalog_engram_heads_not_divisible``.
  * Requests and reverse responses conserve lookup counts.  Diagonal owner
    service is local, not a fabricated remote send.
  * Device table gather reads packed rows (``mte2_bytes``) and writes BF16
    responses (``mte3_bytes``); UVA host offload transfers packed rows over
    H2D and writes BF16 responses on NPU without CPU dequantization.
  * Metadata/ID exchange (INT32 over EDP), EDP row exchange and TP gather are
    separate collectives with exact group axes.

[Timeouts & Locks]
  Pure arithmetic; no I/O or locks.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Sequence

from mindie_sim.catalog.ops import comm_op, compute_dtype_for, gemm_op, vector_op
from mindie_sim.catalog.types import EngramSpec, QuantSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec

#: Native V4.1 hash tensor dtype is INT32 (4 B/id, ``hash_state.py:99``).
ENGRAM_HASH_ID_BYTES = 4
#: RMSNorm(hidden), RMSNorm(key), and channel-weighted dot reductions per branch.
ENGRAM_GATE_REDUCTIONS = 3
#: Elementwise ops per hidden element: h^2, k^2, qw*kw, h*(qw*kw), *k, gate*v, h+gate*v.
ENGRAM_GATE_ELEMENT_OPS = 7
#: Per-branch scalar ops: two rsqrt, scale, signed-root clamp/sqrt, sigmoid, mask.
ENGRAM_GATE_SCALAR_OPS = 19
#: FP32 element width for QuaRot inverse-rotation matmul operands/weights.
FP32_BYTES = 4
#: Pair of 1D channel-weight vectors (q_weight and k_weight) per branch.
ENGRAM_CHANNEL_WEIGHT_TENSORS = 2
#: Input/output MAC factor (2 * M * N * K) for blockwise QuaRot inverse rotation.
MATMUL_FLOPS_PER_MAC = 2


def _requests(views: Sequence[Any]) -> tuple[str, ...]:
    return tuple(view.request_id for view in views)


def _engram_shard_dims(
    engram: EngramSpec, parallel: ParallelSpec, runtime: Any,
    *, node_ranks: int | None = None,
) -> tuple[int, int, int]:
    """Resolve ``(tp_size, edp_size, local_heads)`` and validate head divisibility.

    Matches ``vllm_ascend/models/deepseek_v41/engram/embedding.py``,
    ``vllm_ascend/ascend_config.py:826-835``, and
    ``vllm/distributed/parallel_state.py:1930-1969``: heads shard uniformly
    across ``tp * edp`` ranks.  In Ascend MRV2 (default), ``engram_cpu_offload``
    with ``dp * pcp > 1`` forces ``dp_shared_memory = True`` (``edp = 1``).
    Otherwise ``edp`` spans the node-local DP x PCP shard group.
    """
    tp = max(1, int(parallel.tp))
    dp_pcp = max(1, int(parallel.dp) * int(getattr(parallel, "pcp", 1) or 1))
    meta = dict(getattr(runtime, "metadata", ()) or ())
    cpu_offload = bool(getattr(runtime, "engram_cpu_offload", False))
    v2_runner = meta.get("use_v2_model_runner", True) is not False
    if cpu_offload and dp_pcp > 1 and v2_runner:
        # Ascend MRV2 unconditionally forces shared host tables across DP/PCP
        # peers to eliminate per-step lookup collectives (ascend_config.py:826).
        dp_shared = True
    else:
        dp_shared = bool(meta.get("engram_dp_shared_memory", False))
    explicit_edp = meta.get("engram_dp_size")
    if dp_shared:
        edp = 1
    elif explicit_edp is not None:
        edp = max(1, int(explicit_edp))
    else:
        raw_node_ranks = node_ranks if node_ranks is not None else meta.get("ranks_per_node")
        if raw_node_ranks is not None and int(raw_node_ranks) > 0:
            replica = tp * max(1, int(getattr(parallel, "pp", 1) or 1))
            rpn = int(raw_node_ranks)
            edp = math.gcd(dp_pcp, rpn // replica) if rpn % replica == 0 else 1
        else:
            edp = dp_pcp
    heads = int(engram.lookups_per_token_per_table)
    num_shards = tp * edp
    if heads <= 0 or heads % num_shards != 0:
        raise ModelingError(f"catalog_engram_heads_not_divisible:{heads}%{num_shards}")
    return tp, edp, heads // num_shards


def _edp_group_metadata(
    edp: int, parallel: ParallelSpec,
) -> tuple[tuple[str, Any], ...]:
    """Resolve exact ``Topology.groups`` axis and width metadata for the EDP group."""
    if edp <= 1:
        return (("group_axis", "local"),)
    dp = max(1, int(parallel.dp))
    pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
    edp_pcp = math.gcd(edp, pcp)
    edp_dp = max(1, edp // edp_pcp)
    if edp_dp == 1 and edp_pcp > 1:
        if edp_pcp == pcp:
            return (("group_axis", "pcp"), ("comm_group_size", edp))
        return (
            ("group_axis", "dcp_pcp"),
            ("dcp_pcp_factor", edp_pcp),
            ("comm_group_size", edp),
        )
    if edp_pcp == 1 and edp_dp > 1:
        if edp_dp == dp:
            return (("group_axis", "dp"), ("comm_group_size", edp))
        return (
            ("group_axis", "dcp_dp"),
            ("dcp_dp_width", edp_dp),
            ("comm_group_size", edp),
        )
    return (
        ("group_axis", "cp"),
        ("dcp_tp_gather", 1),
        ("dcp_pcp_factor", edp_pcp),
        ("dcp_dp_width", edp_dp),
        ("comm_group_size", edp),
    )


def _resolve_rotation_block_size(engram: EngramSpec, model: ModelSpec, runtime: Any) -> int:
    """Return declared QuaRot inverse-rotation block size (0 when unrotated)."""
    meta = dict(getattr(runtime, "metadata", ()) or ())
    if meta.get("engram_rotation_block_size") is not None:
        return max(0, int(meta["engram_rotation_block_size"]))
    src = dict(engram.source)
    if src.get("rotation_block_size") is not None:
        return max(0, int(src["rotation_block_size"]))
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    if facts.get("engram_rotation_block_size") is not None:
        return max(0, int(facts["engram_rotation_block_size"]))
    return 0


def _lower_engram_table(
    *, prefix: str, table: Any, engram: EngramSpec, plan: Any, stage: int,
    deps: str, parallel: ParallelSpec, mode: str, padded_tokens: int,
    requests: tuple[str, ...], runtime: Any, response_row_bytes: float,
    node_ranks: int | None = None, edp_slot_tokens: int | None = None,
) -> tuple[list[OpSpec], str]:
    """Lower one Engram table's query, EDP ID gather, local gather, EDP alltoall and TP gather."""
    tp, edp, local_heads = _engram_shard_dims(engram, parallel, runtime, node_ranks=node_ranks)
    tokens = max(0, int(padded_tokens))
    meta = dict(getattr(runtime, "metadata", ()) or ())
    raw_slot = edp_slot_tokens if edp_slot_tokens is not None else meta.get("engram_edp_slot_tokens")
    slot = max(tokens, int(raw_slot)) if (edp > 1 and raw_slot is not None) else tokens
    lookups = tokens * engram.lookups_per_token_per_table
    slot_lookups = slot * engram.lookups_per_token_per_table
    # Each rank embeds ``edp * slot`` gathered rows for its ``local_heads``
    # columns, conserving ``edp * slot * (heads // (tp * edp)) == slot * (heads // tp)``.
    local_lookups = (slot * edp) * local_heads
    query = vector_op(
        op_id=f"{prefix}.engram_query_t{table.table_id}", deps=(deps,),
        stage=stage, layer=plan.index, requests=requests,
        elements=lookups, ops_per_element=2.0, dtype="fp32",
        family="engram_query", phase=mode, recipe="catalog_engram_query",
        semantic_family="engram_ngram_query_construction",
        shape=(tokens, engram.lookups_per_token_per_table),
        metadata=(("table_id", table.table_id), ("max_ngram", engram.max_ngram),
                  ("heads", engram.heads), ("no_dedup", engram.no_dedup),
                  ("lookups", lookups), ("local_lookups", local_lookups),
                  ("lookup_id_bytes", ENGRAM_HASH_ID_BYTES)),
    )
    # ``gather_engram_hashes`` gathers INT32 hashes over the EDP group once
    # across all tables; each table bills its own [slot, heads] INT32 slice.
    id_exchange_bytes = (
        float(slot_lookups * ENGRAM_HASH_ID_BYTES * (edp - 1)) if edp > 1 else 0.0
    )
    edp_axis_meta = _edp_group_metadata(edp, parallel)
    id_exchange = comm_op(
        op_id=f"{prefix}.engram_ids_t{table.table_id}", deps=(query.op_id,),
        stage=stage, layer=plan.index, requests=requests,
        family="engram_ids", phase=mode, recipe="catalog_engram_node_exchange",
        link_bytes=id_exchange_bytes,
        shape=(slot * edp, engram.lookups_per_token_per_table),
        metadata=(("table_id", table.table_id),
                  ("diagonal_is_local", True),
                  ("owner_scope", "engram_dp_group"),
                  *edp_axis_meta,
                  ("edp_size", edp),
                  ("edp_slot_tokens", slot),
                  ("tp_size", tp),
                  ("request_count_conserved", slot_lookups)),
    )
    table_ops: list[OpSpec] = [query, id_exchange]
    cpu_offload = bool(getattr(runtime, "engram_cpu_offload", False))
    packed_bytes = float(local_lookups) * engram.row_bytes
    output_bytes = float(local_lookups) * response_row_bytes
    gather_deps = (id_exchange.op_id,)
    if cpu_offload and packed_bytes > 0.0:
        # Native UVA gather kernel reads packed codes+scales from pinned host
        # memory over H2D and dequantizes on NPU (npu.py:274); host_vector
        # bills only per-lookup address resolution (never CPU dequantization).
        host = vector_op(
            op_id=f"{prefix}.engram_host_t{table.table_id}",
            deps=(id_exchange.op_id,), stage=stage, layer=plan.index,
            requests=requests, family="engram_rh2d", phase=mode,
            recipe="engram_cpu_decode_pinned_staging_h2d",
            elements=0, ops_per_element=0.0, dtype="bf16",
            shape=(local_lookups, engram.width),
            read_bytes_override=0.0, write_bytes_override=0.0,
            metadata=(("resource_work", {"host_memory": packed_bytes,
                                         "host_vector": float(local_lookups),
                                         "h2d": packed_bytes}),),
        )
        table_ops.append(host)
        gather_deps = (host.op_id,)
    device_dtype = {
        "device_bf16": "bf16", "device_int8_fp32": "int8",
        "host_fp8_e8m0": "fp8_e4m3",
    }.get(engram.storage_mode, "bf16")
    quantized_storage = engram.storage_mode != "device_bf16"
    gather_ops_per_elem = 2.0 if (quantized_storage or not cpu_offload) else 0.0
    gather = vector_op(
        op_id=f"{prefix}.engram_gather_t{table.table_id}", deps=gather_deps,
        stage=stage, layer=plan.index, requests=requests,
        elements=local_lookups * engram.width, ops_per_element=gather_ops_per_elem,
        dtype=device_dtype, family="engram_gather", phase=mode,
        recipe="catalog_engram_device_gather",
        semantic_family="engram_device_table_gather_dequant",
        read_bytes_override=0.0 if cpu_offload else packed_bytes,
        write_bytes_override=output_bytes,
        shape=(local_lookups, engram.width),
        metadata=(("table_id", table.table_id),
                  ("storage_mode", engram.storage_mode),
                  ("storage_row_bytes", engram.row_bytes),
                  ("response_bytes_per_lookup", response_row_bytes),
                  ("storage_and_response_distinct", True),
                  ("local_heads", local_heads),
                  ("local_lookups", local_lookups),
                  ("device_table_row_bytes", 0.0 if cpu_offload else engram.row_bytes)),
    )
    table_ops.append(gather)
    broadcast_deps = (gather.op_id,)
    if edp > 1:
        edp_exchange = comm_op(
            op_id=f"{prefix}.engram_response_t{table.table_id}",
            deps=(gather.op_id,), stage=stage, layer=plan.index,
            requests=requests, family="alltoall", phase=mode,
            recipe="catalog_engram_dp_exchange",
            link_bytes=float(output_bytes * (edp - 1) / edp),
            shape=(local_lookups, engram.width),
            metadata=(("table_id", table.table_id),
                      ("scope", "engram_dp_group"),
                      *edp_axis_meta,
                      ("edp_slot_tokens", slot),
                      ("diagonal_is_local", True),
                      ("response_count_conserved", local_lookups)),
        )
        table_ops.append(edp_exchange)
        broadcast_deps = (edp_exchange.op_id,)
    broadcast = comm_op(
        op_id=f"{prefix}.engram_broadcast_t{table.table_id}",
        deps=broadcast_deps, stage=stage, layer=plan.index,
        requests=requests, family="engram_broadcast", phase=mode,
        recipe="catalog_engram_tp_broadcast",
        link_bytes=(
            float(lookups * response_row_bytes * (tp - 1) / tp) if tp > 1 else 0.0
        ),
        shape=(lookups, engram.width),
        metadata=(("table_id", table.table_id), ("scope", "tp_group"),
                  ("group_axis", "tp" if tp > 1 else "local"),
                  *((("comm_group_size", tp),) if tp > 1 else ()),
                  ("response_count_conserved", lookups)),
    )
    table_ops.append(broadcast)
    return table_ops, broadcast.op_id


def lower_engram(
    *, prefix: str, plan: Any, stage: int, deps: str, model: ModelSpec,
    parallel: ParallelSpec, quant: QuantSpec, mode: str, padded_tokens: int,
    views: Sequence[Any], runtime: Any, node_ranks: int | None = None,
    edp_slot_tokens: int | None = None,
) -> tuple[list[OpSpec], str]:
    engram = model.engram
    if engram is None:
        raise ModelingError(f"catalog_engram_missing:layer{plan.index}")
    from mindie_sim.catalog.qwen4_exp import is_qwen4_exp_ple, lower_ple
    if is_qwen4_exp_ple(engram):
        return lower_ple(prefix=prefix, plan=plan, stage=stage, deps=deps, model=model,
                         parallel=parallel, quant=quant, mode=mode,
                         padded_tokens=padded_tokens, views=views, runtime=runtime)
    engram.validate()
    ops: list[OpSpec] = []
    requests = _requests(views)
    tokens = max(0, int(padded_tokens))
    h = model.hidden_size
    hc_mult = plan.hc_mult or 1
    elem_bytes = dtype_bytes(model.activation_dtype)
    response_row_bytes = float(engram.width) * elem_bytes
    response_lists: list[str] = []
    for table in engram.tables:
        if table.layer_id != plan.index:
            continue
        table_ops, broadcast_id = _lower_engram_table(
            prefix=prefix, table=table, engram=engram, plan=plan, stage=stage,
            deps=deps, parallel=parallel, mode=mode, padded_tokens=tokens,
            requests=requests, runtime=runtime, response_row_bytes=response_row_bytes,
            node_ranks=node_ranks, edp_slot_tokens=edp_slot_tokens,
        )
        ops.extend(table_ops)
        response_lists.append(broadcast_id)
    if not response_lists:
        raise ModelingError(f"catalog_engram_table_not_at_layer:{plan.index}")
    join = comm_op(
        op_id=f"{prefix}.engram_response_join", deps=tuple(response_lists),
        stage=stage, layer=plan.index, requests=requests, family="barrier",
        phase=mode, recipe="catalog_engram_reverse_matrix_join",
        link_bytes=0.0, shape=(tokens, engram.columns_per_table * engram.width),
        metadata=(("reverse_count_matrix", True), ("tables", len(response_lists))),
    )
    ops.append(join)
    # Native Engram gate: one WKV projection per table, optional FP32 QuaRot
    # inverse rotation, and fused vector RMSNorm + channel-weighted dot +
    # signed-root sigmoid gate + residual add (single [T, hc_mult, H] store).
    gate_ids: list[str] = []
    for table in engram.tables:
        if table.layer_id != plan.index:
            continue
        gate = gemm_op(
            op_id=f"{prefix}.engram_gate_t{table.table_id}", deps=(join.op_id,),
            stage=stage, layer=plan.index, requests=requests, m=tokens,
            n=(hc_mult + 1) * h, k=engram.columns_per_table * engram.width,
            quant=quant, category="engram_gate", phase=mode, family="gemm",
            semantic_family="engram_gate_projection",
            metadata=(("table_id", table.table_id),
                      ("runtime_dtype", compute_dtype_for(quant, "engram_gate")[0]),
                      ("source_dtype", quant.source_format_for("engram_gate"))),
        )
        ops.append(gate)
        gate_ids.append(gate.op_id)
    branch_rows = tokens * hc_mult
    apply_deps: tuple[str, ...] = tuple(gate_ids) + (deps,)
    rot_block = _resolve_rotation_block_size(engram, model, runtime)
    if rot_block > 0 and branch_rows > 0:
        if h % rot_block != 0:
            raise ModelingError(f"catalog_engram_rotation_block_invalid:{h}%{rot_block}")
        rot_flops = float(MATMUL_FLOPS_PER_MAC * branch_rows * h * rot_block)
        rot_blocks_count = h // rot_block
        rot_read = float(branch_rows * h) * elem_bytes + float(rot_blocks_count * rot_block * rot_block * FP32_BYTES)
        rot_write = float(branch_rows * h * FP32_BYTES)
        rotate = OpSpec(
            op_id=f"{prefix}.engram_rotate", family="gemm", phase=mode,
            stage=stage, layer=plan.index, requests=requests,
            semantic_shape=(branch_rows * rot_blocks_count, rot_block, rot_block),
            local_shape=(branch_rows * rot_blocks_count, rot_block, rot_block),
            padded_shape=(branch_rows * rot_blocks_count, rot_block, rot_block),
            launch_shape=(branch_rows * rot_blocks_count, rot_block, rot_block),
            deps=(deps,), dtype="fp32", flops=rot_flops,
            hbm_bytes=rot_read + rot_write, mte2_bytes=rot_read, mte3_bytes=rot_write,
            recipe="catalog_engram_quarot_inverse",
            matrix_flops=(("fp32", rot_flops),),
            metadata=(
                ("semantic_family", "engram_quarot_inverse_rotation"),
                ("rotation_block_size", rot_block),
                ("hc_mult", hc_mult),
            ),
        )
        ops.append(rotate)
        apply_deps = (*apply_deps, rotate.op_id)
    apply_vec_ops = float(
        branch_rows
        * (
            ENGRAM_GATE_REDUCTIONS * max(0, h - 1)
            + ENGRAM_GATE_ELEMENT_OPS * h
            + ENGRAM_GATE_SCALAR_OPS
        )
    ) if branch_rows > 0 else 0.0
    # Reads hidden [T, C, H], key [T, C, H], value [T, H], and 1D q_weight & k_weight [2, C, H];
    # writes fused output hidden + gate * value [T, C, H] once (engram_gate.py:48).
    apply_read = float(
        (tokens * (ENGRAM_CHANNEL_WEIGHT_TENSORS * hc_mult + 1) + ENGRAM_CHANNEL_WEIGHT_TENSORS * hc_mult) * h
    ) * elem_bytes if tokens > 0 else 0.0
    apply_write = float(branch_rows * h) * elem_bytes
    apply = vector_op(
        op_id=f"{prefix}.engram_apply", deps=apply_deps,
        stage=stage, layer=plan.index, requests=requests,
        elements=branch_rows * h,
        ops_per_element=0.0, vector_ops_override=apply_vec_ops,
        dtype=model.activation_dtype, family="engram_apply",
        phase=mode, recipe="catalog_engram_gate_apply",
        semantic_family="engram_hyper_connection_gate",
        read_bytes_override=apply_read, write_bytes_override=apply_write,
        shape=(tokens, hc_mult, h),
        metadata=(("consumer_width", h), ("hc_mult", hc_mult),
                  ("fused_residual_add", True),
                  ("channel_weight_params", ENGRAM_CHANNEL_WEIGHT_TENSORS * hc_mult * h)),
    )
    ops.append(apply)
    return ops, apply.op_id


__all__ = ["lower_engram"]
