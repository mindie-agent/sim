"""Engram lookup, exchange, gate and optional host-staging path.

[Role]
  Lower V4.1 Engram layers as explicit query construction, ID/count exchange,
  device gather/dequant, reverse response, TP broadcast, gate projection and
  optional CPU->device staging.

[Invariants]
  * ``(max_ngram-1)*n_heads`` lookups per table/token (24 for V4.1); no dedup
    is applied by default.
  * Requests and reverse responses conserve lookup counts.  Diagonal owner
    service is local, not a fabricated remote send.
  * The device table gather path is not replaced by the optional host path.
    Host/source bytes (264 B) do not imply the BF16 response (512 B) or the
    H2D staged bytes.
  * Metadata/ID exchange, response exchange and TP broadcast are separate.
  * Missing CPU decode/H2D timing is surfaced as a measurement gap; it is not
    modeled as a zero-duration operation.

[Timeouts & Locks]
  Pure arithmetic; no I/O or locks.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from typing import Any, Sequence

from .a3_catalog_ops import comm_op, gemm_op, vector_op
from .a3_catalog_types import EngramSpec, QuantSpec
from .a3_model_types import ModelSpec
from .a3_ops import OpSpec
from .a3_types import A3InputError, ParallelSpec, dtype_bytes


def _requests(views: Sequence[Any]) -> tuple[str, ...]:
    return tuple(view.request_id for view in views)


def lower_engram(
    *, prefix: str, plan: Any, stage: int, deps: str, model: ModelSpec,
    parallel: ParallelSpec, quant: QuantSpec, mode: str, padded_tokens: int,
    views: Sequence[Any], runtime: Any,
) -> tuple[list[OpSpec], str]:
    engram = model.engram
    if engram is None:
        raise A3InputError(f"catalog_engram_missing:layer{plan.index}")
    engram.validate()
    ops: list[OpSpec] = []
    requests = _requests(views)
    h = model.hidden_size
    hc_mult = plan.hc_mult or 1
    table_lookup_lists: list[str] = []
    response_lists: list[str] = []
    for table in engram.tables:
        if table.layer_id != plan.index:
            continue
        lookups = max(1, padded_tokens * engram.lookups_per_token_per_table)
        query = vector_op(
            op_id=f"{prefix}.engram_query_t{table.table_id}", deps=(deps,),
            stage=stage, layer=plan.index, requests=requests,
            elements=lookups, ops_per_element=2.0, dtype="fp32",
            family="engram_query", phase=mode, recipe="catalog_engram_query",
            semantic_family="engram_ngram_query_construction",
            shape=(padded_tokens, engram.lookups_per_token_per_table),
            metadata=(("table_id", table.table_id), ("max_ngram", engram.max_ngram),
                      ("heads", engram.heads), ("no_dedup", engram.no_dedup),
                      ("lookups", lookups), ("lookup_id_bytes", engram.id_bytes)),
        )
        id_exchange = comm_op(
            op_id=f"{prefix}.engram_ids_t{table.table_id}", deps=(query.op_id,),
            stage=stage, layer=plan.index, requests=requests,
            family="engram_ids", phase=mode, recipe="catalog_engram_node_exchange",
            link_bytes=float(lookups * engram.id_bytes * (parallel.tp - 1) / max(1, parallel.tp)),
            shape=(lookups,),
            metadata=(("table_id", table.table_id),
                      ("diagonal_is_local", True),
                      ("owner_scope", "node_local"),
                      ("request_count_conserved", lookups)),
        )
        ops.extend([query, id_exchange])
        table_lookup_lists.append(id_exchange.op_id)
        host_bytes = int(getattr(runtime, "engram_transfer_bytes", 0) or 0)
        gather_deps = (id_exchange.op_id,)
        if host_bytes > 0:
            host = comm_op(
                op_id=f"{prefix}.engram_host_t{table.table_id}",
                deps=(id_exchange.op_id,), stage=stage, layer=plan.index,
                requests=requests, family="engram_rh2d", phase=mode,
                recipe="engram_cpu_decode_pinned_staging_h2d",
                link_bytes=float(host_bytes), shape=(host_bytes,),
                hbm_bytes=2.0 * host_bytes,
                metadata=(("scope", "cpu_decode_plus_pinned_h2d"),
                          ("source_table_row_bytes", 264.0),
                          ("response_row_bytes", float(engram.host_response_bytes)),
                          ("cpu_decode_timing", "measurement_gap"),
                          ("h2d_overlap_claim", "none_without_schedule")),
            )
            ops.append(host)
            gather_deps = (host.op_id,)
        device_dtype = {
            "device_bf16": "bf16", "device_int8_fp32": "int8",
            "host_fp8_e8m0": "fp8_e4m3",
        }.get(engram.storage_mode, "bf16")
        gather = vector_op(
            op_id=f"{prefix}.engram_gather_t{table.table_id}", deps=gather_deps,
            stage=stage, layer=plan.index, requests=requests,
            elements=max(1, lookups * engram.width), ops_per_element=2.0,
            dtype=device_dtype, family="engram_gather", phase=mode,
            recipe="catalog_engram_device_gather",
            semantic_family="engram_device_table_gather_dequant",
            read_write=1.0,
            hbm_bytes_override=float(lookups) * engram.row_bytes,
            shape=(lookups, engram.width),
            metadata=(("table_id", table.table_id),
                      ("storage_mode", engram.storage_mode),
                      ("storage_row_bytes", engram.row_bytes),
                      ("response_bytes_per_lookup", engram.host_response_bytes),
                      ("storage_and_response_distinct", True),
                      ("device_table_row_bytes", engram.row_bytes)),
        )
        broadcast = comm_op(
            op_id=f"{prefix}.engram_broadcast_t{table.table_id}",
            deps=(gather.op_id,), stage=stage, layer=plan.index,
            requests=requests, family="engram_broadcast", phase=mode,
            recipe="catalog_engram_tp_broadcast",
            link_bytes=float(lookups * engram.host_response_bytes
                             * (parallel.tp - 1) / max(1, parallel.tp)),
            shape=(lookups, engram.width),
            metadata=(("table_id", table.table_id), ("scope", "tp_group"),
                      ("response_count_conserved", lookups)),
        )
        ops.extend([gather, broadcast])
        response_lists.append(broadcast.op_id)
    if not response_lists:
        raise A3InputError(f"catalog_engram_table_not_at_layer:{plan.index}")
    join = comm_op(
        op_id=f"{prefix}.engram_response_join", deps=tuple(response_lists),
        stage=stage, layer=plan.index, requests=requests, family="barrier",
        phase=mode, recipe="catalog_engram_reverse_matrix_join",
        link_bytes=0.0, shape=(padded_tokens, engram.columns_per_table * engram.width),
        metadata=(("reverse_count_matrix", True), ("tables", len(response_lists))),
    )
    ops.append(join)
    # Gate: one real projection per table, then q/k hyper-connection gate.
    gate_ids: list[str] = []
    for table in engram.tables:
        if table.layer_id != plan.index:
            continue
        gate = gemm_op(
            op_id=f"{prefix}.engram_gate_t{table.table_id}", deps=(join.op_id,),
            stage=stage, layer=plan.index, requests=requests, m=padded_tokens,
            n=(hc_mult + 1) * h, k=engram.columns_per_table * engram.width,
            quant=quant, category="engram_gate", phase=mode, family="gemm",
            semantic_family="engram_gate_projection",
            metadata=(("table_id", table.table_id), ("runtime_dtype", "bf16"),
                      ("source_dtype", "fp8_e4m3")),
        )
        ops.append(gate)
        gate_ids.append(gate.op_id)
    q_gate = gemm_op(
        op_id=f"{prefix}.engram_q", deps=(deps,), stage=stage, layer=plan.index,
        requests=requests, m=padded_tokens, n=hc_mult, k=h, quant=quant,
        category="engram_gate", phase=mode, family="gemm",
        semantic_family="engram_q_gate",
    )
    k_gate = gemm_op(
        op_id=f"{prefix}.engram_k", deps=(deps,), stage=stage, layer=plan.index,
        requests=requests, m=padded_tokens, n=hc_mult, k=h, quant=quant,
        category="engram_gate", phase=mode, family="gemm",
        semantic_family="engram_k_gate",
    )
    ops.extend([q_gate, k_gate])
    apply = vector_op(
        op_id=f"{prefix}.engram_apply", deps=tuple(gate_ids) + (q_gate.op_id, k_gate.op_id),
        stage=stage, layer=plan.index, requests=requests,
        elements=max(1, padded_tokens * (hc_mult + 1) * h),
        ops_per_element=3.0, dtype=model.activation_dtype, family="engram_apply",
        phase=mode, recipe="catalog_engram_gate_apply",
        semantic_family="engram_hyper_connection_gate",
        metadata=(("consumer_width", h), ("hc_mult", hc_mult)),
    )
    add = vector_op(
        op_id=f"{prefix}.engram_add", deps=(apply.op_id, deps), stage=stage,
        layer=plan.index, requests=requests,
        elements=max(1, padded_tokens * h), ops_per_element=2.0,
        dtype=model.activation_dtype, family="elementwise", phase=mode,
        recipe="catalog_engram_residual_add",
        semantic_family="engram_residual_join",
    )
    ops.extend([apply, add])
    return ops, add.op_id


__all__ = ["lower_engram"]
