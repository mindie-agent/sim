"""Semantic mandatory work and optimistic hardware lower bounds.

These are model/token semantics, not a calibrated predictive estimate.  Kept
separate from plan enumeration/scheduling in ``a3_should``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .a3_types import (
    A3InputError, MachineSpec, ModelSpec, ParallelSpec, WorkloadSpec,
    align_up, dtype_bytes,
)

_MIB = 1024 * 1024
_TILE = 64
_LAUNCHES_PER_LAYER = 12


@dataclass(frozen=True)
class SemanticWork:
    prompt_tokens: int
    output_tokens: int
    linear_flops: float
    attention_flops: float
    logits_flops: float
    total_flops: float
    weight_bytes: float
    resident_weight_bytes: float
    kv_bytes: float
    activation_bytes: float
    total_bytes: float
    comm_bytes: float
    linear_positions: int = 0
    decode_positions: int = 0
    per_request: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "linear_flops": self.linear_flops,
            "attention_flops": self.attention_flops,
            "logits_flops": self.logits_flops,
            "total_flops": self.total_flops,
            "weight_bytes": self.weight_bytes,
            "resident_weight_bytes": self.resident_weight_bytes,
            "linear_positions": self.linear_positions,
            "decode_positions": self.decode_positions,
            "kv_bytes": self.kv_bytes,
            "activation_bytes": self.activation_bytes,
            "total_bytes": self.total_bytes,
            "comm_bytes": self.comm_bytes,
            "per_request": dict(self.per_request),
        }


def _attention_pairs(model: ModelSpec, prompt: int, output: int, prefix: int) -> float:
    q = max(0, prompt - prefix)
    decode = max(0, output - 1)
    if q == 0 and output > 0:
        # A KV-only prefix hit has no cached final-position logits.  Recompute
        # the last prompt position once (context = prefix - 1).
        prefill_pairs = max(0, prefix - 1) + 1.0
    else:
        prefill_pairs = q * prefix + q * (q + 1) / 2.0
    decode_pairs = decode * prompt + decode * (decode + 1) / 2.0
    if model.attention == "gqa":
        per_pair = 4.0 * model.num_heads * model.head_dim
    else:
        assert model.mla is not None
        per_pair = 2.0 * model.num_heads * (model.mla.qk_head_dim + model.mla.v_head_dim)
    return per_pair * (prefill_pairs + decode_pairs)


def _active_weight_bytes(model: ModelSpec) -> tuple[float, int, int]:
    """Return (active weight read bytes, active expert count, linear positions unused)."""
    h = model.hidden_size
    wb = dtype_bytes(model.weight_dtype)
    if model.attention == "gqa":
        attn = h * model.num_heads * model.head_dim
        attn += 2 * h * model.num_kv_heads * model.head_dim
        attn += model.num_heads * model.head_dim * h
    else:
        assert model.mla is not None
        m = model.mla
        attn = h * m.q_lora_rank + m.q_lora_rank * model.num_heads * m.qk_head_dim
        attn += h * (m.kv_lora_rank + m.qk_rope_head_dim)
        attn += m.kv_lora_rank * model.num_heads * (m.qk_nope_head_dim + m.v_head_dim)
        attn += model.num_heads * m.v_head_dim * h
    if model.ffn == "dense":
        ffn = 3 * h * model.intermediate_size
        active_experts = 0
    else:
        active_experts = max(1, min(model.top_k, model.num_experts))
        ffn = active_experts * 3 * h * model.moe_intermediate_size
        ffn += model.num_shared_experts * 3 * h * model.moe_intermediate_size
    embedding = model.vocab_size * h * wb
    head = 0.0 if model.tie_word_embeddings else model.vocab_size * h * wb
    active = embedding + head + model.layers * (attn + ffn) * wb
    return float(active), active_experts, model.layers


def semantic_work(model: ModelSpec, workload: WorkloadSpec, parallel: ParallelSpec) -> SemanticWork:
    """Derive mandatory semantic work from model and token counts.

    New linear positions are ``P-K+O-1``: the last prompt position emits the
    first output token.  Prefix KV affects attention only (and one final-position
    recomputation if no cached logits are declared).
    """
    if model.layer_plans:
        from .a3_graph_lowering import catalog_semantic_work

        return catalog_semantic_work(model, workload, parallel)
    prompt_total = sum(request.prompt_tokens for request in workload.requests)
    output_total = sum(request.output_tokens for request in workload.requests)
    linear_positions_total = 0
    decode_positions_total = 0
    attention = 0.0
    kv_bytes = 0.0
    activation_bytes = 0.0
    for request in workload.requests:
        q = max(0, request.prompt_tokens - request.prefix_tokens)
        decode_positions = max(0, request.output_tokens - 1)
        if q == 0 and request.output_tokens > 0:
            # final-position recomputation for a KV-only prefix hit
            linear_positions = 1 + decode_positions
        else:
            linear_positions = q + decode_positions
        linear_positions_total += linear_positions
        decode_positions_total += decode_positions
        attention += _attention_pairs(
            model, request.prompt_tokens, request.output_tokens, request.prefix_tokens,
        )
        kv_tokens = max(1, request.prompt_tokens + request.output_tokens - 1)
        kv_bytes += kv_tokens * model.kv_bytes_per_token_per_layer() * model.layers
        activation_bytes += (
            2.0 * linear_positions * model.hidden_size
            * dtype_bytes(model.activation_dtype) * model.layers
        )
    h = model.hidden_size
    if model.attention == "gqa":
        linear = 2.0 * linear_positions_total * h * (model.num_heads + 2 * model.num_kv_heads) * model.head_dim
        linear += 2.0 * linear_positions_total * model.num_heads * model.head_dim * h
    else:
        assert model.mla is not None
        m = model.mla
        linear = 2.0 * linear_positions_total * (
            h * m.q_lora_rank + m.q_lora_rank * model.num_heads * m.qk_head_dim
            + h * (m.kv_lora_rank + m.qk_rope_head_dim)
            + m.kv_lora_rank * model.num_heads * (m.qk_nope_head_dim + m.v_head_dim)
            + model.num_heads * m.v_head_dim * h
        )
    if model.ffn == "dense":
        linear += 6.0 * linear_positions_total * h * model.intermediate_size
    else:
        routed = linear_positions_total * model.top_k
        linear += 6.0 * routed * h * model.moe_intermediate_size
        linear += 6.0 * linear_positions_total * h * model.moe_intermediate_size * model.num_shared_experts
    logits = 2.0 * output_total * h * model.vocab_size
    flops = linear * model.layers + attention * model.layers + logits
    active_weight, _, _ = _active_weight_bytes(model)
    resident_weight = model.parameter_count() * dtype_bytes(model.weight_dtype)
    tp, pp, ep, dcp = parallel.tp, parallel.pp, parallel.ep, parallel.dcp
    comm = 0.0
    tokens = linear_positions_total
    if tp > 1:
        comm += model.layers * 2.0 * (tp - 1) / tp * tokens * h * dtype_bytes(model.activation_dtype) * 2.0
    if pp > 1:
        comm += (pp - 1) * tokens * h * dtype_bytes(model.activation_dtype)
    if model.ffn == "moe" and ep > 1:
        comm += model.layers * 2.0 * tokens * model.top_k * h * dtype_bytes(model.activation_dtype) * (ep - 1) / ep
    if dcp > 1 and model.attention == "gqa":
        comm += model.layers * tokens * model.num_heads * model.head_dim * dtype_bytes(model.activation_dtype) * (dcp - 1) / dcp
    total_bytes = active_weight + kv_bytes + activation_bytes
    return SemanticWork(
        prompt_tokens=prompt_total, output_tokens=output_total,
        linear_flops=linear * model.layers, attention_flops=attention * model.layers,
        logits_flops=logits, total_flops=flops,
        weight_bytes=active_weight, resident_weight_bytes=resident_weight,
        kv_bytes=kv_bytes, activation_bytes=activation_bytes,
        total_bytes=total_bytes, comm_bytes=comm,
        linear_positions=linear_positions_total,
        decode_positions=decode_positions_total,
        per_request=tuple((request.request_id, float(request.prompt_tokens + request.output_tokens))
                          for request in workload.requests),
    )


def hardware_lower_bound(
    work: SemanticWork, machine: MachineSpec, model: ModelSpec, parallel: ParallelSpec,
) -> dict[str, Any]:
    """Optimistic L for the declared semantics and topology."""
    devices = len(machine.dies)
    peak = machine.peak_flops_for(model.compute_dtype)
    hbm = machine.dies[0].hbm_bw_bytes_per_s
    link = max((item.capacity_bytes_per_ms * 1000.0 for item in machine.links), default=1.0)
    compute_ms = work.total_flops / (devices * peak / 1000.0)
    memory_ms = work.total_bytes / (devices * hbm / 1000.0)
    comm_ms = work.comm_bytes / max(1.0, link / 1000.0)
    # A single-layer dependency bound is optimistic by design: it does not sum
    # layer latency or assume a framework schedule.
    per_layer_flops = work.total_flops / max(1, model.layers)
    dependency_ms = per_layer_flops / (devices * peak / 1000.0)
    lower = max(compute_ms, memory_ms, comm_ms, dependency_ms)
    return {
        "hardware_lower_bound_ms": lower,
        "components_ms": {
            "compute": compute_ms, "memory": memory_ms,
            "communication": comm_ms, "single_layer_dependency": dependency_ms,
        },
        "basis": {
            "devices": devices, "compute_peak_flops_per_s": peak,
            "hbm_bytes_per_s": hbm, "link_bytes_per_s": link,
            "includes_framework_copies": False, "includes_graph_padding": False,
            "includes_queue_wait": False, "is_runnable_target": False,
        },
        "note": "L is an optimistic resource lower bound for semantics, not U and not a global bound.",
    }



__all__ = ["SemanticWork", "semantic_work", "hardware_lower_bound"]
