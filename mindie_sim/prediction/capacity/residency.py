"""Resident weights, offload ratios and workspace estimates.

[Role] Derive fixed and staging memory from model terms and placement.
[Invariants] Payload, quantization metadata, workspaces and reserves retain separate
  accounting. Explicit offload policy overrides case metadata.
[Timeouts & Locks] Bounded by layers, experts and active requests. No locks or I/O.
[Side-Effect Policy] Computes memory requirements from the supplied case.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from mindie_sim.catalog.cache import local_kv_heads
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes, term_payload_bytes
from mindie_sim.catalog.weights import layer_weight_terms, stage_weight_terms
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import dtype_bytes

from .contracts import MemoryPolicy


_FFN_INTERMEDIATE_TENSORS = 3
_VISION_SCORE_ELEMENT_BYTES = 4


def _offload_ratio(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"memory_offload_invalid_ratio:{name}:{value}")
    ratio = float(value)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"memory_offload_invalid_ratio:{name}:{value}")
    return ratio


def _resolve_offload_ratios(case: CaseSpec, policy: MemoryPolicy) -> dict[str, float]:
    """Prefer explicit policy ratios, then the case's offload metadata."""
    raw = dict(case.runtime.metadata).get("offload", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("memory_offload_metadata_mapping_required")
    metadata = dict(raw)

    def pick(name: str) -> float:
        declared = getattr(policy, name)
        if declared != 0.0:
            return _offload_ratio(name, declared)
        if name in metadata:
            return _offload_ratio(name, metadata[name])
        return 0.0

    ratios = {name: pick(name) for name in
              ("expert_ddr_ratio", "kvc_ddr_ratio", "kvc_ssd_ratio")}
    if ratios["kvc_ddr_ratio"] + ratios["kvc_ssd_ratio"] > 1.0 + 1e-12:
        raise ValueError("memory_offload_kvc_ratios_exceed_unity")
    return ratios


def _routed_expert_bytes_by_stage(
    case: CaseSpec, layer_stage: tuple[int, ...], *, workspace_tokens: int = 1,
    slot_plans=None,
) -> tuple[list[float], list[float], list[float]]:
    """Routed-expert total, payload and active staging bytes per PP stage."""
    parallel = case.runtime.parallel
    quant = case.model.quant or QuantSpec("bf16")
    result = [0.0] * max(1, parallel.pp)
    payload = [0.0] * max(1, parallel.pp)
    staging = [0.0] * max(1, parallel.pp)
    ep = max(1, int(parallel.ep))
    tokens = max(1, int(workspace_tokens))
    for plan in case.model.layer_plans:
        if plan.ffn != "moe" or int(plan.num_experts) <= 0 or int(plan.moe_intermediate_size) <= 0:
            continue
        stage = layer_stage[plan.index] if plan.index < len(layer_stage) else 0
        terms = [term for term in layer_weight_terms(plan, case.model)
                 if term.category == "routed_expert"]
        slots = (slot_plans or {}).get(plan.index)
        base_count = extra_count = 0
        if slots is not None:
            from mindie_sim.catalog.expert_slots import routed_expert_weight_terms
            terms = list(routed_expert_weight_terms(terms, slots, ep_rank=0))
            base_count = int(slots.base_slots_by_rank[0])
            extra_count = int(slots.redundant_slots_by_rank[0])
        routed = float(sum(
            term_bytes(term, quant, parallel, case.machine.ranks_per_node)
            for term in terms
        ))
        routed_payload = float(sum(
            term_payload_bytes(term, quant, parallel, case.machine.ranks_per_node)
            for term in terms
        ))
        result[stage] += routed
        payload[stage] += routed_payload
        num_exp = int(plan.num_experts)
        k = min(num_exp, max(1, int(plan.top_k)))
        if slots is not None:
            # ``routed`` is already this rank's physical slot bank, so the active
            # staging of this rank is
            #   sum_{slot in rank} W[logical(slot)] * p[logical(slot)]
            # with NO further EP division: ``active_total`` is already local.
            copies = slots.copies_per_expert()
            local_slots = max(1, base_count + extra_count)
            per_slot = routed / float(local_slots)
            row_share = float(k) / float(num_exp)
            best_active = 0.0
            for ids in slots.expert_ids_by_rank:
                active = 0.0
                for expert in ids:
                    share = row_share / float(copies.get(int(expert), 0) or 1)
                    if 0.0 < share < 1.0:
                        active += -math.expm1(float(tokens) * math.log1p(-share))
                    else:
                        active += float(share >= 1.0)
                best_active = max(best_active, active)
            active_layer_bytes = min(routed, per_slot * best_active)
        else:
            active_total = float(num_exp) if k >= num_exp else float(num_exp) * (
                1.0 - ((1.0 - float(k) / float(num_exp)) ** float(tokens))
            )
            local_exp = max(1, (num_exp + ep - 1) // ep)
            active_layer_bytes = (active_total / float(ep)) * (
                routed / float(local_exp))
        staging[stage] = max(staging[stage], min(routed, active_layer_bytes))
    return result, payload, staging



def _weight_ledger_by_stage(
    case: CaseSpec, layer_stage: tuple[int, ...], *, draft_kind: str,
    include_draft: bool, include_engram_tables: bool, enable_shared_expert_dp: bool,
    include_vision: bool | None = None,
    enable_pcp_o_proj_weight_sharding: bool = False,
    dcp_q_replicate: bool = False,
    slot_plans=None, ep_rank: int | None = None,
) -> tuple[list[float], list[float]]:
    """Persistent weight payload and quant metadata per PP stage.

    The total is checked against ``stage_weight_bytes`` for the same inputs;
    this is a decomposition, never an alternate resident-weight quantity.
    """
    model, parallel = case.model, case.runtime.parallel
    quant = model.quant or QuantSpec("bf16")
    total_by_stage: list[float] = []
    payload_by_stage: list[float] = []
    for terms in stage_weight_terms(
        model, parallel, include_draft=include_draft, draft_kind=draft_kind or None,
        include_engram_tables=include_engram_tables, include_vision=include_vision,
        layer_partition=layer_stage, slot_plans=slot_plans, ep_rank=ep_rank,
    ):
        total = payload = 0.0
        from mindie_sim.catalog.weights import _expand_resident_terms
        for term in terms:
            for t_sub, placed in _expand_resident_terms(
                term, parallel, enable_shared_expert_dp,
                pcp_o_proj_weight_sharding=enable_pcp_o_proj_weight_sharding,
                dcp_q_replicate=dcp_q_replicate, model=model,
            ):
                total += term_bytes(t_sub, quant, placed, case.machine.ranks_per_node)
                payload += term_payload_bytes(
                    t_sub, quant, placed, case.machine.ranks_per_node)
        total_by_stage.append(float(total))
        payload_by_stage.append(float(payload))
    return total_by_stage, payload_by_stage


class _MemoryWorkspaceMixin:
    def _resolve_workspace_tokens(self, tokens=None) -> int:
        runtime = self.case.runtime
        if tokens is None:
            return min(runtime.max_num_batched_tokens, sum(
                max(1, min(r.prompt_tokens, runtime.max_num_batched_tokens)) for r in self.case.workload.requests))
        if type(tokens) is not int or tokens < 0:
            raise ValueError("memory_workspace_tokens_require_nonnegative_integer")
        return tokens

    def _draft_tap_stage_bytes(self, tokens: int) -> list[float]:
        """Place draft-tap hidden buffers only on producing-to-consuming PP stages."""
        model = self.case.model
        pp = max(1, int(self.case.runtime.parallel.pp))
        per_stage = [0.0] * pp
        if not (self.draft_kind and model.auxiliary):
            return per_stage
        from mindie_sim.catalog.draft import optional_draft_spec
        draft = optional_draft_spec(model, self.draft_kind)
        per_tap = float(tokens) * float(model.hidden_size) * dtype_bytes(model.activation_dtype)
        draft_stage = pp - 1
        for layer_id in draft.target_layer_ids:
            idx = int(layer_id)
            if 0 <= idx < len(self.layer_stage):
                prod_stage = int(self.layer_stage[idx])
            elif idx < 0:
                resolved_idx = max(0, len(self.layer_stage) + idx)
                prod_stage = int(self.layer_stage[resolved_idx]) if self.layer_stage else 0
            else:
                prod_stage = draft_stage
            for stage in range(min(prod_stage, draft_stage), draft_stage + 1):
                per_stage[stage] += per_tap
        return per_stage

    def _async_inflight_workspace(self, tokens=None) -> list[float]:
        """Ping-pong step input/residual/output buffer retained across overlapped async steps."""
        model, runtime = self.case.model, self.case.runtime
        p = runtime.parallel
        resolved = self._resolve_workspace_tokens(tokens)
        result = [0.0] * p.pp
        for layer in model.layer_plans:
            stage = self.layer_stage[layer.index]
            residual = model.hidden_size * (2 + max(1, layer.hc_mult))
            step_buffer = resolved * residual * dtype_bytes(model.activation_dtype)
            result[stage] = max(result[stage], step_buffer)
        tap_bytes = self._draft_tap_stage_bytes(resolved)
        result = [value + tap for value, tap in zip(result, tap_bytes)]
        return [value * self.async_workspace_buffer_multiplier for value in result]

    def _workspace(self, tokens=None):
        """Resident simultaneous tensors for one layer, not lifetime traffic."""
        model, runtime = self.case.model, self.case.runtime
        p = runtime.parallel
        tokens = self._resolve_workspace_tokens(tokens)
        result = [0.0] * p.pp
        for layer in model.layer_plans:
            residual = model.hidden_size * (2 + max(1, layer.hc_mult))
            if layer.residual_block_size:
                residual += model.hidden_size * (layer.index // layer.residual_block_size + 1)
            if layer.attention in {"kda", "gdn"}:
                attention = (layer.conv_width + layer.linear_value_heads * layer.linear_head_dim) / p.tp
            elif layer.attention == "gqa":
                attention = (layer.num_heads / p.tp + 2 * local_kv_heads(layer.num_kv_heads, p.tp)) * layer.head_dim
            else:
                attention = layer.q_lora_rank + layer.kv_lora_rank + layer.num_heads / p.tp * (
                    layer.qk_nope_head_dim + layer.qk_rope_head_dim + layer.v_head_dim)
            ffn = (
                _FFN_INTERMEDIATE_TENSORS * layer.intermediate_size / p.tp
                if layer.ffn == "dense" else
                layer.top_k * (
                    2 * (layer.expert_latent_size or model.hidden_size)
                    + _FFN_INTERMEDIATE_TENSORS * layer.moe_intermediate_size / p.etp)
            )
            footprint = tokens * (residual + max(attention, ffn)) * dtype_bytes(model.activation_dtype)
            stage = self.layer_stage[layer.index]
            result[stage] = max(result[stage], footprint)
        tap_bytes = self._draft_tap_stage_bytes(tokens)
        result = [value + tap for value, tap in zip(result, tap_bytes)]
        if model.vision:
            from mindie_sim.catalog.media import media_geometry
            for request in self.case.workload.requests:
                if not request.media_tokens or request.media_encoded:
                    continue
                patches, *_ = media_geometry(runtime, model.vision, request.media_tokens)
                vision = model.vision
                # The default vision recipe materializes FP32 attention scores.
                scores = 2 * patches * patches * vision.num_heads * _VISION_SCORE_ELEMENT_BYTES
                tensors = patches * (vision.qkv_width + 2 * vision.hidden_size
                                     + 2 * vision.intermediate_size) * dtype_bytes(model.activation_dtype)
                result[0] = max(result[0], scores + tensors)
        return result
