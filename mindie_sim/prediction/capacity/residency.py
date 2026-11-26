"""Resident weights, offload ratios and workspace estimates.

[Role] Derive fixed and staging memory from model terms and placement.
[Invariants] Payload, quantization metadata, workspaces and reserves retain separate
  accounting. Explicit offload policy overrides case metadata.
[Timeouts & Locks] Bounded by layers, experts and active requests. No locks or I/O.
[Side-Effect Policy] Computes memory requirements from the supplied case.
"""
from __future__ import annotations

from dataclasses import replace
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


def _quant_has_expert_overrides(quant: QuantSpec | None) -> bool:
    if quant is None:
        return False
    if getattr(quant, "expert_quant_overrides", ()):
        return True
    for _cat, fields in getattr(quant, "category_overrides", ()) or ():
        if dict(fields).get("expert_quant_overrides"):
            return True
    return False


def _model_has_expert_quant_overrides(model) -> bool:
    """Return True when top-level or per-layer quantization carries per-expert overrides."""
    quant = getattr(model, "quant", None)
    if _quant_has_expert_overrides(quant):
        return True
    has_layer_ov = bool(
        (quant is not None and getattr(quant, "layer_quant_overrides", ()))
        or dict(getattr(model, "metadata", ()) or ()).get("layer_quant_overrides")
    )
    if not has_layer_ov:
        return False
    from mindie_sim.catalog.weight_layer_terms import _layer_resolved_quant
    for plan in getattr(model, "layer_plans", ()) or ():
        if plan.ffn == "moe" and int(plan.num_experts) > 0:
            if _quant_has_expert_overrides(_layer_resolved_quant(model, plan)):
                return True
    return False


def _expert_override_bucket_key(quant: QuantSpec | None, eid: int) -> tuple[int, ...]:
    """Classify one expert ID into its exact override bucket tuple for bit-exact rank deduplication."""
    if quant is None:
        return ()
    buckets: list[int] = []
    idx = -1
    for pos, (indices, _fields) in enumerate(getattr(quant, "expert_quant_overrides", ()) or ()):
        if eid in indices:
            idx = pos
            break
    buckets.append(idx)
    for _cat, fields in getattr(quant, "category_overrides", ()) or ():
        ov = dict(fields).get("expert_quant_overrides")
        if not ov:
            continue
        c_idx = -1
        for pos, (indices, _cfields) in enumerate(ov):
            if eid in indices:
                c_idx = pos
                break
        buckets.append(c_idx)
    return tuple(buckets)


def _ep_rank_weight_signature(
    model,
    slot_plans: Mapping[int, object],
    ep_rank: int,
    *,
    mixed_quant: bool,
    layer_quants: Mapping[int, QuantSpec] | None = None,
) -> tuple:
    """Bit-exact deduplication signature for one EP rank's routed-expert weight profile."""
    sig: list = []
    for layer_idx in sorted(slot_plans):
        slots = slot_plans[layer_idx]
        base_n = int(slots.base_slots_by_rank[ep_rank])
        extra_n = int(slots.redundant_slots_by_rank[ep_rank])
        if not mixed_quant:
            sig.append((layer_idx, base_n, extra_n))
            continue
        lquant = (layer_quants or {}).get(layer_idx) or getattr(model, "quant", None)
        base_ids, extra_ids = slots.split_expert_ids(ep_rank)
        base_hist: dict[tuple[int, ...], int] = {}
        for eid in base_ids:
            bk = _expert_override_bucket_key(lquant, int(eid))
            base_hist[bk] = base_hist.get(bk, 0) + 1
        extra_hist: dict[tuple[int, ...], int] = {}
        for eid in extra_ids:
            bk = _expert_override_bucket_key(lquant, int(eid))
            extra_hist[bk] = extra_hist.get(bk, 0) + 1
        sig.append((
            layer_idx, base_n, extra_n,
            tuple(sorted(base_hist.items())),
            tuple(sorted(extra_hist.items())),
        ))
    return tuple(sig)


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
        if declared is not None:
            return _offload_ratio(name, declared)
        if name in metadata:
            return _offload_ratio(name, metadata[name])
        return 0.0

    ratios = {name: pick(name) for name in
              ("expert_ddr_ratio", "kvc_ddr_ratio", "kvc_ssd_ratio")}
    if ratios["kvc_ddr_ratio"] + ratios["kvc_ssd_ratio"] > 1.0 + 1e-12:
        raise ValueError("memory_offload_kvc_ratios_exceed_unity")
    return ratios


def _layer_routed_bytes_for_rank(
    base_terms,
    slots,
    ep_rank: int,
    quant: QuantSpec,
    parallel,
    node_ranks: int,
    *,
    routed_budget: int,
    mixed_quant: bool,
) -> tuple[float, float, float]:
    from mindie_sim.catalog.expert_slots import routed_expert_weight_terms

    terms = list(routed_expert_weight_terms(base_terms, slots, ep_rank=ep_rank))
    routed = float(sum(term_bytes(term, quant, parallel, node_ranks) for term in terms))
    routed_payload = float(sum(term_payload_bytes(term, quant, parallel, node_ranks) for term in terms))
    base_ids, extra_ids = slots.split_expert_ids(ep_rank)
    all_ids = (*base_ids, *extra_ids)
    local_slots = max(1, len(all_ids))
    active_slots = min(len(all_ids), routed_budget)
    if active_slots <= 0 or routed <= 0.0:
        active_bytes = 0.0
    elif mixed_quant and active_slots < len(all_ids):
        per_expert = sorted(
            (
                float(sum(
                    term_bytes(
                        replace(t, count=1, resident_count=1, expert_ids=(int(eid),)),
                        quant, parallel, node_ranks,
                    )
                    for t in base_terms
                ))
                for eid in all_ids
            ),
            reverse=True,
        )
        active_bytes = min(routed, sum(per_expert[:active_slots]))
    else:
        active_bytes = min(routed, (routed / float(local_slots)) * float(active_slots))
    return routed, routed_payload, active_bytes


def _routed_expert_bytes_by_stage(
    case: CaseSpec, layer_stage: tuple[int, ...], *, workspace_tokens: int = 1,
    slot_plans=None, ep_rank: int | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """Routed-expert total, payload and active staging bytes per PP stage."""
    parallel = case.runtime.parallel
    quant = case.model.quant or QuantSpec("bf16")
    pp = max(1, parallel.pp)
    ep = max(1, int(parallel.ep))
    tokens = max(1, int(workspace_tokens))
    mixed_quant = _model_has_expert_quant_overrides(case.model)
    if slot_plans and ep_rank is None and ep > 1:
        from mindie_sim.catalog.weight_layer_terms import _layer_resolved_quant
        layer_quants = (
            {plan.index: _layer_resolved_quant(case.model, plan)
             for plan in case.model.layer_plans if plan.index in slot_plans}
            if mixed_quant else None
        )
        sig_cache: dict[tuple, tuple[list[float], list[float], list[float]]] = {}
        by_rank: list[tuple[list[float], list[float], list[float]]] = []
        for rank_i in range(ep):
            sig = _ep_rank_weight_signature(
                case.model, slot_plans, rank_i,
                mixed_quant=mixed_quant, layer_quants=layer_quants,
            )
            cached = sig_cache.get(sig)
            if cached is None:
                cached = _routed_expert_bytes_by_stage(
                    case, layer_stage, workspace_tokens=tokens,
                    slot_plans=slot_plans, ep_rank=rank_i,
                )
                sig_cache[sig] = cached
            by_rank.append(cached)
        best_res: list[float] = []
        best_pay: list[float] = []
        best_stg: list[float] = []
        for st in range(pp):
            lead = max(range(ep), key=lambda r: by_rank[r][0][st])
            best_res.append(by_rank[lead][0][st])
            best_pay.append(by_rank[lead][1][st])
            best_stg.append(max(by_rank[r][2][st] for r in range(ep)))
        return best_res, best_pay, best_stg

    result = [0.0] * pp
    payload = [0.0] * pp
    staging = [0.0] * pp
    target_ep = 0 if ep_rank is None else int(ep_rank)
    for plan in case.model.layer_plans:
        if plan.ffn != "moe" or int(plan.num_experts) <= 0 or int(plan.moe_intermediate_size) <= 0:
            continue
        stage = layer_stage[plan.index] if plan.index < len(layer_stage) else 0
        terms = [term for term in layer_weight_terms(plan, case.model)
                 if term.category == "routed_expert"]
        num_exp = int(plan.num_experts)
        k = min(num_exp, max(1, int(plan.top_k)))
        ep_dp = max(1, int(getattr(parallel, "ep_dp_width", 1) or 1))
        routed_budget = tokens * ep_dp * k
        slots = (slot_plans or {}).get(plan.index)
        if slots is not None:
            routed, routed_payload, active_layer_bytes = _layer_routed_bytes_for_rank(
                terms, slots, target_ep, quant, parallel, case.machine.ranks_per_node,
                routed_budget=routed_budget, mixed_quant=mixed_quant,
            )
        else:
            routed = float(sum(
                term_bytes(term, quant, parallel, case.machine.ranks_per_node)
                for term in terms
            ))
            routed_payload = float(sum(
                term_payload_bytes(term, quant, parallel, case.machine.ranks_per_node)
                for term in terms
            ))
            local_exp = max(1, (num_exp + ep - 1) // ep)
            worst_local = min(local_exp, routed_budget)
            active_layer_bytes = min(
                routed, float(worst_local) * (routed / float(local_exp)),
            )
        result[stage] += routed
        payload[stage] += routed_payload
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
        runtime=case.runtime,
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
    def _max_speculative_proposals(self) -> int:
        from mindie_sim.catalog.speculative_state import resolve_speculative_proposals
        import json
        from mindie_sim.schema.speculative import SpeculativeSpec

        runtime = self.case.runtime
        meta = dict(getattr(runtime, "metadata", ()) or ())
        spec_method = str(meta.get("spec_method", "") or "").strip().lower()
        mode = str(meta.get("draft_mode", "") or "").strip().lower()
        if spec_method == "none" or mode == "none":
            return 0
        proposal = int(resolve_speculative_proposals(runtime))
        if proposal <= 0:
            raw_spec = meta.get("speculative")
            if raw_spec:
                spec_map = json.loads(raw_spec) if isinstance(raw_spec, str) else raw_spec
                proposal = int(SpeculativeSpec.from_mapping(spec_map).draft_len)
        if proposal <= 0:
            mtp = getattr(getattr(self.case.model, "auxiliary", None), "mtp", None)
            non_neural = spec_method in {"ngram", "ngram_gpu"} or mode in {"ngram", "ngram_gpu"}
            if (not non_neural and mtp is not None
                    and (mode or spec_method or float(getattr(runtime, "mtp_acceptance_length", 1.0) or 1.0) > 1.0)):
                proposal = int(getattr(mtp, "proposal_tokens", 0) or 0)
        return max(0, proposal)

    def _resolve_workspace_tokens(self, tokens=None) -> int:
        runtime = self.case.runtime
        if tokens is None:
            max_batched = int(runtime.max_num_batched_tokens)
            max_input = int(getattr(runtime, "max_num_input_tokens", 0) or 0)
            prefill_cap = min(max_batched, max_input) if max_input > 0 else max_batched
            max_seqs = max(1, int(getattr(runtime, "max_num_seqs", 1) or 1))
            requests = self.case.workload.requests
            prefill_per_req = sorted(
                (max(1, min(int(r.prompt_tokens), prefill_cap)) for r in requests),
                reverse=True,
            )[:max_seqs]
            prefill_tokens = min(prefill_cap, sum(prefill_per_req))
            proposal = self._max_speculative_proposals()
            if proposal <= 0:
                return prefill_tokens
            decode_toks = sorted(
                (
                    1 + min(proposal, max(0, int(r.output_tokens) - 2), max(0, max_batched - 1))
                    for r in requests if int(r.output_tokens) > 1
                ),
                reverse=True,
            )[:max_seqs]
            if not decode_toks:
                return prefill_tokens
            pref_prefix = [0]
            for pt in prefill_per_req:
                pref_prefix.append(min(prefill_cap, pref_prefix[-1] + pt))
            best_mixed = prefill_tokens
            dec_sum = 0
            n_req = len(requests)
            for k in range(len(decode_toks) + 1):
                if k > 0:
                    dec_sum += decode_toks[k - 1]
                p_slots = min(max_seqs - k, n_req - k, len(prefill_per_req))
                p_contrib = pref_prefix[p_slots] if p_slots > 0 else 0
                best_mixed = max(best_mixed, min(max_batched, dec_sum + p_contrib))
            return best_mixed
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
