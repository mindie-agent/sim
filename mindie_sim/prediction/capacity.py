"""Placement-aware resident memory and dynamic request ownership.

[Role] Derive weight/cache geometry and account for live allocations.
[Invariants] Stored observations cannot overwrite geometry. Each replica owns
  its cache; PP stages and physical ranks retain separate capacities. A reserved
  pool and the bytes used inside it are not added twice.
[Timeouts & Locks] Bounded by layers and active requests, no I/O or locks.
[Side-Effect Policy] Mutates only this prediction's cache ledger.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math

from mindie_sim.catalog.cache import layer_cache_components, local_kv_heads
from mindie_sim.catalog.cache_pages import CachePages
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms, stage_weight_bytes
from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.case import Case as CaseSpec


@dataclass(frozen=True)
class MemoryPolicy:
    workspace_bytes: float = 0
    graph_pool_bytes: float = 0
    communication_buffer_bytes: float = 0
    runtime_bytes: float = 0
    derive_workspace: bool = True
    cache_page_groups: tuple[dict, ...] = ()
    async_delayed_release_steps: int | None = None
    async_in_flight_lookahead_steps: int | None = None
    async_workspace_buffer_multiplier: float | None = None
    expert_ddr_ratio: float = 0.0
    kvc_ddr_ratio: float = 0.0
    kvc_ssd_ratio: float = 0.0

    @property
    def reserved(self) -> float:
        return sum((self.workspace_bytes, self.graph_pool_bytes,
                    self.communication_buffer_bytes, self.runtime_bytes))


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
) -> tuple[list[float], list[float]]:
    """Routed-expert total and single-layer active staging bytes per PP stage."""
    parallel = case.runtime.parallel
    quant = case.model.quant or QuantSpec("bf16")
    result = [0.0] * max(1, parallel.pp)
    staging = [0.0] * max(1, parallel.pp)
    ep = max(1, int(parallel.ep))
    tokens = max(1, int(workspace_tokens))
    for plan in case.model.layer_plans:
        if plan.ffn != "moe" or int(plan.num_experts) <= 0 or int(plan.moe_intermediate_size) <= 0:
            continue
        stage = layer_stage[plan.index] if plan.index < len(layer_stage) else 0
        routed = float(sum(
            term_bytes(term, quant, parallel, case.machine.ranks_per_node)
            for term in layer_weight_terms(plan, case.model)
            if term.category == "routed_expert"
        ))
        result[stage] += routed
        num_exp = int(plan.num_experts)
        k = min(num_exp, max(1, int(plan.top_k)))
        active_total = float(num_exp) if k >= num_exp else float(num_exp) * (
            1.0 - ((1.0 - float(k) / float(num_exp)) ** float(tokens))
        )
        local_exp = max(1, (num_exp + ep - 1) // ep)
        active_layer_bytes = (active_total / float(ep)) * (routed / float(local_exp))
        staging[stage] = max(staging[stage], min(routed, active_layer_bytes))
    return result, staging


class Memory:
    def __init__(self, case: CaseSpec, policy: MemoryPolicy | None = None, *,
                 workspace_tokens: int | None = None):
        self.case = case
        self.policy = policy or MemoryPolicy(**case.runtime.memory)
        p = case.runtime.parallel
        partition = case.runtime.pp_layer_partition
        self.layers = tuple(partition or (
            (case.model.layers // p.pp + (stage < case.model.layers % p.pp))
            for stage in range(p.pp)
        ))
        self.layer_stage = tuple(stage for stage, count in enumerate(self.layers) for _ in range(count))
        self.pages = CachePages(
            self.policy.cache_page_groups,
            {layer.index: set(layer_cache_components(
                case.model, layer, p, case.runtime.max_model_len,
                capacity_padded=False)) for layer in case.model.layer_plans},
            self.layer_stage,
        )
        self.draft_kind = (dict(case.runtime.metadata).get("draft_mode") or
                           ("dspark" if case.runtime.dspark else "mtp" if
                            case.runtime.mtp_acceptance_length > 1 or any(
                                value > 1 for _, value in case.runtime.mtp_acceptance_by_output_tokens)
                            or dict(case.runtime.metadata).get("speculative")
                            else ""))
        draft = bool(self.draft_kind)
        self.offload = _resolve_offload_ratios(case, self.policy)
        self.kvc_hbm_ratio = max(
            0.0, 1.0 - self.offload["kvc_ddr_ratio"] - self.offload["kvc_ssd_ratio"]
        )
        self.gross_weights = stage_weight_bytes(
            case.model, p, case.machine.ranks_per_node, include_draft=draft,
            include_engram_tables=not case.runtime.engram_cpu_offload,
            layer_partition=self.layers, draft_kind=self.draft_kind or None,
        )
        resolved_ws_tokens = self._resolve_workspace_tokens(workspace_tokens)
        if self.offload["expert_ddr_ratio"] > 0.0:
            self.routed_expert_bytes, stage_staging_base = _routed_expert_bytes_by_stage(
                case, self.layer_stage, workspace_tokens=max(1, resolved_ws_tokens),
            )
            self.expert_staging_workspace = [
                self.offload["expert_ddr_ratio"] * value for value in stage_staging_base
            ]
        else:
            self.routed_expert_bytes = [0.0] * p.pp
            self.expert_staging_workspace = [0.0] * p.pp
        self.expert_hbm_saved = [
            self.offload["expert_ddr_ratio"] * value for value in self.routed_expert_bytes
        ]
        self.weights = [
            max(0.0, gross - saved)
            for gross, saved in zip(self.gross_weights, self.expert_hbm_saved)
        ]
        async_enabled = bool(case.runtime.async_scheduling)
        self.async_delayed_release_steps = (
            (1 if async_enabled else 0) if self.policy.async_delayed_release_steps is None
            else max(0, int(self.policy.async_delayed_release_steps))
        )
        self.async_in_flight_lookahead_steps = (
            (1 if async_enabled else 0) if self.policy.async_in_flight_lookahead_steps is None
            else max(0, int(self.policy.async_in_flight_lookahead_steps))
        )
        self.async_workspace_buffer_multiplier = (
            (1.0 if async_enabled else 0.0) if self.policy.async_workspace_buffer_multiplier is None
            else max(0.0, float(self.policy.async_workspace_buffer_multiplier))
        )
        width = p.tp * p.pcp
        self.async_workspace = (
            self._async_inflight_workspace(workspace_tokens)
            if self.policy.derive_workspace and self.async_workspace_buffer_multiplier > 0
            else [0.0] * p.pp
        )
        base_workspace = (self._workspace(workspace_tokens) if self.policy.derive_workspace
                          else [0.0] * p.pp)
        staging_workspace = (self.expert_staging_workspace if self.policy.derive_workspace
                             else [0.0] * p.pp)
        self.workspace = [
            base + extra + stage_buf
            for base, extra, stage_buf in zip(base_workspace, self.async_workspace, staging_workspace)
        ]
        self.capacity = [
            [
                min(d.hbm_bytes * case.runtime.gpu_memory_utilization
                    for d in case.machine.dies[(replica * p.pp + stage) * width:
                                               (replica * p.pp + stage + 1) * width])
                - self.weights[stage] - self.policy.reserved - self.workspace[stage]
                for stage in range(p.pp)
            ]
            for replica in range(p.dp)
        ]
        self.peak = [[0.0] * p.pp for _ in range(p.dp)]
        self.live = [[0.0] * p.pp for _ in range(p.dp)]
        self.owners: dict[tuple[int, str], tuple[float, ...]] = {}
        self.delayed_owners: dict[tuple[int, str], tuple[int, tuple[float, ...]]] = {}
        self.peak_delayed_release = [[0.0] * p.pp for _ in range(p.dp)]

    def _resolve_workspace_tokens(self, tokens=None) -> int:
        runtime = self.case.runtime
        if tokens is None:
            return min(runtime.max_num_batched_tokens, sum(
                max(1, min(r.prompt_tokens, runtime.max_num_batched_tokens)) for r in self.case.workload.requests))
        if type(tokens) is not int or tokens < 0:
            raise ValueError("memory_workspace_tokens_require_nonnegative_integer")
        return tokens

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
        if self.draft_kind and model.auxiliary:
            from mindie_sim.catalog.draft import optional_draft_spec
            draft = optional_draft_spec(model, self.draft_kind)
            feature_bytes = resolved * model.hidden_size * len(draft.target_layer_ids) * dtype_bytes(model.activation_dtype)
            result = [value + feature_bytes for value in result]
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
            ffn = (3 * layer.intermediate_size / p.tp if layer.ffn == "dense" else
                   layer.top_k * (2 * (layer.expert_latent_size or model.hidden_size)
                                  + 3 * layer.moe_intermediate_size / p.etp))
            footprint = tokens * (residual + max(attention, ffn)) * dtype_bytes(model.activation_dtype)
            stage = self.layer_stage[layer.index]
            result[stage] = max(result[stage], footprint)
        if self.draft_kind and model.auxiliary:
            from mindie_sim.catalog.draft import optional_draft_spec
            draft = optional_draft_spec(model, self.draft_kind)
            feature_bytes = tokens * model.hidden_size * len(draft.target_layer_ids) * dtype_bytes(model.activation_dtype)
            result = [value + feature_bytes for value in result]
        if model.vision:
            from mindie_sim.catalog.media import media_geometry
            for request in self.case.workload.requests:
                if not request.media_tokens or request.media_encoded:
                    continue
                patches, *_ = media_geometry(runtime, model.vision, request.media_tokens)
                vision = model.vision
                # The default vision recipe materializes FP32 attention scores.
                scores = 2 * patches * patches * vision.num_heads * 4
                tensors = patches * (vision.qkv_width + 2 * vision.hidden_size
                                     + 2 * vision.intermediate_size) * dtype_bytes(model.activation_dtype)
                result[0] = max(result[0], scores + tensors)
        return result

    def cache_bytes(self, tokens: int, *, padded=True) -> tuple[float, ...]:
        case, model, p = self.case, self.case.model, self.case.runtime.parallel
        result = (self.pages.stage_bytes(tokens, case.runtime.max_model_len, p.pp)
                  if padded else [0.0] * p.pp)
        if model.layer_plans:
            for layer in model.layer_plans:
                components = layer_cache_components(model, layer, p, tokens, capacity_padded=padded,
                                                     interleave=case.runtime.kv_block_size)
                covered = self.pages.covered.get(layer.index, ()) if padded else ()
                result[self.layer_stage[layer.index]] += sum(
                    value for name, value in components.items() if name not in covered)
        from mindie_sim.catalog.draft import optional_draft_spec
        mode = self.draft_kind
        if mode and model.auxiliary is not None:
            draft = optional_draft_spec(model, mode)
            row = ((draft.kv_cache_elements or draft.kv_lora_rank + draft.qk_rope_head_dim) if draft.attention_kind == "mla"
                   else 2 * local_kv_heads(draft.num_kv_heads, p.tp) * draft.head_dim)
            draft_tokens = (math.ceil(tokens / case.runtime.kv_block_size) * case.runtime.kv_block_size
                            if padded else tokens)
            result[-1] += draft_tokens * row * dtype_bytes(model.kv_dtype) * draft.layers
        return tuple(result)

    def _max_layer_cache_bytes(self, tokens: int) -> tuple[float, ...]:
        model, p = self.case.model, self.case.runtime.parallel
        result = [0.0] * p.pp
        for layer in model.layer_plans:
            comps = layer_cache_components(
                model, layer, p, tokens, capacity_padded=False,
                interleave=self.case.runtime.kv_block_size)
            st = self.layer_stage[layer.index]
            result[st] = max(result[st], float(sum(comps.values())))
        return tuple(result)

    def resize(self, replica: int, request: str, tokens: int) -> bool:
        key = replica, request
        before = self.owners.get(key, (0.0,) * len(self.weights))
        after = self.cache_bytes(tokens)
        staging = (0.0,) * len(self.weights)
        if self.kvc_hbm_ratio != 1.0:
            staging = tuple(
                val * (1.0 - self.kvc_hbm_ratio)
                for val in self._max_layer_cache_bytes(tokens)
            )
            after = tuple(value * self.kvc_hbm_ratio for value in after)
        needed = [live - old + new for live, old, new in zip(self.live[replica], before, after)]
        if any(val + stg > cap for val, stg, cap in zip(needed, staging, self.capacity[replica])):
            return False
        self.owners[key] = after
        self.live[replica] = needed
        self.peak[replica] = [max(old, new) for old, new in zip(self.peak[replica], needed)]
        return True

    def release(self, replica: int, request: str) -> None:
        self.delayed_owners.pop((replica, request), None)
        sizes = self.owners.pop((replica, request), (0.0,) * len(self.weights))
        self.live[replica] = [live - size for live, size in zip(self.live[replica], sizes)]

    def defer_release(self, replica: int, request: str, steps: int | None = None) -> None:
        """Retain KV ownership for finished requests across async scheduling lookahead steps."""
        delay = self.async_delayed_release_steps if steps is None else max(0, int(steps))
        if delay <= 0 or (replica, request) not in self.owners:
            self.release(replica, request)
            return
        self.delayed_owners[replica, request] = (delay, self.owners[replica, request])
        pinned = [0.0] * len(self.weights)
        for (rep, _), (_, sizes) in self.delayed_owners.items():
            if rep == replica:
                pinned = [cur + sz for cur, sz in zip(pinned, sizes)]
        self.peak_delayed_release[replica] = [
            max(old, new) for old, new in zip(self.peak_delayed_release[replica], pinned)
        ]

    def has_delayed_releases(self, replica: int) -> bool:
        return any(rep == replica for rep, _ in self.delayed_owners)

    def step_delayed_releases(self, replica: int) -> None:
        for (rep, req), (rem, sizes) in list(self.delayed_owners.items()):
            if rep != replica:
                continue
            if rem <= 1:
                self.release(replica, req)
            else:
                self.delayed_owners[rep, req] = (rem - 1, sizes)

    def flush_delayed_releases(self, replica: int | None = None) -> None:
        for rep, req in list(self.delayed_owners):
            if replica is None or rep == replica:
                self.release(rep, req)

    def to_dict(self) -> dict:
        result = {
            "status": "feasible" if all(value >= 0 for row in self.capacity for value in row) else "oom",
            "basis": "declared_geometry_and_live_allocations",
            "weights_bytes_per_stage_rank": self.weights,
            "reserve_bytes_per_rank": self.policy.reserved,
            "activation_workspace_bytes_per_stage_rank": self.workspace,
            "async_scheduling": bool(self.case.runtime.async_scheduling),
            "async_workspace_overhead_bytes_per_stage_rank": self.async_workspace,
            "async_delayed_release_steps": self.async_delayed_release_steps,
            "async_in_flight_lookahead_steps": self.async_in_flight_lookahead_steps,
            "peak_delayed_release_cache_bytes_per_replica_stage": self.peak_delayed_release,
            "kv_capacity_bytes_per_replica_stage": self.capacity,
            "peak_cache_bytes_per_replica_stage": self.peak,
            "live_cache_bytes_per_replica_stage": self.live,
            "peak_resident_bytes_per_replica_stage": [
                [weight + self.policy.reserved + workspace + cache
                 for weight, workspace, cache in zip(self.weights, self.workspace, row)]
                for row in self.peak
            ],
            "layer_partition": self.layers,
            "workspace_basis": "simultaneous_layer_tensors_plus_explicit_reserves",
        }
        if any(value > 0.0 for value in self.offload.values()):
            result["offload"] = {
                "expert_ddr_ratio": self.offload["expert_ddr_ratio"],
                "kvc_ddr_ratio": self.offload["kvc_ddr_ratio"],
                "kvc_ssd_ratio": self.offload["kvc_ssd_ratio"],
                "kvc_hbm_ratio": self.kvc_hbm_ratio,
                "gross_weights_bytes_per_stage_rank": self.gross_weights,
                "routed_expert_bytes_per_stage_rank": self.routed_expert_bytes,
                "expert_hbm_saved_bytes_per_stage_rank": self.expert_hbm_saved,
            }
        return result
