"""Placement-aware resident memory and dynamic request ownership.

[Role] Derive weight/cache geometry and account for live allocations.
[Invariants] Stored observations cannot overwrite geometry. Each replica owns
  its cache; PP stages and physical ranks retain separate capacities. A reserved
  pool and the bytes used inside it are not added twice.
[Timeouts & Locks] Bounded by layers and active requests, no I/O or locks.
[Side-Effect Policy] Mutates only this prediction's cache ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from mindie_sim.catalog.cache import layer_cache_components, local_kv_heads
from mindie_sim.catalog.cache_pages import CachePages
from mindie_sim.catalog.weights import stage_weight_bytes
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

    @property
    def reserved(self) -> float:
        return sum((self.workspace_bytes, self.graph_pool_bytes,
                    self.communication_buffer_bytes, self.runtime_bytes))


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
        self.weights = stage_weight_bytes(
            case.model, p, case.machine.ranks_per_node, include_draft=draft,
            include_engram_tables=not case.runtime.engram_cpu_offload,
            layer_partition=self.layers, draft_kind=self.draft_kind or None,
        )
        width = p.tp * p.pcp
        self.workspace = (self._workspace(workspace_tokens) if self.policy.derive_workspace
                          else [0.0] * p.pp)
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

    def _workspace(self, tokens=None):
        """Resident simultaneous tensors for one layer, not lifetime traffic."""
        model, runtime = self.case.model, self.case.runtime
        p = runtime.parallel
        if tokens is None:
            tokens = min(runtime.max_num_batched_tokens, sum(
                max(1, min(r.prompt_tokens, runtime.max_num_batched_tokens)) for r in self.case.workload.requests))
        elif type(tokens) is not int or tokens < 0:
            raise ValueError("memory_workspace_tokens_require_nonnegative_integer")
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

    def resize(self, replica: int, request: str, tokens: int) -> bool:
        key = replica, request
        before = self.owners.get(key, (0.0,) * len(self.weights))
        after = self.cache_bytes(tokens)
        needed = [live - old + new for live, old, new in zip(self.live[replica], before, after)]
        if any(value > capacity for value, capacity in zip(needed, self.capacity[replica])):
            return False
        self.owners[key] = after
        self.live[replica] = needed
        self.peak[replica] = [max(old, new) for old, new in zip(self.peak[replica], needed)]
        return True

    def release(self, replica: int, request: str) -> None:
        sizes = self.owners.pop((replica, request), (0.0,) * len(self.weights))
        self.live[replica] = [live - size for live, size in zip(self.live[replica], sizes)]

    def to_dict(self) -> dict:
        return {
            "status": "feasible" if all(value >= 0 for row in self.capacity for value in row) else "oom",
            "basis": "declared_geometry_and_live_allocations",
            "weights_bytes_per_stage_rank": self.weights,
            "reserve_bytes_per_rank": self.policy.reserved,
            "activation_workspace_bytes_per_stage_rank": self.workspace,
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
