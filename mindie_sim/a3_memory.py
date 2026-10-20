"""Per-die model weight/KV/state/media/draft memory planning.

[Role]
  Compute per-rank, per-PP-stage resident weights (largest actual stage), exact
  per-request token cache, recurrent/conv state, C2 rings, media expansion,
  optional draft residency and per-request block rounding.

[Invariants]
  * Resources are per die and per PP stage.  The largest actual stage is used,
    not the first stage by assumption.
  * Resident body weights, optional draft weights, vision weights, node-sharded
    Engram tables and per-request state are separate planes.
  * TP/EP/ETP shards and TP>KV-head replication are reflected; Engram tables
    are replicated per node and row-sharded within the node, never divided by
    global EP.
  * State is a per-request endpoint, not a copy per historical token.  The C2
    ring keeps allocated capacity while valid rows update.
  * Cache growth is block-rounded for token-addressed planes; compressed rows
    and windows use their exact completed counts for boundary accounting.
  * Workload fit uses the live single-request peak, not a lifetime sum over
    sequential requests.
  * OOM is explicit; no negative capacity is clamped into a modeled block.

[Timeouts & Locks]
  Pure arithmetic; no I/O or locks.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .a3_types import MachineSpec, ModelSpec, ParallelSpec, RuntimeSpec, WorkloadSpec, dtype_bytes
from .a3_catalog_cache import (
    block_round, cache_bytes_per_token_estimate, live_request_cache_peak,
    request_cache_breakdown, stage_cache_components,
    state_bytes_per_request_per_rank,
)


@dataclass(frozen=True)
class MemoryPlan:
    status: str
    weights_bytes: float
    kv_bytes_per_token: float
    kv_blocks: int
    kv_capacity_tokens: int
    hbm_bytes: float
    available_bytes: float
    reasons: tuple[str, ...] = ()
    details: tuple[tuple[str, Any], ...] = ()
    stage_weights_bytes: tuple[float, ...] = ()
    stage_cache_bytes: tuple[float, ...] = ()
    largest_weight_stage: int = 0
    largest_cache_stage: int = 0
    state_bytes_per_request: float = 0.0
    ring_bytes_per_request: float = 0.0
    state_reserve_bytes: float = 0.0
    optional_draft_weight_bytes: float = 0.0
    external_shared_weight_bytes: float = 0.0
    engram_staging_bytes: float = 0.0
    vision_weight_bytes: float = 0.0
    engram_weight_bytes: float = 0.0
    media_tokens: int = 0
    request_peak_tokens: int = 0
    per_request_cache: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "weights_bytes": self.weights_bytes,
            "weights_gib": self.weights_bytes / 1024 ** 3,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "kv_blocks": self.kv_blocks,
            "kv_capacity_tokens": self.kv_capacity_tokens,
            "hbm_bytes": self.hbm_bytes,
            "hbm_gib": self.hbm_bytes / 1024 ** 3,
            "available_bytes": self.available_bytes,
            "reasons": list(self.reasons),
            "details": dict(self.details),
            "stage_weights_bytes": list(self.stage_weights_bytes),
            "stage_cache_bytes": list(self.stage_cache_bytes),
            "largest_weight_stage": self.largest_weight_stage,
            "largest_cache_stage": self.largest_cache_stage,
            "state_bytes_per_request": self.state_bytes_per_request,
            "ring_bytes_per_request": self.ring_bytes_per_request,
            "state_reserve_bytes": self.state_reserve_bytes,
            "optional_draft_weight_bytes": self.optional_draft_weight_bytes,
            "external_shared_weight_bytes": self.external_shared_weight_bytes,
            "engram_staging_bytes": self.engram_staging_bytes,
            "vision_weight_bytes": self.vision_weight_bytes,
            "engram_weight_bytes": self.engram_weight_bytes,
            "media_tokens": self.media_tokens,
            "request_peak_tokens": self.request_peak_tokens,
            "per_request_cache": list(self.per_request_cache),
        }


def _draft_enabled(model: ModelSpec, runtime: RuntimeSpec) -> bool:
    if model.is_standalone_draft:
        return False  # body already includes draft heads/layers
    return bool(dict(runtime.metadata or ()).get("draft_mode"))


def _media_tokens(runtime: RuntimeSpec) -> int:
    return max(0, int(runtime.media_tokens or 0))


def memory_plan(
    model: ModelSpec, runtime: RuntimeSpec, machine: MachineSpec,
    *, workload: WorkloadSpec | None = None,
) -> MemoryPlan:
    parallel = runtime.parallel
    parallel.validate(model)
    machine.validate_placement(parallel)
    die = machine.dies[0]
    pp = max(1, parallel.pp)
    node_ranks = max(1, int(machine.ranks_per_node or 1))
    draft = _draft_enabled(model, runtime)
    media_tokens = _media_tokens(runtime)
    details: dict[str, Any] = {
        "per_die": True,
        "tp_sharded_weights": parallel.tp,
        "ep_sharded_experts": parallel.ep,
        "etp": parallel.etp,
        "pp_stages": pp,
        "node_ranks": node_ranks,
        "engram_node_sharded": model.engram is not None,
        "engram_storage_mode": model.engram.storage_mode if model.engram else "",
        "draft_resident": draft,
        "media_tokens": media_tokens,
    }
    extras: dict[str, Any] = {}
    if model.quant is not None:
        from .a3_catalog_weights import stage_weight_bytes, weight_breakdown
        stage_weights = stage_weight_bytes(
            model, parallel, node_ranks, include_draft=draft,
        )
        body_weights = stage_weight_bytes(
            model, parallel, node_ranks, include_draft=False,
        )
        breakdown = weight_breakdown(model, parallel, node_ranks)
        weights = max(stage_weights) if stage_weights else 0.0
        largest_weight_stage = max(
            range(len(stage_weights)), key=lambda i: stage_weights[i]
        ) if stage_weights else 0
        draft_stage_delta = [
            with_draft - body for with_draft, body in zip(stage_weights, body_weights)
        ]
        optional_draft = max(draft_stage_delta) if draft_stage_delta else 0.0
        engram_staging = 0.0
        from .a3_catalog_weights import external_shared_weight_bytes as _external
        external_shared = _external(model, parallel, node_ranks)
        vision_bytes = float(breakdown.get("vision", 0.0) + breakdown.get("vision_pos", 0.0)
                             + breakdown.get("projector", 0.0) + breakdown.get("image_token", 0.0))
        engram_bytes = float(breakdown.get("engram_table", 0.0))
        # Per-request token cache: exclude endpoint states and rings.
        state_components = 0.0
        ring_components = 0.0
        token_components = 0.0
        stage_token_components: list[float] = []
        stage_cache = stage_cache_components(model, parallel, runtime.max_model_len)
        for stage in stage_cache:
            state = float(stage.get("recurrent_state", 0.0) + stage.get("conv_state", 0.0))
            ring = float(stage.get("c2_ring", 0.0))
            tokens = float(sum(stage.values()) - state - ring)
            state_components += state
            ring_components += ring
            token_components += tokens
            stage_token_components.append(tokens)
        per_request = request_cache_breakdown(model, parallel, runtime.max_model_len)
        stage_cache_bytes = tuple(per_request["stage_bytes"])
        kv_per_token = (
            max(stage_token_components) / max(1, runtime.max_model_len)
            if stage_token_components else 0.0
        )
        state_per_request = state_bytes_per_request_per_rank(model, parallel)
        details["per_request_cache_at_max_model_len"] = per_request["by_component"]
        details["state_components_at_max"] = state_components
        details["largest_cache_stage"] = per_request["largest_stage"]
        extras.update({
            "state_bytes_per_request": state_per_request,
            "ring_bytes_per_request": ring_components,
        })
    else:
        weights = model.weight_bytes(parallel)
        weights = weights if weights > 0 else 0.0
        stage_weights = tuple([weights] + [0.0] * (pp - 1))
        largest_weight_stage = 0
        local_kv = model.local_kv_heads(parallel.tp)
        if model.attention == "gqa":
            per_layer = 2.0 * local_kv * model.head_dim * dtype_bytes(model.kv_dtype)
        else:
            assert model.mla is not None
            per_layer = (model.mla.kv_lora_rank + model.mla.qk_rope_head_dim) * dtype_bytes(model.kv_dtype)
        stage_cache_bytes = tuple(
            per_layer * model.layers_per_stage(pp) for _ in range(pp)
        )
        kv_per_token = per_layer * model.layers_per_stage(pp) / max(1, parallel.dcp)
        state_per_request = model.state_bytes_per_request / pp / max(1, parallel.tp)
        optional_draft = 0.0
        external_shared = 0.0
        engram_staging = 0.0
        vision_bytes = 0.0
        engram_bytes = 0.0
        ring_components = 0.0
        per_request = {}
        extras.update({
            "state_bytes_per_request": state_per_request,
            "ring_bytes_per_request": 0.0,
        })
    # Activation/expert/media/workspace reserve.
    act_bytes = dtype_bytes(model.activation_dtype)
    batch_tokens = max(1, runtime.max_num_batched_tokens)
    activation_reserve = float(batch_tokens * model.hidden_size * act_bytes * 2)
    if model.vision is not None and media_tokens:
        patch_tokens = model.vision.patch_tokens_for_media(media_tokens)
        activation_reserve += float(
            patch_tokens * (model.vision.hidden_size + model.vision.aligner_input_width)
            * act_bytes * 2
        )
    max_live = max(1, min(
        runtime.max_num_seqs,
        len(workload.requests) if workload is not None else runtime.max_num_seqs,
    ))
    workspace = max(16.0 * 1024 ** 2, 0.02 * die.hbm_bytes) + activation_reserve
    if model.engram is not None:
        # Lookup-bounded staging/response buffers, not a fraction of table size.
        # The token budget is a global per-iteration window shared by all
        # requests; do not multiply it by max_num_seqs.
        global_window_lookups = (
            max(1, runtime.max_num_batched_tokens)
            * model.engram.lookups_per_token
        )
        engram_staging = float(
            global_window_lookups
            * (model.engram.host_response_bytes + model.engram.id_bytes)
        )
        workspace += engram_staging
    state_per_request = float(extras.get("state_bytes_per_request", 0.0))
    ring_per_request = float(extras.get("ring_bytes_per_request", 0.0))
    state_reserve = (state_per_request + ring_per_request) * max_live
    available = (
        die.hbm_bytes * runtime.gpu_memory_utilization
        - weights - workspace - state_reserve
    )
    reasons: list[str] = []
    if available <= 0:
        reasons.append("weights_and_workspace_exceed_hbm")
    block_size = max(1, runtime.kv_block_size)
    block_bytes = max(1.0, max(1.0, kv_per_token) * block_size)
    blocks = max(0, int(available // block_bytes)) if available > 0 else 0
    if blocks <= 0:
        reasons.append("no_kv_blocks")
    status = "fits" if not reasons else "oom"
    peak_tokens = _request_peak_tokens(workload, media_tokens) if workload is not None else 0
    request_rows: list[dict[str, Any]] = []
    if model.quant is not None:
        if workload is not None:
            peak = live_request_cache_peak(
                model, parallel, workload, block_size=block_size,
                media_tokens=media_tokens,
            )
            request_rows = peak["per_request"]
        else:
            req = request_cache_breakdown(
                model, parallel, runtime.max_model_len + media_tokens,
            )
            request_rows = [{
                "request_id": "_max_model_len_request",
                "sequence_tokens": runtime.max_model_len + media_tokens,
                "cache_bytes": req["total_bytes"],
                "max_stage_bytes": req["max_stage_bytes"],
                "block_rounded_tokens": block_round(runtime.max_model_len + media_tokens, block_size),
            }]
    details.update({
        "activation_reserve_bytes": activation_reserve,
        "workspace_bytes": workspace,
        "available_before_kv_bytes": available,
        "kv_block_size": block_size,
        "kv_token_components_at_max_model_len": per_request.get("by_component") if isinstance(per_request, dict) else {},
        "max_live_requests": max_live,
        "state_reserve_bytes": state_reserve,
        "largest_stage_weights": largest_weight_stage,
        "request_peak_tokens": peak_tokens,
        "request_live_peak_not_lifetime_sum": True,
        "optional_draft_weight_bytes": optional_draft,
        "vision_weight_bytes": vision_bytes,
        "engram_weight_bytes": engram_bytes,
        "media_tokens": media_tokens,
    })
    return MemoryPlan(
        status=status, weights_bytes=float(weights),
        kv_bytes_per_token=float(kv_per_token), kv_blocks=blocks,
        kv_capacity_tokens=blocks * block_size,
        hbm_bytes=float(die.hbm_bytes), available_bytes=float(available),
        reasons=tuple(reasons), details=tuple(sorted(details.items())),
        stage_weights_bytes=tuple(float(v) for v in stage_weights),
        stage_cache_bytes=tuple(float(v) for v in stage_cache_bytes),
        largest_weight_stage=int(largest_weight_stage),
        largest_cache_stage=int(per_request.get("largest_stage", 0)) if isinstance(per_request, dict) else 0,
        state_bytes_per_request=float(state_per_request),
        ring_bytes_per_request=float(ring_per_request),
        state_reserve_bytes=float(state_reserve),
        optional_draft_weight_bytes=float(optional_draft),
        external_shared_weight_bytes=float(external_shared),
        engram_staging_bytes=float(engram_staging),
        vision_weight_bytes=float(vision_bytes),
        engram_weight_bytes=float(engram_bytes),
        media_tokens=int(media_tokens),
        request_peak_tokens=int(peak_tokens),
        per_request_cache=tuple(request_rows),
    )


def _request_peak_tokens(workload: WorkloadSpec, media_tokens: int) -> int:
    if not workload.requests:
        return 0
    return max(
        request.prompt_tokens + request.output_tokens
        for request in workload.requests
    )


def workload_required_tokens(workload: WorkloadSpec) -> int:
    """Live single-request peak committed tokens (not a lifetime sum)."""
    if not workload.requests:
        return 0
    return max(request.prompt_tokens + request.output_tokens for request in workload.requests)


def workload_lifetime_tokens(workload: WorkloadSpec) -> int:
    return sum(request.prompt_tokens + request.output_tokens for request in workload.requests)


def fits_workload(workload: WorkloadSpec, memory: MemoryPlan) -> bool:
    if memory.status != "fits":
        return False
    return workload_required_tokens(workload) <= memory.kv_capacity_tokens


__all__ = [
    "MemoryPlan", "memory_plan", "workload_required_tokens",
    "workload_lifetime_tokens", "fits_workload",
]
