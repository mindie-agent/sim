"""Hardware-limit target-modeling ``RuntimeStackAdapter`` (``ideal-target``).

[Role] Implement the built-in ``ideal-target`` runtime stack plugin for pure
  hardware-limit and Pillar A/B/C target modeling on ``Engine`` across LLM,
  VLM, and Diffusion workloads while preserving exact on-disk
  ``CheckpointQuantMap`` precision boundaries.
[Invariants] Strictly honors ``CheckpointQuantMap`` lookup precedence
  (``layer_exceptions`` -> ``ignored_modules`` -> ``rules``), never guesses
  unknown weight tags, allocates zero redundant prepared weight copies, never
  fabricates LLM KV-Cache on diffusion phases, and never imports ``export/``
  or ``compose/``.
[Timeouts & Locks] Bounded read of sibling ``manifest.json`` (< 50 MiB); pure
  in-memory planning thereafter; no locks.
[Side-Effect Policy] Read-only over local manifest and immutable L0 contracts;
  zero filesystem writes or device context initialization.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Literal

from mindie_sim.catalog.cache_ownership import _draft_cache_parallel
from mindie_sim.catalog.checkpoint_reader import resolve_checkpoint_tensor_tag
from mindie_sim.catalog.draft import draft_quant, optional_draft_spec
from mindie_sim.catalog.draft_geometry import draft_layer_plans
from mindie_sim.catalog.draft_spec import effective_draft_context_cap
from mindie_sim.catalog.npu_checkpoints import _classify_from_quant_spec
from mindie_sim.schema.checkpoint import CheckpointQuantMap, TensorQuantRule
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.operators import KernelReadiness, TensorPrecisionABI
from mindie_sim.schema.parallel import ParallelSpec, local_kv_heads
from mindie_sim.schema.stack import (
    CAPTURE_MODES,
    CachePlacementRule,
    CachePlaneABI,
    CacheStateABI,
    CaptureBucketRule,
    CaptureMode,
    FrameworkRewriteKind,
    IterationDecision,
    IterationSchedulerState,
    LaunchConfigCodec,
    ResolvedStackOptions,
    RewriteTrack,
    RuntimeQuantPlan,
    SchedulerDone,
    SchedulingContract,
    StackCapabilitySpec,
    StackLoweringPlan,
    StackTrack,
    StarsSubmitPolicy,
    StreamLaneBudget,
    StreamOverlapPolicy,
    WaitForEvent,
    resolve_checkpoint_cache_abis,
    resolve_checkpoint_tensor_abi,
    resolve_synthetic_preset_tag,
)
from mindie_sim.schema.workload import IterationPlan, RequestView

MANIFEST_PATH = Path(__file__).resolve().with_name("manifest.json")
MAX_MANIFEST_BYTES = 50 * 1024 * 1024
DEFAULT_KV_BLOCK_SIZE = 128
DEFAULT_LATENT_PATCHES = 4096
DEFAULT_CFG_BRANCHES = 2
DEFAULT_TEXT_ENCODE_TOKENS = 256
EXPERT_FULL_ENUM_LIMIT = 16
DEFAULT_FULL_CAPTURE_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_PIECEWISE_BUCKETS: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)
DEFAULT_STARS_FIFO_LANES = 2
DEFAULT_STARS_QUEUE_DEPTH = 8
MX_BLOCK_ELEMENTS = 32
FP8_BLOCK_ELEMENTS = 128
INT4_GROUP_ELEMENTS = 128
FP32_SCALE_BYTES = 4
FP16_SCALE_BYTES = 2
E8M0_SCALE_BYTES = 1
KV_CACHE_PLANES = 2
DIFFUSION_PHASES = frozenset({"denoise_step", "vae_decode", "text_encode"})
LLM_PHASES = frozenset({"prefill", "decode", "mixed"})
_ZERO_NEURAL_DRAFT_METHODS = frozenset({"ngram", "ngram_gpu"})
_NONCAUSAL_BLOCK_DRAFT_METHODS = frozenset({"dspark", "dflash"})
_EXPERT_ID_RE = re.compile(r"\.experts\.(\d+)\.")

_DIT_LAYER_ROLES: tuple[str, ...] = (
    "attn.qkv_proj",
    "attn.o_proj",
    "ffn.fc1",
    "ffn.fc2",
    "adaln.modulation",
)
_GQA_ATTN_ROLES: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
_MLA_Q_LORA_ATTN_ROLES: tuple[str, ...] = (
    "self_attn.q_a_proj",
    "self_attn.q_b_proj",
    "self_attn.kv_a_proj_with_mqa",
    "self_attn.kv_b_proj",
    "self_attn.o_proj",
)
_MLA_DIRECT_ATTN_ROLES: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.kv_a_proj_with_mqa",
    "self_attn.kv_b_proj",
    "self_attn.o_proj",
)
_GDN_ATTN_ROLES: tuple[str, ...] = (
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "linear_attn.out_proj",
)
_KDA_ATTN_ROLES: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.b_proj",
    "self_attn.f_a_proj",
    "self_attn.f_b_proj",
    "self_attn.g_proj",
    "self_attn.o_proj",
)
_KDA_GATED_ATTN_ROLES: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.b_proj",
    "self_attn.f_a_proj",
    "self_attn.g_a_proj",
    "self_attn.f_b_proj",
    "self_attn.g_b_proj",
    "self_attn.o_proj",
)
_DENSE_MLP_ROLES: tuple[str, ...] = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_SHARED_EXPERT_ROLES: tuple[str, ...] = (
    "mlp.shared_experts.gate_proj",
    "mlp.shared_experts.up_proj",
    "mlp.shared_experts.down_proj",
)
_EXPERT_PROJECTION_SUFFIXES: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")


@lru_cache(maxsize=1)
def _load_manifest() -> dict[str, Any]:
    try:
        size = MANIFEST_PATH.stat().st_size
        if size > MAX_MANIFEST_BYTES:
            raise ModelingError(f"stack_manifest_too_large:{MANIFEST_PATH.name}:{size}")
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except TimeoutError:
        raise
    except (OSError, ValueError) as exc:
        raise ModelingError(f"stack_manifest_load_failed:{MANIFEST_PATH.name}:{exc}") from exc
    if not isinstance(data, dict) or data.get("stack_id") != "ideal-target":
        raise ModelingError("ideal_target_manifest_invalid")
    return data


def _native_float_dtype(model: ModelSpec, checkpoint: CheckpointQuantMap) -> str:
    raw = (checkpoint.native_float_dtype or model.activation_dtype or "bf16").strip().lower()
    if raw == "bfloat16":
        return "bf16"
    if raw == "float16":
        return "fp16"
    if raw == "float32":
        return "fp32"
    return raw


_abi_for_raw_tag = resolve_checkpoint_tensor_abi


def _layer_specs_for(model: ModelSpec) -> tuple[LayerPlan, ...]:
    if model.layer_plans:
        return model.layer_plans
    mla = model.mla
    return tuple(
        LayerPlan(
            index=idx,
            native_layer_id=idx,
            attention=model.attention,
            ffn=model.ffn,
            num_heads=model.num_heads,
            num_kv_heads=model.num_kv_heads or model.num_heads,
            head_dim=model.head_dim,
            q_lora_rank=mla.q_lora_rank if mla is not None else 0,
            kv_lora_rank=mla.kv_lora_rank if mla is not None else 0,
            qk_nope_head_dim=mla.qk_nope_head_dim if mla is not None else 0,
            qk_rope_head_dim=mla.qk_rope_head_dim if mla is not None else 0,
            v_head_dim=mla.v_head_dim if mla is not None else 0,
            num_experts=model.num_experts,
            top_k=model.top_k,
            num_shared_experts=model.num_shared_experts,
        )
        for idx in range(model.layers)
    )


def _resolve_tag_and_rule(
    checkpoint: CheckpointQuantMap,
    tensor_path: str,
    *,
    scope: str = "target",
    source_native_layer_id: int | None = None,
    expert_id: int | None = None,
) -> tuple[str, TensorQuantRule | None]:
    raw = str(tensor_path or "").strip()
    mod_path = raw[:-len(".weight")] if raw.endswith(".weight") else raw
    weight_path = f"{mod_path}.weight"
    probe_paths = (mod_path, weight_path) if mod_path != weight_path else (mod_path,)
    for path in probe_paths:
        exc_or_ign = checkpoint.resolve_exception_or_ignored(
            path, scope=scope, source_native_layer_id=source_native_layer_id, expert_id=expert_id,  # type: ignore[arg-type]
        )
        if exc_or_ign is not None:
            return exc_or_ign, None
    for path in probe_paths:
        for rule in checkpoint.rules:
            if rule.matches(
                path, scope=scope, source_native_layer_id=source_native_layer_id, expert_id=expert_id,  # type: ignore[arg-type]
            ):
                return rule.raw_tag, rule
    if checkpoint.source_format == "unquantized_native":
        return "FLOAT", None
    if checkpoint.source_format == "hf_quantization_config":
        parts = set(mod_path.lower().split("."))
        if "experts" in parts:
            for rule in checkpoint.rules:
                if rule.pattern == "*.experts.*":
                    return rule.raw_tag, rule
        for rule in checkpoint.rules:
            if rule.pattern == "*":
                return rule.raw_tag, rule
    for path in probe_paths:
        tag = resolve_checkpoint_tensor_tag(
            checkpoint, path, scope=scope, source_native_layer_id=source_native_layer_id, expert_id=expert_id,  # type: ignore[arg-type]
        )
        if tag is not None:
            return tag, None
    if checkpoint.source_format == "explicit_synthetic_preset" and checkpoint.model_quant_type:
        return (
            resolve_synthetic_preset_tag(
                checkpoint.model_quant_type, is_routed_expert=(expert_id is not None),
            ),
            None,
        )
    return (checkpoint.model_quant_type or "FLOAT"), None


def _expert_boundary_ids(checkpoint: CheckpointQuantMap, num_experts: int) -> tuple[int, ...]:
    count = max(1, int(num_experts))
    if count <= EXPERT_FULL_ENUM_LIMIT:
        return tuple(range(count))
    ids: set[int] = {0, 1, count - 1}
    for exc in checkpoint.layer_exceptions:
        for val in exc.expert_indices:
            if 0 <= val < count:
                ids.add(val)
        for m in _EXPERT_ID_RE.findall(exc.pattern):
            val = int(m)
            if 0 <= val < count:
                ids.add(val)
    for entry in checkpoint.ignored_modules:
        for m in _EXPERT_ID_RE.findall(entry):
            val = int(m)
            if 0 <= val < count:
                ids.add(val)
    for rule in checkpoint.rules:
        for m in _EXPERT_ID_RE.findall(rule.pattern):
            val = int(m)
            if 0 <= val < count:
                ids.add(val)
        if rule.presence is not None and rule.presence.expert_indices:
            present = [v for v in rule.presence.expert_indices if 0 <= v < count]
            if present:
                ids.add(min(present))
                ids.add(max(present))
    return tuple(sorted(ids))


def _layer_prefix_for(checkpoint: CheckpointQuantMap, *, is_diffusion: bool) -> str:
    if is_diffusion:
        return "transformer_blocks"
    for candidate in ("model.language_model.layers", "language_model.model.layers", "model.layers"):
        marker = candidate + "."
        if any(r.pattern.startswith(marker) for r in checkpoint.rules) or any(
            e.pattern.startswith(marker) for e in checkpoint.layer_exceptions
        ):
            return candidate
    return "model.layers"


def _layer_roles_for(
    plan: LayerPlan, checkpoint: CheckpointQuantMap, *, is_diffusion: bool,
) -> list[tuple[str, int | None]]:
    if is_diffusion:
        return [(role, None) for role in _DIT_LAYER_ROLES]
    attn = str(plan.attention or "").lower()
    if attn == "mla":
        attn_roles = _MLA_Q_LORA_ATTN_ROLES if int(plan.q_lora_rank or 0) > 0 else _MLA_DIRECT_ATTN_ROLES
    elif attn == "gdn":
        attn_roles = _GDN_ATTN_ROLES
    elif attn == "kda":
        attn_roles = _KDA_GATED_ATTN_ROLES if int(plan.gate_rank or 0) > 0 else _KDA_ATTN_ROLES
    else:
        attn_roles = _GQA_ATTN_ROLES
    roles: list[tuple[str, int | None]] = [(role, None) for role in attn_roles]
    if str(plan.ffn or "").lower() == "moe" and int(plan.num_experts or 0) > 0:
        for eid in _expert_boundary_ids(checkpoint, int(plan.num_experts)):
            for suffix in _EXPERT_PROJECTION_SUFFIXES:
                roles.append((f"mlp.experts.{eid}.{suffix}", eid))
        if int(plan.num_shared_experts or 0) > 0:
            roles.extend((role, None) for role in _SHARED_EXPERT_ROLES)
    else:
        roles.extend((role, None) for role in _DENSE_MLP_ROLES)
    return roles


def _draft_role_aliases(logical_role: str) -> tuple[str, ...]:
    """Expand canonical unfused draft role names to checkpoint naming conventions."""
    if logical_role == "self_attn.q_a_proj":
        return (logical_role, "attn.wq_a")
    if logical_role == "self_attn.q_b_proj":
        return (logical_role, "attn.wq_b")
    if logical_role == "self_attn.kv_a_proj_with_mqa":
        return (logical_role, "attn.wkv")
    if logical_role == "self_attn.o_proj":
        return (logical_role, "attn.wo_a", "attn.wo_b")
    if logical_role.startswith("mlp.experts."):
        parts = logical_role.split(".")
        if len(parts) == 4:
            eid, proj = parts[2], parts[3]
            short = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}.get(proj)
            if short:
                return (logical_role, f"ffn.experts.{eid}.{short}")
    if logical_role.startswith("mlp.shared_experts."):
        proj = logical_role.rsplit(".", 1)[-1]
        short = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}.get(proj)
        singular = f"mlp.shared_expert.{proj}"
        return (
            (logical_role, singular, f"ffn.shared_experts.{short}")
            if short else (logical_role, singular)
        )
    return (logical_role,)


def _draft_native_base_layers(model: ModelSpec) -> int:
    """Resolve the untrimmed target body layer count for appended-layer MTP lookup."""
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    return max(
        int(model.body_layers or model.layers),
        int(facts.get("source_body_layers") or 0),
        len(facts.get("mlp_layer_types") or ()),
        len(facts.get("layer_types") or ()),
    )


def _resolve_draft_tag_and_rule(
    checkpoint: CheckpointQuantMap,
    *,
    draft_kind: str,
    prefix: str,
    local_idx: int,
    offset_id: int,
    logical_role: str,
    expert_id: int | None,
    draft_qspec: Any = None,
    native_dtype: str = "bf16",
) -> tuple[str, str, TensorQuantRule | None]:
    """Resolve one neural draft tensor against checkpoint draft/offset paths."""
    draft_prefix = "dspark.layers" if draft_kind == "dspark" else "mtp.layers"
    role_aliases = _draft_role_aliases(logical_role)
    candidates: list[tuple[str, str, int]] = []
    for role_alias in role_aliases:
        candidates.append((f"{draft_prefix}.{local_idx}.{role_alias}", "draft", local_idx))
        candidates.append((f"{draft_kind}.{local_idx}.{role_alias}", "draft", local_idx))
    for role_alias in role_aliases:
        candidates.append((f"{draft_prefix}.{offset_id}.{role_alias}", "draft", offset_id))
        candidates.append((f"{prefix}.{offset_id}.{role_alias}", "draft", offset_id))
        candidates.append((f"{prefix}.{offset_id}.{role_alias}", "target", offset_id))
    for tensor_path, scope_try, lid_try in candidates:
        weight_path = f"{tensor_path}.weight"
        for probe in (tensor_path, weight_path):
            exc = checkpoint.resolve_exception_or_ignored(
                probe, scope=scope_try, source_native_layer_id=lid_try, expert_id=expert_id,  # type: ignore[arg-type]
            )
            if exc is not None:
                return tensor_path, exc, None
        for probe in (tensor_path, weight_path):
            for rule in checkpoint.rules:
                if rule.matches(
                    probe, scope=scope_try, source_native_layer_id=lid_try, expert_id=expert_id,  # type: ignore[arg-type]
                ):
                    return tensor_path, rule.raw_tag, rule
    primary_path = candidates[0][0]
    if checkpoint.source_format == "modelslim_description":
        return primary_path, "FLOAT", None
    if (
        checkpoint.source_format in {"explicit_synthetic_preset", "unquantized_native"}
        and draft_qspec is not None
    ):
        role_cat = "routed_expert" if expert_id is not None else "dense"
        classified = _classify_from_quant_spec(role_cat, draft_qspec, native_format=native_dtype)
        return primary_path, classified.raw_tag, None
    raw_tag, matched_rule = _resolve_tag_and_rule(
        checkpoint,
        primary_path,
        scope="draft",
        source_native_layer_id=local_idx,
        expert_id=expert_id,
    )
    return primary_path, raw_tag, matched_rule


def _draft_invocation_count(spec_method: str, spec_k: int) -> int:
    k = max(0, int(spec_k))
    if k <= 0:
        return 0
    norm = str(spec_method or "").strip().lower()
    if norm in _ZERO_NEURAL_DRAFT_METHODS:
        return 0
    if norm in _NONCAUSAL_BLOCK_DRAFT_METHODS:
        return 1
    return k


def _stars_replay_mode(
    capture_rule: CaptureBucketRule | None, fallback_mode: CaptureMode,
) -> Literal["eager", "captured_segment", "full_graph"]:
    mode = capture_rule.mode if capture_rule is not None else fallback_mode
    has_full = bool(capture_rule.full_keys) if capture_rule is not None else (mode in {"full", "full_decode_only", "full_and_piecewise"})
    has_pw = bool(capture_rule.piecewise_keys) if capture_rule is not None else (mode in {"piecewise", "full_and_piecewise"})
    if mode == "none":
        return "eager"
    if mode in {"full", "full_decode_only"}:
        return "full_graph" if has_full else "eager"
    if mode in {"piecewise", "full_and_piecewise"}:
        if has_pw:
            return "captured_segment"
        if mode == "full_and_piecewise" and has_full:
            return "full_graph"
        return "eager"
    return "eager"


def _positive_option_int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"{code}:{value!r}")
    return int(value)


def _build_llm_cache_abis(
    layers: tuple[LayerPlan, ...],
    checkpoint: CheckpointQuantMap,
    parallel: ParallelSpec,
    native_dtype: str,
    *,
    role: str,
    phase: str,
    kv_block_size: int = DEFAULT_KV_BLOCK_SIZE,
    stage: int | None = None,
    pp_layer_partition: tuple[int, ...] | None = None,
    total_layers: int | None = None,
    default_kv_tag: str = "",
    default_index_tag: str = "",
    kv_tag_override_explicit: bool = False,
    index_tag_override_explicit: bool = False,
) -> tuple[tuple[CacheStateABI, ...], tuple[CacheStateABI, ...], tuple[CacheStateABI, ...]]:
    return resolve_checkpoint_cache_abis(
        layers, checkpoint, parallel, native_dtype,
        role=role, phase=phase, kv_block_size=kv_block_size,
        stage=stage, pp_layer_partition=pp_layer_partition,
        total_layers=total_layers,
        default_kv_tag=default_kv_tag, default_index_tag=default_index_tag,
        kv_tag_override_explicit=kv_tag_override_explicit,
        index_tag_override_explicit=index_tag_override_explicit,
    )


def _normalize_capture_mode(raw: Any, default: CaptureMode) -> CaptureMode:
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text == "eager":
        return "none"
    if text not in CAPTURE_MODES:
        raise ModelingError(f"ideal_target_capture_mode_invalid:{raw!r}")
    return text  # type: ignore[return-value]


def _schedule_one_step(
    state: IterationSchedulerState,
    *,
    phase: str,
    speculative_k: int,
    effective: Mapping[str, Any],
) -> IterationDecision:
    if not state.ready_request_ids:
        if state.in_flight_completions:
            next_ms = min(ts for _, ts in state.in_flight_completions)
            return WaitForEvent(reason="completion", until_ms=next_ms)
        return SchedulerDone(completed_requests=len(state.request_progress))
    priorities = dict(state.request_priorities)
    progress = {rid: (ctx, rem) for rid, ctx, rem in state.request_progress}
    prefixes = dict(state.prefix_hits)
    ordered = sorted(state.ready_request_ids, key=lambda rid: (-priorities.get(rid, 0), rid))
    views: list[RequestView] = []
    batch_budget = state.max_num_batched_tokens
    used_tokens = 0
    if phase in DIFFUSION_PHASES:
        if phase == "denoise_step":
            q_len = int(effective.get("tokens_per_denoise_step", DEFAULT_LATENT_PATCHES * DEFAULT_CFG_BRANCHES))
            exec_mode = "denoise_timestep"
        elif phase == "text_encode":
            q_len = int(effective.get("text_encode_tokens", DEFAULT_TEXT_ENCODE_TOKENS))
            exec_mode = "text_encode"
        else:
            q_len = int(effective.get("latent_patches", DEFAULT_LATENT_PATCHES))
            exec_mode = "vae_decode"
        for rid in ordered:
            if len(views) >= state.max_num_seqs:
                break
            if used_tokens + q_len > batch_budget:
                continue
            views.append(
                RequestView(
                    request_id=rid,
                    q_len=q_len,
                    context=0,
                    samples=False,
                    phase=phase,
                    semantic_phase=phase,
                    execution_mode=exec_mode,
                    priority=priorities.get(rid, 0),
                    draft_verify=False,
                    draft_prepare=False,
                )
            )
            used_tokens += q_len
    else:
        input_budget = (
            min(batch_budget, state.max_num_input_tokens)
            if state.max_num_input_tokens is not None
            else batch_budget
        )
        used_input_tokens = 0
        for rid in ordered:
            if len(views) >= state.max_num_seqs or used_tokens >= batch_budget:
                break
            ctx, rem = progress.get(rid, (0, 0 if phase == "decode" else 1))
            is_decode_req = phase == "decode" or (phase == "mixed" and rem <= 0)
            if is_decode_req:
                q_len = 1
                draft_proposals = speculative_k
                step_tokens = q_len + draft_proposals
                if used_tokens + step_tokens > batch_budget:
                    continue
                req_phase = "decode"
                samples = True
            else:
                raw_q = max(1, rem)
                avail = min(batch_budget - used_tokens, input_budget - used_input_tokens)
                if avail <= 0:
                    continue
                if state.chunked_prefill:
                    q_len = min(raw_q, avail)
                elif raw_q <= avail:
                    q_len = raw_q
                else:
                    continue
                draft_proposals = 0
                step_tokens = q_len
                req_phase = "prefill"
                samples = q_len >= raw_q
                used_input_tokens += q_len
            views.append(
                RequestView(
                    request_id=rid,
                    q_len=q_len,
                    context=ctx,
                    samples=samples,
                    phase=req_phase,
                    prefix_tokens=prefixes.get(rid, 0),
                    priority=priorities.get(rid, 0),
                    draft_verify=(draft_proposals > 0),
                    draft_prepare=(draft_proposals > 0),
                    draft_proposal_tokens=draft_proposals,
                )
            )
            used_tokens += step_tokens
    if not views:
        if state.in_flight_completions:
            next_ms = min(ts for _, ts in state.in_flight_completions)
            return WaitForEvent(reason="completion", until_ms=next_ms)
        raise ModelingError(
            f"iteration_step_exceeds_token_budget:stack=ideal-target:phase={phase}:budget={batch_budget}"
        )
    return IterationPlan(
        requests=tuple(views),
        reason="scheduled",
        all_request_ids=state.ready_request_ids,
    )


class IdealTargetStackAdapter:
    """Pure hardware-limit / target-modeling ``RuntimeStackAdapter``."""

    def __init__(self, manifest: Mapping[str, Any] | None = None) -> None:
        raw = dict(_load_manifest()) if manifest is None else dict(manifest)
        self._manifest = raw
        self._capabilities = StackCapabilitySpec(
            stack_id=str(raw["stack_id"]),
            manifest_revision=str(raw["manifest_revision"]),
            supported_domains=tuple(raw["supported_domains"]),
            supported_work_units=tuple(raw["supported_work_units"]),
            supported_hardware_backends=tuple(raw["supported_hardware_backends"]),
            supported_phases=tuple(raw["supported_phases"]),
            supported_checkpoint_formats=tuple(raw["supported_checkpoint_formats"]),
            supported_tracks=tuple(raw["supported_tracks"]),
            supported_recipe_ids=frozenset(raw.get("supported_recipe_ids", ())),
            has_launch_codec=bool(raw.get("has_launch_codec", False)),
            protocol_version=str(raw.get("protocol_version", "mindie-sim.stack-adapter/1")),
        )

    def capabilities(self) -> StackCapabilitySpec:
        return self._capabilities

    def default_checkpoint_for(
        self, canonical_model: str, hardware: Hardware
    ) -> str | None:
        del canonical_model, hardware
        return None

    def resolve_options(
        self,
        model: ModelSpec,
        hardware: Hardware,
        parallel: ParallelSpec,
        *,
        role: str,
        track: StackTrack,
        requested_options: Mapping[str, Any],
        field_origins: Mapping[str, str],
    ) -> ResolvedStackOptions:
        del model, field_origins
        if track not in self._capabilities.supported_tracks:
            raise ModelingError(
                f"stack_track_unsupported:stack={self._capabilities.stack_id}:track={track}"
            )
        req = dict(requested_options)
        dispositions: list[tuple[str, str, str]] = []
        enforce_eager = bool(req.get("enforce_eager", False)) or track == "eager"
        default_cap: CaptureMode = "none" if enforce_eager else "full_and_piecewise"
        cap_mode = "none" if enforce_eager else _normalize_capture_mode(req.get("capture_mode"), default_cap)
        spec_k = int(req.get("speculative_k") or req.get("num_speculative_tokens") or 0)
        spec_cfg_req = req.get("speculative_config")
        spec_method = str(
            req.get("spec_method")
            or (spec_cfg_req.get("method") if isinstance(spec_cfg_req, Mapping) else "")
            or ""
        ).strip().lower()
        verify_stride = max(1, 1 + spec_k)
        full_keys = (
            ()
            if cap_mode in {"none", "piecewise"}
            else tuple(k * verify_stride for k in DEFAULT_FULL_CAPTURE_BUCKETS)
        )
        pw_keys = () if cap_mode in {"none", "full", "full_decode_only"} else DEFAULT_PIECEWISE_BUCKETS
        capture_rule = CaptureBucketRule(
            mode=cap_mode,
            full_keys=full_keys,
            piecewise_keys=pw_keys,
            verify_stride=verify_stride,
            fallback_mode="piecewise" if cap_mode == "full_and_piecewise" else "eager",
        )
        raw_kv_block = req.get("kv_block_size")
        kv_block_size = (
            _positive_option_int(raw_kv_block, "ideal_target_kv_block_size_invalid")
            if raw_kv_block is not None
            else DEFAULT_KV_BLOCK_SIZE
        )
        overlaps: tuple[str, ...] = ()
        stream_policy: StreamOverlapPolicy | None = None
        if track == "target" and not enforce_eager:
            overlaps = ("shared_expert", "comm_compute") if parallel.world_size > 1 else ("shared_expert",)
            stream_policy = StreamOverlapPolicy(
                lanes=(
                    StreamLaneBudget(stream_id="main", die_id=0),
                    StreamLaneBudget(stream_id="aux", die_id=0),
                ),
                service_terminal="step_done",
                enabled_overlaps=overlaps,
            )
        opt_mode = str(req.get("optimization_mode", "modeling" if track == "target" else "simulation")).strip().lower()
        if opt_mode not in {"simulation", "modeling"}:
            raise ModelingError(f"ideal_target_optimization_mode_invalid:{opt_mode!r}")
        for key in sorted(req):
            dispositions.append((key, "accepted", "ideal_target_option_applied"))
        effective: dict[str, Any] = {
            "capture_mode": "eager" if cap_mode == "none" else cap_mode,
            "kv_block_size": kv_block_size,
            "optimization_mode": opt_mode,
            "role": role,
            "arch_family": hardware.arch_family,
            "pillar_abc_exploration": (track == "target" and opt_mode == "modeling"),
            "speculative_k": spec_k,
        }
        if spec_method:
            effective["spec_method"] = spec_method
        if req.get("kv_cache_dtype"):
            effective["kv_cache_dtype"] = str(req["kv_cache_dtype"]).strip()
        if req.get("kv_cache_dtype_explicit"):
            effective["kv_cache_dtype_explicit"] = bool(req["kv_cache_dtype_explicit"])
        if req.get("indexer_kv_dtype"):
            effective["indexer_kv_dtype"] = str(req["indexer_kv_dtype"]).strip()
        if req.get("indexer_kv_dtype_explicit"):
            effective["indexer_kv_dtype_explicit"] = bool(req["indexer_kv_dtype_explicit"])
        if req.get("pp_layer_partition"):
            part_vals: list[int] = []
            for v in req["pp_layer_partition"]:
                if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                    raise ModelingError(f"ideal_target_pp_partition_invalid:{v!r}")
                part_vals.append(int(v))
            effective["pp_layer_partition"] = tuple(part_vals)
        if req.get("draft_max_model_len") is not None:
            effective["draft_max_model_len"] = _positive_option_int(
                req["draft_max_model_len"], "ideal_target_draft_max_model_len_invalid",
            )
        if any(k in req for k in ("latent_patches", "cfg_branches", "text_encode_tokens")):
            patches = _positive_option_int(req.get("latent_patches", DEFAULT_LATENT_PATCHES), "diffusion_latent_patches_invalid")
            cfg = _positive_option_int(req.get("cfg_branches", DEFAULT_CFG_BRANCHES), "diffusion_cfg_branches_invalid")
            text_toks = _positive_option_int(req.get("text_encode_tokens", DEFAULT_TEXT_ENCODE_TOKENS), "diffusion_text_tokens_invalid")
            effective["latent_patches"] = patches
            effective["cfg_branches"] = cfg
            effective["text_encode_tokens"] = text_toks
            effective["tokens_per_denoise_step"] = patches * cfg
        return ResolvedStackOptions(
            stack_id=self._capabilities.stack_id,
            track=track,
            requested_mechanisms=tuple(sorted(req.items())),
            effective_mechanisms=tuple(sorted(effective.items())),
            dispositions=tuple(dispositions),
            capture_rule=capture_rule,
            graph_capture_mode=cap_mode,
            graph_capture_keys=full_keys,
            piecewise_capture_keys=pw_keys,
            stream_policy=stream_policy,
            collective_policy=(("overlap_enabled", bool(overlaps)),),
        )

    def resolve_quant_instantiation(
        self,
        model: ModelSpec,
        checkpoint: CheckpointQuantMap,
        hardware: Hardware,
        parallel: ParallelSpec,
        options: ResolvedStackOptions,
        *,
        role: str,
        phase: str,
    ) -> RuntimeQuantPlan:
        """[Cohesive-Pipeline] Resolve target-stack tensor, cache, and recurrent quantization ABIs in one pass."""
        del hardware
        if phase not in self._capabilities.supported_phases:
            raise ModelingError(
                f"stack_phase_unsupported:stack={self._capabilities.stack_id}:phase={phase}"
            )
        native_dtype = _native_float_dtype(model, checkpoint)
        layers = _layer_specs_for(model)
        is_diffusion = phase in DIFFUSION_PHASES
        prefix = _layer_prefix_for(checkpoint, is_diffusion=is_diffusion)
        port_abis: list[tuple[tuple[str, int, str], TensorPrecisionABI]] = []
        dispositions: list[tuple[str, str, str]] = []
        for plan in layers:
            native_id = plan.native_layer_id if plan.native_layer_id >= 0 else plan.index
            for logical_role, expert_id in _layer_roles_for(plan, checkpoint, is_diffusion=is_diffusion):
                tensor_path = f"{prefix}.{native_id}.{logical_role}"
                raw_tag, matched_rule = _resolve_tag_and_rule(
                    checkpoint,
                    tensor_path,
                    scope="target",
                    source_native_layer_id=native_id,
                    expert_id=expert_id,
                )
                abi = _abi_for_raw_tag(
                    raw_tag, native_dtype, matched_rule,
                    role=role, is_routed_expert=(expert_id is not None),
                )
                port_abis.append((("target", plan.index, logical_role), abi))
                dispositions.append((tensor_path, raw_tag, "checkpoint_precision_honored"))
        for rule in checkpoint.rules:
            _abi_for_raw_tag(rule.raw_tag, native_dtype, rule)
        for exc in checkpoint.layer_exceptions:
            _abi_for_raw_tag(exc.raw_tag, native_dtype)

        eff = dict(options.effective_mechanisms)
        spec_k = int(eff.get("speculative_k", 0) or 0)
        spec_method = str(eff.get("spec_method", "") or "")
        draft_invocations = _draft_invocation_count(spec_method, spec_k)
        aux = getattr(model, "auxiliary", None)
        draft_kind = ""
        draft_plans: tuple[LayerPlan, ...] = ()
        draft_spec: Any = None
        draft_qspec: Any = None
        if not is_diffusion and draft_invocations > 0 and aux is not None:
            draft_kind = (
                "dspark"
                if (spec_method in _NONCAUSAL_BLOCK_DRAFT_METHODS and aux.dspark is not None)
                else ("mtp" if aux.mtp is not None else ("dspark" if aux.dspark is not None else ""))
            )
            if draft_kind:
                draft_spec = optional_draft_spec(model, draft_kind)
                draft_qspec = (
                    draft_quant(model, draft_spec)
                    if (draft_spec is not None and model.quant is not None)
                    else None
                )
                draft_plans = draft_layer_plans(model, draft_kind)
                start_id = _draft_native_base_layers(model)
                for d_plan in draft_plans:
                    local_idx = int(d_plan.index)
                    offset_id = (
                        int(d_plan.native_layer_id)
                        if int(d_plan.native_layer_id) >= 0
                        else (start_id + local_idx)
                    )
                    for logical_role, expert_id in _layer_roles_for(d_plan, checkpoint, is_diffusion=False):
                        d_path, d_tag, d_rule = _resolve_draft_tag_and_rule(
                            checkpoint,
                            draft_kind=draft_kind,
                            prefix=prefix,
                            local_idx=local_idx,
                            offset_id=offset_id,
                            logical_role=logical_role,
                            expert_id=expert_id,
                            draft_qspec=draft_qspec,
                            native_dtype=native_dtype,
                        )
                        d_abi = _abi_for_raw_tag(
                            d_tag, native_dtype, d_rule,
                            role=role, is_routed_expert=(expert_id is not None),
                        )
                        port_abis.append((("draft", local_idx, logical_role), d_abi))
                        dispositions.append((d_path, d_tag, "checkpoint_precision_honored"))

        if is_diffusion:
            kv_abis: tuple[CacheStateABI, ...] = ()
            idx_abis: tuple[CacheStateABI, ...] = ()
            rec_abis: tuple[CacheStateABI, ...] = ()
        else:
            kv_block = int(eff.get("kv_block_size") or DEFAULT_KV_BLOCK_SIZE)
            raw_pp_part = eff.get("pp_layer_partition")
            pp_part = tuple(int(v) for v in raw_pp_part) if raw_pp_part else None
            default_kv_tag = str(eff.get("kv_cache_dtype") or "")
            default_idx_tag = str(eff.get("indexer_kv_dtype") or "")
            kv_abis, idx_abis, rec_abis = _build_llm_cache_abis(
                layers, checkpoint, parallel, native_dtype,
                role=role, phase=phase, kv_block_size=kv_block,
                pp_layer_partition=pp_part, total_layers=model.layers,
                default_kv_tag=default_kv_tag, default_index_tag=default_idx_tag,
                kv_tag_override_explicit=bool(eff.get("kv_cache_dtype_explicit", False)),
                index_tag_override_explicit=bool(eff.get("indexer_kv_dtype_explicit", False)),
            )
            if draft_plans:
                raw_dml = int(eff.get("draft_max_model_len") or 0)
                cap_hint = (
                    type("_DraftCapHint", (), {"metadata": (("draft_max_model_len", raw_dml),)})()
                    if raw_dml > 0 else None
                )
                d_cap_val = effective_draft_context_cap(draft_spec, cap_hint)
                d_cap = d_cap_val if d_cap_val > 0 else None
                draft_kv_tag = str(getattr(draft_qspec, "kv_dtype", "") or "").strip() if draft_qspec is not None else ""
                draft_idx_tag = str(getattr(draft_qspec, "index_dtype", "") or "").strip() if draft_qspec is not None else ""
                target_kv_low = str(getattr(model, "kv_dtype", "") or "").strip().lower()
                target_idx_low = (
                    str(getattr(model.quant, "index_dtype", "") or "").strip().lower()
                    if model.quant is not None else ""
                )
                eff_draft_kv = (
                    draft_kv_tag.upper()
                    if (draft_kv_tag and (draft_kv_tag.lower() != target_kv_low or not checkpoint.cache_rule.kv_tag))
                    else checkpoint.cache_rule.kv_tag
                )
                eff_draft_idx = (
                    draft_idx_tag.upper()
                    if (draft_idx_tag and (draft_idx_tag.lower() != target_idx_low or not checkpoint.cache_rule.index_tag))
                    else checkpoint.cache_rule.index_tag
                )
                draft_cache_rule = replace(
                    checkpoint.cache_rule,
                    kv_tag=eff_draft_kv,
                    index_tag=eff_draft_idx,
                    layer_scoped_kv=(),
                    layer_scoped_index=(),
                )
                draft_ckpt = replace(checkpoint, cache_rule=draft_cache_rule)
                draft_parallel = _draft_cache_parallel(model, parallel, draft_kind, interleave=kv_block)
                orig_digest = checkpoint.digest()
                last_stage = max(0, int(parallel.pp) - 1)
                d_kv, d_idx, d_rec = _build_llm_cache_abis(
                    draft_plans, draft_ckpt, draft_parallel, native_dtype,
                    role=role, phase=phase, kv_block_size=kv_block,
                    stage=last_stage,
                    default_kv_tag=draft_kv_tag or default_kv_tag,
                    default_index_tag=draft_idx_tag or default_idx_tag,
                )
                d_kv = tuple(replace(abi, checkpoint_digest=orig_digest, draft_context_cap=d_cap) for abi in d_kv)
                d_idx = tuple(replace(abi, checkpoint_digest=orig_digest, draft_context_cap=d_cap) for abi in d_idx)
                d_rec = tuple(replace(abi, checkpoint_digest=orig_digest, draft_context_cap=d_cap) for abi in d_rec)
                kv_abis = (*kv_abis, *d_kv)
                idx_abis = (*idx_abis, *d_idx)
                rec_abis = (*rec_abis, *d_rec)
        return RuntimeQuantPlan(
            checkpoint_digest=checkpoint.digest(),
            stack_id=self._capabilities.stack_id,
            port_abis=tuple(port_abis),
            packed_groups=(),
            kv_cache_abi=kv_abis,
            index_cache_abi=idx_abis,
            recurrent_state_abi=rec_abis,
            prepared_weights=(),
            dispositions=tuple(dispositions),
            provenance=checkpoint.provenance.origin,
        )

    def select_lowering_plan(
        self,
        model: ModelSpec,
        quant_plan: RuntimeQuantPlan,
        hardware: Hardware,
        parallel: ParallelSpec,
        options: ResolvedStackOptions,
    ) -> StackLoweringPlan:
        del model, quant_plan, hardware, parallel
        eff = dict(options.effective_mechanisms)
        opt_mode = str(eff.get("optimization_mode", "modeling" if options.track == "target" else "simulation"))
        spec_k = int(eff.get("speculative_k", 0) or 0)
        spec_method = str(eff.get("spec_method", "") or "")
        draft_invocations = _draft_invocation_count(spec_method, spec_k)
        if options.track == "eager":
            allowed_tracks: tuple[RewriteTrack, ...] = ()
            allowed_classes: frozenset[KernelReadiness] = frozenset({"shipping"})
            allowed_fw: tuple[FrameworkRewriteKind, ...] = ()
            passes: tuple[str, ...] = ()
        elif opt_mode == "simulation" or options.track in {"shipping", "proposed"}:
            allowed_tracks = ("config",)
            allowed_classes = (
                frozenset({"shipping", "proposed_pr"})
                if options.track == "proposed"
                else frozenset({"shipping"})
            )
            allowed_fw = ()
            passes = ("activation_quantization",)
        else:
            allowed_tracks = tuple(self._manifest["allowed_rewrite_tracks"])
            allowed_classes = frozenset(self._manifest["allowed_rewrite_classes"])
            allowed_fw = tuple(self._manifest["allowed_framework_rewrites"])
            passes = tuple(self._manifest.get("lowering_passes", ("activation_quantization",)))
        replay_mode = _stars_replay_mode(options.capture_rule, options.graph_capture_mode)
        scheduling = SchedulingContract(
            capture_bucket_rule=options.capture_rule,
            stars_submit_policy=(
                StarsSubmitPolicy(
                    die_id=0,
                    stream_id="main",
                    owner_kind="aic",
                    fifo_lanes=DEFAULT_STARS_FIFO_LANES,
                    queue_depth=DEFAULT_STARS_QUEUE_DEPTH,
                    replay_mode=replay_mode,
                ),
            ),
            stream_overlap_policy=options.stream_policy,
            chunked_prefill_enabled=True,
            speculative_proposal_count=spec_k,
            draft_invocation_count=draft_invocations,
        )
        return StackLoweringPlan(
            stack_id=self._capabilities.stack_id,
            track=options.track,
            recipe_bindings=(),
            lowering_passes=passes,
            allowed_rewrite_classes=allowed_classes,
            allowed_rewrite_tracks=allowed_tracks,
            allowed_framework_rewrites=allowed_fw,
            scheduling=scheduling,
            binding_origins=(("stack", "stack_profile"),),
        )

    def plan_iteration(
        self,
        model: ModelSpec,
        state: IterationSchedulerState,
        options: ResolvedStackOptions,
        *,
        phase: str,
    ) -> IterationDecision:
        del model
        if phase not in self._capabilities.supported_phases:
            raise ModelingError(
                f"stack_phase_unsupported:stack={self._capabilities.stack_id}:phase={phase}"
            )
        eff = dict(options.effective_mechanisms)
        spec_k = int(eff.get("speculative_k", 0) or 0)
        return _schedule_one_step(state, phase=phase, speculative_k=spec_k, effective=eff)

    @property
    def launch_codec(self) -> LaunchConfigCodec | None:
        return None


__all__ = ["IdealTargetStackAdapter", "MANIFEST_PATH"]
