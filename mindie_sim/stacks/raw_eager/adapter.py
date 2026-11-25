"""Pure eager baseline ``RuntimeStackAdapter`` (``raw-eager``).

[Role] Implement the built-in ``raw-eager`` baseline stack plugin with
  ``capture_mode="eager"``, zero graph capture, zero optional fusions, zero
  prepared weight copies, and ``allowed_rewrite_tracks=()``.
[Invariants] Accepts only ``track="eager"``, rejects capture/fusion requests in
  explicit ``dispositions`` without enabling them, honors on-disk
  ``CheckpointQuantMap`` precision verbatim, never fabricates LLM KV-Cache on
  diffusion phases, and never imports ``export/``, ``compose/``, or
  ``stacks.registry``.
[Timeouts & Locks] Bounded read of sibling ``manifest.json`` (< 50 MiB); pure
  in-memory planning thereafter; no locks.
[Side-Effect Policy] Read-only over local manifest and immutable L0 contracts;
  zero filesystem writes or device context initialization.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

from mindie_sim.catalog.checkpoint_reader import resolve_checkpoint_tensor_tag
from mindie_sim.schema.checkpoint import CheckpointQuantMap, TensorQuantRule
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.operators import TensorPrecisionABI
from mindie_sim.schema.parallel import ParallelSpec, local_kv_heads
from mindie_sim.schema.stack import (
    CachePlacementRule,
    CachePlaneABI,
    CacheStateABI,
    CaptureBucketRule,
    IterationDecision,
    IterationSchedulerState,
    LaunchConfigCodec,
    ResolvedStackOptions,
    RuntimeQuantPlan,
    SchedulerDone,
    SchedulingContract,
    StackCapabilitySpec,
    StackLoweringPlan,
    StackTrack,
    StarsSubmitPolicy,
    WaitForEvent,
    resolve_checkpoint_cache_abis,
    resolve_checkpoint_tensor_abi,
)
from mindie_sim.schema.workload import IterationPlan, RequestView

MANIFEST_PATH = Path(__file__).resolve().with_name("manifest.json")
MAX_MANIFEST_BYTES = 50 * 1024 * 1024
DEFAULT_KV_BLOCK_SIZE = 128
DEFAULT_LATENT_PATCHES = 4096
DEFAULT_CFG_BRANCHES = 2
DEFAULT_TEXT_ENCODE_TOKENS = 256
EXPERT_FULL_ENUM_LIMIT = 16
MX_BLOCK_ELEMENTS = 32
FP8_BLOCK_ELEMENTS = 128
INT4_GROUP_ELEMENTS = 128
FP32_SCALE_BYTES = 4
FP16_SCALE_BYTES = 2
E8M0_SCALE_BYTES = 1
KV_CACHE_PLANES = 2
DIFFUSION_PHASES = frozenset({"denoise_step", "vae_decode", "text_encode"})
_EXPERT_ID_RE = re.compile(r"\.experts\.(\d+)\.")
_REJECTED_EAGER_OPTION_KEYS = frozenset({
    "capture_mode",
    "graph_capture_mode",
    "aclgraph",
    "enable_fusion",
    "enable_multistream",
    "enable_fused_mc2",
    "lowering_passes",
    "speculative_k",
})
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
    if not isinstance(data, dict) or data.get("stack_id") != "raw-eager":
        raise ModelingError("raw_eager_manifest_invalid")
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
    exc_or_ign = checkpoint.resolve_exception_or_ignored(
        mod_path, scope=scope, source_native_layer_id=source_native_layer_id, expert_id=expert_id,  # type: ignore[arg-type]
    )
    if exc_or_ign is not None:
        return exc_or_ign, None
    for rule in checkpoint.rules:
        if rule.matches(
            mod_path, scope=scope, source_native_layer_id=source_native_layer_id, expert_id=expert_id,  # type: ignore[arg-type]
        ):
            return rule.raw_tag, rule
    if mod_path != raw:
        exc_raw = checkpoint.resolve_exception_or_ignored(
            raw, scope=scope, source_native_layer_id=source_native_layer_id, expert_id=expert_id,  # type: ignore[arg-type]
        )
        if exc_raw is not None:
            return exc_raw, None
        for rule in checkpoint.rules:
            if rule.matches(
                raw, scope=scope, source_native_layer_id=source_native_layer_id, expert_id=expert_id,  # type: ignore[arg-type]
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
    tag = resolve_checkpoint_tensor_tag(
        checkpoint, tensor_path, scope=scope, source_native_layer_id=source_native_layer_id, expert_id=expert_id,  # type: ignore[arg-type]
    )
    return (tag if tag is not None else (checkpoint.model_quant_type or "FLOAT")), None


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
    attn_roles = (
        (_MLA_Q_LORA_ATTN_ROLES if int(plan.q_lora_rank or 0) > 0 else _MLA_DIRECT_ATTN_ROLES)
        if str(plan.attention or "").lower() == "mla"
        else _GQA_ATTN_ROLES
    )
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


def _positive_option_int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"{code}:{value!r}")
    return int(value)


def _eager_cache_abis(
    layers: tuple[LayerPlan, ...],
    checkpoint: CheckpointQuantMap,
    parallel: ParallelSpec,
    native_dtype: str,
    *,
    role: str,
    phase: str,
    kv_block_size: int = DEFAULT_KV_BLOCK_SIZE,
) -> tuple[tuple[CacheStateABI, ...], tuple[CacheStateABI, ...], tuple[CacheStateABI, ...]]:
    return resolve_checkpoint_cache_abis(
        layers, checkpoint, parallel, native_dtype,
        role=role, phase=phase, kv_block_size=kv_block_size,
    )


class RawEagerStackAdapter:
    """Pure eager baseline ``RuntimeStackAdapter`` with zero rewrites and zero capture."""

    capture_mode: str = "eager"

    def __init__(self, manifest: Mapping[str, Any] | None = None) -> None:
        raw = dict(_load_manifest()) if manifest is None else dict(manifest)
        self._manifest = raw
        self.capture_mode = str(raw.get("capture_mode", "eager"))
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
        del model, hardware, parallel, field_origins
        if track != "eager":
            raise ModelingError(
                f"stack_track_unsupported:stack={self._capabilities.stack_id}:track={track}"
            )
        req = dict(requested_options)
        dispositions: list[tuple[str, str, str]] = []
        for key, value in sorted(req.items()):
            is_eager_noop = (
                key in {"capture_mode", "graph_capture_mode"}
                and str(value).strip().lower() in {"eager", "none"}
            )
            if key in _REJECTED_EAGER_OPTION_KEYS and not is_eager_noop and value not in (None, False, 0, "", ()):
                dispositions.append((key, "rejected", "raw_eager_disables_graph_capture_and_fusions"))
            else:
                dispositions.append((key, "accepted", "raw_eager_baseline"))
        capture_rule = CaptureBucketRule(mode="none", fallback_mode="eager")
        raw_kv_block = req.get("kv_block_size")
        kv_block_size = (
            _positive_option_int(raw_kv_block, "raw_eager_kv_block_size_invalid")
            if raw_kv_block is not None
            else DEFAULT_KV_BLOCK_SIZE
        )
        effective: dict[str, Any] = {
            "capture_mode": "eager",
            "automatic_optional_fusions": False,
            "kv_block_size": kv_block_size,
            "role": role,
        }
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
            track="eager",
            requested_mechanisms=tuple(sorted(req.items())),
            effective_mechanisms=tuple(sorted(effective.items())),
            dispositions=tuple(dispositions),
            capture_rule=capture_rule,
            graph_capture_mode="none",
            graph_capture_keys=(),
            piecewise_capture_keys=(),
            splitting_ops=(),
            speculative=None,
            stream_policy=None,
            collective_policy=(("overlap_enabled", False),),
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
                abi = _abi_for_raw_tag(raw_tag, native_dtype, matched_rule)
                port_abis.append((("target", plan.index, logical_role), abi))
                dispositions.append((tensor_path, raw_tag, "raw_eager_unfused_checkpoint_precision"))
        for rule in checkpoint.rules:
            _abi_for_raw_tag(rule.raw_tag, native_dtype, rule)
        for exc in checkpoint.layer_exceptions:
            _abi_for_raw_tag(exc.raw_tag, native_dtype)

        if is_diffusion:
            kv_abis: tuple[CacheStateABI, ...] = ()
            idx_abis: tuple[CacheStateABI, ...] = ()
            rec_abis: tuple[CacheStateABI, ...] = ()
        else:
            eff = dict(options.effective_mechanisms)
            kv_block = int(eff.get("kv_block_size") or DEFAULT_KV_BLOCK_SIZE)
            kv_abis, idx_abis, rec_abis = _eager_cache_abis(
                layers, checkpoint, parallel, native_dtype,
                role=role, phase=phase, kv_block_size=kv_block,
            )
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
        if options.track != "eager":
            raise ModelingError(
                f"stack_track_unsupported:stack={self._capabilities.stack_id}:track={options.track}"
            )
        scheduling = SchedulingContract(
            capture_bucket_rule=CaptureBucketRule(mode="none", fallback_mode="eager"),
            stars_submit_policy=(
                StarsSubmitPolicy(
                    die_id=0,
                    stream_id="main",
                    owner_kind="aic",
                    fifo_lanes=1,
                    queue_depth=1,
                    replay_mode="eager",
                ),
            ),
            stream_overlap_policy=None,
            chunked_prefill_enabled=True,
            speculative_proposal_count=0,
            draft_invocation_count=0,
        )
        return StackLoweringPlan(
            stack_id=self._capabilities.stack_id,
            track="eager",
            recipe_bindings=(),
            lowering_passes=(),
            allowed_rewrite_classes=frozenset({"shipping"}),
            allowed_rewrite_tracks=(),
            allowed_framework_rewrites=(),
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
        if not state.ready_request_ids:
            if state.in_flight_completions:
                next_ms = min(ts for _, ts in state.in_flight_completions)
                return WaitForEvent(reason="completion", until_ms=next_ms)
            return SchedulerDone(completed_requests=len(state.request_progress))
        eff = dict(options.effective_mechanisms)
        priorities = dict(state.request_priorities)
        progress = {rid: (ctx, rem) for rid, ctx, rem in state.request_progress}
        ordered = sorted(state.ready_request_ids, key=lambda rid: (-priorities.get(rid, 0), rid))
        views: list[RequestView] = []
        batch_budget = state.max_num_batched_tokens
        used_tokens = 0
        if phase in DIFFUSION_PHASES:
            if phase == "denoise_step":
                q_len = int(eff.get("tokens_per_denoise_step", DEFAULT_LATENT_PATCHES * DEFAULT_CFG_BRANCHES))
                exec_mode = "denoise_timestep"
            elif phase == "text_encode":
                q_len = int(eff.get("text_encode_tokens", DEFAULT_TEXT_ENCODE_TOKENS))
                exec_mode = "text_encode"
            else:
                q_len = int(eff.get("latent_patches", DEFAULT_LATENT_PATCHES))
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
                    if used_tokens + q_len > batch_budget:
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
                        priority=priorities.get(rid, 0),
                        draft_verify=False,
                        draft_prepare=False,
                    )
                )
                used_tokens += q_len
        if not views:
            if state.in_flight_completions:
                next_ms = min(ts for _, ts in state.in_flight_completions)
                return WaitForEvent(reason="completion", until_ms=next_ms)
            raise ModelingError(
                f"iteration_step_exceeds_token_budget:stack={self._capabilities.stack_id}:phase={phase}:budget={batch_budget}"
            )
        return IterationPlan(
            requests=tuple(views),
            reason="scheduled",
            all_request_ids=state.ready_request_ids,
        )

    @property
    def launch_codec(self) -> LaunchConfigCodec | None:
        return None


__all__ = ["MANIFEST_PATH", "RawEagerStackAdapter"]
