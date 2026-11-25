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
MX_BLOCK_ELEMENTS = 32
FP8_BLOCK_ELEMENTS = 128
INT4_GROUP_ELEMENTS = 128
FP32_SCALE_BYTES = 4
FP16_SCALE_BYTES = 2
E8M0_SCALE_BYTES = 1
KV_CACHE_PLANES = 2
DIFFUSION_PHASES = frozenset({"denoise_step", "vae_decode", "text_encode"})
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
_STANDARD_LAYER_ROLES: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_MOE_EXPERT_ROLE = "mlp.experts.0.down_proj"


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
    return tuple(
        LayerPlan(
            index=idx,
            native_layer_id=idx,
            attention=model.attention,
            ffn=model.ffn,
            num_heads=model.num_heads,
            num_kv_heads=model.num_kv_heads or model.num_heads,
            head_dim=model.head_dim,
            kv_lora_rank=model.mla.kv_lora_rank if model.mla is not None else 0,
            qk_rope_head_dim=model.mla.qk_rope_head_dim if model.mla is not None else 0,
            num_experts=model.num_experts,
            top_k=model.top_k,
        )
        for idx in range(model.layers)
    )


def _eager_cache_abis(
    layers: tuple[LayerPlan, ...],
    checkpoint: CheckpointQuantMap,
    parallel: ParallelSpec,
    native_dtype: str,
    *,
    role: str,
    phase: str,
) -> tuple[tuple[CacheStateABI, ...], tuple[CacheStateABI, ...], tuple[CacheStateABI, ...]]:
    return resolve_checkpoint_cache_abis(
        layers, checkpoint, parallel, native_dtype,
        role=role, phase=phase, kv_block_size=DEFAULT_KV_BLOCK_SIZE,
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
        effective: dict[str, Any] = {
            "capture_mode": "eager",
            "automatic_optional_fusions": False,
            "role": role,
        }
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
        del hardware, options
        if phase not in self._capabilities.supported_phases:
            raise ModelingError(
                f"stack_phase_unsupported:stack={self._capabilities.stack_id}:phase={phase}"
            )
        native_dtype = _native_float_dtype(model, checkpoint)
        layers = _layer_specs_for(model)
        rule_by_pattern = {rule.pattern: rule for rule in checkpoint.rules}
        port_abis: list[tuple[tuple[str, int, str], TensorPrecisionABI]] = []
        dispositions: list[tuple[str, str, str]] = []
        for plan in layers:
            native_id = plan.native_layer_id if plan.native_layer_id >= 0 else plan.index
            roles = list(_STANDARD_LAYER_ROLES)
            if plan.num_experts > 0 or model.num_experts > 0:
                roles.append(_MOE_EXPERT_ROLE)
            for logical_role in roles:
                tensor_path = f"model.layers.{native_id}.{logical_role}"
                raw_tag = resolve_checkpoint_tensor_tag(
                    checkpoint,
                    tensor_path,
                    scope="target",
                    source_native_layer_id=native_id,
                    expert_id=0 if "experts." in logical_role else None,
                )
                matched_rule = (
                    rule_by_pattern.get(f"model.layers.*.{logical_role}")
                    or (rule_by_pattern.get("*.experts.*") if "experts." in logical_role else None)
                    or rule_by_pattern.get("*")
                )
                if raw_tag is None:
                    raw_tag = checkpoint.model_quant_type or "FLOAT"
                abi = _abi_for_raw_tag(raw_tag, native_dtype, matched_rule)
                port_abis.append((("target", plan.index, logical_role), abi))
                dispositions.append((tensor_path, raw_tag, "raw_eager_unfused_checkpoint_precision"))
        for rule in checkpoint.rules:
            _abi_for_raw_tag(rule.raw_tag, native_dtype, rule)
        for exc in checkpoint.layer_exceptions:
            _abi_for_raw_tag(exc.raw_tag, native_dtype)

        if phase in DIFFUSION_PHASES:
            kv_abis: tuple[CacheStateABI, ...] = ()
            idx_abis: tuple[CacheStateABI, ...] = ()
            rec_abis: tuple[CacheStateABI, ...] = ()
        else:
            kv_abis, idx_abis, rec_abis = _eager_cache_abis(
                layers, checkpoint, parallel, native_dtype, role=role, phase=phase,
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
        del model, options
        if phase not in self._capabilities.supported_phases:
            raise ModelingError(
                f"stack_phase_unsupported:stack={self._capabilities.stack_id}:phase={phase}"
            )
        if not state.ready_request_ids:
            if state.in_flight_completions:
                next_ms = min(ts for _, ts in state.in_flight_completions)
                return WaitForEvent(reason="completion", until_ms=next_ms)
            return SchedulerDone(completed_requests=len(state.request_progress))
        priorities = dict(state.request_priorities)
        progress = {rid: (ctx, rem) for rid, ctx, rem in state.request_progress}
        ordered = sorted(state.ready_request_ids, key=lambda rid: (-priorities.get(rid, 0), rid))
        views: list[RequestView] = []
        used_tokens = 0
        for rid in ordered:
            if len(views) >= state.max_num_seqs or used_tokens >= state.max_num_batched_tokens:
                break
            ctx, rem = progress.get(rid, (0, 1))
            q_len = 1 if phase == "decode" else min(max(1, rem), state.max_num_batched_tokens - used_tokens)
            if used_tokens + q_len > state.max_num_batched_tokens:
                continue
            views.append(
                RequestView(
                    request_id=rid,
                    q_len=q_len,
                    context=0 if phase in DIFFUSION_PHASES else ctx,
                    samples=(phase == "decode"),
                    phase=phase,
                    priority=priorities.get(rid, 0),
                )
            )
            used_tokens += q_len
        if not views:
            return WaitForEvent(reason="grant_release", until_ms=state.now_ms)
        return IterationPlan(
            requests=tuple(views),
            reason="scheduled",
            all_request_ids=state.ready_request_ids,
        )

    @property
    def launch_codec(self) -> LaunchConfigCodec | None:
        return None


__all__ = ["MANIFEST_PATH", "RawEagerStackAdapter"]
