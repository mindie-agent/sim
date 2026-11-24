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
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from mindie_sim.catalog.checkpoint_reader import resolve_checkpoint_tensor_tag
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
)
from mindie_sim.schema.workload import IterationPlan, RequestView

MANIFEST_PATH = Path(__file__).resolve().with_name("manifest.json")
MAX_MANIFEST_BYTES = 50 * 1024 * 1024
DEFAULT_KV_BLOCK_SIZE = 128
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


def _build_llm_cache_abis(
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
    budget = state.max_num_batched_tokens
    if phase == "prefill" and state.max_num_input_tokens is not None:
        budget = min(budget, state.max_num_input_tokens)
    used_tokens = 0
    for rid in ordered:
        if len(views) >= state.max_num_seqs or used_tokens >= budget:
            break
        ctx, rem = progress.get(rid, (0, 1))
        if phase == "decode":
            q_len = 1
        else:
            raw_q = max(1, rem)
            q_len = min(raw_q, budget - used_tokens) if state.chunked_prefill else raw_q
        step_tokens = q_len + (speculative_k if phase == "decode" else 0)
        if used_tokens + step_tokens > budget:
            continue
        views.append(
            RequestView(
                request_id=rid,
                q_len=q_len,
                context=ctx,
                samples=(phase == "decode"),
                phase=phase,
                prefix_tokens=prefixes.get(rid, 0),
                priority=priorities.get(rid, 0),
                draft_proposal_tokens=speculative_k if phase == "decode" else 0,
            )
        )
        used_tokens += step_tokens
    if not views:
        return WaitForEvent(reason="grant_release", until_ms=state.now_ms)
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
        spec_k = int(req.get("speculative_k", 0) or 0)
        verify_stride = max(1, 1 + spec_k)
        full_keys = () if cap_mode == "none" else tuple(k * verify_stride for k in DEFAULT_FULL_CAPTURE_BUCKETS)
        pw_keys = () if cap_mode in {"none", "full", "full_decode_only"} else DEFAULT_PIECEWISE_BUCKETS
        capture_rule = CaptureBucketRule(
            mode=cap_mode,
            full_keys=full_keys,
            piecewise_keys=pw_keys,
            verify_stride=verify_stride,
            fallback_mode="eager",
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
            "optimization_mode": opt_mode,
            "role": role,
            "arch_family": hardware.arch_family,
            "pillar_abc_exploration": (track == "target" and opt_mode == "modeling"),
            "speculative_k": spec_k,
        }
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
                dispositions.append((tensor_path, raw_tag, "checkpoint_precision_honored"))
        for rule in checkpoint.rules:
            _abi_for_raw_tag(rule.raw_tag, native_dtype, rule)
        for exc in checkpoint.layer_exceptions:
            _abi_for_raw_tag(exc.raw_tag, native_dtype)

        if phase in DIFFUSION_PHASES:
            kv_abis: tuple[CacheStateABI, ...] = ()
            idx_abis: tuple[CacheStateABI, ...] = ()
            rec_abis: tuple[CacheStateABI, ...] = ()
        else:
            kv_abis, idx_abis, rec_abis = _build_llm_cache_abis(
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
        eff = dict(options.effective_mechanisms)
        opt_mode = str(eff.get("optimization_mode", "modeling" if options.track == "target" else "simulation"))
        spec_k = int(eff.get("speculative_k", 0) or 0)
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
        replay_mode = "eager" if options.graph_capture_mode == "none" else "full_graph"
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
            draft_invocation_count=spec_k,
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
        spec_k = int(dict(options.effective_mechanisms).get("speculative_k", 0) or 0)
        return _schedule_one_step(state, phase=phase, speculative_k=spec_k)

    @property
    def launch_codec(self) -> LaunchConfigCodec | None:
        return None


__all__ = ["IdealTargetStackAdapter", "MANIFEST_PATH"]
