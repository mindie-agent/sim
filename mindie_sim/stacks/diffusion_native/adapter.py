"""Built-in Diffusion / DiT ``RuntimeStackAdapter`` (``diffusion-native``).

[Role] Implement the built-in ``diffusion-native`` stack plugin for Diffusion
  and DiT workloads (``supported_domains=("diffusion",)``,
  ``work_unit="denoise_timestep"``), declaring denoise timesteps (with optional
  step-cache skipping), latent patches, CFG branches, and optional text-encoder
  and VAE-decode phases while binding ``prediction/operators/diffusion/``.
[Invariants] NEVER fabricates LLM KV-Cache, Indexer cache, or recurrent state
  (all three cache ABI tuples are strictly empty). Rejects LLM phases
  (``prefill``/``decode``/``mixed``) and validates every emitted recipe binding
  against ``MINDIE_SD_CUSTOM_OPS``. Never imports ``export/``, ``compose/``,
  or ``stacks.registry``.
[Timeouts & Locks] Bounded read of sibling ``manifest.json`` (< 50 MiB); pure
  in-memory planning thereafter; no locks.
[Side-Effect Policy] Read-only over local manifest and immutable L0 contracts;
  zero filesystem writes or device context initialization.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

from mindie_sim.prediction.operators.diffusion import (
    MINDIE_SD_CUSTOM_OPS,
    compute_mindie_sd_effective_denoising_steps,
)
from mindie_sim.schema.checkpoint import CheckpointQuantMap, TensorQuantRule
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.operators import KernelReadiness, TensorPrecisionABI
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.stack import (
    CaptureBucketRule,
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
    WaitForEvent,
    WorkUnitKind,
    resolve_checkpoint_tensor_abi,
)
from mindie_sim.schema.workload import IterationPlan, RequestView

MANIFEST_PATH = Path(__file__).resolve().with_name("manifest.json")
MAX_MANIFEST_BYTES = 50 * 1024 * 1024
DEFAULT_DENOISE_TIMESTEPS = 50
DEFAULT_LATENT_PATCHES = 4096
DEFAULT_CFG_BRANCHES = 2
DEFAULT_TEXT_ENCODE_TOKENS = 256
DEFAULT_SPARSE_BLOCK_SIZE = 128
DEFAULT_SPARSITY_RATIO = 0.0
DEFAULT_STARS_FIFO_LANES = 2
DEFAULT_STARS_QUEUE_DEPTH = 8
MX_BLOCK_ELEMENTS = 32
FP8_BLOCK_ELEMENTS = 128
INT4_GROUP_ELEMENTS = 128
FP32_SCALE_BYTES = 4
FP16_SCALE_BYTES = 2
E8M0_SCALE_BYTES = 1

_SPARSE_ATTENTION_OPS = frozenset({"rainfusion", "ada_bsa", "eagle_bsa", "eagle_qbsa"})
_DIT_LAYER_ROLES: tuple[str, ...] = (
    "attn.qkv_proj",
    "attn.o_proj",
    "ffn.fc1",
    "ffn.fc2",
    "adaln.modulation",
)
_LLM_ONLY_OPTION_KEYS = frozenset({
    "kv_cache_dtype",
    "kv_block_size",
    "speculative_k",
    "enable_prefix_caching",
    "chunked_prefill",
})


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
    if not isinstance(data, dict) or data.get("stack_id") != "diffusion-native":
        raise ModelingError("diffusion_native_manifest_invalid")
    return data


def _pos_int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"{code}:{value!r}")
    return int(value)


def _nonneg_int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingError(f"{code}:{value!r}")
    return int(value)


@dataclass(frozen=True, slots=True)
class DiffusionWorkloadConfig:
    """Explicit diffusion / DiT work-unit contract (no fabricated LLM KV-Cache)."""

    denoise_timesteps: int = DEFAULT_DENOISE_TIMESTEPS
    effective_denoise_timesteps: int = DEFAULT_DENOISE_TIMESTEPS
    latent_patches: int = DEFAULT_LATENT_PATCHES
    cfg_branches: int = DEFAULT_CFG_BRANCHES
    include_text_encode: bool = False
    include_vae_decode: bool = False
    text_encode_tokens: int = DEFAULT_TEXT_ENCODE_TOKENS
    sparse_attention_op: str = "rainfusion"
    sparsity_ratio: float = DEFAULT_SPARSITY_RATIO
    block_size: int = DEFAULT_SPARSE_BLOCK_SIZE
    laurel_low_rank: int = 0
    step_cache_warmup: int = 0
    step_cache_interval: int = 1
    step_cache_hit_ratio: float | None = None

    def __post_init__(self) -> None:
        steps = _pos_int(self.denoise_timesteps, "diffusion_denoise_timesteps_invalid")
        eff_steps = _pos_int(self.effective_denoise_timesteps, "diffusion_effective_timesteps_invalid")
        if eff_steps > steps:
            raise ModelingError(f"diffusion_effective_steps_exceed_total:{eff_steps}>{steps}")
        patches = _pos_int(self.latent_patches, "diffusion_latent_patches_invalid")
        cfg = _pos_int(self.cfg_branches, "diffusion_cfg_branches_invalid")
        text_toks = _pos_int(self.text_encode_tokens, "diffusion_text_tokens_invalid")
        block = _pos_int(self.block_size, "diffusion_sparse_block_size_invalid")
        laurel = _nonneg_int(self.laurel_low_rank, "diffusion_laurel_rank_invalid")
        warmup = _nonneg_int(self.step_cache_warmup, "diffusion_cache_warmup_invalid")
        interval = _pos_int(self.step_cache_interval, "diffusion_cache_interval_invalid")
        if isinstance(self.sparsity_ratio, bool):
            raise ModelingError(f"diffusion_sparsity_ratio_invalid:{self.sparsity_ratio!r}")
        ratio = float(self.sparsity_ratio)
        if not math.isfinite(ratio) or ratio < 0.0 or ratio >= 1.0:
            raise ModelingError(f"diffusion_sparsity_ratio_invalid:{ratio!r}")
        op_short = str(self.sparse_attention_op or "").strip().lower().removeprefix("mindie_sd_")
        if op_short and op_short not in _SPARSE_ATTENTION_OPS:
            raise ModelingError(f"diffusion_sparse_attention_op_unknown:{self.sparse_attention_op!r}")
        object.__setattr__(self, "denoise_timesteps", steps)
        object.__setattr__(self, "effective_denoise_timesteps", eff_steps)
        object.__setattr__(self, "latent_patches", patches)
        object.__setattr__(self, "cfg_branches", cfg)
        object.__setattr__(self, "text_encode_tokens", text_toks)
        object.__setattr__(self, "sparse_attention_op", op_short)
        object.__setattr__(self, "sparsity_ratio", ratio)
        object.__setattr__(self, "block_size", block)
        object.__setattr__(self, "laurel_low_rank", laurel)
        object.__setattr__(self, "step_cache_warmup", warmup)
        object.__setattr__(self, "step_cache_interval", interval)

    @property
    def tokens_per_denoise_step(self) -> int:
        return self.latent_patches * self.cfg_branches

    @property
    def active_phases(self) -> tuple[str, ...]:
        phases: list[str] = []
        if self.include_text_encode:
            phases.append("text_encode")
        phases.append("denoise_step")
        if self.include_vae_decode:
            phases.append("vae_decode")
        return tuple(phases)

    @classmethod
    def from_options(cls, requested: Mapping[str, Any]) -> "DiffusionWorkloadConfig":
        steps = _pos_int(requested.get("denoise_timesteps", DEFAULT_DENOISE_TIMESTEPS), "diffusion_denoise_timesteps_invalid")
        cache_interval = _pos_int(requested.get("step_cache_interval", 1), "diffusion_cache_interval_invalid")
        cache_hit = requested.get("step_cache_hit_ratio")
        warmup = _nonneg_int(requested.get("step_cache_warmup", 0), "diffusion_cache_warmup_invalid")
        if cache_interval > 1 or cache_hit is not None:
            summary = compute_mindie_sd_effective_denoising_steps(
                infer_steps=steps,
                warmup_steps=warmup,
                cache_interval=cache_interval,
                cache_hit_ratio=float(cache_hit) if cache_hit is not None else None,
            )
            eff_steps = int(summary["computed_steps"])
        else:
            eff_steps = steps
        return cls(
            denoise_timesteps=steps,
            effective_denoise_timesteps=eff_steps,
            latent_patches=int(requested.get("latent_patches", DEFAULT_LATENT_PATCHES)),
            cfg_branches=int(requested.get("cfg_branches", DEFAULT_CFG_BRANCHES)),
            include_text_encode=bool(requested.get("include_text_encode", False)),
            include_vae_decode=bool(requested.get("include_vae_decode", False)),
            text_encode_tokens=int(requested.get("text_encode_tokens", DEFAULT_TEXT_ENCODE_TOKENS)),
            sparse_attention_op=str(requested.get("sparse_attention_op", "rainfusion")),
            sparsity_ratio=float(requested.get("sparsity_ratio", DEFAULT_SPARSITY_RATIO)),
            block_size=int(requested.get("block_size", DEFAULT_SPARSE_BLOCK_SIZE)),
            laurel_low_rank=int(requested.get("laurel_low_rank", 0)),
            step_cache_warmup=warmup,
            step_cache_interval=cache_interval,
            step_cache_hit_ratio=float(cache_hit) if cache_hit is not None else None,
        )


def _native_float_dtype(model: ModelSpec, checkpoint: CheckpointQuantMap) -> str:
    raw = (checkpoint.native_float_dtype or model.activation_dtype or "bf16").strip().lower()
    if raw == "bfloat16":
        return "bf16"
    if raw == "float16":
        return "fp16"
    if raw == "float32":
        return "fp32"
    return raw


_abi_for_diffusion_tag = resolve_checkpoint_tensor_abi


def _is_a5_hardware(hardware: Hardware) -> bool:
    tokens = f"{hardware.arch_family} {hardware.sku} {hardware.name}".lower()
    return any(tok in tokens for tok in ("a5", "arch35", "dav_3510", "ascend950", "950"))


class DiffusionNativeStackAdapter:
    """Built-in Diffusion / DiT ``RuntimeStackAdapter`` bound to ``operators/diffusion/``."""

    work_unit: WorkUnitKind = "denoise_timestep"

    def __init__(self, manifest: Mapping[str, Any] | None = None) -> None:
        raw = dict(_load_manifest()) if manifest is None else dict(manifest)
        self._manifest = raw
        declared_recipes = frozenset(raw.get("supported_recipe_ids", ()))
        known_diffusion_recipes = frozenset(f"mindie_sd_{name}" for name in MINDIE_SD_CUSTOM_OPS)
        if not declared_recipes.issubset(known_diffusion_recipes):
            unknown = sorted(declared_recipes - known_diffusion_recipes)
            raise ModelingError(f"diffusion_native_unknown_recipes:{unknown}")
        self._capabilities = StackCapabilitySpec(
            stack_id=str(raw["stack_id"]),
            manifest_revision=str(raw["manifest_revision"]),
            supported_domains=tuple(raw["supported_domains"]),
            supported_work_units=tuple(raw["supported_work_units"]),
            supported_hardware_backends=tuple(raw["supported_hardware_backends"]),
            supported_phases=tuple(raw["supported_phases"]),
            supported_checkpoint_formats=tuple(raw["supported_checkpoint_formats"]),
            supported_tracks=tuple(raw["supported_tracks"]),
            supported_recipe_ids=declared_recipes,
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
        if track not in self._capabilities.supported_tracks:
            raise ModelingError(
                f"stack_track_unsupported:stack={self._capabilities.stack_id}:track={track}"
            )
        req = dict(requested_options)
        cfg = DiffusionWorkloadConfig.from_options(req)
        dispositions: list[tuple[str, str, str]] = []
        for key in sorted(req):
            if key in _LLM_ONLY_OPTION_KEYS:
                dispositions.append((key, "rejected", "diffusion_forbids_llm_kv_cache_and_speculation"))
            else:
                dispositions.append((key, "accepted", "diffusion_workload_option_applied"))
        is_eager = track == "eager" or bool(req.get("enforce_eager", False))
        cap_mode = "none" if is_eager else "full"
        full_keys = () if is_eager else (cfg.tokens_per_denoise_step,)
        capture_rule = CaptureBucketRule(
            mode=cap_mode,
            full_keys=full_keys,
            fallback_mode="eager",
        )
        effective: dict[str, Any] = {
            "work_unit": self.work_unit,
            "denoise_timesteps": cfg.denoise_timesteps,
            "effective_denoise_timesteps": cfg.effective_denoise_timesteps,
            "latent_patches": cfg.latent_patches,
            "cfg_branches": cfg.cfg_branches,
            "tokens_per_denoise_step": cfg.tokens_per_denoise_step,
            "include_text_encode": cfg.include_text_encode,
            "include_vae_decode": cfg.include_vae_decode,
            "text_encode_tokens": cfg.text_encode_tokens,
            "active_phases": cfg.active_phases,
            "sparse_attention_op": cfg.sparse_attention_op,
            "sparsity_ratio": cfg.sparsity_ratio,
            "block_size": cfg.block_size,
            "laurel_low_rank": cfg.laurel_low_rank,
            "capture_mode": "eager" if is_eager else "full",
            "role": role,
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
            piecewise_capture_keys=(),
            speculative=None,
            stream_policy=None,
            collective_policy=(("cfg_parallel_branches", cfg.cfg_branches),),
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
        del hardware, parallel, options, role
        if phase not in self._capabilities.supported_phases:
            raise ModelingError(
                f"stack_phase_unsupported:stack={self._capabilities.stack_id}:phase={phase}"
            )
        native_dtype = _native_float_dtype(model, checkpoint)
        rule_by_pattern = {rule.pattern: rule for rule in checkpoint.rules}
        port_abis: list[tuple[tuple[str, int, str], TensorPrecisionABI]] = []
        dispositions: list[tuple[str, str, str]] = []
        for idx in range(model.layers):
            for logical_role in _DIT_LAYER_ROLES:
                tensor_path = f"transformer_blocks.{idx}.{logical_role}"
                raw_tag = checkpoint.resolve_raw_tag(
                    tensor_path,
                    scope="target",
                    source_native_layer_id=idx,
                )
                matched_rule = rule_by_pattern.get(f"transformer_blocks.*.{logical_role}")
                if raw_tag is None:
                    raw_tag = checkpoint.model_quant_type or "FLOAT"
                abi = _abi_for_diffusion_tag(raw_tag, native_dtype, matched_rule)
                port_abis.append((("target", idx, logical_role), abi))
                dispositions.append((tensor_path, raw_tag, "diffusion_checkpoint_precision_honored"))
        for rule in checkpoint.rules:
            _abi_for_diffusion_tag(rule.raw_tag, native_dtype, rule)
        for exc in checkpoint.layer_exceptions:
            _abi_for_diffusion_tag(exc.raw_tag, native_dtype)
        if checkpoint.cache_rule.kv_tag:
            dispositions.append((
                "cache_rule.kv_tag",
                checkpoint.cache_rule.kv_tag,
                "ignored_diffusion_has_no_llm_kv_cache",
            ))
        return RuntimeQuantPlan(
            checkpoint_digest=checkpoint.digest(),
            stack_id=self._capabilities.stack_id,
            port_abis=tuple(port_abis),
            packed_groups=(),
            kv_cache_abi=(),
            index_cache_abi=(),
            recurrent_state_abi=(),
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
        del model, parallel
        eff = dict(options.effective_mechanisms)
        bindings: list[tuple[str, str, tuple[tuple[str, Any], ...]]] = []
        if options.track != "eager":
            bindings.extend((
                ("adalayernorm", "mindie_sd_adalayernorm", ()),
                ("norm_rope_concat", "mindie_sd_norm_rope_concat", ()),
                ("eagle_ffn", "mindie_sd_eagle_ffn", ()),
            ))
            sparse_op = str(eff.get("sparse_attention_op", "") or "").strip().lower()
            if sparse_op and sparse_op in MINDIE_SD_CUSTOM_OPS:
                bindings.append((
                    sparse_op,
                    f"mindie_sd_{sparse_op}",
                    (
                        ("sparsity_ratio", float(eff.get("sparsity_ratio", DEFAULT_SPARSITY_RATIO))),
                        ("block_size", int(eff.get("block_size", DEFAULT_SPARSE_BLOCK_SIZE))),
                    ),
                ))
            laurel_rank = int(eff.get("laurel_low_rank", 0) or 0)
            if laurel_rank > 0:
                bindings.append(("laurel", "mindie_sd_laurel", (("low_rank", laurel_rank),)))
            uses_mxfp4 = any(abi.linear_method == "mx_fp4" for _, abi in quant_plan.port_abis)
            if uses_mxfp4 and _is_a5_hardware(hardware):
                bindings.append(("quant_four_over_six_a5", "mindie_sd_quant_four_over_six_a5", ()))
        if options.track == "eager":
            allowed_tracks: tuple[RewriteTrack, ...] = ()
            allowed_classes: frozenset[KernelReadiness] = frozenset({"shipping"})
            allowed_fw: tuple[FrameworkRewriteKind, ...] = ()
        elif options.track == "target":
            allowed_tracks = ("config", "framework", "kernel_new", "kernel_tune")
            allowed_classes = frozenset({"shipping", "proposed_pr", "synthesized_target", "tiler_extension"})
            allowed_fw = ("merge_sibling_projections", "bind_shipping_op", "multistream_overlap")
        elif options.track == "proposed":
            allowed_tracks = ("config",)
            allowed_classes = frozenset({"shipping", "proposed_pr"})
            allowed_fw = ()
        else:
            allowed_tracks = ("config",)
            allowed_classes = frozenset({"shipping"})
            allowed_fw = ()
        scheduling = SchedulingContract(
            capture_bucket_rule=options.capture_rule,
            stars_submit_policy=(
                StarsSubmitPolicy(
                    die_id=0,
                    stream_id="main",
                    owner_kind="aic",
                    fifo_lanes=DEFAULT_STARS_FIFO_LANES,
                    queue_depth=DEFAULT_STARS_QUEUE_DEPTH,
                    replay_mode="eager" if options.graph_capture_mode == "none" else "full_graph",
                ),
            ),
            stream_overlap_policy=None,
            chunked_prefill_enabled=False,
            speculative_proposal_count=0,
            draft_invocation_count=0,
        )
        return StackLoweringPlan(
            stack_id=self._capabilities.stack_id,
            track=options.track,
            recipe_bindings=tuple(bindings),
            lowering_passes=(),
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
        if not state.ready_request_ids:
            if state.in_flight_completions:
                next_ms = min(ts for _, ts in state.in_flight_completions)
                return WaitForEvent(reason="completion", until_ms=next_ms)
            return SchedulerDone(completed_requests=len(state.request_progress))
        eff = dict(options.effective_mechanisms)
        if phase == "denoise_step":
            q_len = int(eff.get("tokens_per_denoise_step", DEFAULT_LATENT_PATCHES * DEFAULT_CFG_BRANCHES))
            exec_mode = "denoise_timestep"
        elif phase == "text_encode":
            q_len = int(eff.get("text_encode_tokens", DEFAULT_TEXT_ENCODE_TOKENS))
            exec_mode = "text_encode"
        else:
            q_len = int(eff.get("latent_patches", DEFAULT_LATENT_PATCHES))
            exec_mode = "vae_decode"
        priorities = dict(state.request_priorities)
        ordered = sorted(state.ready_request_ids, key=lambda rid: (-priorities.get(rid, 0), rid))
        selected = ordered[: state.max_num_seqs]
        views = tuple(
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
            for rid in selected
        )
        return IterationPlan(
            requests=views,
            reason="scheduled_diffusion",
            all_request_ids=state.ready_request_ids,
        )

    @property
    def launch_codec(self) -> LaunchConfigCodec | None:
        return None


__all__ = [
    "DiffusionNativeStackAdapter",
    "DiffusionWorkloadConfig",
    "MANIFEST_PATH",
]
