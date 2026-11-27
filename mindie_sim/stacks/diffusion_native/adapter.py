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
from mindie_sim.prediction.program.implementation_registry import IMPLEMENTATIONS, select
from mindie_sim.schema.checkpoint import CheckpointQuantMap, TensorQuantRule
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.operators import KernelReadiness, TensorPrecisionABI
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.physical_mechanisms import DAV_2201_PRESET, DAV_3510_PRESET
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


def _normalize_custom_ops(
    raw: Any,
) -> tuple[tuple[str, tuple[tuple[str, Any], ...]], ...]:
    """Validate optional ``mindie_sd_custom_ops`` selector bindings through ``select()``."""
    if not raw:
        return ()
    entries = list(raw.items()) if isinstance(raw, Mapping) else list(raw)
    normalized: list[tuple[str, tuple[tuple[str, Any], ...]]] = []
    seen: set[str] = set()
    for selector_raw, opts_raw in entries:
        selector = str(selector_raw or "").strip()
        if not selector or selector in seen:
            raise ModelingError(f"diffusion_custom_op_selector_invalid:{selector_raw!r}")
        if not isinstance(opts_raw, Mapping) and not isinstance(opts_raw, (tuple, list)):
            raise ModelingError(f"diffusion_custom_op_options_invalid:{selector}")
        opts_dict = dict(opts_raw)
        op_name = opts_dict.get("op_name")
        if op_name is not None:
            clean = str(op_name).strip().lower().removeprefix("mindie_sd_")
            if clean not in MINDIE_SD_CUSTOM_OPS:
                raise ModelingError(f"diffusion_custom_op_unknown:{op_name!r}")
            opts_dict["op_name"] = clean
        elif selector.lower().removeprefix("mindie_sd_") not in MINDIE_SD_CUSTOM_OPS:
            raise ModelingError(f"diffusion_custom_op_requires_op_name:{selector}")
        select(
            {selector: {"implementation": "mindie_sd_custom_ops", "options": opts_dict}},
            selector,
        )
        seen.add(selector)
        normalized.append((selector, tuple(sorted(opts_dict.items()))))
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _validate_custom_ops_for_hardware(
    custom_ops: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...],
    hardware: Hardware,
    *,
    track: StackTrack,
) -> None:
    if track == "target" or not _is_a5_hardware(hardware):
        return
    for selector, opt_pairs in custom_ops:
        opts_dict = dict(opt_pairs)
        op_short = str(opts_dict.get("op_name") or selector).strip().lower().removeprefix("mindie_sd_")
        if op_short in _A5_UNSUPPORTED_SPARSE_OPS:
            raise ModelingError(
                f"diffusion_sparse_attention_op_unsupported_on_hardware:a5:{op_short}"
            )


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
    custom_ops: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] = ()
    kv_dtype: str | None = None

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
        custom_ops = _normalize_custom_ops(self.custom_ops)
        kv_dtype = str(self.kv_dtype).strip().lower() if self.kv_dtype is not None else None
        if kv_dtype == "":
            kv_dtype = None
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
        object.__setattr__(self, "custom_ops", custom_ops)
        object.__setattr__(self, "kv_dtype", kv_dtype)

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
            custom_ops=_normalize_custom_ops(requested.get("custom_ops")),
            kv_dtype=requested.get("kv_dtype"),
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
_A5_UNSUPPORTED_SPARSE_OPS = frozenset({"ada_bsa"})


def _resolve_tag_and_rule(
    checkpoint: CheckpointQuantMap,
    tensor_path: str,
    *,
    scope: str = "target",
    source_native_layer_id: int | None = None,
) -> tuple[str, TensorQuantRule | None]:
    raw = str(tensor_path or "").strip()
    mod_path = raw[:-len(".weight")] if raw.endswith(".weight") else raw
    exc_or_ign = checkpoint.resolve_exception_or_ignored(
        mod_path, scope=scope, source_native_layer_id=source_native_layer_id,  # type: ignore[arg-type]
    )
    if exc_or_ign is not None:
        return exc_or_ign, None
    for rule in checkpoint.rules:
        if rule.matches(
            mod_path, scope=scope, source_native_layer_id=source_native_layer_id,  # type: ignore[arg-type]
        ):
            return rule.raw_tag, rule
    if mod_path != raw:
        exc_raw = checkpoint.resolve_exception_or_ignored(
            raw, scope=scope, source_native_layer_id=source_native_layer_id,  # type: ignore[arg-type]
        )
        if exc_raw is not None:
            return exc_raw, None
        for rule in checkpoint.rules:
            if rule.matches(
                raw, scope=scope, source_native_layer_id=source_native_layer_id,  # type: ignore[arg-type]
            ):
                return rule.raw_tag, rule
    if checkpoint.source_format == "unquantized_native":
        return "FLOAT", None
    if checkpoint.source_format == "hf_quantization_config":
        for rule in checkpoint.rules:
            if rule.pattern == "*":
                return rule.raw_tag, rule
    tag = checkpoint.resolve_raw_tag(
        tensor_path, scope=scope, source_native_layer_id=source_native_layer_id,  # type: ignore[arg-type]
    )
    return (tag if tag is not None else (checkpoint.model_quant_type or "FLOAT")), None


def _soc_family_for_hardware(hardware: Hardware) -> str | None:
    target = hardware.dies[0] if getattr(hardware, "dies", None) else hardware
    arch_obj = getattr(target, "compute_arch", None)
    arch_id = str(
        getattr(arch_obj, "arch_id", "") or getattr(arch_obj, "preset", "") or ""
    ).strip().lower()
    tokens = " ".join(
        str(getattr(obj, attr, "") or "")
        for obj in (hardware, target)
        for attr in ("arch_family", "sku", "name")
    ).lower()
    if (
        arch_id in {DAV_3510_PRESET, "arch35", "a5"}
        or "3510" in arch_id
        or any(tok in tokens for tok in ("a5", "arch35", "dav_3510", "ascend950", "950"))
    ):
        return "Ascend950"
    if any(tok in tokens for tok in ("910b", "a2")) or arch_id == "a2":
        return "Ascend910B"
    if (
        arch_id in {DAV_2201_PRESET, "arch22", "a3"}
        or "2201" in arch_id
        or any(tok in tokens for tok in ("910_93", "9391", "9382", "9362", "a3", "arch22", "2201"))
    ):
        return "Ascend910_93"
    return None


def _is_a5_hardware(hardware: Hardware) -> bool:
    return _soc_family_for_hardware(hardware) == "Ascend950"


class DiffusionNativeStackAdapter:
    """Built-in Diffusion / DiT ``RuntimeStackAdapter`` bound to ``operators/diffusion/``."""

    work_unit: WorkUnitKind = "denoise_timestep"

    def __init__(self, manifest: Mapping[str, Any] | None = None) -> None:
        raw = dict(_load_manifest()) if manifest is None else dict(manifest)
        self._manifest = raw
        declared_recipes = frozenset(raw.get("supported_recipe_ids", ()))
        known_diffusion_recipes = (
            frozenset(f"mindie_sd_{name}" for name in MINDIE_SD_CUSTOM_OPS)
            | {"mindie_sd_custom_ops"}
        )
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
        del model, parallel, field_origins
        if track not in self._capabilities.supported_tracks:
            raise ModelingError(
                f"stack_track_unsupported:stack={self._capabilities.stack_id}:track={track}"
            )
        req = dict(requested_options)
        cfg = DiffusionWorkloadConfig.from_options(req)
        if track != "target" and _is_a5_hardware(hardware) and cfg.sparse_attention_op in _A5_UNSUPPORTED_SPARSE_OPS:
            raise ModelingError(
                f"diffusion_sparse_attention_op_unsupported_on_hardware:a5:{cfg.sparse_attention_op}"
            )
        if track != "eager":
            _validate_custom_ops_for_hardware(cfg.custom_ops, hardware, track=track)
        dispositions: list[tuple[str, str, str]] = []
        for key in sorted(req):
            if key in _LLM_ONLY_OPTION_KEYS:
                dispositions.append((key, "rejected", "diffusion_forbids_llm_kv_cache_and_speculation"))
            elif key in {"custom_ops", "kv_dtype"} and track == "eager":
                dispositions.append((key, "rejected", "diffusion_eager_track_disables_custom_ops"))
            elif key == "kv_dtype" and cfg.sparse_attention_op != "eagle_qbsa":
                dispositions.append((key, "rejected", "diffusion_kv_dtype_requires_eagle_qbsa"))
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
        if track != "eager" and cfg.kv_dtype is not None and cfg.sparse_attention_op == "eagle_qbsa":
            effective["kv_dtype"] = cfg.kv_dtype
        if track != "eager" and cfg.custom_ops:
            effective["custom_ops"] = cfg.custom_ops
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
        layers: tuple[LayerPlan, ...] = model.layer_plans or tuple(
            LayerPlan(
                index=idx,
                native_layer_id=idx,
                attention=model.attention,
                ffn=model.ffn,
                num_heads=model.num_heads,
                num_kv_heads=model.num_kv_heads or model.num_heads,
                head_dim=model.head_dim,
            )
            for idx in range(model.layers)
        )
        port_abis: list[tuple[tuple[str, int, str], TensorPrecisionABI]] = []
        dispositions: list[tuple[str, str, str]] = []
        for plan in layers:
            native_id = plan.native_layer_id if plan.native_layer_id >= 0 else plan.index
            for logical_role in _DIT_LAYER_ROLES:
                tensor_path = f"transformer_blocks.{native_id}.{logical_role}"
                raw_tag, matched_rule = _resolve_tag_and_rule(
                    checkpoint,
                    tensor_path,
                    scope="target",
                    source_native_layer_id=native_id,
                )
                abi = _abi_for_diffusion_tag(raw_tag, native_dtype, matched_rule)
                port_abis.append((("target", plan.index, logical_role), abi))
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
            hw_soc = _soc_family_for_hardware(hardware)
            bindings.extend((
                ("adalayernorm", "mindie_sd_adalayernorm", ()),
                ("norm_rope_concat", "mindie_sd_norm_rope_concat", ()),
                ("eagle_ffn", "mindie_sd_eagle_ffn", ()),
            ))
            sparse_op = str(eff.get("sparse_attention_op", "") or "").strip().lower()
            if (
                options.track != "target"
                and _is_a5_hardware(hardware)
                and sparse_op in _A5_UNSUPPORTED_SPARSE_OPS
            ):
                raise ModelingError(
                    f"diffusion_sparse_attention_op_unsupported_on_hardware:a5:{sparse_op}"
                )
            if sparse_op and sparse_op in MINDIE_SD_CUSTOM_OPS:
                sparse_opts: list[tuple[str, Any]] = [
                    ("sparsity_ratio", float(eff.get("sparsity_ratio", DEFAULT_SPARSITY_RATIO))),
                    ("block_size", int(eff.get("block_size", DEFAULT_SPARSE_BLOCK_SIZE))),
                ]
                if sparse_op == "eagle_qbsa" and eff.get("kv_dtype") is not None:
                    sparse_opts.append(("kv_dtype", str(eff["kv_dtype"])))
                bindings.append((
                    sparse_op,
                    f"mindie_sd_{sparse_op}",
                    tuple(sparse_opts),
                ))
            laurel_rank = int(eff.get("laurel_low_rank", 0) or 0)
            if laurel_rank > 0:
                bindings.append(("laurel", "mindie_sd_laurel", (("low_rank", laurel_rank),)))
            uses_mxfp4 = any(abi.linear_method == "mx_fp4" for _, abi in quant_plan.port_abis)
            if uses_mxfp4 and _is_a5_hardware(hardware):
                bindings.append(("quant_four_over_six_a5", "mindie_sd_quant_four_over_six_a5", ()))
            custom_ops = _normalize_custom_ops(eff.get("custom_ops"))
            _validate_custom_ops_for_hardware(custom_ops, hardware, track=options.track)
            if custom_ops and "mindie_sd_custom_ops" in IMPLEMENTATIONS:
                binding_idx = {sel: idx for idx, (sel, _, _) in enumerate(bindings)}
                for selector, opt_pairs in custom_ops:
                    merged_opts: dict[str, Any] = (
                        dict(bindings[binding_idx[selector]][2])
                        if selector in binding_idx
                        else {}
                    )
                    merged_opts.update(opt_pairs)
                    if hw_soc is not None and "soc_family" not in merged_opts:
                        merged_opts["soc_family"] = hw_soc
                    select(
                        {selector: {"implementation": "mindie_sd_custom_ops", "options": merged_opts}},
                        selector,
                    )
                    entry = (selector, "mindie_sd_custom_ops", tuple(sorted(merged_opts.items())))
                    if selector in binding_idx:
                        bindings[binding_idx[selector]] = entry
                    else:
                        binding_idx[selector] = len(bindings)
                        bindings.append(entry)
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
        budget = state.max_num_batched_tokens
        used_tokens = 0
        views: list[RequestView] = []
        for rid in ordered:
            if len(views) >= state.max_num_seqs:
                break
            if used_tokens + q_len > budget:
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
        if not views:
            if state.in_flight_completions:
                next_ms = min(ts for _, ts in state.in_flight_completions)
                return WaitForEvent(reason="completion", until_ms=next_ms)
            raise ModelingError(
                f"iteration_step_exceeds_token_budget:stack={self._capabilities.stack_id}:phase={phase}:required={q_len}:budget={budget}"
            )
        return IterationPlan(
            requests=tuple(views),
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
