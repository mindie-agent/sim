"""Runtime/model feature projection for the canonical prediction frontend.

[Role]
  Normalise ``Runtime``/``ParallelSpec``/``QuantSpec`` into the immutable
  ``FeatureConfig`` used by frontend lowering and operator analysis.  Project
  only declared source, normalisation, capture and parallel facts.

[Invariants]
  * Every field is derived from the existing runtime/model records; this is
    not a second user-facing schema.
  * Metadata carries only explicitly parsed source facts, never defaults,
    width guesses, measured supply records or model-name branches.
  * Two identical declarations project to equal immutable records.
  * No numeric constant is implicit: declared tiles and defaults are named.

[Timeouts & Locks]
  Bounded in-memory projections; one small source-summary cache is capped.

[Side-Effect Policy]
  Reads declared source side files only through the cached summary; returns
  new immutable records and never mutates the caller's runtime or model.

"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.parallel import ParallelSpec

_DEFAULT_MT = 128
_DEFAULT_KV_TILE = 128
_DEFAULT_MC2_CHUNKS = 4
_SUMMARY_CACHE_LIMIT = 32
_DEFAULT_MOE_ROWS = 128
_DEFAULT_SPLIT_TARGET = 4096
_DEFAULT_MAX_SPLITS = 8


def _declared_kda_use_beta_sigmoid(runtime: Any) -> int | None:
    """Return the declared KDA option, or None when the case does not set it.

    Native ``recurrent.linear`` removes the separate ``linear_beta_sigmoid``
    producer only when this option is true on a decode source bind. The
    frontend reads the same declaration from feature metadata.
    """
    kernels = getattr(runtime, "kernels", None)
    if not isinstance(kernels, Mapping):
        return None
    lowerings = kernels.get("lowerings")
    if not isinstance(lowerings, Mapping):
        return None
    binding = lowerings.get("kda")
    if not isinstance(binding, Mapping):
        return None
    options = binding.get("options")
    if not isinstance(options, Mapping) or "use_beta_sigmoid" not in options:
        return None
    value = options["use_beta_sigmoid"]
    if isinstance(value, bool) or (isinstance(value, int) and value in (0, 1)):
        return int(value)
    raise ModelingError("kda_use_beta_sigmoid_invalid")


def _pool_indexer_lowering(runtime: Any) -> str:
    """Implementation name of the pool-key lowering, or an empty string."""
    kernels = getattr(runtime, "kernels", None) or {}
    if not isinstance(kernels, dict):
        return ""
    lowerings = kernels.get("lowerings") or {}
    if not isinstance(lowerings, dict):
        return ""
    binding = lowerings.get("pool_key_indexer")
    if isinstance(binding, str):
        return binding
    if isinstance(binding, dict):
        return str(binding.get("implementation") or "")
    return ""


def _identity_metadata(machine: Any, model: Any) -> dict[str, Any]:
    """Actual deployment identity for compiled feature records.

    This is derived from the unaltered case machine/model, never from a
    measured supply record or a model/case label.  The keys are metadata only;
    a consumer that has no source-backed use for them must leave selection
    unchanged.
    """
    if machine is None:
        return {}
    key = str(getattr(machine, "hardware_key", "") or
              getattr(machine, "sku", "") or getattr(machine, "name", "") or "")
    sku = str(getattr(machine, "sku", "") or "")
    name = str(getattr(machine, "name", "") or "")
    arch = str(getattr(machine, "arch_family", "") or "")
    device = sku or key or name
    model_meta = dict(getattr(model, "metadata", ()) or ()) if model is not None else {}
    model_name = str(getattr(model, "name", "") or "") if model is not None else ""
    model_id = str(model_meta.get("catalog_key") or
                   model_meta.get("model_catalog_key") or model_name or "")
    result = {}
    for field, value in (("hardware_key", key), ("hardware_sku", sku),
                         ("hardware_name", name), ("arch_family", arch),
                         ("device", device), ("model_identifier", model_id)):
        if value:
            result[field] = value
    return result


def _effective_graph_buckets(runtime: Any) -> tuple[int, ...]:
    """Capture sizes selected by the same declared mode as GraphPolicy.

    Full, uniform decode may use full-capture keys; piecewise/eager modes use
    their own keys.  The feature record is not a capture policy fork: only the
    declared runtime fields choose, with the historical graph_buckets fallback
    retained when a legacy case has no explicit capture key list.
    """
    mode = str(getattr(runtime, "graph_mode", "") or "").lower()
    if mode in {"none", "off", "eager", "disabled"} or bool(
            getattr(runtime, "deepep_high_throughput", False)):
        return ()
    capture_mode = str(getattr(runtime, "graph_capture_mode", "") or "").lower()
    keys = getattr(runtime, "graph_capture_keys", ()) or ()
    piecewise = getattr(runtime, "piecewise_capture_keys", ()) or ()
    buckets = getattr(runtime, "graph_buckets", ()) or ()
    if capture_mode == "full" and not bool(getattr(runtime, "invalid_full_modes", False)):
        return tuple(int(value) for value in (keys or buckets))
    if capture_mode == "piecewise" or capture_mode == "mixed":
        return tuple(int(value) for value in piecewise)
    return tuple(int(value) for value in buckets)


def _model_source_metadata(model: Any) -> dict[str, Any]:
    """Promote only explicitly parsed HF source facts into feature metadata.

    No defaults, width guesses or boolean aliases.  The nested
    ``unconsumed_config_fields`` record is the parsed-source carrier for the
    native config; top-level model metadata is accepted only when the parser
    already put the exact key there.
    """
    if model is None:
        return {}
    model_meta = dict(getattr(model, "metadata", ()) or ())
    unconsumed = model_meta.get("unconsumed_config_fields")
    sources = [model_meta]
    if isinstance(unconsumed, Mapping):
        sources.append(unconsumed)
    meta: dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        rope = source.get("rope_parameters")
        if isinstance(rope, Mapping):
            rope_copy = dict(rope)
            if rope_copy.get("mrope_section") is not None:
                try:
                    rope_copy["mrope_section"] = tuple(int(item) for item in rope_copy["mrope_section"])
                except (TypeError, ValueError):
                    pass
            meta["rope_parameters"] = rope_copy
        for key in ("mrope_section", "mrope_interleaved",
                    "partial_rotary_factor", "rope_theta", "rope_type"):
            value = source.get(key, rope.get(key) if isinstance(rope, Mapping) else None)
            if value is not None:
                if key == "mrope_section":
                    try:
                        value = tuple(int(item) for item in value)
                    except (TypeError, ValueError):
                        continue
                meta[key] = value
        value = source.get("routed_scaling_factor")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) and float(value) > 0):
            meta["routed_scaling_factor"] = float(value)
        source_block = source.get("glm51_source")
        if isinstance(source_block, Mapping):
            meta["glm51_source"] = dict(source_block)
    eps = getattr(model, "norm_eps", None)
    if isinstance(eps, (int, float)) and not isinstance(eps, bool) and eps > 0:
        meta["norm_eps"] = float(eps)
    return meta


_NORMAL_SUMMARY_CACHE: dict[tuple[str, int, int, str, int, int], dict[str, Any]] = {}


def _source_stat(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except TimeoutError:
        raise
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _normal_summary(source_path: Path) -> dict[str, Any]:
    """Parse one real source once and cache only its normalized small summary."""
    quant_path = source_path.parent / "quant_model_description.json"
    index_path = source_path.parent / "quant_model_weights.safetensors.index.json"
    qstat = _source_stat(quant_path)
    istat = _source_stat(index_path)
    key = (str(quant_path), *(qstat or (-1, -1)), str(index_path),
           *(istat or (-1, -1)))
    cached = _NORMAL_SUMMARY_CACHE.get(key)
    if cached is not None:
        return cached
    quant: Any = None
    index: Any = None
    q_status = "ok"
    idx_status = "ok"
    if qstat is None:
        q_status = "unavailable"
    else:
        try:
            quant = json.loads(quant_path.read_text(encoding="utf-8"))
        except TimeoutError:
            raise
        except (OSError, ValueError, TypeError):
            q_status = "unreadable_or_malformed"
    if istat is None:
        idx_status = "unavailable"
    else:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except TimeoutError:
            raise
        except (OSError, ValueError, TypeError):
            idx_status = "unreadable_or_malformed"
    summary: dict[str, Any] = {
        "quant_source_status": q_status,
        "weight_index_status": idx_status,
        "quant_manifest_path": str(quant_path) if qstat is not None else None,
        "weight_index_path": str(index_path) if istat is not None else None,
        "quant_key_count": None,
        "weight_key_count": None,
        "weight_index_complete": False,
        "constructor_wants_bias": None,
        "role_bias": {},
        "trained_layernorm_bias": False,
    }
    if q_status == "ok" and isinstance(quant, Mapping):
        quant_keys = [str(key) for key in quant]
        summary["quant_key_count"] = len(quant_keys)
        summary["constructor_wants_bias"] = any(
            "norm.bias" in key for key in quant_keys)
        layers: list[int] = []
        for qkey in quant_keys:
            match = re.match(r"model\.layers\.(\d+)\.", qkey)
            if match:
                layers.append(int(match.group(1)))
        layers = sorted(set(layers))
        weight_map: dict[str, Any] = {}
        if idx_status == "ok" and isinstance(index, Mapping):
            raw_map = index.get("weight_map")
            if isinstance(raw_map, Mapping):
                weight_map = {str(key): value for key, value in raw_map.items()}
        summary["weight_key_count"] = len(weight_map)
        index_complete = bool(weight_map) and bool(layers) and all(
            f"model.layers.{layer}.input_layernorm.weight" in weight_map
            and f"model.layers.{layer}.post_attention_layernorm.weight" in weight_map
            for layer in layers) and "model.norm.weight" in weight_map
        summary["weight_index_complete"] = index_complete
        roles: dict[str, dict[str, Any]] = {}
        specs = {
            "input_layernorm": [f"model.layers.{layer}.input_layernorm.bias"
                                for layer in layers],
            "post_attention_layernorm": [
                f"model.layers.{layer}.post_attention_layernorm.bias"
                for layer in layers],
            "final_norm": ["model.norm.bias"],
            "indexer_k_norm": [
                f"model.layers.{layer}.self_attn.indexer.k_norm.bias"
                for layer in layers],
        }
        for role_name, expected in specs.items():
            loaded = [key for key in expected if key in weight_map]
            if index_complete:
                absent = [key for key in expected if key not in weight_map]
                unobserved: list[str] = []
            else:
                absent = []
                unobserved = [key for key in expected if key not in weight_map]
            if loaded:
                state = "loaded"
            elif index_complete:
                state = "absent"
            else:
                state = "unobserved"
            roles[role_name] = {
                "state": state,
                "loaded": loaded,
                "absent": absent,
                "unobserved": unobserved,
            }
        summary["role_bias"] = roles
        summary["trained_layernorm_bias"] = any(
            roles.get(name, {}).get("state") == "loaded"
            for name in ("input_layernorm", "post_attention_layernorm"))
    if len(_NORMAL_SUMMARY_CACHE) >= _SUMMARY_CACHE_LIMIT:
        _NORMAL_SUMMARY_CACHE.clear()
    _NORMAL_SUMMARY_CACHE[key] = summary
    del quant, index
    return summary


def _normal_source_metadata(model: Any, runtime: Any = None) -> dict[str, Any]:
    """Normalize declared normalization semantics without synthesizing identity."""
    if model is None:
        return {}
    meta = dict(getattr(model, "metadata", ()) or ())
    if runtime is not None:
        meta = {**dict(getattr(runtime, "metadata", ()) or ()), **meta}
    source = dict(meta.get("glm51_source") or {}) if isinstance(
        meta.get("glm51_source"), Mapping) else {}
    model_type = str(meta.get("source_model_type") or "")
    classes = {
        "qwen3_5_text": ("AscendGemmaRMSNorm", 1.0, "input_layernorm.initial"),
        "glm_moe_dsa": ("AscendRMSNorm", 0.0, "fused_residual"),
    }
    resolved = classes.get(model_type, (None, None, None))
    explicit_class = str(meta.get("rms_norm_source_class") or "") or None
    explicit_offset = meta.get("rms_norm_source_weight_offset")
    explicit_role = str(meta.get("rms_norm_source_role") or "") or None
    source_class = explicit_class or resolved[0]
    offset = explicit_offset if explicit_offset is not None else resolved[1]
    role = explicit_role or resolved[2]
    conflict = bool(explicit_class and resolved[0] and explicit_class != resolved[0])
    normal: dict[str, Any] = {
        "source_model_type": model_type or None,
        "source_class": source_class,
        "source_weight_offset": offset,
        "source_role": role,
        "source_conflict": conflict,
        "package": {},
        "quant_manifest_sha256": str(source.get("quant_manifest_sha256") or "") or None,
    }
    for key in ("vllm_ascend_commit", "vllm_ascend_package",
                "vllm_ascend_layernorm_sha256", "vllm_ascend_device_op_sha256",
                "vllm_commit", "vllm_qwen3_5_sha256"):
        if meta.get(key):
            normal[key] = str(meta[key])
    if explicit_class:
        normal["explicit_source_class"] = explicit_class
    if explicit_offset is not None:
        normal["explicit_source_weight_offset"] = explicit_offset
    if explicit_role:
        normal["explicit_source_role"] = explicit_role
    # Only a nonempty declared semantic source path; never Path('')/cwd.
    raw_source_path = str(source.get("model_config") or "")
    source_path = Path(raw_source_path) if raw_source_path else None
    if source_path is None:
        fallback = str(meta.get("source_model_path") or "")
        if fallback and Path(fallback).is_dir():
            source_path = Path(fallback) / "config.json"
    if source_path is None or not source_path.is_file():
        normal["quant_source_status"] = (
            "no_declared_semantic_source" if source_path is None
            else "semantic_source_path_unavailable")
        normal["weight_index_status"] = normal["quant_source_status"]
        normal["constructor_wants_bias"] = None
        normal["role_bias"] = {}
        return {"normal_source": normal}
    normal.update(_normal_summary(source_path))
    return {"normal_source": normal}


def _runtime_metadata(runtime: Any, machine: Any = None, model: Any = None) -> tuple:
    meta = dict(getattr(runtime, "metadata", ()) or ())
    dspark = getattr(runtime, "dspark", None)
    if dspark is not None:
        # A directly supplied block-drafter specification is the declared
        # speculative width; an explicit metadata entry keeps precedence.
        meta.setdefault("draft_proposal_tokens", dspark.draft_len)
    meta.update(_identity_metadata(machine, model))
    meta.update(_model_source_metadata(model))
    _normal = _normal_source_metadata(model, runtime)
    if _normal.get("normal_source"):
        _ns = dict(_normal["normal_source"])
        meta["normal_source"] = _ns
        if _ns.get("source_model_type"):
            meta["source_model_type"] = _ns["source_model_type"]
        if _ns.get("source_class"):
            meta["rms_norm_source_class"] = _ns["source_class"]
        if _ns.get("source_weight_offset") is not None:
            meta["rms_norm_source_weight_offset"] = _ns["source_weight_offset"]
        if _ns.get("source_role"):
            meta["rms_norm_source_role"] = _ns["source_role"]
        for key, value in (_ns.get("package") or {}).items():
            meta.setdefault(key, value)
        if _ns.get("native_norm"):
            meta["native_norm"] = dict(_ns["native_norm"])
    for flag in ("multistream_kda_proj", "multistream_attention_indexer",
                 "multistream_window_compressed"):
        if bool(getattr(runtime, flag, False)):
            meta[flag] = True
    declared = _declared_kda_use_beta_sigmoid(runtime)
    if declared is not None:
        meta["kda_use_beta_sigmoid"] = declared
    # The analytical producer and the CANN KeyPool caller are different
    # contracts. Record only the CANN selection so a missing lowering stays
    # on the analytical default.
    if _pool_indexer_lowering(runtime) == "vllm_ascend_cann_keypool_caller":
        meta["pool_key_indexer"] = "vllm_ascend_cann_keypool_caller"
    # Deployment block size.  Attention uses it only for incremental q=1
    # page-table index traffic.  A missing or zero value stays off that path.
    block = getattr(runtime, "kv_block_size", 0) or 0
    if isinstance(block, bool) or not isinstance(block, int) or block < 0:
        raise ModelingError("kv_block_size_invalid")
    if block > 0:
        meta["kv_block_size"] = block
    if "cp_kv_cache_interleave_size" not in meta and block > 0:
        meta["cp_kv_cache_interleave_size"] = block
    max_model_len = getattr(runtime, "max_model_len", 0) or 0
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len < 0:
        raise ModelingError("max_model_len_invalid")
    if max_model_len > 0:
        meta["max_model_len"] = max_model_len
    # Source MC2 capacity reads the configured scheduler budgets directly.
    # These are runtime declarations, not timing inputs; a missing value keeps
    # the historical static recipe and the selector's own branch.
    for key in ("max_num_batched_tokens", "max_num_seqs"):
        value = getattr(runtime, key, 0) or 0
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ModelingError(f"runtime_{key}_invalid")
        if value > 0:
            meta[key] = value
    kernels = getattr(runtime, "kernels", None)
    if isinstance(kernels, Mapping) and "sequence_parallel" in kernels:
        decl = kernels.get("sequence_parallel")
        if not isinstance(decl, Mapping):
            raise ModelingError("runtime_sequence_parallel_invalid")
        meta["sequence_parallel"] = dict(decl)
    if isinstance(kernels, Mapping) and "lowerings" in kernels:
        lowerings = kernels.get("lowerings") or {}
        if not isinstance(lowerings, Mapping):
            raise ModelingError("runtime_lowerings_invalid")
        meta["source_lowerings"] = tuple(sorted(str(key) for key in lowerings))
        fusion = lowerings.get("native_fusion")
        if isinstance(fusion, Mapping):
            options = fusion.get("options")
            if isinstance(options, Mapping):
                roles = options.get("roles")
                if roles:
                    meta["native_fusion_roles"] = tuple(
                        str(role) for role in roles)
                profiles = options.get("program_profiles")
                if isinstance(profiles, Mapping):
                    meta["native_fusion_profiles"] = tuple(sorted(
                        (str(key), str(value)) for key, value in profiles.items()))
    return tuple(sorted(meta.items()))
def _machine_aic_cores(machine: Any) -> int | None:
    """Declared AIC count; structure-only callers retain an unresolved value."""
    dies = getattr(machine, "dies", ()) if machine is not None else ()
    if dies and getattr(dies[0], "aic_count", 0):
        return int(dies[0].aic_count)
    return None


@dataclass(frozen=True)
class FeatureConfig:
    """Normalised projection of Runtime/ParallelSpec/QuantSpec.

    This is deliberately not a second user-facing schema: it is constructed
    from the existing runtime/model records and only adds implementation
    tiles that the static contract must fix.
    """

    tp: int = 1
    ep: int = 1
    etp: int = 1
    pp: int = 1
    dp: int = 1
    cp: int = 1
    pcp: int = 1
    cp_plan: str = "none"
    activation_dtype: str = "bf16"
    quant_activation_format: str = ""
    weight_dtype: str = "bf16"
    kv_dtype: str = "bf16"
    index_dtype: str = "int8"
    quant: QuantSpec | None = None
    graph_buckets: tuple[int, ...] = ()
    multistream_shared_expert: bool = False
    multistream_kda_proj: bool = False
    multistream_attention_indexer: bool = False
    multistream_window_compressed: bool = False
    moe_recipe: str = "alltoall"
    mc2_chunks: int = _DEFAULT_MC2_CHUNKS
    lowering_passes: tuple[str, ...] = ("activation_quantization",)
    implementation: str = "analytical"
    final_norm_scope: str = "all"
    token_tile: int = _DEFAULT_MT
    kv_tile: int = _DEFAULT_KV_TILE
    cores: int | None = None
    moe_row_tile: int = _DEFAULT_MOE_ROWS
    split_target: int = _DEFAULT_SPLIT_TARGET
    max_splits: int = _DEFAULT_MAX_SPLITS
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for name in ("tp", "ep", "etp", "pp", "dp", "cp", "pcp"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ModelingError(f"feature_parallel_invalid:{name}")
        tiles = ("token_tile", "kv_tile", "moe_row_tile", "split_target", "max_splits")
        for name in tiles + (("cores",) if self.cores is not None else ()):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ModelingError(f"feature_tile_invalid:{name}")
        for name in ("activation_dtype", "weight_dtype", "kv_dtype", "index_dtype"):
            dtype_bytes(getattr(self, name))
        if self.quant_activation_format:
            dtype_bytes(self.quant_activation_format)
        if self.final_norm_scope not in {"all", "selected"}:
            raise ModelingError("feature_final_norm_scope_invalid")
        if self.cp_plan not in {"none", "dcp", "dsa_cp", "pcp", "pcp_dcp"}:
            raise ModelingError(f"feature_cp_plan_invalid:{self.cp_plan}")
        if self.moe_recipe not in {"alltoall", "mc2", "tp_join_reduce",
                                   "local_expert_reduce"}:
            raise ModelingError(f"feature_moe_recipe_invalid:{self.moe_recipe}")
        if self.mc2_chunks < 1:
            raise ModelingError("feature_mc2_chunks_invalid")
        object.__setattr__(self, "graph_buckets", tuple(sorted(set(self.graph_buckets))))
        object.__setattr__(self, "lowering_passes", tuple(self.lowering_passes))
        if isinstance(self.metadata, Mapping):
            object.__setattr__(self, "metadata", tuple(sorted(self.metadata.items())))
        else:
            object.__setattr__(self, "metadata", tuple(self.metadata))

    @property
    def activation_format(self) -> str:
        return self.quant_activation_format or self.activation_dtype or "bf16"

    @property
    def world_size(self) -> int:
        return self.dp * self.pp * self.tp * self.pcp

    @property
    def etp_width(self) -> int: return max(1, self.etp)

    @property
    def max_bucket(self) -> int: return max(self.graph_buckets) if self.graph_buckets else 0

    def project(self, *, tp: int = 0, dp: int = 0, ep: int = 0, pp: int = 0,
                cores: int | None = None, graph_buckets: Iterable[int] | None = None,
                kv_dtype: str = "", kv_bf16_override: bool = False) -> "FeatureConfig":
        tp_v, dp_v = max(1, tp or self.tp), max(1, dp or self.dp)
        ep_v = max(1, int(ep or self.ep))
        # ETP is the remaining TP shard of each expert, not DP*TP/EP.  EP
        # groups already live in the DP*TP domain; EP beyond TP leaves ETP=1.
        etp_v = (tp_v if ep_v <= 1 else
                 (tp_v // ep_v if ep_v <= tp_v and tp_v % ep_v == 0 else 1))
        meta = dict(self.metadata)
        if kv_bf16_override: meta["kv_bf16_override"] = 1
        updates: dict[str, Any] = dict(tp=tp_v, dp=dp_v, ep=ep_v, etp=etp_v,
                                       pp=max(1, pp or self.pp),
                                       graph_buckets=(tuple(self.graph_buckets)
                                                      if graph_buckets is None or not tuple(graph_buckets)
                                                      else tuple(graph_buckets)),
                                       metadata=tuple(sorted(meta.items())))
        if cores is not None and cores > 0: updates["cores"] = int(cores)
        if kv_dtype: updates["kv_dtype"] = kv_dtype
        return replace(self, **updates)

    @classmethod
    def from_runtime(cls, runtime: Any, model: ModelSpec | None = None,
                     *, machine: Any = None) -> "FeatureConfig":
        parallel = getattr(runtime, "parallel", ParallelSpec()) if runtime is not None else ParallelSpec()
        if not isinstance(parallel, ParallelSpec):
            raise ModelingError("feature_parallel_record_invalid")
        quant = getattr(model, "quant", None) if model is not None else None
        declared_etp = int(getattr(parallel, "etp", 0) or 0)
        if declared_etp < 1:
            declared_etp = (parallel.tp if parallel.ep <= 1 else
                            (parallel.tp // parallel.ep
                             if parallel.ep <= parallel.tp and parallel.tp % parallel.ep == 0
                             else 1))
        return cls(
            tp=parallel.tp, ep=parallel.ep, etp=declared_etp,
            pp=parallel.pp, dp=parallel.dp, cp=parallel.cp, pcp=parallel.pcp,
            cp_plan=str(getattr(parallel, "cp_plan", "none") or "none"),
            activation_dtype=getattr(model, "activation_dtype", "bf16") if model else "bf16",
            quant_activation_format=(str(quant.activation_format)
                                     if quant is not None and quant.activation_format else ""),
            weight_dtype=getattr(model, "weight_dtype", "bf16") if model else "bf16",
            kv_dtype=((str(getattr(quant, "kv_dtype", "") or "") if quant else "") or
                      (getattr(model, "kv_dtype", "bf16") if model else "bf16")),
            index_dtype=(str(getattr(quant, "index_dtype", "") or "") if quant else "") or "int8",
            quant=quant,
            graph_buckets=_effective_graph_buckets(runtime),
            multistream_shared_expert=bool(getattr(runtime, "multistream_overlap_shared_expert", False)),
            multistream_kda_proj=bool(getattr(runtime, "multistream_kda_proj", False)),
            multistream_attention_indexer=bool(getattr(runtime, "multistream_attention_indexer", False)),
            multistream_window_compressed=bool(getattr(runtime, "multistream_window_compressed", False)),
            moe_recipe=str(getattr(runtime, "moe_recipe", "alltoall")),
            mc2_chunks=int(getattr(runtime, "mc2_chunks", _DEFAULT_MC2_CHUNKS)),
            lowering_passes=tuple(getattr(runtime, "lowering_passes", ("activation_quantization",))),
            final_norm_scope=str(getattr(runtime, "final_norm_scope", "all")),
            cores=_machine_aic_cores(machine),
            metadata=_runtime_metadata(runtime, machine, model),
        )

    @classmethod
    def from_model(cls, model: ModelSpec, *, machine: Any = None,
                   **overrides: Any) -> "FeatureConfig":
        quant = getattr(model, "quant", None)
        base = dict(
            activation_dtype=model.activation_dtype or "bf16",
            quant_activation_format=(str(quant.activation_format)
                                     if quant is not None and quant.activation_format else ""),
            weight_dtype=model.weight_dtype or "bf16",
            kv_dtype=((getattr(quant, "kv_dtype", "") if quant else "")
                      or model.kv_dtype or "bf16"),
            index_dtype=((getattr(quant, "index_dtype", "") if quant else "")
                         or "int8"),
            quant=quant,
            cp_plan=(("dcp" if int(overrides.get("cp", 1) or 1) > 1 else "none")
                     if "cp_plan" not in overrides else str(overrides["cp_plan"])),
            metadata=tuple(sorted({**_identity_metadata(machine, model),
                                  **_model_source_metadata(model)}.items())),
        )
        base.setdefault("cores", _machine_aic_cores(machine))
        if "etp" not in overrides:
            tp = int(overrides.get("tp", 1) or 1)
            dp = int(overrides.get("dp", 1) or 1)
            ep = int(overrides.get("ep", 1) or 1)
            base["etp"] = ((tp if ep <= 1 else
                            (tp // ep if ep <= tp and tp % ep == 0 else 1))
                           if tp > 0 else 1)
        base.update(overrides)
        return cls(**base)

    def with_parallel(self, parallel: ParallelSpec) -> "FeatureConfig":
        eff_etp = min(self.etp, parallel.tp) if parallel.tp > 0 else self.etp
        return replace(self, tp=parallel.tp, ep=parallel.ep, pp=parallel.pp,
                       dp=parallel.dp, cp=parallel.cp, pcp=parallel.pcp,
                       etp=eff_etp,
                       cp_plan=str(getattr(parallel, "cp_plan", self.cp_plan)))

__all__ = ["FeatureConfig"]
