"""Deterministic instantiation of canonical subgraphs and static ``Theta``.

[Role]
  Project ``(ModelSpec, LayerPlan, FeatureConfig)`` into immutable, typed
  ``SubgraphInstance`` records.  Each operator receives a fully fixed
  ``StaticSignature``: weight dimensions, local head/expert geometry,
  quantisation role bytes, parallel axes and declared tiles.  Only sequence
  geometry and explicit state ports remain dynamic.

[Invariants]
  * Instantiation is a pure function of model/layer/features and declared
    implementation tiles; no measurement, model-name branch or request id is
    consulted.
  * Semantic macros reference child subgraphs instead of duplicating their
    work.  A child kind present in ``SubgraphTemplate.children`` contributes no
    second operator list here.
  * Quantisation is resolved per role; an override on one role cannot rewrite
    another role's weight payload, scale or zero-point accounting.
  * Repeated layers intern the same template object; instance identity and
    layer ids stay outside the template and performance key.
  * Unsupported mechanisms remain ``CoverageHole`` records, never fabricated
    operators or measured latency.

[Timeouts & Locks]
  Bounded per-layer construction; no iteration history, waits or locks.

[Side-Effect Policy]
  Returns new immutable records and mutates only the caller-supplied bounded
  template cache.  No I/O, device access or global model state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from mindie_sim.catalog.cache import local_kv_heads
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.dynamic_quant_schedule import dynamic_quant_traffic
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.subgraphs import (
    BoundOperator, CoverageHole, CoverageReport, FusionContract,
    OperatorTemplate, StatePortSpec, StaticSignature, SubgraphInstance,
    SubgraphTemplate,
)
from .registry import (
    TemplateRegistry, build, port, register_builder, template,
    validate_combination,
)

_DEFAULT_MT = 128
_DEFAULT_NT = 128
_DEFAULT_KV_TILE = 128
_DEFAULT_CORES = 24
_DEFAULT_MOE_ROWS = 128
_DEFAULT_SPLIT_TARGET = 4096
_DEFAULT_MAX_SPLITS = 8

def _declared_kda_use_beta_sigmoid(runtime: Any) -> int | None:
    """Return the declared KDA option, or None when the case does not set it.

    Native ``recurrent.linear`` removes the separate ``linear_beta_sigmoid``
    producer only when this option is true on a decode source bind. The
    piecewise walk reads the same declaration from feature metadata.
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
            meta["rope_parameters"] = dict(rope)
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
    if len(_NORMAL_SUMMARY_CACHE) >= 32:
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
    from mindie_sim.prediction.mx_source_provenance import source_metadata
    meta.update(dict(source_metadata(meta)))
    return tuple(sorted(meta.items()))


def source_lowering_selected(ctx: Any, name: str) -> bool:
    """True only when the deployment/source declares this native lowering.

    A legacy feature record without a lowering table keeps the existing
    template contract; once a real runtime table is present, absence means the
    causal template stays unfused.  The template alone is not selection.
    """
    keys = dict(getattr(ctx.features, "metadata", ()) or ()).get("source_lowerings")
    if keys is None:
        return True
    declared = {str(value) for value in keys}
    return str(name) in declared

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
    mc2_chunks: int = 4
    lowering_passes: tuple[str, ...] = ("activation_quantization",)
    implementation: str = "analytical"
    final_norm_scope: str = "all"
    token_tile: int = _DEFAULT_MT
    kv_tile: int = _DEFAULT_KV_TILE
    cores: int = _DEFAULT_CORES
    moe_row_tile: int = _DEFAULT_MOE_ROWS
    split_target: int = _DEFAULT_SPLIT_TARGET
    max_splits: int = _DEFAULT_MAX_SPLITS
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for name in ("tp", "ep", "etp", "pp", "dp", "cp", "pcp"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ModelingError(f"feature_parallel_invalid:{name}")
        for name in ("token_tile", "kv_tile", "cores", "moe_row_tile",
                     "split_target", "max_splits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ModelingError(f"feature_tile_invalid:{name}")
        for name in ("activation_dtype", "weight_dtype", "kv_dtype", "index_dtype"):
            dtype_bytes(getattr(self, name))
        if self.quant_activation_format:
            dtype_bytes(self.quant_activation_format)
        if self.final_norm_scope not in {"all", "selected"}:
            raise ModelingError("feature_final_norm_scope_invalid")
        if self.cp_plan not in {"none", "dcp", "dsa_cp", "pcp"}:
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
                cores: int = 0, graph_buckets: Iterable[int] | None = None,
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
        if cores > 0: updates["cores"] = int(cores)
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
            mc2_chunks=int(getattr(runtime, "mc2_chunks", 4)),
            lowering_passes=tuple(getattr(runtime, "lowering_passes", ("activation_quantization",))),
            final_norm_scope=str(getattr(runtime, "final_norm_scope", "all")),
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
        return replace(self, tp=parallel.tp, ep=parallel.ep, pp=parallel.pp,
                       dp=parallel.dp, cp=parallel.cp, pcp=parallel.pcp,
                       cp_plan=str(getattr(parallel, "cp_plan", self.cp_plan)))

@dataclass(frozen=True)
class _Ctx:
    model: ModelSpec
    layer: LayerPlan
    features: FeatureConfig
    stage: int
    role: str = "body"
    variant: str = ""
    layers: int = 0

    def __post_init__(self) -> None:
        if self.layer.index < 0:
            raise ModelingError("layer_index_invalid")
        if self.stage < 0 or self.stage >= self.features.pp:
            raise ModelingError(f"layer_stage_invalid:{self.stage}")
        if self.layers and self.layer.index >= self.layers:
            raise ModelingError(f"layer_index_exceeds_model:{self.layer.index}")

    @property
    def hidden(self) -> int: return int(self.model.hidden_size)
    @property
    def tp(self) -> int: return self.features.tp
    @property
    def ep(self) -> int: return self.features.ep
    @property
    def etp(self) -> int: return self.features.etp
    @property
    def heads(self) -> int: return int(self.layer.num_heads or self.model.num_heads)
    @property
    def local_heads(self) -> int: return max(1, self.heads // self.tp)
    @property
    def kv_heads(self) -> int: return int(self.layer.num_kv_heads or self.model.num_kv_heads or self.heads)
    @property
    def local_kv_heads(self) -> int: return local_kv_heads(self.kv_heads, self.tp)
    @property
    def group_heads(self) -> int: return max(1, self.local_heads // self.local_kv_heads)
    @property
    def width(self) -> int: return int(self.layer.head_dim or self.model.head_dim)
    @property
    def intermediate(self) -> int: return int(self.layer.intermediate_size or self.model.intermediate_size)
    @property
    def local_intermediate(self) -> int: return max(1, math.ceil(self.intermediate / self.tp))
    @property
    def local_experts(self) -> int:
        experts = int(self.layer.num_experts or self.model.num_experts or 1)
        return max(1, math.ceil(experts / self.ep))

    @property
    def expert_width(self) -> int:
        width = int(self.layer.moe_intermediate_size or self.model.moe_intermediate_size)
        return max(1, math.ceil(width / max(1, self.etp)))

    @property
    def latent(self) -> int:
        return int(self.layer.expert_latent_size or self.hidden)

    @property
    def shared_width(self) -> int:
        width = int(self.layer.shared_intermediate_size
                    or self.layer.moe_intermediate_size or self.model.moe_intermediate_size)
        count = int(self.layer.num_shared_experts or self.model.num_shared_experts or 1)
        return width * max(1, count)

    @property
    def top_k(self) -> int:
        return int(self.layer.top_k or self.model.top_k or 1)

    @property
    def quant(self) -> QuantSpec | None:
        return self.features.quant

    @property
    def kv_dtype(self) -> str:
        return self.features.kv_dtype or self.model.kv_dtype

    @property
    def kv_row_elements(self) -> int:
        return int(self.layer.cache_row_elements)

    @property
    def kv_row_bytes(self) -> float:
        # Ordinary GQA/MQA owns one local KV row per rank (replicated when the
        # global KV head count is below TP); using the global cache row would
        # charge every rank for all logical KV heads.  MLA and compressed
        # representations keep their declared cache-row layout.
        if (self.layer.attention in {"gqa", "mha", "mqa"}
                and self.layer.cache_kind != "window_compressed"):
            elements = self.local_kv_heads * (self.width + self.width)
        else:
            elements = self.kv_row_elements
        return elements * dtype_bytes(self.kv_dtype)

    @property
    def shared_consumer(self) -> bool:
        owner = int(self.layer.kv_owner_layer_id)
        return 0 <= owner < self.layer.index and not self.layer.owns_long_kv

    @property
    def attention_kind(self) -> str:
        if self.layer.cache_kind == "window_compressed":
            return "A2"
        if self.layer.attention == "mla":
            return "A1"
        if self.layer.attention == "gdn":
            return "A3"
        if self.layer.attention == "kda":
            return "A4"
        return "A0"

def _sig(role: str, **groups: Any) -> StaticSignature:
    return StaticSignature(role, **groups)

def _parallel(ctx: _Ctx) -> tuple[tuple[str, Any], ...]:
    return (("tp", ctx.tp), ("ep", ctx.ep), ("etp", ctx.etp),
            ("pp", ctx.features.pp), ("dp", ctx.features.dp),
            ("cp", ctx.features.cp), ("pcp", ctx.features.pcp),
            ("cp_plan", ctx.features.cp_plan),
            ("world", ctx.features.world_size), ("ranks_per_node", int(dict(ctx.features.metadata).get("ranks_per_node", 8) or 8)))

def _tiles(ctx: _Ctx) -> tuple[tuple[str, Any], ...]:
    features = ctx.features
    return (("mt", features.token_tile), ("nt", features.kv_tile),
            ("cores", features.cores), ("moe_rows", features.moe_row_tile),
            ("split_target", features.split_target),
            ("max_splits", features.max_splits),
            ("graph_buckets", features.graph_buckets))

_LOW_ACT = frozenset({"mxfp8", "mxfp4", "fp8_e4m3", "int8"})
_HIGH_ACT = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})

def _matrix_act_format(ctx: _Ctx, category: str) -> str:
    stream = ctx.features.activation_dtype or "bf16"
    quant = ctx.quant
    if quant is None:
        declared = ctx.features.quant_activation_format
        return declared if declared in _LOW_ACT else stream
    resolved = quant.for_category(category)
    fmt = str(resolved.format_for(category))
    # compute_dtype_for returns fp32 activations for an fp32 category.  The
    # model stream dtype is the pre-cast value, not the matrix operand.
    if fmt in {"fp32", "float32"}:
        return "fp32"
    if fmt in _HIGH_ACT:
        return stream
    act = str(resolved.activation_format or ctx.features.quant_activation_format or "")
    return act if act in _LOW_ACT else stream


def _quant_fields(ctx: _Ctx, category: str, rows: int, cols: int) -> tuple[tuple[str, Any], ...]:
    rows, cols = int(rows), int(cols)
    quant = ctx.quant
    act_format = _matrix_act_format(ctx, category)
    fallback = dtype_bytes(act_format)
    if rows <= 0 or cols <= 0:
        return (("weight_format", ctx.features.weight_dtype), ("act_format", act_format),
                ("weight_payload", 0.0), ("weight_scale", 0.0), ("weight_zero", 0.0),
                ("weight_total", 0.0), ("residency", 0.0), ("scale_group", 0),
                ("act_bytes", fallback), ("quantized", 0))
    if quant is None:
        weight_bytes = float(rows * cols) * dtype_bytes(ctx.features.weight_dtype)
        return (("weight_format", ctx.features.weight_dtype),
                ("act_format", act_format),
                ("weight_payload", weight_bytes), ("weight_scale", 0.0),
                ("weight_zero", 0.0), ("weight_total", weight_bytes),
                ("residency", weight_bytes), ("scale_group", 0),
                ("act_bytes", fallback), ("quantized", 0))
    resolved = quant.for_category(category)
    payload = resolved.matrix_payload_bytes(rows, cols, category)
    total = resolved.matrix_bytes(rows, cols, category)
    zero = 0.0
    fmt = resolved.format_for(category)
    runtime_meta = resolved.execution_int8_weight_metadata()
    runtime_scale_bytes = resolved.execution_weight_scale_bytes()
    if fmt in {"int8", "w8a8"} and runtime_meta == "scale_offset":
        zero = float(rows) * float(runtime_scale_bytes)
    scale = max(0.0, total - payload - zero)
    group = 0
    if fmt not in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
        group = int(resolved.weight_block[1] if resolved.weight_block else 0)
        if not group:
            group = int(resolved.expert_block or 0)
    fields = (("weight_format", fmt), ("act_format", act_format),
              ("weight_payload", payload), ("weight_scale", scale),
              ("weight_zero", zero), ("weight_total", total),
              ("residency", total), ("scale_group", group),
              ("act_bytes", dtype_bytes(act_format)),
              ("quantized", int(fmt not in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})))
    method = resolved.linear_method(category)
    prequant = str(getattr(ctx.features, "activation_dtype", "") or
                   getattr(ctx.features, "activation_format", "") or "bf16")
    declared_output = resolved.output_format_for(
        category, prequant=prequant)
    fields += (("linear_method", method), ("prequant_format", prequant),
               ("category", str(category)))
    if method == "mx_fp8":
        # The consumer's second input is the producer's E8M0 activation scale,
        # M x (K/64) x 2.  Keep it as a separate tensor; never fold it into
        # the E4M3 payload width.
        blocks = (int(cols) + 31) // 32
        padded = (blocks + 1) // 2 * 2
        fields += (("activation_payload_bytes", float(int(cols))),
                   ("activation_scale_bytes", float(padded)),
                   ("activation_scale_shape", (padded // 2, 2)),
                   ("activation_scale_format", "fp8_e8m0"))
    if declared_output:
        fields += (("output_format", declared_output),)
    if fmt not in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
        fields += (("weight_scale_format", resolved.execution_weight_scale_format()),
                   ("weight_scale_bytes", float(runtime_scale_bytes)),
                   ("weight_scale_storage_format", resolved.scale_format),
                   ("weight_scale_storage_bytes", float(resolved.scale_bytes)))
    return fields


def _declared_compute_quant(rows: int, cols: int, dtype: str) -> tuple[tuple[str, Any], ...]:
    """Weight and activation of a compute dtype declared beside the quant plan.

    Native router construction replaces the model quant with
    ``QuantSpec(weight_format=router_dtype, activation_format=router_dtype)``
    and category ``router_runtime``.  The ``router`` exclusion is BF16 and
    does not carry that declaration.
    """
    if dtype not in {"fp32", "bf16", "fp16"}:
        raise ModelingError(f"unsupported_declared_compute_dtype:{dtype}")
    width = float(dtype_bytes(dtype))
    payload = float(int(rows)) * float(int(cols)) * width
    return (
        ("weight_format", dtype), ("act_format", dtype),
        ("weight_payload", payload), ("weight_scale", 0.0),
        ("weight_zero", 0.0), ("weight_total", payload),
        ("residency", payload), ("scale_group", 0),
        ("act_bytes", width), ("quantized", 0),
    )

def _result_bytes(ctx: _Ctx, category: str) -> float:
    """Materialized consumer bytes for a role, with the prior fallback."""
    prequant = str(getattr(ctx.features, "activation_dtype", "") or
                   getattr(ctx.features, "activation_format", "") or "bf16")
    fmt = (ctx.quant.output_format_for(category, prequant=prequant)
           if ctx.quant is not None else "")
    return dtype_bytes(fmt or ctx.features.activation_format)


def _matrix(role: str, k: int, n: int, ctx: _Ctx, category: str,
            *, extra: tuple[tuple[str, Any], ...] = (),
            slots: tuple[str, ...] = ("weight_total",)) -> StaticSignature:
    return _sig(role, matrix=(("K", int(k)), ("N", int(n))),
                quant=_quant_fields(ctx, category, n, k),
                parallel=_parallel(ctx), tiles=_tiles(ctx), extras=extra)

def _norm_gamma(ctx: _Ctx, width: int, role: str) -> tuple[tuple[str, Any], ...]:
    """Declare one normalization gamma read in its actual loaded format."""
    quant = ctx.quant
    if quant is not None:
        runtime = quant.norm_gamma_bytes(width, role)
        storage = quant.norm_gamma_bytes(width, role, source=True)
        runtime_format = quant.norm_gamma_format(role)
        storage_format = quant.norm_gamma_format(role, source=True)
    else:
        runtime = storage = width * dtype_bytes(ctx.features.activation_dtype or "bf16")
        runtime_format = storage_format = ctx.features.activation_dtype or "bf16"
    return (
        ("gamma_bytes", runtime),
        ("gamma_storage_bytes", storage),
        ("gamma_format", runtime_format),
        ("gamma_storage_format", storage_format),
        ("gamma_role", role),
    )


def _elementwise(role: str, ctx: _Ctx, *, elements: int, ops: int,
                 read_bytes: float, write_bytes: float,
                 extra: tuple[tuple[str, Any], ...] = ()) -> StaticSignature:
    return _sig(role, matrix=(("elements", int(elements)),),
                quant=(("act_bytes", dtype_bytes(ctx.features.activation_format)),),
                parallel=_parallel(ctx), tiles=_tiles(ctx),
                extras=(("ops_per_element", ops), ("read_bytes", read_bytes),
                        ("write_bytes", write_bytes), *extra))

def _copy(role: str, ctx: _Ctx, *, elements: int, bytes_per_element: float,
          extra: tuple[tuple[str, Any], ...] = ()) -> StaticSignature:
    return _sig(role, matrix=(("elements", int(elements)),),
                quant=(("act_bytes", bytes_per_element),),
                parallel=_parallel(ctx), tiles=_tiles(ctx), extras=extra)

def _collective(role: str, ctx: _Ctx, *, axis: str, members: int,
                bytes_per_token: float, stages: int = 2,
                extra: tuple[tuple[str, Any], ...] = ()) -> StaticSignature:
    return _sig(role, parallel=_parallel(ctx), tiles=_tiles(ctx),
                quant=(("wire_bytes", float(bytes_per_token)),),
                extras=(("group_axis", axis), ("members", int(members)),
                        ("stages", int(stages)), *extra))


def _dense_flashcomm(ctx: _Ctx) -> bool:
    """v0.30 dense FlashComm sequence parallelism. Not a timed branch."""
    from mindie_sim.prediction.piecewise.flashcomm_dense import dense_flashcomm_enabled
    return dense_flashcomm_enabled(ctx.model, ctx.features)


def _activation_hidden_bytes(ctx: _Ctx) -> float:
    """Residual payload. The collective moves the activation stream, not the quant stream."""
    return float(ctx.hidden) * float(dtype_bytes(ctx.features.activation_dtype or "bf16"))


def _native_group2_declared(ctx: _Ctx) -> bool:
    """Only an explicit structured AICPU/BF16 AG-RS opt-in admits the curve."""
    meta = dict(ctx.features.metadata or {})
    decl = meta.get("a5_native_group2")
    if not isinstance(decl, dict):
        return False
    primitives = decl.get("primitives")
    if not isinstance(primitives, (list, tuple)):
        return False
    return bool(
        ctx.tp == 2
        and str(ctx.features.activation_dtype or "").lower() == "bf16"
        and str(decl.get("schema") or "") == "mindie-sim/a5-native-group2/1"
        and int(decl.get("group_size") or 0) == 2
        and str(decl.get("dtype") or "").lower() == "bf16"
        and str(decl.get("execution_core") or "").lower() == "aicpu"
        and str(decl.get("stack") or "").lower() == "rack08"
        and str(decl.get("cann") or "") == "9.2.0"
        and bool(decl.get("source_path"))
        and str(decl.get("deployment_image_id") or "") == str(meta.get("image_id") or "")
        and str(decl.get("deployment_cann") or "") in str(meta.get("image") or "")
        and {"all_gather", "reduce_scatter"} <= {str(v) for v in primitives}
        and "cann9.2.0" in str(meta.get("image") or ""))


def _sp_collective(ctx: _Ctx, kind: str) -> StaticSignature:
    rows = "shard" if kind == "all_gather" else "padded"
    name = "tp_all_gather" if kind == "all_gather" else "tp_reduce_scatter"
    extra = (("collective", kind), ("sp_rows", rows),
             ("payload_dtype", str(ctx.features.activation_dtype or "bf16")))
    if _native_group2_declared(ctx):
        decl = dict(ctx.features.metadata or {}).get("a5_native_group2")
        extra += (("a5_native_group2", "native-20261107"),
                  ("a5_native_stack", str(decl.get("stack") or "rack08")),
                  ("a5_native_cann", str(decl.get("cann") or "9.2.0")),
                  ("a5_native_source", str(decl.get("source_id")
                                            or "a5-hccl-native-20261107")),
                  ("a5_native_source_path", str(decl.get("source_path") or "")),
                  ("a5_native_execution_core", str(decl.get("execution_core") or "")),
                  ("a5_native_placement", str(decl.get("placement") or "")),
                  ("a5_native_deployment_image_id",
                   str(decl.get("deployment_image_id") or "")))
    return _collective(
        name, ctx, axis="tp", members=ctx.tp,
        bytes_per_token=_activation_hidden_bytes(ctx), extra=extra)


def _dense_sp_block(specs: list, signatures: dict, ctx: _Ctx) -> None:
    """Gather the sequence shard, then reduce-scatter partial sums.

    This replaces one TP allreduce. The gather is first because the shard
    norm is the preceding subgraph. Do not also append ``tp_allreduce``.
    """
    specs.insert(0, ("tp_all_gather", "collective", "tp_all_gather"))
    signatures["tp_all_gather"] = _sp_collective(ctx, "all_gather")
    specs.append(("tp_reduce_scatter", "collective", "tp_reduce_scatter"))
    signatures["tp_reduce_scatter"] = _sp_collective(ctx, "reduce_scatter")


def _dense_sp_exit(specs: list, signatures: dict, ctx: _Ctx) -> None:
    """One hidden all-gather after the final norm and before the LM head."""
    specs.append(("tp_all_gather", "collective", "tp_all_gather"))
    signatures["tp_all_gather"] = _sp_collective(ctx, "all_gather")

def _state(role: str, ctx: _Ctx, *, row_bytes: float, rows: int = 1,
           extra: tuple[tuple[str, Any], ...] = ()) -> StaticSignature:
    return _sig(role, matrix=(("rows", int(rows)), ("row_bytes", float(row_bytes))),
                parallel=_parallel(ctx), tiles=_tiles(ctx), extras=extra)

def _attention_sig(role: str, ctx: _Ctx, *, local_heads: int, local_kv_heads: int,
                   group_heads: int, width: int, value_width: int,
                   kv_row_bytes: float, extra: tuple[tuple[str, Any], ...] = ()) -> StaticSignature:
    return _sig(
        role,
        heads=(("q", int(local_heads)), ("kv", int(local_kv_heads)),
               ("g", int(group_heads)), ("dk", int(width)), ("dv", int(value_width))),
        quant=(("kv_bytes", float(kv_row_bytes)),
               ("act_bytes", dtype_bytes(ctx.features.activation_format))),
        parallel=_parallel(ctx), tiles=_tiles(ctx),
        extras=(("window", int(ctx.layer.window)),
                ("causal", int(not ctx.layer.noncausal_attention)),
                ("noncausal", int(ctx.layer.noncausal_attention)),
                ("cp", int(ctx.features.cp)),
                ("cp_plan", str(ctx.features.cp_plan)),
                ("cp_interleave", int(dict(ctx.features.metadata).get(
                    "cp_kv_cache_interleave_size", 128) or 128)),
                *extra),
    )

# ---------------------------------------------------------------------------
# Reusable ports

def _in(name: str, representation: str, **kwargs: Any) -> StatePortSpec:
    return port(name, "in", representation, **kwargs)

def _out(name: str, representation: str, **kwargs: Any) -> StatePortSpec:
    return port(name, "out", representation, **kwargs)

_HIDDEN_IN = _in("hidden", "activation")
_HIDDEN_OUT = _out("hidden", "activation")
_RESIDUAL_IN = _in("residual", "activation")
_RESIDUAL_OUT = _out("residual", "activation")
_KV_IN = _in("kv_cache", "kv_cache")
_KV_OUT = _out("kv_cache", "kv_cache")
_KV_READY_OUT = _out("kv_ready", "readiness", scope="stage")

@dataclass(frozen=True)
class _Built:
    template: SubgraphTemplate
    theta: tuple[tuple[str, StaticSignature], ...]

    def signature(self, name: str) -> StaticSignature:
        for key, value in self.theta:
            if key == name:
                return value
        raise ModelingError(f"builder_signature_missing:{self.template.kind}:{name}")

def _cross_facts(ctx: _Ctx) -> dict[str, Any]:
    routed = str(dict(_quant_fields(ctx, "routed_expert", 1, 1))["weight_format"]) if int(ctx.model.num_experts or 0) else ""
    kv_bf16 = int(bool(dict(ctx.features.metadata).get("kv_bf16_override")))
    return {"kv_dtype": ctx.kv_dtype, "kv_bf16": kv_bf16, "routed_weight_format": routed, "max_bucket": ctx.features.max_bucket}

def _built(kind: str, ctx: _Ctx, variant: str, title: str,
           specs: Iterable[tuple[str, str, str]], signatures: Mapping[str, StaticSignature],
           *, children: tuple[str, ...] = (), fusions: tuple[FusionContract, ...] = (),
           state_in: tuple[StatePortSpec, ...] = (), state_out: tuple[StatePortSpec, ...] = (),
           coverage: str = "structural", contract: tuple[tuple[str, str], ...] = (),
           streams: Mapping[str, str] | None = None) -> _Built:
    operators, theta, facts, smap = [], [], _cross_facts(ctx), dict(streams or ())
    for name, family, role in specs:
        if name not in signatures:
            raise ModelingError(f"builder_signature_missing:{kind}:{name}")
        operators.append(OperatorTemplate(name=name, family=family, role=role, stream=smap.get(name, "main")))
        theta.append((name, signatures[name].relocated(**facts)))
    return _Built(SubgraphTemplate(kind=kind, variant=variant, title=title, operators=tuple(operators),
                  state_in=state_in, state_out=state_out, children=children, fusions=fusions,
                  coverage=coverage, contract=contract), tuple(theta))

def _variant(ctx: _Ctx, default: str) -> str:
    return ctx.variant or default

def _guard_low_act(specs: list, signatures: dict, ctx: _Ctx, name: str, *,
                   elements: int, category: str, dynamic_only: bool = False) -> None:
    if not name:
        return
    fmt = _matrix_act_format(ctx, category)
    if fmt not in _LOW_ACT or (specs and specs[-1][1] == "quantize"):
        return
    count = int(elements)
    if fmt in {"int8", "int4"}:
        # Per-token npu_dynamic_quant. Static int8 and FP8/MX do not qualify
        # and must not fall through to a one-pass stand-in.
        traffic = dynamic_quant_traffic(
            ctx.quant, category, count, fmt,
            str(ctx.features.activation_dtype or "bf16"))
        if traffic is None:
            return
        read_bytes, write_bytes, extra = traffic
        qname = f"{name}_QuantV2"
        specs.append((qname, "quantize", "quantize"))
        signatures[qname] = _elementwise(
            "quantize", ctx, elements=count, ops=2,
            read_bytes=read_bytes, write_bytes=write_bytes, extra=extra)
        return
    # New dense/GDN/qkv sites ask only for the dynamic int producer. FP8 and
    # MX stay on the call sites that already named them.
    if dynamic_only:
        return
    # MX is a resolved runtime method, not a bare E4M3 label.  The header
    # normalizes the stored weight dtype to fp8_e4m3; only linear_method
    # distinguishes the group-32 E8M0 DynamicMxQuant producer from regular
    # FP8 or INT8 quant rows.
    method = (ctx.quant.linear_method(category)
              if ctx.quant is not None else "dense")
    if method == "mx_fp8":
        from mindie_sim.prediction.mx_dynamic_quant import producer_contract
        stream_dtype = str(ctx.features.activation_dtype or "bf16")
        source_revision = str(dict(ctx.features.metadata).get(
            "mx_quant_source_revision", "unpinned"))
        contract = producer_contract(count, input_dtype=stream_dtype,
                                     source_revision=source_revision)
        # A1.o keeps the pre-existing QuantV2 graph slot/name so downstream
        # state/fusion identity is stable; other MX roles use the native name.
        # In both cases the program and ABI are DynamicMxQuant.
        qname = (f"{name}_QuantV2"
                 if category in {"attn.o", "attn.qkv"} else
                 f"{name}_DynamicMxQuant")
        specs.append((qname, "quantize", "quantize"))
        signatures[qname] = _elementwise(
            "quantize", ctx, elements=count, ops=1,
            read_bytes=float(contract["input_bytes"]),
            write_bytes=float(contract["write_bytes"]),
            extra=(("quant_op", "DynamicMxQuant"),
                   ("native_kind", "DynamicMxQuant"),
                   ("out_format", "fp8_e4m3"),
                   ("output_format", "fp8_e4m3"),
                   ("input_format", str(contract["input_format"])),
                   ("tiling_key", int(contract["tiling_key"])),
                   ("source_branch", str(contract["branch"])),
                   ("source_revision", str(contract["source_revision"])),
                   ("source_certified", int(bool(contract.get("source_certified")))),
                   ("activation_scale_format", "fp8_e8m0"),
                   ("activation_scale_shape", tuple(contract["scale_shape"])),
                   ("activation_payload_bytes", int(contract["payload_bytes"])),
                   ("activation_scale_bytes", int(contract["scale_bytes"])),
                   ("row_scale_bytes", int(contract["scale_bytes"])),
                   ("mx_scale_row_bytes", int(contract["scale_bytes"])),
                   ("mx_groups_per_row", int(contract["scale_bytes"])),
                   ("mx_vector_ops_per_row", int(contract["vector_ops"])),
                   ("mx_register_calls", int(contract["work"]["register_calls"]["total"])),
                   ("mx_work_kind", str(contract["work"]["work_kind"])),
                   ("mx_issue_cycles_known", int(bool(contract["work"]["issue_cycles_known"]))),
                   ("mx_algorithm", str(contract["work"]["algorithm"])),
                   ("mx_source", f"{contract['source_revision']}:{contract['source'].get('kernel', 'unpinned')}:{contract['source'].get('kernel_sha256', contract['source'].get('sha256', ''))}"),
                   ("category", category)))
        if name in signatures:
            signatures[name] = signatures[name].relocated(
                scale_producer=qname,
                activation_payload_input=f"{qname}_output_payload",
                activation_scale_input=f"{qname}_output_scale")
        return
    # No raw FP4 DynamicMxQuant producer is claimed here; only the resolved
    # mx_fp8 branch above has the source ABI and group-32 E8M0 contract.
    op = "QuantV2"
    qname = f"{name}_{op}"
    specs.append((qname, "quantize", "quantize"))
    signatures[qname] = _elementwise(
        "quantize", ctx, elements=count, ops=2,
        read_bytes=count * dtype_bytes(ctx.features.activation_dtype or "bf16"),
        write_bytes=count * dtype_bytes(fmt),
        extra=(("quant_op", op), ("out_format", fmt), ("category", category)))


def _precede(specs: list, signatures: dict, ctx: _Ctx, name: str,
             elements: int, category: str, *, dynamic_only: bool = False) -> None:
    """Insert the quant guard immediately before ``name`` when that operator exists."""
    if not name:
        return
    index = next((i for i, item in enumerate(specs) if item[0] == name), None)
    if index is None:
        return
    head = list(specs[:index])
    _guard_low_act(head, signatures, ctx, name, elements=elements, category=category,
                   dynamic_only=dynamic_only)
    if len(head) > index:
        specs[index:index] = head[index:]


from .builders_attention import BUILDERS as _ATTENTION_BUILDERS
from .builders_ffn import BUILDERS as _FFN_BUILDERS

_BUILDERS = {**_ATTENTION_BUILDERS, **_FFN_BUILDERS}

for _kind, _builder in _BUILDERS.items():
    register_builder(_kind, _builder)

@dataclass(frozen=True)
class LayerSlot:
    kind: str
    variant: str
    role: str

@dataclass(frozen=True)
class LayerInstances:
    layer: int
    stage: int
    subgraphs: tuple[SubgraphInstance, ...]
    slots: tuple[LayerSlot, ...]
    holes: tuple[CoverageHole, ...] = ()

    def find(self, kind: str, role: str | None = None) -> SubgraphInstance:
        for instance in self.subgraphs:
            if instance.kind != kind:
                continue
            inst_role = str(dict(instance.metadata).get("role", ""))
            if role is None or inst_role == role or (role in {"pre", "post"} and inst_role.endswith("_" + role)):
                return instance
        raise ModelingError(f"subgraph_instance_missing:{kind}:{self.layer}:{role}")

@dataclass(frozen=True)
class InstantiatedModel:
    model: ModelSpec
    features: FeatureConfig
    layers: tuple[LayerInstances, ...]
    globals: tuple[SubgraphInstance, ...]
    coverage: CoverageReport
    stage_of: tuple[int, ...]

    @property
    def subgraphs(self) -> tuple[SubgraphInstance, ...]:
        return tuple(instance for layer in self.layers for instance in layer.subgraphs) + self.globals

    def find(self, kind: str, *, layer: int | None = None, role: str | None = None) -> SubgraphInstance:
        candidates = self.globals if layer is None else self.layers[layer].subgraphs
        for instance in candidates:
            if instance.kind != kind:
                continue
            inst_role = str(dict(instance.metadata).get("role", ""))
            if role is None or inst_role == role or (role in {"pre", "post"} and inst_role.endswith("_" + role)):
                return instance
        raise ModelingError(f"subgraph_instance_missing:{kind}:{layer}:{role}")

    def performance_keys(self) -> tuple[tuple, ...]:
        return tuple(instance.performance_key() for instance in self.subgraphs)

    def fingerprints(self) -> tuple[str, ...]:
        return tuple(instance.fingerprint() for instance in self.subgraphs)

def _stage_for(layer: int, layers: int, pp: int) -> int:
    if pp <= 1 or layers <= 1:
        return 0
    return min(pp - 1, layer * pp // layers)

def _validate_parallel(model: ModelSpec, features: FeatureConfig) -> None:
    """Reject layouts whose local ownership would silently drop heads."""
    tp = max(1, features.tp)
    if tp > 1:
        heads = int(model.num_heads)
        if heads % tp:
            raise ModelingError(f"heads_not_divisible_by_tp:{heads}%{tp}")
        for layer in model.layer_plans:
            local = int(layer.num_heads or heads)
            if local % tp:
                raise ModelingError(
                    f"layer_heads_not_divisible_by_tp:{layer.index}:{local}%{tp}")
    ep = max(1, features.ep)
    if ep > 1:
        declared = [int(model.num_experts or 0)]
        declared.extend(int(layer.num_experts or 0) for layer in model.layer_plans)
        for count in declared:
            if count and count % ep:
                raise ModelingError(f"experts_not_divisible_by_ep:{count}%{ep}")

def _gemma_facts(model: ModelSpec) -> dict:
    return dict(model.catalog_facts)

def _layer_has_ple(model: ModelSpec, layer: LayerPlan) -> bool:
    facts = _gemma_facts(model)
    layers = facts.get("gemma4_ple_layers")
    if layers is not None:
        return layer.index in {int(value) for value in layers}
    if facts.get("ple") is not None:
        return bool(facts.get("ple"))
    return bool(layer.engram and facts.get("gemma4"))

def _layer_slots(model: ModelSpec, layer: LayerPlan,
                 features: FeatureConfig) -> tuple[LayerSlot, ...]:
    slots: list[LayerSlot] = []
    hc = bool(layer.hyper_connections)

    def boundary(name: str) -> None:
        if hc:
            slots.append(LayerSlot("R0", name, name))

    if layer.residual_block_size:
        slots.append(LayerSlot("R1", "block_stack", "residual"))
    slots.append(LayerSlot("N0", "rmsnorm", "attn_norm"))
    boundary("attn_pre")
    attention_kind = ("A2" if layer.cache_kind == "window_compressed"
                      else "A1" if layer.attention == "mla"
                      else "A3" if layer.attention == "gdn"
                      else "A4" if layer.attention == "kda" else "A0")
    shared_consumer = (0 <= layer.kv_owner_layer_id < layer.index
                       and not layer.owns_long_kv)
    kv_slot = LayerSlot("K0", "read" if shared_consumer else "store", "kv")
    if attention_kind == "A0":
        variant = "mqa" if (layer.num_kv_heads or 0) == 1 and layer.num_heads > 1 else "gqa"
        if shared_consumer:
            variant = "shared_consumer"
        slots.append(LayerSlot(attention_kind, variant, "attention"))
        slots.append(kv_slot)
    elif attention_kind == "A1":
        if layer.compress_ratio or layer.cache_kind == "window_compressed":
            slots.append(LayerSlot("K1", str(layer.compressor_kind or "weighted_pool"), "compress"))
        if layer.indexer:
            slots.append(LayerSlot("I0", str(layer.indexer_type or "full"), "index_produce"))
            slots.append(LayerSlot("I1", "candidate" if layer.candidate_filter else "topk", "index_select"))
        slots.append(LayerSlot(attention_kind,
                               "latent_mqa" if layer.mla_variant == "latent_mqa"
                               else "latent_mla", "attention"))
        if shared_consumer:
            slots.append(kv_slot)
    elif attention_kind in {"A3", "A4"}:
        slots.append(LayerSlot(attention_kind, "", "attention"))
    else:
        slots.append(LayerSlot("A2", "window_compressed", "attention_prolog"))
        slots.append(LayerSlot("K0", "layout", "kv"))
        if layer.compress_ratio or layer.cache_kind == "window_compressed":
            slots.append(LayerSlot("K1", str(layer.compressor_kind or "weighted_pool"),
                                   "compress"))
        if layer.indexer:
            slots.append(LayerSlot("I0", str(layer.indexer_type or "full"), "index_produce"))
            slots.append(LayerSlot("I1", "candidate" if layer.candidate_filter else "topk",
                                   "index_select"))
        slots.append(LayerSlot("A2", "", "attention"))
    if attention_kind not in {"A1", "A2"}:
        if layer.compress_ratio or layer.cache_kind == "window_compressed":
            slots.append(LayerSlot("K1", str(layer.compressor_kind or "weighted_pool"),
                                   "compress"))
        if layer.indexer:
            slots.append(LayerSlot("I0", str(layer.indexer_type or "full"), "index_produce"))
            slots.append(LayerSlot("I1", "candidate" if layer.candidate_filter else "topk",
                                   "index_select"))
    boundary("attn_post")
    # The post-attention residual is folded into the pre-FFN AddRmsNorm;
    # there is no third standalone residual N0 per layer.
    slots.append(LayerSlot("N0", "layernorm" if layer.ffn_activation == "gelu" and False
                           else "rmsnorm", "ffn_norm"))
    boundary("ffn_pre")
    if layer.ffn == "moe":
        slots.append(LayerSlot("F1", str(layer.router_mode or "softmax"), "router"))
        slots.append(LayerSlot("F2", features.moe_recipe, "transport"))
        slots.append(LayerSlot("F3", "latent" if layer.expert_latent_size else "packed",
                               "routed_ffn"))
        slots.append(LayerSlot("F2", features.moe_recipe, "transport_combine"))
        if layer.num_shared_experts:
            slots.append(LayerSlot("F4", "gated" if layer.shared_expert_gate else "plain",
                                   "shared_ffn"))
            slots.append(LayerSlot("F4", "join", "join"))
    else:
        activation = str(layer.ffn_activation or "silu")
        if activation not in {"silu", "gelu", "gelu_tanh", "situ", "ungated"}:
            activation = "silu"
        slots.append(LayerSlot("F0", activation, "dense_ffn"))
    boundary("ffn_post")
    if layer.engram:
        slots.append(LayerSlot("E1", "engram", "engram"))
    elif _layer_has_ple(model, layer):
        slots.append(LayerSlot("E1", "ple", "ple"))
    return tuple(slots)

def instantiate_layer(model: ModelSpec, layer: LayerPlan,
                      features: FeatureConfig, *,
                      registry: TemplateRegistry | None = None,
                      layers: int | None = None,
                      stage: int | None = None) -> LayerInstances:
    total = int(layers if layers is not None else len(model.layer_plans) or model.layers)
    _validate_parallel(model, features)
    stage_value = _stage_for(layer.index, total, features.pp) if stage is None else int(stage)
    registry = registry if registry is not None else _DEFAULT_REGISTRY
    slots = _layer_slots(model, layer, features)
    validate_combination(slot.kind for slot in slots)
    subgraphs: list[SubgraphInstance] = []
    holes: list[CoverageHole] = []
    for slot in slots:
        ctx = _Ctx(model=model, layer=layer, features=features, stage=stage_value,
                   role=slot.role, variant=slot.variant, layers=total)
        built = build(slot.kind, ctx)
        template = registry.intern(built.template)
        bound = tuple(BoundOperator(op, built.signature(op.name))
                      for op in template.operators)
        instance = SubgraphInstance(
            instance_id=f"{slot.kind}:{template.variant}:L{layer.index}:{slot.role}",
            template=template, layer=layer.index, stage=stage_value,
            operators=bound, state_in=template.state_in,
            state_out=template.state_out,
            owner_layer=(int(layer.kv_owner_layer_id)
                         if 0 <= layer.kv_owner_layer_id < layer.index else None),
            metadata=(("role", slot.role), ("variant", slot.variant),
                      ("attention_kind", "A1" if layer.attention == "mla" else layer.attention)),
        )
        subgraphs.append(instance)
        if template.coverage != "structural":
            holes.append(CoverageHole(slot.kind, "declared_partial", template.title))
    return LayerInstances(layer.index, stage_value, tuple(subgraphs), slots, tuple(holes))

_DEFAULT_REGISTRY = TemplateRegistry()

def _has_ple(model: ModelSpec) -> bool:
    facts = dict(model.catalog_facts)
    metadata = dict(model.metadata)
    return bool(facts.get("ple") or metadata.get("ple") or facts.get("gemma_ple"))

def instantiate_model(model: ModelSpec, features: FeatureConfig, *,
                      registry: TemplateRegistry | None = None) -> InstantiatedModel:
    if not model.layer_plans:
        raise ModelingError("instantiate_requires_explicit_layer_plans")
    registry = registry if registry is not None else _DEFAULT_REGISTRY
    total = len(model.layer_plans)
    layers = tuple(
        instantiate_layer(model, layer, features, registry=registry,
                          layers=total, stage=_stage_for(layer.index, total, features.pp))
        for layer in model.layer_plans)
    globals_: list[SubgraphInstance] = []
    holes: list[CoverageHole] = []

    def add_global(kind: str, variant: str, role: str, stage: int) -> None:
        ctx = _Ctx(model=model, layer=model.layer_plans[0], features=features,
                   stage=min(features.pp - 1, max(0, stage)), role=role,
                   variant=variant, layers=total)
        built = build(kind, ctx)
        template = registry.intern(built.template)
        bound = tuple(BoundOperator(op, built.signature(op.name))
                      for op in template.operators)
        globals_.append(SubgraphInstance(
            instance_id=f"{kind}:{template.variant}:G:{role}", template=template,
            layer=0, stage=min(features.pp - 1, max(0, stage)), operators=bound,
            state_in=template.state_in, state_out=template.state_out,
            metadata=(("role", role), ("global", True))))
        if template.coverage != "structural":
            holes.append(CoverageHole(kind, "declared_partial", template.title))

    add_global("E0", "media" if model.vision is not None else "token", "input", 0)
    if model.vision is not None:
        add_global("V0", "vision", "media", 0)
    if model.has_optional_draft or model.is_standalone_draft:
        draft_var = ("dspark" if model.auxiliary and model.auxiliary.dspark and not model.auxiliary.mtp
                     else ("mtp" if model.has_optional_draft else "standalone"))
        add_global("D0", draft_var, "draft", features.pp - 1)
    add_global("O0", features.final_norm_scope, "output", features.pp - 1)
    entries = []
    for layer in layers:
        for instance in layer.subgraphs:
            entries.append((instance.instance_id, f"{instance.kind}:{instance.variant}"))
    for instance in globals_:
        entries.append((instance.instance_id, f"{instance.kind}:{instance.variant}"))
    holes.extend(layer_hole for layer in layers for layer_hole in layer.holes)
    if _has_ple(model) and not any(layer_slot.kind == "E1" for layer in layers
                                   for layer_slot in layer.slots):
        if not dict(model.catalog_facts).get("gemma4"):
            holes.append(CoverageHole("E1", "ple_not_instantiated",
                                      "PLE declared without an explicit per-layer table record"))
    coverage = CoverageReport(tuple(entries), tuple(holes))
    return InstantiatedModel(
        model=model, features=features, layers=layers, globals=tuple(globals_),
        coverage=coverage, stage_of=tuple(layer.stage for layer in layers))

def instantiate(model: ModelSpec, features: FeatureConfig | None = None, *,
                runtime: Any = None, registry: TemplateRegistry | None = None,
                machine: Any | None = None) -> InstantiatedModel:
    return instantiate_model(
        model, features or FeatureConfig.from_runtime(runtime, model, machine=machine),
        registry=registry)
