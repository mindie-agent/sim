"""Model, request/workload, and SLO typed contracts.

[Role]
  Own Dense/GQA/MLA and Dense/MoE model semantics, request workloads and SLO
  records used by both predictive and normative paths.

[Invariants]
  * ``hidden_size`` is the residual width; Q/O projection width may differ.
  * KV heads are replicated when TP exceeds the KV-head count.
  * Expert routing distributions are explicit or uniform; quantized weights
    carry a declared weight dtype and explicit dequant work downstream.
  * Revision/source fields are metadata and do not alter semantics.

[Timeouts & Locks]
  Pure validation.  No I/O, locks or device probes.

[Side-Effect Policy]
  No filesystem, network or global mutation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import ModelingError, _DTYPE_BYTES, _int, _mapping, _metadata, _normalize_dtype, _positive_float, _unknown, dtype_bytes
from mindie_sim.catalog.types import (
    AuxiliarySpec, DraftSpec, EngramSpec, QuantSpec, VisionSpec,
)


@dataclass(frozen=True)
class MlaSpec:
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    # Optional DeepSeek-style fields; default 0 keeps catalog entries unchanged.
    o_lora_rank: int = 0
    o_groups: int = 0
    index_topk: int = 0
    index_head_dim: int = 0
    index_n_heads: int = 0
    compress_ratio: int = 0
    sliding_window: int = 0

    @classmethod
    def from_mapping(cls, value: Any) -> "MlaSpec":
        raw = _mapping(value, "model.mla_invalid")
        allowed = {"q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
                   "o_lora_rank", "o_groups", "index_topk", "index_head_dim", "index_n_heads",
                   "compress_ratio", "sliding_window"}
        _unknown(raw, allowed, "model.mla")
        return cls(
            q_lora_rank=_int(raw.get("q_lora_rank") or 0, "model.mla.q_lora_rank_invalid", minimum=0),
            kv_lora_rank=_int(raw.get("kv_lora_rank", 0), "model.mla.kv_lora_rank_invalid", minimum=1),
            qk_nope_head_dim=_int(raw.get("qk_nope_head_dim", 0), "model.mla.qk_nope_invalid", minimum=1),
            qk_rope_head_dim=_int(raw.get("qk_rope_head_dim", 0), "model.mla.qk_rope_invalid", minimum=0),
            v_head_dim=_int(raw.get("v_head_dim", 0), "model.mla.v_head_dim_invalid", minimum=1),
            o_lora_rank=_int(raw.get("o_lora_rank", 0), "model.mla.o_lora_rank_invalid", minimum=0),
            o_groups=_int(raw.get("o_groups", 0), "model.mla.o_groups_invalid", minimum=0),
            index_topk=_int(raw.get("index_topk", 0), "model.mla.index_topk_invalid", minimum=0),
            index_head_dim=_int(raw.get("index_head_dim", 0), "model.mla.index_head_dim_invalid", minimum=0),
            index_n_heads=_int(raw.get("index_n_heads", 0), "model.mla.index_n_heads_invalid", minimum=0),
            compress_ratio=_int(raw.get("compress_ratio", 0), "model.mla.compress_ratio_invalid", minimum=0),
            sliding_window=_int(raw.get("sliding_window", 0), "model.mla.sliding_window_invalid", minimum=0),
        )

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def to_dict(self) -> dict[str, Any]:
        return {
            "q_lora_rank": self.q_lora_rank, "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim, "qk_rope_head_dim": self.qk_rope_head_dim,
            "qk_head_dim": self.qk_head_dim, "v_head_dim": self.v_head_dim,
        }


@dataclass(frozen=True)
class LayerPlan:
    index: int
    attention: str = "gqa"          # gqa | mla | kda | gdn
    ffn: str = "dense"              # dense | moe
    compress_ratio: int = 0
    indexer: bool = False
    window: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    intermediate_size: int = 0
    num_experts: int = 0
    top_k: int = 0
    moe_intermediate_size: int = 0
    num_shared_experts: int = 0
    linear_heads: int = 0
    linear_head_dim: int = 0
    conv_kernel: int = 0
    state_bytes: float = 0.0
    mtp: bool = False
    # --- catalog-rich structural fields (defaults preserve generic callers) ---
    architecture: str = "gqa"
    indexer_type: str = "none"          # none | c4 | full | shared
    candidate_filter: bool = False
    o_lora_rank: int = 0
    o_groups: int = 0
    mla_variant: str = "none"           # none | latent_mqa | latent_mla
    mla_output_gate: bool = False
    q_output_gate: bool = False
    noncausal_attention: bool = False
    expert_latent_size: int = 0
    expert_latent_norm_position: str = "pre"  # pre | post | none
    router_input_size: int = 0
    router_mode: str = "softmax"
    router_dtype: str = ""
    router_output_dtype: str = ""       # independent of operands and matrix compute
    router_precast_weight_copy: bool = False
    router_norm_topk: bool = False
    router_groups: int = 0
    router_group_topk: int = 0
    shared_intermediate_size: int = 0
    shared_expert_gate: bool = False
    linear_key_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_heads: int = 0
    packed_input_projection: bool = False
    gate_rank: int = 0               # zero keeps a direct hidden->gate projection
    projection_dtype: str = ""      # source default; explicit quant roles override
    conv_width: int = 0
    recurrent_state_elements: int = 0
    recurrent_state_dtype: str = "fp32"
    conv_state_elements: int = 0
    conv_state_dtype: str = "bf16"
    kv_cache_elements: int = 0
    kv_owner_layer_id: int = -1
    owns_long_kv: bool = False
    index_k_owner_layer_id: int = -1
    owns_index_k: bool = False
    topk_owner_layer_id: int = -1
    owns_topk: bool = False
    owns_candidate_blocks: bool = False
    candidate_source_layer_id: int = -1
    index_source_layer_id: int = -1
    index_source_status: str = "exact"
    engram: bool = False
    hyper_connections: bool = False
    hc_mult: int = 0
    hc_sinkhorn_iters: int = 0
    residual_block_size: int = 0
    native_layer_id: int = -1
    scope: str = "body"                 # body | mtp | dspark | standalone_draft
    cache_kind: str = "dense"           # dense | window_compressed
    compressor_kind: str = "none"      # none | weighted_pool | gated_ring
    compressor_width: int = 1
    compressor_state_rows: int = 0
    compressor_norm_dtype: str = "bf16"
    index_input: str = "hidden"         # hidden | compressed
    index_query_per_layer: bool = False
    index_heads: int = 0
    index_dim: int = 0
    index_topk: int = 0
    candidate_block_size: int = 1
    candidate_topk_blocks: int = 0
    qk_norm: bool = False
    ffn_activation: str = "silu"
    activation_situ_beta: float = 1.0
    activation_situ_linear_beta: float | None = None

    @property
    def indexer_executes(self) -> bool:
        return bool(self.indexer)

    @property
    def cache_row_elements(self) -> int:
        if self.kv_cache_elements:
            return int(self.kv_cache_elements)
        if self.attention == "mla":
            return int(self.kv_lora_rank + self.qk_rope_head_dim)
        return int(2 * self.num_kv_heads * self.head_dim)

    @property
    def state_bytes_total(self) -> float:
        return float(self.recurrent_state_elements) * float(dtype_bytes(self.recurrent_state_dtype)) \
            + float(self.conv_state_elements) * float(dtype_bytes(self.conv_state_dtype))

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layers: int
    hidden_size: int
    num_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    attention: str = "gqa"
    ffn: str = "dense"
    num_kv_heads: int = 0
    activation_dtype: str = "bf16"
    weight_dtype: str = "bf16"
    kv_dtype: str = "bf16"
    compute_dtype: str = "bf16"
    norm_eps: float = 1e-6
    quantization: str = "none"
    num_experts: int = 0
    top_k: int = 0
    moe_intermediate_size: int = 0
    num_shared_experts: int = 0
    expert_routing: tuple[float, ...] = ()
    expert_routing_source: str = "declared"
    mla: MlaSpec | None = None
    tie_word_embeddings: bool = False
    layer_plans: tuple[LayerPlan, ...] = ()
    state_bytes_per_request: float = 0.0
    catalog_facts: tuple[tuple[str, Any], ...] = ()
    metadata: tuple[tuple[str, Any], ...] = ()
    quant: QuantSpec | None = None
    vision: VisionSpec | None = None
    engram: EngramSpec | None = None
    auxiliary: AuxiliarySpec | None = None
    body_layers: int = 0

    @property
    def is_standalone_draft(self) -> bool:
        return bool(self.auxiliary is not None and self.auxiliary.standalone_draft)

    @property
    def has_optional_draft(self) -> bool:
        return bool(
            self.auxiliary is not None
            and not self.auxiliary.standalone_draft
            and (self.auxiliary.mtp is not None or self.auxiliary.dspark is not None)
        )

    @staticmethod
    def canonical_fields(value: Any) -> dict:
        raw = _mapping(value, "model_invalid")
        aliases = {
            "n_layers": "layers", "num_hidden_layers": "layers",
            "d_model": "hidden_size", "hidden_dim": "hidden_size",
            "n_heads": "num_heads", "num_attention_heads": "num_heads",
            "n_kv_heads": "num_kv_heads", "num_key_value_heads": "num_kv_heads",
            "d_head": "head_dim", "ffn_dim": "intermediate_size",
            "ffn_intermediate_size": "intermediate_size", "n_experts": "num_experts",
            "n_routed_experts": "num_experts", "num_experts_per_tok": "top_k",
            "moe_inter_dim": "moe_intermediate_size", "dtype": "activation_dtype",
            "attention_type": "attention", "ffn_type": "ffn",
            "rms_norm_eps": "norm_eps",
        }
        result = {}
        for key, item in raw.items():
            name = aliases.get(key, key)
            if name in result and result[name] != item:
                raise ModelingError(f"model_field_alias_conflict:{name}")
            result[name] = item
        return result

    @classmethod
    def from_mapping(cls, value: Any) -> "ModelSpec":
        value = _mapping(value, "model_invalid")
        # External HF configs have a separate interpretation boundary. Explicit
        # simulator records retain strict field checking and roundtrip semantics.
        if "layer_plans" not in value and (
            "hf_config" in value or any(key in value for key in
                                       ("model_type", "architectures", "text_config"))
        ):
            from mindie_sim.catalog.specs import build_model_spec
            config = _mapping(value.get("hf_config", value), "model.hf_config_invalid")
            key = str(value.get("name", config.get("_name_or_path",
                      config.get("name", config.get("model_type", "external-model")))))
            model = build_model_spec(key, config)
            if "hf_config" in value:
                from .serialization import apply_model_overrides
                overrides = cls.canonical_fields({k: v for k, v in value.items() if k != "hf_config"})
                return apply_model_overrides(model, overrides)
            return model
        raw = cls.canonical_fields(value)
        if "layer_plans" in raw and not any(key in raw for key in ("catalog_key", "model_catalog_key", "preset")):
            from .serialization import explicit_model
            return explicit_model(raw)
        allowed = {
            "name", "layers", "hidden_size", "num_heads", "num_attention_heads",
            "head_dim", "num_kv_heads", "num_key_value_heads", "intermediate_size",
            "ffn_intermediate_size", "vocab_size", "attention", "ffn", "architecture",
            "activation_dtype", "weight_dtype", "kv_dtype", "compute_dtype", "quantization",
            "num_experts", "n_routed_experts", "top_k", "moe_intermediate_size",
            "num_shared_experts", "expert_routing", "expert_routing_histogram",
            "expert_routing_distribution", "mla", "tie_word_embeddings",
            "catalog_key", "model_catalog_key", "preset", "metadata",
            "n_layers", "num_hidden_layers", "d_model", "hidden_dim", "n_heads",
            "n_kv_heads", "d_head", "ffn_dim", "n_experts", "num_experts_per_tok",
            "moe_inter_dim", "dtype", "attention_type", "ffn_type",
            "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
            "v_head_dim", "o_lora_rank", "o_groups", "index_topk", "index_head_dim",
            "index_n_heads", "compress_ratio", "sliding_window",
            "norm_eps", "layer_plans", "layer_overrides", "auxiliary", "quant",
        }
        _unknown(raw, allowed, "model")
        catalog_key = raw.get("catalog_key", raw.get("model_catalog_key", raw.get("preset")))
        if catalog_key:
            from mindie_sim.model_catalog import load_model

            model = load_model(str(catalog_key))
            from .serialization import apply_model_overrides
            return apply_model_overrides(model, raw)
        architecture = str(raw.get("architecture", "")).strip().lower()
        attention = str(raw.get("attention", "")).strip().lower()
        ffn = str(raw.get("ffn", "")).strip().lower()
        if architecture == "mla+moe":
            attention, ffn = "mla", "moe"
        elif architecture in {"dense", "gqa", "mla", "moe"}:
            if architecture == "mla":
                attention = "mla"
            elif architecture == "moe":
                ffn = "moe"
            elif architecture in {"dense", "gqa"}:
                attention = attention or "gqa"
        elif architecture:
            raise ModelingError(f"architecture_requires_layer_plans:{architecture}")
        attention = attention or "gqa"
        ffn = ffn or ("moe" if raw.get("num_experts") else "dense")
        if attention not in {"gqa", "mla"}:
            raise ModelingError(f"model.attention_unsupported:{attention}")
        if ffn not in {"dense", "moe"}:
            raise ModelingError(f"model.ffn_unsupported:{ffn}")
        hidden = _int(raw.get("hidden_size"), "model.hidden_size_required", minimum=1)
        heads = _int(raw.get("num_heads", raw.get("num_attention_heads")), "model.num_heads_required", minimum=1)
        head_dim = _int(raw.get("head_dim", hidden // heads if hidden % heads == 0 else 0), "model.head_dim_required", minimum=1)
        # hidden_size is the residual/model width; Q/O projection width may be
        # num_heads*head_dim and is intentionally allowed to differ.
        num_kv_heads = _int(raw.get("num_kv_heads", raw.get("num_key_value_heads", heads)), "model.num_kv_heads_invalid", minimum=1)
        if attention == "gqa" and heads % num_kv_heads != 0:
            raise ModelingError(f"model.gqa_group_not_integral:{heads}%{num_kv_heads}")
        mla_raw = raw.get("mla")
        if mla_raw is None and any(
            key in raw for key in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "v_head_dim")
        ):
            mla_raw = {key: raw[key] for key in (
                "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim",
                "qk_rope_head_dim", "v_head_dim",
                "o_lora_rank", "o_groups", "index_topk", "index_head_dim",
                "index_n_heads", "compress_ratio", "sliding_window",
            ) if key in raw}
        mla = MlaSpec.from_mapping(mla_raw) if mla_raw is not None else None
        if attention == "mla" and mla is None:
            raise ModelingError("model.mla_required")
        quantization_raw = raw.get("quantization", "none")
        if isinstance(quantization_raw, Mapping):
            quantization = str(quantization_raw.get("weight", quantization_raw.get("name", "none"))).strip().lower()
        else:
            quantization = str(quantization_raw).strip().lower()
        from .serialization import auxiliary, immutable_sequences
        from mindie_sim.catalog.quant import QUANTIZATION_PRESETS, quant_preset

        quant = QuantSpec(**immutable_sequences(raw["quant"])) if raw.get("quant") else None
        if quant is None and quantization in QUANTIZATION_PRESETS:
            quant = quant_preset(quantization)
            # Legacy explicit storage fields override a preset, matching the
            # catalog wrapper. An explicit QuantSpec remains authoritative.
            changes = {}
            if "weight_dtype" in raw:
                declared_weight = _normalize_dtype(raw["weight_dtype"], "model.weight_dtype_invalid")
                changes.update(weight_format=declared_weight, expert_weight_format=declared_weight)
            if "kv_dtype" in raw:
                changes["kv_dtype"] = _normalize_dtype(raw["kv_dtype"], "model.kv_dtype_invalid")
            if changes:
                quant = replace(quant, **changes)
        if quant is None and "fp8" in quantization:
            raise ModelingError("fp8_requires_explicit_quant_block_and_scale_geometry")
        # Matrix operands belong to QuantSpec. Residual/norm/attention tensors
        # stay BF16 unless the caller explicitly declares a different dtype.
        # This is the same representation boundary used by HF/catalog inputs.
        weight_default = quant.weight_format if quant else _quant_weight_dtype(quantization)
        activation_default = "bf16" if quant else _quant_activation_dtype(quantization)
        weight_dtype = _normalize_dtype(raw.get("weight_dtype", weight_default), "model.weight_dtype_invalid")
        activation_dtype = _normalize_dtype(raw.get("activation_dtype", activation_default), "model.activation_dtype_invalid")
        kv_dtype = _normalize_dtype(raw.get("kv_dtype", quant.kv_dtype if quant else "bf16"), "model.kv_dtype_invalid")
        compute_dtype = _normalize_dtype(raw.get("compute_dtype", activation_dtype), "model.compute_dtype_invalid")
        layers = _int(raw.get("layers"), "model.layers_required", minimum=1)
        intermediate = _int(raw.get("intermediate_size", raw.get("ffn_intermediate_size")), "model.intermediate_size_required", minimum=1)
        vocab = _int(raw.get("vocab_size"), "model.vocab_size_required", minimum=1)
        experts = _int(raw.get("num_experts", raw.get("n_routed_experts", 0)), "model.num_experts_invalid")
        top_k = _int(raw.get("top_k", 0), "model.top_k_invalid")
        moe_intermediate = _int(raw.get("moe_intermediate_size", 0), "model.moe_intermediate_invalid")
        shared = _int(raw.get("num_shared_experts", 0), "model.shared_experts_invalid")
        if ffn == "moe":
            if experts < 2 or top_k < 1 or top_k > experts or moe_intermediate < 1:
                raise ModelingError("model.moe_configuration_invalid")
        routing, routing_source = _parse_routing(raw, experts)
        return cls(
            name=str(raw.get("name", "model")).strip(), layers=layers, hidden_size=hidden,
            num_heads=heads, head_dim=head_dim, intermediate_size=intermediate,
            vocab_size=vocab, attention=attention, ffn=ffn, num_kv_heads=num_kv_heads,
            activation_dtype=activation_dtype, weight_dtype=weight_dtype, kv_dtype=kv_dtype,
            compute_dtype=compute_dtype, quantization=quantization, num_experts=experts,
            top_k=top_k, moe_intermediate_size=moe_intermediate,
            num_shared_experts=shared, expert_routing=routing,
            expert_routing_source=routing_source, mla=mla,
            norm_eps=float(raw.get("norm_eps", 1e-6)), quant=quant,
            auxiliary=auxiliary(immutable_sequences(raw["auxiliary"])) if raw.get("auxiliary") else None,
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            metadata=_metadata(raw),
        )

    @property
    def attention_label(self) -> str:
        return self.attention

    def weight_bytes(self, parallel) -> float:
        from mindie_sim.prediction.structure import normalize_model
        from mindie_sim.catalog.weights import stage_weight_bytes
        model = normalize_model(self)
        partition = tuple(model.layers // parallel.pp + int(i < model.layers % parallel.pp)
                          for i in range(parallel.pp))
        return max(stage_weight_bytes(model, parallel, layer_partition=partition))

    def parameter_count(self) -> float:
        from mindie_sim.prediction.structure import normalize_model
        from mindie_sim.catalog.weights import body_parameter_count
        return body_parameter_count(normalize_model(self))

    def to_dict(self):
        from dataclasses import asdict
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        data["catalog_facts"] = dict(self.catalog_facts)
        return data


def _quant_weight_dtype(quantization: str) -> str:
    q = quantization.lower()
    if q in {"fp8", "w8a8_fp8", "fp8_e4m3"}:
        return "fp8"
    if q in {"int8", "w8a8"}:
        return "int8"
    if q in {"int4", "awq", "gptq"}:
        return "int4"
    return "bf16"


def _quant_activation_dtype(quantization: str) -> str:
    q = quantization.lower()
    if "fp8" in q:
        return "fp8"
    if "int8" in q or q == "w8a8":
        return "int8"
    return "bf16"


def _parse_routing(raw: Mapping[str, Any], experts: int) -> tuple[tuple[float, ...], str]:
    value = raw.get("expert_routing", raw.get("expert_routing_histogram"))
    if value is None:
        return (), "uniform"
    if isinstance(value, Mapping):
        entries = [float(value.get(str(i), value.get(i, 0.0))) for i in range(experts)]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        entries = [float(item) for item in value]
    else:
        raise ModelingError("model.expert_routing_invalid")
    if len(entries) != experts or any(not math.isfinite(v) or v < 0 for v in entries):
        raise ModelingError("model.expert_routing_invalid")
    total = sum(entries)
    if total <= 0:
        return (), "uniform"
    return tuple(v / total for v in entries), "declared_histogram"


@dataclass(frozen=True)
class SloSpec:
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    itl_ms: float | None = None
    e2e_ms: float | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "SloSpec":
        if value is None:
            return cls()
        raw = _mapping(value, "slo_invalid")
        _unknown(raw, {"ttft_ms", "tpot_ms", "itl_ms", "e2e_ms"}, "slo")
        result = cls(
            ttft_ms=_optional_positive(raw.get("ttft_ms"), "slo.ttft_invalid"),
            tpot_ms=_optional_positive(raw.get("tpot_ms"), "slo.tpot_invalid"),
            itl_ms=_optional_positive(raw.get("itl_ms"), "slo.itl_invalid"),
            e2e_ms=_optional_positive(raw.get("e2e_ms"), "slo.e2e_invalid"),
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"ttft_ms": self.ttft_ms, "tpot_ms": self.tpot_ms, "itl_ms": self.itl_ms, "e2e_ms": self.e2e_ms}


def _optional_positive(value: Any, reason: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, reason)


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int
    prefix_tokens: int = 0
    priority: int = 0
    text_prompt_tokens: int = 0
    media_tokens: int = 0
    media_encoded: bool = False
    # Closed-loop chain: request i+1 is released at request i's PREDICTED completion
    # plus ``gap_ms`` (never from a measured finish).  chain_id 0 = open loop.
    chain_id: int = 0
    gap_ms: float = 0.0

    @classmethod
    def from_mapping(cls, value: Any, index: int) -> "RequestSpec":
        raw = _mapping(value, f"workload.requests[{index}]_invalid")
        _unknown(raw, {"id", "request_id", "arrival_ms", "prompt_tokens", "prompt_len",
                       "text_tokens", "media_tokens", "media_encoded",
                       "output_tokens", "output_len", "prefix_tokens", "prefix_len", "priority",
                       "chain_id", "gap_ms", "text_prompt_tokens"}, "workload.request")
        request_id = str(raw.get("request_id", raw.get("id", f"r{index}")))
        text_prompt = _int(
            raw.get("text_prompt_tokens",
                    raw.get("text_tokens", raw.get("prompt_tokens", raw.get("prompt_len")))),
            "workload.prompt_tokens_invalid", minimum=1,
        )
        media_tokens = _int(raw.get("media_tokens", 0), "workload.media_tokens_invalid", minimum=0)
        media_encoded = bool(raw.get("media_encoded", False))
        # Encoding status may skip vision work, but media positions remain part
        # of the sequence budget and cache.
        prompt = text_prompt + media_tokens
        output = _int(raw.get("output_tokens", raw.get("output_len")), "workload.output_tokens_invalid", minimum=1)
        prefix = _int(raw.get("prefix_tokens", raw.get("prefix_len", 0)), "workload.prefix_tokens_invalid")
        if prefix > prompt:
            raise ModelingError("workload.prefix_exceeds_prompt")
        return cls(
            request_id=request_id,
            arrival_ms=_positive_float(raw.get("arrival_ms", 0.0), "workload.arrival_invalid", allow_zero=True),
            prompt_tokens=prompt, output_tokens=output, prefix_tokens=prefix,
            priority=_int(raw.get("priority", 0), "workload.priority_invalid", minimum=0),
            text_prompt_tokens=text_prompt, media_tokens=media_tokens,
            media_encoded=media_encoded,
            chain_id=_int(raw.get("chain_id", 0), "workload.chain_id_invalid", minimum=0),
            gap_ms=_positive_float(raw.get("gap_ms", 0.0), "workload.gap_ms_invalid", allow_zero=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id, "arrival_ms": self.arrival_ms,
            "prompt_tokens": self.prompt_tokens, "output_tokens": self.output_tokens,
            "prefix_tokens": self.prefix_tokens, "priority": self.priority,
            # from_mapping computes prompt = text + media; emitting both a total and a
            # text count would double media on a round trip, so the canonical text
            # value is the non-media remainder.
            "text_prompt_tokens": self.prompt_tokens - self.media_tokens,
            "media_tokens": self.media_tokens,
            "media_encoded": self.media_encoded,
            "chain_id": self.chain_id,
            "gap_ms": self.gap_ms,
        }


@dataclass(frozen=True)
class WorkloadSpec:
    requests: tuple[RequestSpec, ...]
    synthetic: bool = False
    metadata: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mapping(cls, value: Any) -> "WorkloadSpec":
        raw = _mapping(value, "workload_invalid")
        _unknown(raw, {"requests", "num_requests", "prompt_tokens", "prompt_len", "output_tokens",
                       "output_len", "arrival_interval_ms", "prefix_tokens", "synthetic"}, "workload")
        if raw.get("requests") is not None:
            items = raw["requests"]
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or not items:
                raise ModelingError("workload.requests_invalid")
            requests = tuple(RequestSpec.from_mapping(item, i) for i, item in enumerate(items))
            synthetic = bool(raw.get("synthetic", False))
        else:
            count = _int(raw.get("num_requests"), "workload.num_requests_required", minimum=1)
            prompts = _lengths(raw.get("prompt_tokens", raw.get("prompt_len")), count, "workload.prompt_tokens_invalid")
            outputs = _lengths(raw.get("output_tokens", raw.get("output_len")), count, "workload.output_tokens_invalid")
            prefixes = _lengths(raw.get("prefix_tokens", 0), count, "workload.prefix_tokens_invalid", minimum=0)
            interval = _positive_float(raw.get("arrival_interval_ms", 0.0), "workload.arrival_interval_invalid", allow_zero=True)
            requests = tuple(
                RequestSpec(f"r{i}", i * interval, prompts[i], outputs[i], prefixes[i])
                for i in range(count)
            )
            synthetic = bool(raw.get("synthetic", True))
        ids = [request.request_id for request in requests]
        if len(ids) != len(set(ids)):
            raise ModelingError("workload.request_id_duplicate")
        return cls(requests=requests, synthetic=synthetic, metadata=_metadata(raw))

    def validate_model(self, model: ModelSpec, runtime: "RuntimeSpec") -> None:
        for request in self.requests:
            total = request.prompt_tokens + request.output_tokens
            if total > runtime.max_model_len:
                raise ModelingError(
                    f"request_exceeds_max_model_len:{request.request_id}:{total}>{runtime.max_model_len}"
                )
            if request.prompt_tokens > runtime.max_model_len:
                raise ModelingError("request_prompt_exceeds_max_model_len")
            if not runtime.enable_chunked_prefill and request.prompt_tokens > runtime.max_num_batched_tokens:
                raise ModelingError("prompt_exceeds_batch_budget_without_chunking")

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": [request.to_dict() for request in self.requests],
            "num_requests": len(self.requests),
            "synthetic": self.synthetic,
            "metadata": dict(self.metadata),
        }


def _lengths(value: Any, count: int, reason: str, *, minimum: int = 1) -> list[int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != count:
            raise ModelingError(f"{reason}:length")
        return [_int(item, reason, minimum=minimum) for item in value]
    return [_int(value, reason, minimum=minimum)] * count



__all__ = ["LayerPlan", "MlaSpec", "ModelSpec", "SloSpec", "RequestSpec", "WorkloadSpec"]
