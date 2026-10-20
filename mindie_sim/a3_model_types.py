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
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .a3_types import (
    A3InputError, _DTYPE_BYTES, _int, _mapping, _metadata, _normalize_dtype,
    _positive_float, _unknown, dtype_bytes,
)
from .a3_catalog_types import (
    AuxiliarySpec, DraftSpec, EngramSpec, QuantSpec, VisionSpec,
)


@dataclass(frozen=True)
class MlaSpec:
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    @classmethod
    def from_mapping(cls, value: Any) -> "MlaSpec":
        raw = _mapping(value, "model.mla_invalid")
        allowed = {"q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"}
        _unknown(raw, allowed, "model.mla")
        return cls(
            q_lora_rank=_int(raw.get("q_lora_rank", 0), "model.mla.q_lora_rank_invalid", minimum=1),
            kv_lora_rank=_int(raw.get("kv_lora_rank", 0), "model.mla.kv_lora_rank_invalid", minimum=1),
            qk_nope_head_dim=_int(raw.get("qk_nope_head_dim", 0), "model.mla.qk_nope_invalid", minimum=1),
            qk_rope_head_dim=_int(raw.get("qk_rope_head_dim", 0), "model.mla.qk_rope_invalid", minimum=0),
            v_head_dim=_int(raw.get("v_head_dim", 0), "model.mla.v_head_dim_invalid", minimum=1),
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
    router_input_size: int = 0
    router_mode: str = "softmax"
    router_norm_topk: bool = False
    router_groups: int = 0
    router_group_topk: int = 0
    shared_intermediate_size: int = 0
    shared_expert_gate: bool = False
    linear_key_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_heads: int = 0
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
    resident_weight_bytes_override: float | None = None
    kv_bytes_per_token_override: float | None = None
    state_bytes_per_request: float = 0.0
    catalog_facts: tuple[tuple[str, Any], ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()
    parameter_count_override: float | None = None
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

    @classmethod
    def from_mapping(cls, value: Any) -> "ModelSpec":
        raw = _mapping(value, "model_invalid")
        allowed = {
            "name", "layers", "hidden_size", "num_heads", "num_attention_heads",
            "head_dim", "num_kv_heads", "num_key_value_heads", "intermediate_size",
            "ffn_intermediate_size", "vocab_size", "attention", "ffn", "architecture",
            "activation_dtype", "weight_dtype", "kv_dtype", "compute_dtype", "quantization",
            "num_experts", "n_routed_experts", "top_k", "moe_intermediate_size",
            "num_shared_experts", "expert_routing", "expert_routing_histogram",
            "expert_routing_distribution", "mla", "tie_word_embeddings",
            "catalog_key", "model_catalog_key", "preset",
            "n_layers", "num_hidden_layers", "d_model", "hidden_dim", "n_heads",
            "n_kv_heads", "d_head", "ffn_dim", "n_experts", "num_experts_per_tok",
            "moe_inter_dim", "dtype", "attention_type", "ffn_type",
            "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
            "v_head_dim", "rope_theta", "max_position_embeddings", "rms_norm_eps",
            "hidden_act", "bos_token_id", "eos_token_id", "torch_dtype",
        }
        _unknown(raw, allowed, "model")
        catalog_key = raw.get("catalog_key", raw.get("model_catalog_key", raw.get("preset")))
        if catalog_key:
            from .model_catalog import load_model

            return load_model(str(catalog_key))
        if "layers" not in raw:
            for alias in ("n_layers", "num_hidden_layers"):
                if alias in raw:
                    raw = {**raw, "layers": raw[alias]}
                    break
        if "hidden_size" not in raw:
            for alias in ("d_model", "hidden_dim"):
                if alias in raw:
                    raw = {**raw, "hidden_size": raw[alias]}
                    break
        if "num_heads" not in raw and "num_attention_heads" not in raw:
            if raw.get("n_heads") is not None:
                raw = {**raw, "num_heads": raw["n_heads"]}
        if "num_kv_heads" not in raw and "num_key_value_heads" not in raw:
            if raw.get("n_kv_heads") is not None:
                raw = {**raw, "num_kv_heads": raw["n_kv_heads"]}
        if "head_dim" not in raw and raw.get("d_head") is not None:
            raw = {**raw, "head_dim": raw["d_head"]}
        if "intermediate_size" not in raw and raw.get("ffn_dim") is not None:
            raw = {**raw, "intermediate_size": raw["ffn_dim"]}
        if "num_experts" not in raw and raw.get("n_experts") is not None:
            raw = {**raw, "num_experts": raw["n_experts"]}
        if "top_k" not in raw and raw.get("num_experts_per_tok") is not None:
            raw = {**raw, "top_k": raw["num_experts_per_tok"]}
        if "moe_intermediate_size" not in raw and raw.get("moe_inter_dim") is not None:
            raw = {**raw, "moe_intermediate_size": raw["moe_inter_dim"]}
        if "activation_dtype" not in raw and raw.get("dtype") is not None:
            raw = {**raw, "activation_dtype": raw["dtype"]}
        if "attention" not in raw and raw.get("attention_type") is not None:
            raw = {**raw, "attention": raw["attention_type"]}
        if "ffn" not in raw and raw.get("ffn_type") is not None:
            raw = {**raw, "ffn": raw["ffn_type"]}
        architecture = str(raw.get("architecture", "")).strip().lower()
        attention = str(raw.get("attention", "")).strip().lower()
        ffn = str(raw.get("ffn", "")).strip().lower()
        if architecture in {"dense", "gqa", "mla", "moe"}:
            if architecture == "mla":
                attention = "mla"
            elif architecture == "moe":
                ffn = "moe"
            elif architecture in {"dense", "gqa"}:
                attention = attention or "gqa"
        attention = attention or "gqa"
        ffn = ffn or ("moe" if raw.get("num_experts") else "dense")
        if attention not in {"gqa", "mla"}:
            raise A3InputError(f"model.attention_unsupported:{attention}")
        if ffn not in {"dense", "moe"}:
            raise A3InputError(f"model.ffn_unsupported:{ffn}")
        hidden = _int(raw.get("hidden_size"), "model.hidden_size_required", minimum=1)
        heads = _int(raw.get("num_heads", raw.get("num_attention_heads")), "model.num_heads_required", minimum=1)
        head_dim = _int(raw.get("head_dim", hidden // heads if hidden % heads == 0 else 0), "model.head_dim_required", minimum=1)
        # hidden_size is the residual/model width; Q/O projection width may be
        # num_heads*head_dim and is intentionally allowed to differ.
        num_kv_heads = _int(raw.get("num_kv_heads", raw.get("num_key_value_heads", heads)), "model.num_kv_heads_invalid", minimum=1)
        if attention == "gqa" and heads % num_kv_heads != 0:
            raise A3InputError(f"model.gqa_group_not_integral:{heads}%{num_kv_heads}")
        mla_raw = raw.get("mla")
        if mla_raw is None and any(
            key in raw for key in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "v_head_dim")
        ):
            mla_raw = {key: raw[key] for key in (
                "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim",
                "qk_rope_head_dim", "v_head_dim",
            ) if key in raw}
        mla = MlaSpec.from_mapping(mla_raw) if mla_raw is not None else None
        if attention == "mla" and mla is None:
            raise A3InputError("model.mla_required")
        quantization_raw = raw.get("quantization", "none")
        if isinstance(quantization_raw, Mapping):
            quantization = str(quantization_raw.get("weight", quantization_raw.get("name", "none"))).strip().lower()
        else:
            quantization = str(quantization_raw).strip().lower()
        weight_dtype = _normalize_dtype(raw.get("weight_dtype", _quant_weight_dtype(quantization)), "model.weight_dtype_invalid")
        activation_dtype = _normalize_dtype(raw.get("activation_dtype", _quant_activation_dtype(quantization)), "model.activation_dtype_invalid")
        kv_dtype = _normalize_dtype(raw.get("kv_dtype", "bf16"), "model.kv_dtype_invalid")
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
                raise A3InputError("model.moe_configuration_invalid")
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
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            metadata=_metadata(raw),
        )

    @property
    def attention_label(self) -> str:
        return self.attention

    def local_heads(self, tp: int) -> int:
        return max(1, self.num_heads // tp)

    def local_kv_heads(self, tp: int) -> int:
        # KV heads are replicated when TP exceeds the KV-head count; never
        # truncate the per-rank cache/weight count below one.
        return max(1, (self.num_kv_heads + tp - 1) // tp)

    def layers_per_stage(self, pp: int) -> int:
        if self.layers % pp != 0:
            raise A3InputError(f"layers_not_divisible_by_pp:{self.layers}%{pp}")
        return self.layers // pp

    def weight_bytes_per_layer(self, parallel: ParallelSpec) -> float:
        if self.layer_plans:
            # Per-layer catalog weights are supplied by the explicit graph
            # catalog; PP splits the layer count and TP/EP shards are applied
            # there.  This method is a fallback proportional view.
            if self.resident_weight_bytes_override is not None:
                return self.resident_weight_bytes_override / max(1, self.layers)
        wb = dtype_bytes(self.weight_dtype)
        h, tp = self.hidden_size, parallel.tp
        local_kv = self.local_kv_heads(tp)
        if self.attention == "gqa":
            q = h * self.num_heads * self.head_dim / tp
            kv = 2 * h * local_kv * self.head_dim
            o = self.num_heads * self.head_dim * h / tp
        else:
            assert self.mla is not None
            mla = self.mla
            q = h * mla.q_lora_rank / tp + mla.q_lora_rank * self.num_heads * mla.qk_head_dim / tp
            kv = h * (mla.kv_lora_rank + mla.qk_rope_head_dim) / tp
            kv += mla.kv_lora_rank * self.num_heads * (mla.qk_nope_head_dim + mla.v_head_dim) / tp
            o = self.num_heads * mla.v_head_dim * h / tp
        if self.ffn == "dense":
            ffn = 3 * h * self.intermediate_size / tp
        else:
            ffn = self.num_experts * 3 * h * self.moe_intermediate_size / parallel.ep
            ffn = ffn / max(1, parallel.etp)
            ffn += self.num_shared_experts * 3 * h * self.moe_intermediate_size / tp
        return float((q + kv + o + ffn) * wb)

    def weight_bytes(self, parallel: ParallelSpec) -> float:
        if self.layer_plans:
            from .model_catalog import sharded_weight_bytes

            return sharded_weight_bytes(self, parallel)
        if self.resident_weight_bytes_override is not None:
            return float(self.resident_weight_bytes_override)
        wb = dtype_bytes(self.weight_dtype)
        embedding = self.vocab_size * self.hidden_size * wb / parallel.tp
        head = 0.0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size * wb / parallel.tp
        per_stage = self.layers_per_stage(parallel.pp) * self.weight_bytes_per_layer(parallel)
        return float(embedding + head + per_stage)

    def kv_bytes_per_token_per_layer(self) -> float:
        if self.kv_bytes_per_token_override is not None:
            return float(self.kv_bytes_per_token_override) / max(1, self.layers)
        if self.attention == "gqa":
            return 2.0 * self.num_kv_heads * self.head_dim * dtype_bytes(self.kv_dtype)
        assert self.mla is not None
        return (self.mla.kv_lora_rank + self.mla.qk_rope_head_dim) * dtype_bytes(self.kv_dtype)

    def parameter_count(self) -> float:
        if self.parameter_count_override is not None:
            return float(self.parameter_count_override)
        if self.resident_weight_bytes_override is not None:
            return float(self.resident_weight_bytes_override) / max(1e-12, dtype_bytes(self.weight_dtype))
        h = self.hidden_size
        if self.attention == "gqa":
            attn = h * self.num_heads * self.head_dim + 2 * h * self.num_kv_heads * self.head_dim + self.num_heads * self.head_dim * h
        else:
            assert self.mla is not None
            m = self.mla
            attn = h * m.q_lora_rank + m.q_lora_rank * self.num_heads * m.qk_head_dim
            attn += h * (m.kv_lora_rank + m.qk_rope_head_dim) + m.kv_lora_rank * self.num_heads * (m.qk_nope_head_dim + m.v_head_dim)
            attn += self.num_heads * m.v_head_dim * h
        if self.ffn == "dense":
            ffn = 3 * h * self.intermediate_size
        else:
            ffn = self.num_experts * 3 * h * self.moe_intermediate_size + self.num_shared_experts * 3 * h * self.moe_intermediate_size
        emb = 2 * self.vocab_size * h
        return float(self.layers * (attn + ffn) + emb)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "architecture": f"{self.attention}+{self.ffn}",
            "layers": self.layers, "hidden_size": self.hidden_size,
            "num_heads": self.num_heads, "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim, "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "activation_dtype": self.activation_dtype, "weight_dtype": self.weight_dtype,
            "norm_eps": self.norm_eps,
            "kv_dtype": self.kv_dtype, "compute_dtype": self.compute_dtype,
            "quantization": self.quantization,
            "num_experts": self.num_experts, "top_k": self.top_k,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_shared_experts": self.num_shared_experts,
            "expert_routing_source": self.expert_routing_source,
            "expert_routing": list(self.expert_routing),
            "mla": self.mla.to_dict() if self.mla is not None else None,
            "parameter_count": self.parameter_count(),
            "layer_plans": [plan.to_dict() for plan in self.layer_plans],
            "state_bytes_per_request": self.state_bytes_per_request,
            "catalog_facts": dict(self.catalog_facts),
            "metadata": dict(self.metadata),
            "parameter_count_override": self.parameter_count_override,
            "quant": self.quant.to_dict() if self.quant is not None else None,
            "vision": self.vision.to_dict() if self.vision is not None else None,
            "engram": self.engram.to_dict() if self.engram is not None else None,
            "auxiliary": self.auxiliary.to_dict() if self.auxiliary is not None else None,
            "body_layers": self.body_layers,
        }


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
        raise A3InputError("model.expert_routing_invalid")
    if len(entries) != experts or any(not math.isfinite(v) or v < 0 for v in entries):
        raise A3InputError("model.expert_routing_invalid")
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
        if all(v is None for v in (result.ttft_ms, result.tpot_ms, result.itl_ms, result.e2e_ms)):
            raise A3InputError("slo_empty")
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

    @classmethod
    def from_mapping(cls, value: Any, index: int) -> "RequestSpec":
        raw = _mapping(value, f"workload.requests[{index}]_invalid")
        _unknown(raw, {"id", "request_id", "arrival_ms", "prompt_tokens", "prompt_len",
                       "text_tokens", "media_tokens", "media_encoded",
                       "output_tokens", "output_len", "prefix_tokens", "prefix_len", "priority"}, "workload.request")
        request_id = str(raw.get("request_id", raw.get("id", f"r{index}")))
        text_prompt = _int(
            raw.get("text_tokens", raw.get("prompt_tokens", raw.get("prompt_len"))),
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
            raise A3InputError("workload.prefix_exceeds_prompt")
        return cls(
            request_id=request_id,
            arrival_ms=_positive_float(raw.get("arrival_ms", 0.0), "workload.arrival_invalid", allow_zero=True),
            prompt_tokens=prompt, output_tokens=output, prefix_tokens=prefix,
            priority=_int(raw.get("priority", 0), "workload.priority_invalid", minimum=0),
            text_prompt_tokens=text_prompt, media_tokens=media_tokens,
            media_encoded=media_encoded,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id, "arrival_ms": self.arrival_ms,
            "prompt_tokens": self.prompt_tokens, "output_tokens": self.output_tokens,
            "prefix_tokens": self.prefix_tokens, "priority": self.priority,
            "text_prompt_tokens": self.text_prompt_tokens,
            "media_tokens": self.media_tokens,
            "media_encoded": self.media_encoded,
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
                raise A3InputError("workload.requests_invalid")
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
            raise A3InputError("workload.request_id_duplicate")
        return cls(requests=requests, synthetic=synthetic, metadata=_metadata(raw))

    def validate_model(self, model: ModelSpec, runtime: "RuntimeSpec") -> None:
        for request in self.requests:
            total = request.prompt_tokens + request.output_tokens
            if total > runtime.max_model_len:
                raise A3InputError(
                    f"request_exceeds_max_model_len:{request.request_id}:{total}>{runtime.max_model_len}"
                )
            if request.prompt_tokens > runtime.max_model_len:
                raise A3InputError("request_prompt_exceeds_max_model_len")
            if not runtime.enable_chunked_prefill and request.prompt_tokens > runtime.max_num_batched_tokens:
                raise A3InputError("prompt_exceeds_batch_budget_without_chunking")

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
            raise A3InputError(f"{reason}:length")
        return [_int(item, reason, minimum=minimum) for item in value]
    return [_int(value, reason, minimum=minimum)] * count



__all__ = ["LayerPlan", "MlaSpec", "ModelSpec", "SloSpec", "RequestSpec", "WorkloadSpec"]
