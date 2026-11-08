"""Finite canonical-subgraph vocabulary, builder registry and legality.

[Role]
  Register the 22 typed standard subgraph families, the per-kind builder
  functions supplied by ``instantiate``, an interning cache keyed by semantic
  structure, and the declared combination legality of the vocabulary.

[Invariants]
  * The vocabulary is finite and explicit; a missing mechanism is a coverage
    hole, never an unknown silent kind.
  * Interning keys contain kind/variant/operator structure and declared
    contracts, never model names, layer ids or request identities.
  * The cache is bounded (LRU) so a long-lived process cannot accumulate
    arbitrary template history.
  * A combination is rejected only for declared structural reasons (for
    example an indexer consumer without a producer), never because a model
    label or measurement is absent.

[Timeouts & Locks]
  Bounded in-memory dictionaries; no waits, I/O, locks or device probes.

[Side-Effect Policy]
  Registers builders in a module-level table and mutates only the caller-owned
  bounded cache; no filesystem or network access.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.subgraphs import (
    CoverageHole, OperatorTemplate, StatePortSpec, SubgraphKind,
    SubgraphTemplate, SUBGRAPH_KINDS, SUBGRAPH_KIND_SET, validate_kind,
)


def _kind(kind, title, variants, operators, inputs=(), outputs=(), coverage="structural",
          notes=""):
    return SubgraphKind(kind, title, tuple(variants), tuple(operators),
                        tuple(inputs), tuple(outputs), coverage, notes)


#: The complete canonical LLM vocabulary.  ``C``-like mechanisms that are only
#: partially covered keep an explicit ``partial`` label and a note.
VOCABULARY: tuple[SubgraphKind, ...] = (
    _kind("E0", "Input Embeddings (输入嵌入)", ("token", "media"),
          ("embedding", "input_copy", "media_splice"),
          outputs=("hidden",), notes="media splice requires V0 geometry"),
    _kind("N0", "Normalization and Residual Addition (归一化与残差相加)",
          ("rmsnorm", "layernorm", "residual", "qk_norm", "fused_add_rmsnorm"),
          ("rmsnorm", "layernorm", "residual_add", "qk_norm", "cast"),
          inputs=("hidden", "residual"), outputs=("hidden", "residual")),
    _kind("A0", "Multi-Head / Grouped-Query / Multi-Query Attention (多头／分组查询／多查询注意力)",
          ("gqa", "mha", "mqa", "shared_consumer"),
          ("qkv_projection", "qk_norm", "rotary", "kv_store", "kv_read",
           "attention", "output_gate", "o_projection", "tp_allreduce"),
          inputs=("hidden", "kv_cache"), outputs=("hidden", "kv_cache"),
          notes="shared_consumer omits KV projection and pays reads only"),
    _kind("A1", "Multi-head Latent Attention (MLA 多头潜在注意力)",
          ("latent_mla", "direct_q", "latent_mqa"),
          ("q_down", "q_up", "kv_down", "rotary", "mla_attention",
           "v_expand", "o_projection", "tp_allreduce"),
          inputs=("hidden", "kv_cache"), outputs=("hidden", "kv_cache")),
    _kind("A2", "Sliding-Window Attention with Compressed KV Attention (滑动窗口注意力与压缩 KV 注意力)",
          ("window_compressed", "window_only", "shared_consumer"),
          ("window_attention", "compressed_attention", "merge", "grouped_o",
           "output_gate"),
          inputs=("hidden", "compressed_kv", "index_selection"),
          outputs=("hidden",), coverage="partial",
          notes="compressed branch is distinct from A1 latent MLA"),
    _kind("A3", "Gated DeltaNet (GDN 门控 DeltaNet)", ("chunked_prefill", "recurrent_decode"),
          ("qkvz_projection", "ba_projection", "conv", "prepare",
           "gated_delta", "gated_norm", "o_projection"),
          inputs=("hidden", "recurrent_state", "conv_state"),
          outputs=("hidden", "recurrent_state", "conv_state"),
          coverage="partial", notes="implementation domain constraints stay explicit"),
    _kind("A4", "Kimi Delta Attention (KDA Kimi Delta 注意力)", ("chunked_prefill", "recurrent_decode"),
          ("qkv_projection", "gate_projection", "conv", "prepare",
           "kda_recurrent", "output_norm", "o_projection"),
          inputs=("hidden", "recurrent_state", "conv_state", "accepted_state"),
          outputs=("hidden", "recurrent_state", "conv_state"),
          coverage="partial", notes="accepted-state contract is explicit"),
    _kind("K0", "KV Cache Read, Write and Transfer (KV 缓存读写与传输)", ("store", "read", "transfer"),
          ("kv_store", "kv_read", "layout_convert", "stage_transfer"),
          inputs=("kv_cache",), outputs=("kv_cache", "kv_ready")),
    _kind("K1", "KV Cache Compression and Key Pooling (KV 缓存压缩与键池化)",
          ("weighted_pool", "gated_ring", "kpool"),
          ("compress", "kpool", "tail_store", "state_store"),
          inputs=("kv_cache", "compress_state"),
          outputs=("compressed_kv", "compress_state")),
    _kind("I0", "Attention Indexer Projections and Cache Update (注意力索引器投影与缓存更新)",
          ("c4", "full", "shared"),
          ("index_q_projection", "index_k_projection", "index_rope",
           "index_quant", "index_cache_store"),
          inputs=("hidden", "index_cache"), outputs=("index_cache", "index_ready")),
    _kind("I1", "Attention Index Scoring and Top-K Selection (注意力索引评分与 Top-K 选择)",
          ("topk", "pooled_tail", "candidate"),
          ("index_score", "topk", "candidate_filter", "cp_gather"),
          inputs=("index_cache", "index_selection"),
          outputs=("index_selection",), coverage="partial",
          notes="candidate reuse and CP gather stay separate contracts"),
    _kind("F0", "Dense Feed-Forward Network (FFN 稠密前馈网络)",
          ("silu", "gelu", "situ", "ungated"),
          ("gate_up", "activation", "down", "tp_allreduce"),
          inputs=("hidden",), outputs=("hidden",)),
    _kind("F1", "MoE Routing and Token Permutation (MoE 路由与 Token 重排)",
          ("softmax", "sigmoid", "hash", "grouped"),
          ("router_gemm", "topk", "histogram", "sort"),
          inputs=("hidden",), outputs=("routing_ledger", "expert_rows"),
          coverage="partial", notes="native router variants remain explicit gaps"),
    _kind("F2", "MoE Token Dispatch and Combine (MoE Token 分发与结果合并)",
          ("alltoall", "mc2", "allgather", "local"),
          ("dispatch", "combine", "quantize", "notify"),
          inputs=("hidden", "expert_rows"), outputs=("hidden", "peer_traffic"),
          coverage="partial"),
    _kind("F3", "Routed Expert FFN (路由专家前馈网络)", ("packed", "latent", "mxfp4"),
          ("expert_gate_up", "expert_activation", "expert_down"),
          inputs=("hidden", "expert_rows"), outputs=("hidden",),
          coverage="partial", notes="expected and integer execution domains stay split"),
    _kind("F4", "Shared Expert FFN and Output Merge (共享专家前馈网络与输出合并)",
          ("plain", "gated", "sp_gather"),
          ("shared_gate_up", "shared_activation", "shared_down", "join"),
          inputs=("hidden",), outputs=("hidden",)),
    _kind("R0", "Hyper-Connections (HC 超连接; mHC 流形约束超连接)", ("hc", "hc_arch35"),
          ("hc_projection", "inv_rms", "affine", "sinkhorn", "mix"),
          inputs=("hidden",), outputs=("hidden",)),
    _kind("R1", "Attention Residuals (AttnRes 注意力残差连接)", ("block_stack", "slot_mix"),
          ("slot_push", "score", "softmax_mix", "cast"),
          inputs=("hidden", "residual_slots"), outputs=("hidden", "residual_slots")),
    _kind("E1", "Engram / Per-Layer Embeddings (PLE Engram 条件记忆／逐层嵌入)",
          ("engram", "ple"),
          ("table_lookup", "gate", "projection"),
          inputs=("hidden", "lookup_tokens"), outputs=("hidden",),
          coverage="partial", notes="Engram and Gemma PLE are distinct contracts"),
    _kind("O0", "Final Normalization, LM Head and Sampling (最终归一化、语言模型输出头与采样)",
          ("all", "selected"),
          ("final_norm", "hidden_select", "lm_head", "logits_gather", "sampling"),
          inputs=("hidden",), outputs=("logits", "sampled_tokens")),
    _kind("D0", "Speculative Decoding: Draft and Verify (推测解码：起草与验证)",
          ("mtp", "dspark", "standalone"),
          ("draft_projection", "draft_heads", "target_verify", "accept_commit"),
          inputs=("hidden", "draft_state", "accept_state"),
          outputs=("hidden", "accept_state"), coverage="partial",
          notes="target verify is never discounted by acceptance"),
    _kind("V0", "Multimodal Encoder and Projector (多模态编码器与投影)",
          ("vision", "audio", "unified"),
          ("vision_encoder", "projector", "splice"),
          inputs=("media_tokens",), outputs=("hidden",), coverage="partial"),
)

VOCABULARY_BY_KIND = {entry.kind: entry for entry in VOCABULARY}

if tuple(VOCABULARY_BY_KIND) != SUBGRAPH_KINDS:
    raise RuntimeError("subgraph_vocabulary_order")


@dataclass(frozen=True)
class CombinationRule:
    kind: str
    requires: tuple[str, ...] = ()
    requires_any: tuple[str, ...] = ()
    reason: str = ""

    def validate(self, kinds: set[str]) -> None:
        if self.kind not in kinds:
            return
        missing = [item for item in self.requires if item not in kinds]
        if missing:
            raise ModelingError(
                f"subgraph_combination_invalid:{self.kind}:missing={sorted(missing)}")
        if self.requires_any and not (set(self.requires_any) & kinds):
            raise ModelingError(
                f"subgraph_combination_invalid:{self.kind}:requires_any="
                f"{sorted(self.requires_any)}")


#: Structural legality only.  It says nothing about which implementation is
#: available for a (kind, variant, hardware) triple.
COMBINATION_RULES: tuple[CombinationRule, ...] = (
    CombinationRule("I1", requires=("I0",),
                    reason="selection needs a produced index cache"),
    CombinationRule("K1", requires=("K0",),
                    reason="compression needs a declared KV cache"),
    CombinationRule("A2", requires_any=("K1", "K0"),
                    reason="compressed attention needs a cache producer"),
    CombinationRule("F4", requires_any=("F3", "F0"),
                    reason="shared branch joins a routed or dense branch"),
    CombinationRule("D0", requires_any=("A0", "A1", "A2"),
                    reason="draft/verify needs an attention macro"),
)


def validate_combination(kinds: Iterable[str]) -> tuple[str, ...]:
    ordered = tuple(dict.fromkeys(kinds))
    unknown = set(ordered) - SUBGRAPH_KIND_SET
    if unknown:
        raise ModelingError(f"subgraph_kind_unknown:{sorted(unknown)}")
    present = set(ordered)
    for rule in COMBINATION_RULES:
        rule.validate(present)
    return ordered


def vocabulary() -> tuple[SubgraphKind, ...]:
    return VOCABULARY


def kind_spec(kind: str) -> SubgraphKind:
    validate_kind(kind)
    return VOCABULARY_BY_KIND[kind]


def port(name: str, direction: str, representation: str, **kwargs) -> StatePortSpec:
    return StatePortSpec(name=name, direction=direction,
                         representation=representation, **kwargs)


def operator(name: str, family: str, role: str, **kwargs) -> OperatorTemplate:
    return OperatorTemplate(name=name, family=family, role=role, **kwargs)


def template(kind: str, variant: str, title: str,
             operators: Iterable[OperatorTemplate], **kwargs) -> SubgraphTemplate:
    return SubgraphTemplate(kind=kind, variant=variant, title=title,
                            operators=tuple(operators), **kwargs)


class TemplateRegistry:
    """Bounded LRU cache of immutable subgraph templates."""

    def __init__(self, max_entries: int = 256):
        if not isinstance(max_entries, int) or max_entries < 1:
            raise ModelingError("template_registry_capacity_invalid")
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple, SubgraphTemplate] = OrderedDict()
        self.evictions = 0

    def intern(self, value: SubgraphTemplate) -> SubgraphTemplate:
        if not isinstance(value, SubgraphTemplate):
            raise ModelingError("template_registry_value_invalid")
        key = value.key()
        existing = self._entries.get(key)
        if existing is not None:
            self._entries.move_to_end(key)
            return existing
        self._entries[key] = value
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1
        return value

    def get(self, key: tuple) -> SubgraphTemplate | None:
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()


KIND_BUILDERS: dict[str, Callable[..., SubgraphTemplate]] = {}


def register_builder(kind: str, builder: Callable[..., SubgraphTemplate]) -> None:
    validate_kind(kind)
    if not callable(builder):
        raise ModelingError(f"subgraph_builder_invalid:{kind}")
    if kind in KIND_BUILDERS:
        raise ModelingError(f"subgraph_builder_duplicate:{kind}")
    KIND_BUILDERS[kind] = builder


def build(kind: str, context: Any) -> Any:
    """Invoke the registered builder; it may return a template or a binding."""
    validate_kind(kind)
    builder = KIND_BUILDERS.get(kind)
    if builder is None:
        raise ModelingError(f"subgraph_builder_missing:{kind}")
    return builder(context)


def build_template(kind: str, context: Any) -> SubgraphTemplate:
    value = build(kind, context)
    template = getattr(value, "template", value)
    if not isinstance(template, SubgraphTemplate):
        raise ModelingError(f"subgraph_builder_return_invalid:{kind}")
    return template


def coverage_holes(kinds: Iterable[str], *, reason: str = "declared_partial") -> tuple[CoverageHole, ...]:
    holes = []
    for kind in kinds:
        spec = kind_spec(kind)
        if spec.coverage != "structural":
            holes.append(CoverageHole(kind, reason, spec.notes))
    return tuple(holes)
