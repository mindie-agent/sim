"""Draft matrix and shared-owner residency terms.

[Role] Derive optional draft parameters, packing and placement.
[Invariants] Shared aliases count as runtime residency but not checkpoint bytes.
[Timeouts & Locks] Bounded by draft layers, no I/O.
[Side-Effect Policy] Pure arithmetic.
"""
from dataclasses import replace
from mindie_sim.catalog.types import DraftSpec, QuantSpec
from mindie_sim.catalog.draft import draft_quant
from mindie_sim.catalog.weight_terms import WeightTerm, _ceil_div, _t, term_bytes
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.parallel import ParallelSpec

def _draft_attention_terms(draft: DraftSpec, quant: QuantSpec) -> list[WeightTerm]:
    h = draft.hidden_size
    terms: list[WeightTerm] = []
    if draft.input_mode == "dual_normalized":
        # 4a headers (V4 Hc MTP only): q_norm alias q_a_layernorm F32[1024], kv_norm alias
        # kv_a_layernorm F32[512]; gate BF16[8,4096]+F32 bias[8]; enorm/hnorm
        # BF16[4096] each.  All replicated.
        if draft.q_lora_rank:
            terms.append(_t("draft_q_norm", draft.q_lora_rank, 1, "norm_f32"))
        if draft.kv_lora_rank:
            terms.append(_t("draft_kv_norm", draft.kv_lora_rank, 1, "norm_f32"))
        terms.append(_t("draft_enorm", h, 1, "linear_param_bf16"))
        terms.append(_t("draft_hnorm", h, 1, "linear_param_bf16"))
    if draft.attention_kind == "mla":
        qk = draft.qk_nope_head_dim + draft.qk_rope_head_dim
        # latent_mla owns a kv_b projection; latent_mqa uses wkv directly and
        # has head_dim == kv_lora_rank == v_head_dim (V4/V4.1 DSA).
        latent_mla = draft.mla_variant == "latent_mla"
        if latent_mla:
            terms.extend([
                _t("draft_q_a", draft.q_lora_rank, h, "mla"),
                _t("draft_q_b", draft.num_heads * qk, draft.q_lora_rank,
                   "mla", "out_tp"),
                _t("draft_wkv", draft.kv_lora_rank, h, "mla"),
                _t("draft_kv_b", draft.num_heads * (draft.qk_nope_head_dim + draft.v_head_dim),
                   draft.kv_lora_rank, "mla", "out_tp"),
            ])
            if draft.o_groups:
                group_width = draft.num_heads * draft.head_dim // draft.o_groups
                terms.extend([
                    _t("draft_wo_a", draft.o_groups * draft.o_lora_rank,
                       group_width, "mla", "group_out", groups=draft.o_groups),
                    _t("draft_wo_b", h, draft.o_groups * draft.o_lora_rank,
                       "mla", "group_in", groups=draft.o_groups),
                ])
            else:
                terms.append(_t("draft_o", h, draft.num_heads * draft.head_dim, "mla", "in_tp"))
        else:
            terms.extend([
                _t("draft_q_a", draft.q_lora_rank, h, "mla"),
                _t("draft_q_b", draft.num_heads * qk, draft.q_lora_rank,
                   "mla", "out_tp"),
                _t("draft_wkv", draft.kv_lora_rank, h, "mla"),
            ])
            if draft.o_groups:
                group_width = draft.num_heads * draft.head_dim // draft.o_groups
                terms.extend([
                    _t("draft_wo_a", draft.o_groups * draft.o_lora_rank,
                       group_width, "mla", "group_out", groups=draft.o_groups),
                    _t("draft_wo_b", h, draft.o_groups * draft.o_lora_rank,
                       "mla", "group_in", groups=draft.o_groups),
                ])
            else:
                terms.append(_t("draft_o", h, draft.num_heads * draft.head_dim, "mla", "in_tp"))
    else:
        terms.extend([
            _t("draft_q", draft.q_width, h, "attn.qkv", "out_tp"),
            _t("draft_k", draft.kv_width, h, "attn.qkv", "kv_out",
               units=draft.num_kv_heads, unit_width=draft.head_dim),
            _t("draft_v", draft.kv_width, h, "attn.qkv", "kv_out",
               units=draft.num_kv_heads, unit_width=draft.head_dim),
            _t("draft_o", h, draft.num_heads * draft.head_dim, "attn.o", "in_tp"),
        ])
        if draft.qk_norm:
            terms.extend([_t("draft_q_norm", draft.head_dim, 1, "norm"),
                          _t("draft_k_norm", draft.head_dim, 1, "norm")])
    terms.extend([_t("draft_input_norm", h, 1, "norm"),
                  _t("draft_post_norm", h, 1, "norm"),
                  _t("draft_final_norm", h, 1, "norm")])
    return terms


def _draft_ffn_terms(draft: DraftSpec, model: ModelSpec) -> list[WeightTerm]:
    h = draft.hidden_size
    terms: list[WeightTerm] = []
    if draft.num_experts > 0:
        terms.extend([
            _t("draft_router", draft.num_experts, h, "router"),
            _t("draft_expert_gate", draft.expert_intermediate_size, h,
               "dspark_routed_expert", "expert_gate_up", count=draft.num_experts),
            _t("draft_expert_up", draft.expert_intermediate_size, h,
               "dspark_routed_expert", "expert_gate_up", count=draft.num_experts),
            _t("draft_expert_down", h, draft.expert_intermediate_size,
               "dspark_routed_expert", "expert_down", count=draft.num_experts),
        ])
        if draft.num_shared_experts:
            shared_i = draft.expert_intermediate_size * draft.num_shared_experts
            terms.extend([
                _t("draft_shared_gate", shared_i, h, "shared_expert", "dense_up"),
                _t("draft_shared_up", shared_i, h, "shared_expert", "dense_up"),
                _t("draft_shared_down", h, shared_i, "shared_expert", "dense_down"),
            ])
    else:
        i = draft.intermediate_size
        terms.extend([
            _t("draft_gate", i, h, "dense_mlp", "dense_up"),
            _t("draft_up", i, h, "dense_mlp", "dense_up"),
            _t("draft_down", h, i, "dense_mlp", "dense_down"),
        ])
    if draft.mhc_mult:
        mix = (2 + draft.mhc_mult) * draft.mhc_mult
        hc_dim = draft.mhc_mult * h
        for part in ("attn", "ffn"):
            terms.append(_t(f"draft_hc_{part}_fn", mix, hc_dim, "mhc"))
            terms.append(_t(f"draft_hc_{part}_base", mix, 1, "param_fp32"))
            terms.append(_t(f"draft_hc_{part}_scale", 3, 1, "param_fp32"))
    return terms


def draft_terms(
    draft: DraftSpec, model: ModelSpec, *, standalone_heads_only: bool = False,
) -> list[WeightTerm]:
    h = draft.hidden_size
    terms: list[WeightTerm] = []
    if standalone_heads_only and draft.kind == "standalone_draft":
        # The standalone draft always consumes concatenated target features.
        terms.append(_t("draft_input_proj", h, draft.input_width,
                        draft.input_projection_category, "out_tp"))
        terms.extend([_t("draft_input_norm", h, 1, "norm"),
                      _t("draft_final_norm", h, 1, "norm")])
        if draft.target_shared_embedding:
            terms.append(WeightTerm(
                "target_embedding_alias", draft.vocab_size, h,
                "target_alias_embedding", "replicated", 1, 0, 0, 0, True,
                False,
            ))
        if draft.target_shared_lm_head:
            terms.append(WeightTerm(
                "target_lm_head_alias", draft.vocab_size, h,
                "target_alias_lm_head", "replicated", 1, 0, 0, 0, True,
                False,
            ))
    if not standalone_heads_only:
        if draft.input_mode == "dual_normalized":
            # Deployed V4 MTP input: enorm(decoder h) + hnorm(target h), then two
            # replicated BF16 HxH projections, then add.  Not a DSpark
            # concatenated fp8 input projection.
            terms.append(_t("draft_e_proj", h, h, "linear_param_bf16",
                            "replicated"))
            terms.append(_t("draft_h_proj", h, h, "linear_param_bf16",
                            "replicated"))
        else:
            terms.append(_t("draft_input_proj", h, draft.input_width,
                            draft.input_projection_category, "out_tp"))
        for _ in range(max(0, draft.layers)):
            terms.extend(_draft_attention_terms(draft, model.quant or QuantSpec("bf16")))
            terms.extend(_draft_ffn_terms(draft, model))
    if draft.kind == "mtp" and draft.target_shared_lm_head:
        terms.append(WeightTerm(
            "target_lm_head_alias", draft.vocab_size, h,
            "target_alias_lm_head", "replicated", 1, 0, 0, 0, True, False,
        ))
    if draft.markov_rank:
        terms.append(_t("markov_embed", draft.vocab_size, draft.markov_rank,
                        "embedding"))
        terms.append(_t("markov_head", draft.vocab_size, draft.markov_rank,
                        "lm_head"))
        if draft.confidence_enabled and draft.confidence_executes:
            width = draft.confidence_input_width or (h + draft.markov_rank)
            terms.append(_t("confidence_head", 1, width, "norm"))
    return [replace(term, quant=draft_quant(model, draft)) for term in terms]


def draft_weight_summary(
    model: ModelSpec, *, standalone_heads_only: bool = False,
) -> tuple[float, float]:
    if model.auxiliary is None:
        return 0.0, 0.0
    quant = model.quant or QuantSpec("bf16")
    parallel = ParallelSpec()
    drafts: list[DraftSpec] = []
    aux = model.auxiliary
    if aux.standalone_draft and aux.dspark is not None:
        # Body layers are counted separately; only extra heads/projection here.
        drafts = [aux.dspark]
        standalone_heads_only = True
    else:
        if aux.mtp is not None:
            drafts.append(aux.mtp)
        if aux.dspark is not None and aux.dspark is not aux.mtp:
            drafts.append(aux.dspark)
    params = 0.0
    weights = 0.0
    for draft in drafts:
        for term in draft_terms(draft, model, standalone_heads_only=standalone_heads_only):
            if not term.source_owned:
                continue
            params += term.logical_parameters
            weights += term_bytes(term, quant, parallel, 1)
    return params, weights
