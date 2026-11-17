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
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.parallel import ParallelSpec

ZERO = 0
ONE = 1
#: ``WeightTerm.groups``/``units``/``unit_width`` for a term with no grouping.

def _local_draft_experts(num_experts: int, ep: int) -> int:
    """Experts held on one EP rank.

    An exact split requires ``num_experts % ep == 0``.  A smaller expert set
    is also legal when ``ep % num_experts == 0`` (DeepSeek-V4.1-Flash DSpark
    with 16 experts under EP32/EP64): each rank holds
    ``max(1, num_experts // ep)`` and that expert is replicated across
    ``ep // num_experts`` ranks.
    """
    experts = int(num_experts)
    width = max(1, int(ep))
    if experts <= 0:
        return 0
    return max(ONE, (experts + width - ONE) // width)


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
            wkv_rows = draft.kv_lora_rank + (
                draft.qk_rope_head_dim if draft.qk_rope_head_dim > 0 else 0)
            o_cols = draft.num_heads * (draft.v_head_dim or draft.head_dim)
            terms.extend([
                _t("draft_q_a", draft.q_lora_rank, h, "mla"),
                _t("draft_q_b", draft.num_heads * qk, draft.q_lora_rank,
                   "mla", "out_tp"),
                _t("draft_wkv", wkv_rows, h, "mla"),
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
                terms.append(_t("draft_o", h, o_cols, "mla", "in_tp"))
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
        if draft.q_output_gate:
            terms.append(_t("draft_q_gate", draft.num_heads, h,
                            "attn.gate", "out_tp"))
        terms.extend([
            _t("draft_q", draft.q_width, h, "attn.qkv", "out_tp"),
            _t("draft_k", draft.kv_width, h, "attn.qkv", "kv_out",
               units=draft.num_kv_heads, unit_width=draft.head_dim),
            _t("draft_v", draft.kv_width, h, "attn.qkv", "kv_out",
               units=draft.num_kv_heads, unit_width=draft.head_dim),
            _t("draft_o", h, draft.num_heads * draft.head_dim, "attn.o", "in_tp"),
        ])
        if draft.qk_norm:
            terms.extend([_t("draft_q_norm", draft.head_dim, ONE, "norm"),
                          _t("draft_k_norm", draft.head_dim, ONE, "norm")])
    if draft.input_mode == "normalized_concat":
        # Each native NextN block owns its decoder/target normalisations and one
        # ``eh_proj[H, 2H]`` on the concatenated normalised pair.
        terms.extend([
            _t("draft_enorm", h, ONE, "linear_param_bf16"),
            _t("draft_hnorm", h, ONE, "linear_param_bf16"),
            _t("draft_eh_proj", h, 2 * h, "linear_param_bf16", "replicated"),
        ])
    if draft.mhc_lowrank <= ZERO:
        terms.extend([_t("draft_input_norm", h, ONE, "norm"),
                      _t("draft_post_norm", h, ONE, "norm"),
                      _t("draft_final_norm", h, ONE, "norm")])
    return terms


def _draft_ffn_terms(
    draft: DraftSpec, model: ModelSpec, *, ep: int = ONE,
) -> list[WeightTerm]:
    h = draft.hidden_size
    terms: list[WeightTerm] = []
    if draft.num_experts > 0:
        # Per-rank banks are max(1, num_experts // ep).  The term keeps the
        # global population so residency sharding applies EP once.  When
        # ep > num_experts and ep % num_experts == 0, each rank holds one
        # expert replicated across ep // num_experts ranks.
        _local_draft_experts(draft.num_experts, ep)
        terms.extend([
            _t("draft_router", draft.num_experts, h, "router"),
            _t("draft_expert_gate", draft.expert_intermediate_size, h,
               "dspark_routed_expert", "expert_gate_up", count=draft.num_experts),
            _t("draft_expert_up", draft.expert_intermediate_size, h,
               "dspark_routed_expert", "expert_gate_up", count=draft.num_experts),
            _t("draft_expert_down", h, draft.expert_intermediate_size,
               "dspark_routed_expert", "expert_down", count=draft.num_experts),
            # The draft MoE router inherits the target noaux_tc correction
            # bias, one FP32 parameter per expert per draft layer.
            *([_t("draft_router_correction_bias", draft.num_experts, ONE,
                  "param_fp32")]
              if any(plan.router_noaux_tc for plan in model.layer_plans) else []),
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
    if draft.mhc_lowrank > ZERO:
        from mindie_sim.catalog.qwen4_exp import lowrank_hc_draft_terms
        terms.extend(lowrank_hc_draft_terms(draft))
    elif draft.mhc_mult:
        mix = (2 + draft.mhc_mult) * draft.mhc_mult
        hc_dim = draft.mhc_mult * h
        for part in ("attn", "ffn"):
            terms.append(_t(f"draft_hc_{part}_fn", mix, hc_dim, "mhc"))
            terms.append(_t(f"draft_hc_{part}_base", mix, 1, "param_fp32"))
            terms.append(_t(f"draft_hc_{part}_scale", 3, 1, "param_fp32"))
    return terms


def draft_terms(
    draft: DraftSpec, model: ModelSpec, *, standalone_heads_only: bool = False,
    ep: int = ONE,
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
        if draft.input_mode == "projected_concat":
            # Native DSpark input: one ``main_proj[H, len(targets) * H]`` and one
            # ``main_norm[H]``.  The 3H-wide projection is a different operator
            # from two independent H x H projections, so it must not be modelled
            # as ``dual_normalized``.
            terms.append(_t("draft_main_proj", h, draft.input_width,
                            draft.input_projection_category, "out_tp"))
            terms.append(_t("draft_main_norm", h, ONE, "norm"))
        elif draft.input_mode == "dual_normalized":
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
        if draft.embedding_owned and int(draft.embedding_count) > ZERO:
            terms.append(WeightTerm(
                "draft_embedding", draft.vocab_size, h, "embedding",
                "embedding", int(draft.embedding_count), 0, 0, 0,
                False, True))
        for _ in range(max(ZERO, draft.layers)):
            terms.extend(_draft_attention_terms(draft, model.quant or QuantSpec("bf16")))
            terms.extend(_draft_ffn_terms(draft, model, ep=ep))
            if draft.output_head_owned:
                terms.append(_t("draft_output_norm", h, ONE, "norm"))
                if int(draft.output_head_count) > ZERO:
                    terms.append(WeightTerm(
                        "draft_output_head", draft.vocab_size, h, "lm_head",
                        "replicated", ONE, 0, 0, 0, False, True))
        if draft.mhc_lowrank > ZERO:
            from mindie_sim.catalog.qwen4_exp import lowrank_hc_draft_head_terms
            terms.extend(lowrank_hc_draft_head_terms(draft))
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
    from mindie_sim.catalog.projection_placement import projection_role
    return [replace(term, quant=draft_quant(model, draft),
                    placement_role=projection_role(
                        None, term.name, scope="draft"))
            for term in terms]


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
        for term in draft_terms(
            draft, model, standalone_heads_only=standalone_heads_only,
            ep=parallel.ep,
        ):
            if not term.source_owned:
                continue
            params += term.logical_parameters
            weights += term_bytes(term, quant, parallel, 1)
    return params, weights
