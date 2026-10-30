# Standard DeepSeek and GLM structural adapters

These adapters turn official Hugging Face checkpoint configs into the existing
`LayerPlan` vocabulary. They are pure structural readers: no model-name timing
branch, no latency multiplier and no fitted correction.

## DeepSeek V2 / V2-Lite / V3 / R1

`mindie_sim.catalog.deepseek` keeps the V4/V4.1 functions and adds:

- `_standard_deepseek_plans(c) -> list[LayerPlan]`
- `_aux_standard_deepseek(c) -> AuxiliarySpec | None`
- `_standard_deepseek_gaps(c) -> tuple[str, ...]`

The body is latent MLA only. `qk_nope_head_dim + qk_rope_head_dim` is the
attention width, `v_head_dim` is independent, and the cache row is
`kv_lora_rank + qk_rope_head_dim` for one latent KV head. A null
`q_lora_rank` is a real direct-Q checkpoint: the plan carries `0`, and the
direct projection is one hidden -> `heads * (nope + rope)` matrix.

The MoE condition is copied from the native schedule:

```
n_routed_experts is not None
and layer_idx >= first_k_dense_replace
and layer_idx % moe_layer_freq == 0
```

Dense layers use `intermediate_size`. MoE layers use `n_routed_experts`,
`num_experts_per_tok`, `moe_intermediate_size`, and a shared MLP of
`n_shared_experts * moe_intermediate_size`. Router input is the residual width,
mode is `scoring_func` (`softmax` or `sigmoid`), norm-top-k is read, and the
router dtype is FP32 as in the source.

V4/V4.1 compression, indexer, mHC and grouped output are not invented for
these standard checkpoints. Optional MTP is not added to the body.

Explicit gaps:

- `attention_bias_not_in_layer_plan`
- `router_noaux_tc_correction_bias_not_in_layer_plan`
- `router_noaux_tc_group_score_top2_not_in_layer_plan`
- `router_routed_scaling_factor_not_in_layer_plan`
- `optional_mtp_not_modeled`

V2/V2-Lite do not use `noaux_tc`; V3/R1 do. Their correction bias and routed
scaling are therefore reported as gaps rather than silently asserted complete.

## GLM-4 modern / legacy ChatGLM-4 / GLM-4.5 / Air

`mindie_sim.catalog.glm` keeps the GLM-5/Flash functions and adds the same
three-call protocol:

- `_standard_glm_plans(c)`
- `_aux_standard_glm(c)`
- `_standard_glm_gaps(c)`

All four checkpoints are GQA, not MLA. Modern GLM-4 reads
`num_hidden_layers/hidden_size/num_attention_heads/num_key_value_heads/head_dim`
and dense `intermediate_size`. Legacy ChatGLM reads the native aliases
`num_layers`, `kv_channels`, `multi_query_group_num` and `ffn_hidden_size`;
`rmsnorm=true` is RMSNorm. `padded_vocab_size` is a generic model-registry
field (a layer plan has no vocabulary dimension), not a body-geometry input.
Legacy `ChatGLMModel` also applies half-width RoPE: the pinned source constructs
`RotaryEmbedding(rotary_dim // 2)` at lines 839-844. Its missing explicit
`partial_rotary_factor` field does not imply full-width RoPE; plans retain the
half-width geometry and report the same current execution gap.

GLM-4.5/Air use the native MoE condition `layer_idx >= first_k_dense_replace`
with no assumed `moe_layer_freq`. They keep the shared-expert count and the
sigmoid router, and read `use_qk_norm` from the config. MTP layers stay outside
the body; when reliable auxiliary support is unavailable the adapter returns
`None` and reports `optional_mtp_not_modeled`.

Explicit gaps (degradable, never raised):

- `additional_post_self_attn_and_post_mlp_norms_not_in_layer_schema` for the
  four RMSNorm calls per modern GLM-4 layer
- `partial_rope_<rope>_of_<head>_not_in_attention_plan` for GLM-4/4.5 partial
  RoPE (64 of 128 in the official configs)
- `attention_bias_qkv_not_in_layer_plan` / `linear_bias_not_in_layer_plan`
- `router_sigmoid_correction_bias_and_routed_scaling_not_in_layer_plan`
- legacy `rmsnorm=false` and residual-post-LayerNorm gaps

Quantization is intentionally not selected here. The generic catalog/registry
and quantization assembly remain outside this adapter; standard BF16
checkpoints and FP8 expert checkpoints are represented by the same structural
plans and resolved by the caller.

## Direct-Q MLA execution

`mindie_sim.prediction.mla` now distinguishes `q_lora_rank == 0`. It creates a
single `q_proj` matrix from hidden to `heads * (nope + rope)` and no zero-sized
`q_a`, `q_norm` or `q_b`. The low-rank path is unchanged. If a caller selects
an explicit source prolog that cannot represent direct Q, lowering raises
`mla_prolog_v3_direct_q_unsupported`; it never fabricates a query rank.

`mindie_sim.catalog.weights` mirrors that distinction: direct-Q layers own one
real `q_proj` weight term and no `q_a_norm`; low-rank layers keep
`q_a/q_a_norm/q_b`.

## Fixtures

`tests/fixtures/hf-standard-families/` contains the eight official
`config.json` files plus `manifest.json` (repository, commit revision,
resolved URL, SHA-256 and source manifest) and `source-manifest.json`
(modeling source URL, revision, size and SHA-256). `tests/test_standard_model_families.py`
checks those digests, then verifies schedules, cache row arithmetic, weight
terms, gaps, direct-Q graph costs/liveness and the unchanged low-rank path.

Independent acceptance on the DSH commit also ran the actual V2-Lite first
layer at TP=1/2/4 for prefill=3 and decode=1, context=8. All six executions
preserved a single direct query matrix with local output width `3072 / TP`,
weight parameters `2048 * 3072`, and FLOPs `2 * query_tokens * 2048 *
(3072 / TP)`. Each query waited for the declared 7 ms input readiness; all
executions released tasks, programs and buffers. These controls used declared
synthetic device rates and are geometry/execution evidence, not measured
checkpoint performance.

Adapter and MLA-selection checks passed 22 tests. The broader native-fusion,
layout and structure selection passed 45 tests and skipped 10 unavailable
profiler-dependent checks; two unrelated HC tests failed because the base
revision imported the concurrently developed `family_hc` module before that
module was committed. The parent integration owns that dependency and must
include it before its final package verification.
