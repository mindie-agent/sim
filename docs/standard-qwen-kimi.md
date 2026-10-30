# Standard Qwen and Kimi text adapters

This document covers the ordinary dense/MoE Qwen and Kimi text bodies added by
`mindie_sim/catalog/qwen.py` and `mindie_sim/catalog/kimi.py`. The hybrid
Qwen3.5/GDN and Kimi K3/KDA/latent-MoE/DSpark paths are unchanged and remain
selected by their own native identities.

The adapters produce the existing `LayerPlan`/`AuxiliarySpec` records. They do
not add an execution engine, timing table, model-name performance branch or
measurement-admission rule. Root catalog dispatch and public `Case` assembly are
owned separately; this change only exposes the structural adapter APIs.

## APIs

```python
_standard_qwen_plans(c) -> list[LayerPlan]
_aux_standard_qwen(c) -> AuxiliarySpec
_standard_qwen_gaps(c) -> tuple[str, ...]

_standard_kimi_plans(c) -> list[LayerPlan]
_aux_standard_kimi(c) -> AuxiliarySpec
_standard_kimi_gaps(c) -> tuple[str, ...]
```

`c` is the native Hugging Face configuration mapping. The Kimi adapter accepts
the top-level VL config and flattens `text_config` itself. The root may instead
pass an already flattened mapping retaining `vision_config`.

## Standard Qwen semantics

Supported `model_type` values: `qwen2`, `qwen2_moe`, `qwen3`, `qwen3_moe`.
Qwen2.5 checkpoints use `qwen2`.

* **Attention geometry.** Qwen2/Qwen3 use an explicit `head_dim` when present,
  otherwise `hidden_size // num_attention_heads`, which must be integral.
  Qwen2-MoE follows its reference source and derives the head width; an explicit
  differing `head_dim` is reported as ignored rather than silently applied.
  `Qwen3-30B-A3B` is the discriminating case: `hidden_size=2048`,
  `num_attention_heads=32`, explicit `head_dim=128`, so the Q projection is 4096
  wide, not 2048.
* **QK norm.** Qwen3 dense and MoE set `qk_norm=True`; Qwen2/Qwen2.5/Qwen2-MoE
  do not. This is a layer-plan field, so `layer_weight_terms` charges Q/K norm
  weights only on Qwen3.
* **Bias gaps.** Qwen2/Qwen2.5 attention hard-codes Q/K/V bias and uses an
  unbiased output projection. Qwen2-MoE uses `qkv_bias` (source default true).
  `LayerPlan` has no bias fields, so these paths emit `qkv_bias_unmodeled`.
  Qwen3 uses `attention_bias` for Q/K/V and the output projection; when true the
  adapter emits `attention_bias_unmodeled` and does not claim completeness.
* **MoE schedule.** A layer is sparse exactly when
  `index not in mlp_only_layers`
  and `num_experts > 0`
  and `(index + 1) % decoder_sparse_step == 0`.
  Dense-only overrides are retained even when the one-based step would otherwise
  select them. Standard Qwen routers are softmax with `norm_topk_prob` preserved. The router
  Linear keeps the model precision; the reference FP32 softmax is not a reason
  to promote router weights or add an FP32 input cast.
* **Shared experts.** Qwen2-MoE always constructs one shared MLP of
  `shared_expert_intermediate_size` plus a sigmoid shared-expert gate. Qwen3-MoE
  has no shared branch in the reference source; it remains zero unless the config
  explicitly declares a shared width/count, and its gate is false unless
  explicitly declared.
* **Full attention without `layer_types`.** A standard Qwen2/Qwen3 config needs no
  `layer_types` or linear-attention fields. Every layer is full GQA.
  `kv_cache_elements` is `2 * num_key_value_heads * head_dim`.
* **Ordinary layer schedules.** An explicit `layer_types` sequence is honored:
  `full_attention` has window zero; `sliding_attention` keeps the declared
  `sliding_window` width. A length mismatch is a visible gap: existing entries remain
  authoritative and missing entries retain only their known projections/FFNs. An unknown custom kind becomes a layer plan with
  `attention="projection_only"` and an
  `unsupported_attention_layer:<index>:<kind>` gap; it
  is never coerced to full attention. The layer count and known FFN structure
  remain present, so one custom layer does not erase the rest of the body.
  No GQA cache is invented for an unknown core. If QKV dimensions are missing,
  `attention="unmodeled"` keeps FFN/norm work
  without inventing projection dimensions. The original architecture tag remains.
* **Window placement.** Without explicit `layer_types`, the v4.51.3 FlashAttention
  condition is reproduced:
  `use_sliding_window and sliding_window is not None and
  layer_idx >= max_window_layers`. The declared window is preserved only on those
  layers. This is the source's top-layer condition, not a generic "bottom layers
  are sliding" assumption. A missing `max_window_layers`, missing window value,
  or non-positive window becomes a gap; affected cores remain omitted rather than
  being reinterpreted as full attention.
* **Activation.** `hidden_act` is preserved on each plan. `silu` is supported;
  another value is retained and reported as `hidden_act_unsupported:<name>`,
  rather than rewritten to `silu`.
* **Optional MTP.** Standard Qwen2/3 sources have no supported draft schema.
  Auxiliary work stays empty; either `mtp_num_hidden_layers` or
  `num_nextn_predict_layers` is reported as `optional_mtp_schema_not_modeled`
  when positive. A count alone cannot prove a draft architecture.

## Standard Kimi semantics

Supported top-level/merged `model_type` values: `kimi_k2`, `kimi_vl`.

* **Kimi-K2.** `model_type=kimi_k2`, architecture `DeepseekV3ForCausalLM`,
  `first_k_dense_replace=1`, `moe_layer_freq=1`, 61 layers, 384 routed experts,
  one shared expert, `q_lora_rank=1536`. Layer 0 is dense; layers 1-60 are MoE.
  The adapter carries no latent-expert width/norm, mHC, KDA or DSpark fields.
  The declared FP8 `quantization_config` is left to the root generic quantization
  path, not converted into a structural gap or a model-specific branch.
* **Kimi-VL.** The top-level `model_type=kimi_vl` and `vision_config` are
  retained while `text_config` is flattened. The text body has a missing
  `model_type`, `q_lora_rank=null`, 27 layers, 64 routed experts, two shared
  experts, and ordinary DeepSeek MLA dimensions. Layer 0 is dense; layers 1-26
  are MoE. Plans are runnable without claiming the vision tower.
* **Shared dependency.** Kimi plans, auxiliary and gaps delegate directly to
  `mindie_sim.catalog.deepseek._standard_deepseek_plans`,
  `_aux_standard_deepseek` and `_standard_deepseek_gaps`. This lane does not
  duplicate MLA math or provide a dynamic fallback. The shared dependency is
  mandatory; its absence remains an import error.
* **Auxiliary.** Standard K2/VL default to an empty `AuxiliarySpec`. No K3 DSpark
  is inherited. A native DeepSeek aux is returned only when that shared helper
  declares it; a declared-but-unmodeled native next-token module remains in the
  gap tuple.
* **Vision gap.** Standard VL always includes `vision_tower_unmodeled` and
  `vision_projector_unmodeled`. Text plans remain valid; the adapter does not
  invent projector widths or media work.
* **Unsupported hybrid fields.** If a standard Kimi config unexpectedly contains
  KDA/linear-attention, latent-expert or DSpark fields, the gap tuple names the
  presence instead of silently dropping them.

## Gap names

| Gap | Meaning |
| --- | --- |
| `qkv_bias_unmodeled` | Qwen2/Qwen2.5/Qwen2-MoE Q/K/V biases have no `LayerPlan` field. |
| `attention_bias_unmodeled` | Qwen3 Q/K/V and output-projection biases are declared but unmodeled. |
| `head_dim_ignored_by_qwen2_moe_source:<declared>!=<derived>` | Qwen2-MoE source derives `hidden/heads`; explicit config value is not used. |
| `layer_types_length_mismatch:<n>!=<layers>` | Known schedule entries retained; unspecified attention cores omitted. |
| `layer_types_invalid` | `layer_types` is not an iterable sequence. |
| `unsupported_attention_layer:<index>:<kind>` | Unknown custom core; known projection/FFN work remains explicit. |
| `sliding_window_enabled_without_window` | `use_sliding_window` true but no window declared. |
| `sliding_window_max_window_layers_missing` | Source condition needs `max_window_layers`; no window invented. |
| `sliding_layer_type_without_declared_window:<index>` | Explicit sliding kind lacks a declared width. |
| `sliding_window_invalid:<value>` | Non-positive declared window. |
| `hidden_act_unsupported:<name>` | Activation preserved but not supported by the standard model path. |
| `q_output_gate_unsupported_for_standard_qwen` | Standard Qwen body does not have the hybrid output gate. |
| `optional_mtp_schema_not_modeled` | Next-token layer count declared without a modeled standard MTP schema. |
| `vision_tower_unmodeled` | Kimi-VL vision encoder is outside this text adapter. |
| `vision_projector_unmodeled` | Kimi-VL projector is outside this text adapter. |
| `kda_fields_present_in_standard_kimi_config` | A standard Kimi config unexpectedly carries hybrid KDA fields. |
| `latent_expert_fields_present_in_standard_kimi_config` | Unexpected latent-MoE fields are visible, not silently absorbed. |
| `dspark_fields_present_in_standard_kimi_config` | Unexpected DSpark fields are visible, not turned into K3 aux. |

Shared DeepSeek gaps (for example `router_noaux_tc_correction_bias_not_in_layer_plan`,
`router_noaux_tc_group_score_top2_not_in_layer_plan` and
`router_routed_scaling_factor_not_in_layer_plan`) are preserved unchanged.

## Official fixtures and provenance

The seven exact `config.json` files are in
`tests/fixtures/standard-qwen-kimi/`. `manifest.json` records the repository,
40-character revision, resolved URL, SHA-256 and byte count without any local
`/tmp` paths. `source-manifest.json` records the inspected source revisions and
hashes.

| Fixture | Repository | Revision | Config SHA-256 |
| --- | --- | --- | --- |
| `qwen2-7b.json` | `Qwen/Qwen2-7B` | `453ed1575b739b5b03ce3758b23befdb0967f40e` | `d318b49c4a29771e259d93053084eb752f5bfe69c20c4f2e041bd1017884f810` |
| `qwen2.5-7b-instruct.json` | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | `7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c` |
| `qwen2-57b-a14b.json` | `Qwen/Qwen2-57B-A14B` | `f29d6a182178ddf39f68f403361779b2d37fdce8` | `96bc7f0331f186f96e25ad939b6791cec71a46cce101726d7333e4ec66ded621` |
| `qwen3-8b.json` | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | `f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30` |
| `qwen3-30b-a3b.json` | `Qwen/Qwen3-30B-A3B` | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | `2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297` |
| `kimi-k2-instruct.json` | `moonshotai/Kimi-K2-Instruct` | `fd1984e2b7a3350dbf7305fe73a4ede25c14de50` | `8c13ae1049df55f29b3bdcae69a562433f243ff70dac251d819ecad8dbdf7439` |
| `kimi-vl-a3b-instruct.json` | `moonshotai/Kimi-VL-A3B-Instruct` | `398eede0903cd983a2bfa0cc634e9ac1d843f375` | `af3810295de6b08232776b74110ed482fa0183f4aeaec586ff4f7593910951cb` |

The Qwen modeling/configuration sources are pinned to transformers
`v4.51.3`. The exact URLs and SHA-256 values are in `source-manifest.json`.
Relevant source lines:

* Qwen2: `modeling_qwen2.py:135-150` (explicit/derived `head_dim`, hard-coded
  QKV bias), `:176-182` (window activation).
* Qwen2-MoE: `modeling_qwen2_moe.py:277-306` (derived `head_dim`,
  `qkv_bias`), `:580-627` (one shared expert plus sigmoid gate),
  `:649-650` (one-based sparse step).
* Qwen3: `modeling_qwen3.py:170-203` (explicit `head_dim`, Q/K norm,
  `attention_bias`, window activation).
* Qwen3-MoE: `modeling_qwen3_moe.py:144-163` (explicit `head_dim`, Q/K norm,
  bias), `:237-289` (softmax router), `:322-325` (sparse schedule).
* Kimi-K2: `modeling_deepseek.py:648-683` (MLA dimensions, q-LoRA/direct-Q,
  biases), `:1152-1158` (`first_k_dense_replace`/`moe_layer_freq` schedule).

Kimi-VL's actual `modeling_kimi_vl.py` and `configuration_kimi_vl.py` are pinned
in the source manifest. The source contains its own standard DeepSeek text model:
`:1235-1245` selects direct Q versus q-LoRA, `:1731-1737` selects dense/MoE, and
`:2441` constructs the text model. The vision tower and projector remain explicit
gaps even though the text body is supported.

## Verification boundary

`tests/test_standard_qwen_family.py` and `tests/test_standard_kimi_family.py`
check fixture hashes, layer counts, head geometry, QK-norm counts, bias gaps,
MoE schedules, shared widths/gates, window placement, and weight-term arithmetic
from `layer_weight_terms`. They do not assemble a fake public `ModelSpec`/case to
claim end-to-end registry coverage.

Independent review ran 32 family tests with the actual standard DeepSeek helper
and no dependency skips. Review corrected the router precision, unsupported
attention/cache contract and invented standard MTP branch, and verified direct-Q
weight terms for Kimi-VL.

`tests/test_standard_qwen_kimi_integration.py` adds 26 tests against the real
public HF-config boundary after core revision `6379a6fa`. Seven official fixtures
are reduced to two-layer synthetic geometry while retaining their structural
choices and declared quantization. They exercise 14 balanced prefill/decode graphs,
seven pairs of raw/wrapped HF-config serving predictions, four nondivisible-token
expected-routing graphs, and one partial custom-attention graph. Independent
checks cover QKV/direct-Q FLOPs, router precision, absent K3 mechanisms,
name-independent predictions, input interpretation, bounds and released resources.
These are software/geometry acceptance controls, not measurements of the original
checkpoints or claims about their timing accuracy.

The equal-group balanced routing domain requires equal positive expert rows.
The initial three-token/four-expert probe with top-2 assignments was outside that
domain (rows 2/2/1/1). Balanced checks use four tokens (rows 2/2/2/2); separate
three-token controls use the declared expected-routing path. No routing restriction
or physical work was weakened to make these controls pass.
