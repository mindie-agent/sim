# deepseek-v4-pro-0813 — model reference card

## Identity and provenance

- Catalog key: `deepseek-v4-pro-0813`.
- Source config: `models/deepseek-v4-pro-0813/config.json` in the read-only se-tools checkout at `ae8779fdd5a602bd3a4358bca93b2161d134a47b`.
- Archived HF attribution: `deepseek-ai/DeepSeek-V4-Pro-0813` @ `72e1d3230f6c080a530b0a1d46f8eb4602340597`.
- Provenance is not a runtime admission gate; structurally valid newer configs of the same families are accepted.

## Structure and representation

- Body layers: **61** (native text body; optional draft/MTP is outside this count).
- Residual width: **7168**; vocab: 129280; attention families: dsa_compressed_128=31, dsa_compressed_4=30.
- FFN families: moe=61; routed experts: 384, top-k: 6, shared: 1.
- Parameter count (logical, all resident body/vision/Engram tensors): **1573.00B**; packed body bytes: **850.7 GB** logical across ranks.
- Example placement used for reference: TP16 DP1 EP16 PP1, 8 ranks/node; per-rank PP stage bytes: 56.5GB.
- DeepSeek V4 DSA: local SWA128 on every layer; C4 and C128 compressed branches are distinct. C4 owns its index plane; C128 has no C4 indexer.
- mHC: hc_mult=4, sinkhorn=20; grouped output keeps o_groups=16, o_rank=1024.
- Body schedule: 30 C4 + 31 C128. Total long rows at S=127/128/129 are 930/991/991.
- `num_hash_layers=3` is represented as a hash-router stage; router weight terms stay explicit.
- Optional MTP: layers=3; plain forward excludes it. Draft/verify work is selected only by explicit `runtime.metadata.draft_mode`.

## Quantization and cache/state representation

- `QuantSpec`: fp8_dynamic+mxfp4_experts; weight format fp8_e4m3, activation fp8_e4m3, expert format mxfp4, block (128, 128), scale fp8_e8m0, KV bf16, index int8.
- Packed FP4 uses rows*ceil(cols/2) data plus E8M0 rows*ceil(cols/32); FP8 blocks include their declared scale shape/dtype. Source and runtime formats are separate; unquantized exclusions never downgrade native FP32 categories.
- KV/cache storage is counted per owner before consumer reads; per-request state is an endpoint, not one copy per token. Per-rank placement uses TP/EP/ETP and node-local Engram sharding.

## Native execution profile boundaries

- One native MTP/DSpark module is selected (target E/topK, block5); MTP and DSpark names are not two additive resident modes.

## Optimizer and asynchronous-preparation assumptions

- Ops carry `semantic_family`, `weight_format`, `activation_format`, `compute_format`, `quant_category`, packed weight bytes, and owner/cache keys. Fusion must consume those records rather than infer from names.
- Shared long-KV/index producers are stored once; every consumer retains its own read. Cross-stage consumers require an explicit transfer. Grouped output keeps the group axis; MLA low-rank q_a/kv_a are replicated while q_b/kv_b output heads are sharded once.
- Prepare-eligible work is exposed through `mindie_sim.a3_catalog_prepare.catalog_prepare_candidates` with known-at/lifetime/owner/consumer fields. Model-load layout/scale/ring-capacity work can be lifted; request-token Engram ID metadata and supplied-grid vision position tables are request-arrival work. Future tokens/QKV/routing/acceptance are never marked early. Missing CPU/H2D rates are measurement gaps.

## Reproduce

```bash
python -m mindie_sim predict --input examples/model-reference/cases/deepseek-v4-pro-0813.json
python -m mindie_sim should  --input examples/model-reference/cases/deepseek-v4-pro-0813.json
python -m mindie_sim search  --input examples/model-reference/cases/deepseek-v4-pro-0813.json --search-spec examples/model-reference/cases/deepseek-v4-pro-0813.search.json
```

The case uses the full archived model dimensions and a declared A3 topology; it does not shrink the architecture. Request lengths are representative and can be changed without changing the model card.

## Scope

- Structural/analytic implementation coverage, not measured hardware performance. No A3/NPU execution or heldout data was used in this reference.
- Known boundary: GLM shared-index owner remains derived/conservative; raw image/video resize/crop rules require processor data; host CPU/H2D service rates are explicit inputs. Missing A3 calibration yields analytic estimates plus measurement gaps.
