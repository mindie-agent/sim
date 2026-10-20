# kimi-k3-dspark — model reference card

## Identity and provenance

- Catalog key: `kimi-k3-dspark`.
- Source config: `models/kimi-k3-dspark/config.json` in the read-only se-tools checkout at `ae8779fdd5a602bd3a4358bca93b2161d134a47b`.
- Archived HF attribution: `RadixArk/Kimi-K3-DSpark` @ `3c5bac301d9cf392706189d82ed947feca6c2f0f`.
- Provenance is not a runtime admission gate; structurally valid newer configs of the same families are accepted.

## Structure and representation

- Body layers: **5** (native text body; optional draft/MTP is outside this count).
- Residual width: **7168**; vocab: 163840; attention families: gqa=5.
- FFN families: dense=5; routed experts: 0, top-k: 0, shared: 0.
- Parameter count (logical, all resident body/vision/Engram tensors): **2.25B**; packed body bytes: **4.5 GB** logical across ranks.
- Example placement used for reference: TP1 DP1 EP1 PP1, 8 ranks/node; per-rank PP stage bytes: 4.5GB.
- Standalone target-aware draft: 5 dense GQA layers, H7168, I14336, 64Q/16KV heads, head_dim64, Q4096/KV1024. Target features concatenate 5*7168=35840 ->7168.
- block_size7, Markov rank256, confidence input7168+256=7424. Plain forward executes the five body layers plus target-feature input and Markov/confidence/proposal heads; no fictitious token embedding or full LM head exists.
- Optional DSpark: layers=5, block=7, markov=256, experts/topK=0/0. Explicit `runtime.metadata.draft_mode` only.

## Quantization and cache/state representation

- `QuantSpec`: bf16_dspark; weight format bf16, activation bf16, expert format bf16, block (0, 0), scale fp32, KV bf16, index bf16.
- Packed FP4 uses rows*ceil(cols/2) data plus E8M0 rows*ceil(cols/32); FP8 blocks include their declared scale shape/dtype. Source and runtime formats are separate; unquantized exclusions never downgrade native FP32 categories.
- KV/cache storage is counted per owner before consumer reads; per-request state is an endpoint, not one copy per token. Per-rank placement uses TP/EP/ETP and node-local Engram sharding.

## Native execution profile boundaries

- Native target-shared embed_tokens/LM head aliases are accounted once as shared owners. They are external owner bytes (4,697,620,480B at BF16), not draft checkpoint bytes: body parameters/weights exclude them, and `external_shared_weight_bytes` reports them for union/dedup with target residency. The draft retains base-logit and target-embedding read work. Markov lookup/bias/add/sample is sequential, Markov weights are replicated, and the 7-token query block is noncausal (49 pairs) with target-context KV precompute. Confidence is declared geometry but unexecuted on this upstream profile; draft PP>1 is rejected.

## Optimizer and asynchronous-preparation assumptions

- Ops carry `semantic_family`, `weight_format`, `activation_format`, `compute_format`, `quant_category`, packed weight bytes, and owner/cache keys. Fusion must consume those records rather than infer from names.
- Shared long-KV/index producers are stored once; every consumer retains its own read. Cross-stage consumers require an explicit transfer. Grouped output keeps the group axis; MLA low-rank q_a/kv_a are replicated while q_b/kv_b output heads are sharded once.
- Prepare-eligible work is exposed through `mindie_sim.a3_catalog_prepare.catalog_prepare_candidates` with known-at/lifetime/owner/consumer fields. Model-load layout/scale/ring-capacity work can be lifted; request-token Engram ID metadata and supplied-grid vision position tables are request-arrival work. Future tokens/QKV/routing/acceptance are never marked early. Missing CPU/H2D rates are measurement gaps.

## Reproduce

```bash
python -m mindie_sim predict --input examples/model-reference/cases/kimi-k3-dspark.json
python -m mindie_sim should  --input examples/model-reference/cases/kimi-k3-dspark.json
python -m mindie_sim search  --input examples/model-reference/cases/kimi-k3-dspark.json --search-spec examples/model-reference/cases/kimi-k3-dspark.search.json
```

The case uses the full archived model dimensions and a declared A3 topology; it does not shrink the architecture. Request lengths are representative and can be changed without changing the model card.

## Scope

- Structural/analytic implementation coverage, not measured hardware performance. No A3/NPU execution or heldout data was used in this reference.
- Known boundary: GLM shared-index owner remains derived/conservative; raw image/video resize/crop rules require processor data; host CPU/H2D service rates are explicit inputs. Missing A3 calibration yields analytic estimates plus measurement gaps.

