# deepseek-v4.1-flash — model reference card

## Identity and provenance

- Catalog key: `deepseek-v4.1-flash`.
- Source config: `models/deepseek-v4.1-flash/config.json` in the read-only se-tools checkout at `ae8779fdd5a602bd3a4358bca93b2161d134a47b`.
- Archived HF attribution: `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a40aa45a94ad051997016db3960a90277`.
- Provenance is not a runtime admission gate; structurally valid newer configs of the same families are accepted.

## Structure and representation

- Body layers: **40** (native text body; optional draft/MTP is outside this count).
- Residual width: **5120**; vocab: 129280; attention families: swa_only=2, dsa_shared_compressed_2=18, dsa_shared_compressed_1=20.
- FFN families: moe=40; routed experts: 384, top-k: 6, shared: 1.
- Parameter count (logical, all resident body/vision/Engram tensors): **748.98B**; packed body bytes: **693.2 GB** logical across ranks.
- Example placement used for reference: TP16 DP1 EP16 PP4, 8 ranks/node; per-rank PP stage bytes: 56.1GB, 55.0GB, 54.7GB, 54.7GB.
- V4.1 owners: long-KV/index-K `[2,8,14,20]`; TopK producers `[2,8,14,20,24,28,32,36]`; candidate source 20. Layer26 consumes KV20/TopK24/candidate20; storage is shared, consumer reads are not.
- C2 parity is floor-based across chunks; ring capacity32 rows is separate from valid `min(S,32)` rows. Index-K creation consumes unrotated latent before in-place long-KV RoPE/store.
- Engram at layers1/14: 24 lookups/table/token, 48 across tables; device gather and optional CPU/H2D staging are distinct paths.
- Vision: 32 blocks, H1024, I2816, packed QKV3072, head_dim64, patch14, 3x3 merge to 5120. Decode does not re-run the tower.
- Optional MTP: layers=3; plain forward excludes it. Draft/verify work is selected only by explicit `runtime.metadata.draft_mode`.

## Quantization and cache/state representation

- `QuantSpec`: fp8_dynamic+mxfp4_experts; weight format fp8_e4m3, activation fp8_e4m3, expert format mxfp4, block (32, 32), scale fp8_e8m0, KV bf16, index int8.
- Packed FP4 uses rows*ceil(cols/2) data plus E8M0 rows*ceil(cols/32); FP8 blocks include their declared scale shape/dtype. Source and runtime formats are separate; unquantized exclusions never downgrade native FP32 categories.
- KV/cache storage is counted per owner before consumer reads; per-request state is an endpoint, not one copy per token. Per-rank placement uses TP/EP/ETP and node-local Engram sharding.

## Native execution profile boundaries

- Native MTP/DSpark is one selected execution mode (3 layers, E128/topK3), not two additive modes. Base logits use the target-shared LM head; Markov lookup/bias/add/sample is sequential per proposal; PP>1 draft execution is rejected for the inspected backend.
- Compressor runtime precision: ratio2 wkv/wgate are FP32 runtime copies of BF16 checkpoint tensors (10,485,760B runtime vs 5,242,880B source); ratio1 wkv stays BF16. Device Engram tables default to BF16 512B rows; source host FP8/E8M0 264B rows and INT8+FP32 288B device rows remain explicit modes.

## Optimizer and asynchronous-preparation assumptions

- Ops carry `semantic_family`, `weight_format`, `activation_format`, `compute_format`, `quant_category`, packed weight bytes, and owner/cache keys. Fusion must consume those records rather than infer from names.
- Shared long-KV/index producers are stored once; every consumer retains its own read. Cross-stage consumers require an explicit transfer. Grouped output keeps the group axis; MLA low-rank q_a/kv_a are replicated while q_b/kv_b output heads are sharded once.
- Prepare-eligible work is exposed through `mindie_sim.a3_catalog_prepare.catalog_prepare_candidates` with known-at/lifetime/owner/consumer fields. Model-load layout/scale/ring-capacity work can be lifted; request-token Engram ID metadata and supplied-grid vision position tables are request-arrival work. Future tokens/QKV/routing/acceptance are never marked early. Missing CPU/H2D rates are measurement gaps.

## Reproduce

```bash
python -m mindie_sim predict --input examples/model-reference/cases/deepseek-v4.1-flash.json
python -m mindie_sim should  --input examples/model-reference/cases/deepseek-v4.1-flash.json
python -m mindie_sim search  --input examples/model-reference/cases/deepseek-v4.1-flash.json --search-spec examples/model-reference/cases/deepseek-v4.1-flash.search.json
```

The case uses the full archived model dimensions and a declared A3 topology; it does not shrink the architecture. Request lengths are representative and can be changed without changing the model card.

## Scope

- Structural/analytic implementation coverage, not measured hardware performance. No A3/NPU execution or heldout data was used in this reference.
- Known boundary: GLM shared-index owner remains derived/conservative; raw image/video resize/crop rules require processor data; host CPU/H2D service rates are explicit inputs. Missing A3 calibration yields analytic estimates plus measurement gaps.

