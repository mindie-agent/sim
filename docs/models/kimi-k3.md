# kimi-k3 — model reference card

## Identity and provenance

- Catalog key: `kimi-k3`.
- Source config: `models/kimi-k3/config.json` in the read-only se-tools checkout at `ae8779fdd5a602bd3a4358bca93b2161d134a47b`.
- Archived HF attribution: `moonshotai/Kimi-K3` @ `f831ab66814297da540d832a5235f8e904f29d06`.
- Provenance is not a runtime admission gate; structurally valid newer configs of the same families are accepted.

## Structure and representation

- Body layers: **93** (native text body; optional draft/MTP is outside this count).
- Residual width: **7168**; vocab: 163840; attention families: kda=69, mla=24.
- FFN families: dense=1, moe=92; routed experts: 896, top-k: 16, shared: 2.
- Parameter count (logical, all resident body/vision/Engram tensors): **2779.93B**; packed body bytes: **1560.9 GB** logical across ranks.
- Example placement used for reference: TP16 DP1 EP16 PP3, 8 ranks/node; per-rank PP stage bytes: 36.4GB, 36.3GB, 36.5GB.
- KDA 69 + MLA24 using native one-based lists; final layer93 is zero-based92 MLA. KDA recurrent state=96*128*128 FP32 per request plus a bounded 3-row BF16 conv ring.
- Latent MoE: router on7168, dispatch/combine3584, experts [3584,3072], shared branch on7168 with 2 experts of I6144, output norm then up3584->7168.
- MXFP4 group32 applies to routed expert matrices only; attention/shared/dense/LM head/vision/projector stay BF16. Residual blocks use attn_res_block_size12.
- Vision: 27 blocks, H1024, I4096, packed QKV4608 (per-Q1536), head_dim128, patch14, 2x2 merge, patchmergerv2 to7168. External DSpark draft is optional and separate.
- Optional DSpark: layers=5, block=7, markov=256, experts/topK=0/0. Explicit `runtime.metadata.draft_mode` only.

## Quantization and cache/state representation

- `QuantSpec`: mxfp4_routed_only; weight format bf16, activation bf16, expert format mxfp4, block (0, 0), scale fp8_e8m0, KV bf16, index bf16.
- Packed FP4 uses rows*ceil(cols/2) data plus E8M0 rows*ceil(cols/32); FP8 blocks include their declared scale shape/dtype. Source and runtime formats are separate; unquantized exclusions never downgrade native FP32 categories.
- KV/cache storage is counted per owner before consumer reads; per-request state is an endpoint, not one copy per token. Per-rank placement uses TP/EP/ETP and node-local Engram sharding.

## Native execution profile boundaries

- Vision projector is proj0 -> GELU -> proj2 -> RMSNorm; vision blocks are GELU RMSNorm. A separate A3 7168x7168 rotation is a conditional backend preparation, not a universal checkpoint tensor.

## Optimizer and asynchronous-preparation assumptions

- Ops carry `semantic_family`, `weight_format`, `activation_format`, `compute_format`, `quant_category`, packed weight bytes, and owner/cache keys. Fusion must consume those records rather than infer from names.
- Shared long-KV/index producers are stored once; every consumer retains its own read. Cross-stage consumers require an explicit transfer. Grouped output keeps the group axis; MLA low-rank q_a/kv_a are replicated while q_b/kv_b output heads are sharded once.
- Prepare-eligible work is exposed through `mindie_sim.a3_catalog_prepare.catalog_prepare_candidates` with known-at/lifetime/owner/consumer fields. Model-load layout/scale/ring-capacity work can be lifted; request-token Engram ID metadata and supplied-grid vision position tables are request-arrival work. Future tokens/QKV/routing/acceptance are never marked early. Missing CPU/H2D rates are measurement gaps.

## Reproduce

```bash
python -m mindie_sim predict --input examples/model-reference/cases/kimi-k3.json
python -m mindie_sim should  --input examples/model-reference/cases/kimi-k3.json
python -m mindie_sim search  --input examples/model-reference/cases/kimi-k3.json --search-spec examples/model-reference/cases/kimi-k3.search.json
```

The case uses the full archived model dimensions and a declared A3 topology; it does not shrink the architecture. Request lengths are representative and can be changed without changing the model card.

## Scope

- Structural/analytic implementation coverage, not measured hardware performance. No A3/NPU execution or heldout data was used in this reference.
- Known boundary: GLM shared-index owner remains derived/conservative; raw image/video resize/crop rules require processor data; host CPU/H2D service rates are explicit inputs. Missing A3 calibration yields analytic estimates plus measurement gaps.

