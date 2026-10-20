# qwen3.8-2.4t-a95b — model reference card

## Identity and provenance

- Catalog key: `qwen3.8-2.4t-a95b`.
- Source config: `models/qwen3.8-2.4t-a95b/config.json` in the read-only se-tools checkout at `ae8779fdd5a602bd3a4358bca93b2161d134a47b`.
- Archived HF attribution: `Qwen/Qwen3.8-2.4T-A95B` @ `207bd685a7e3696cfaff12ded7c6a7ea0f88c996`.
- Provenance is not a runtime admission gate; structurally valid newer configs of the same families are accepted.

## Structure and representation

- Body layers: **92** (native text body; optional draft/MTP is outside this count).
- Residual width: **8192**; vocab: 248320; attention families: gdn=69, gated_gqa=23.
- FFN families: moe=92; routed experts: 512, top-k: 10, shared: 0.
- Parameter count (logical, all resident body/vision/Engram tensors): **2419.80B**; packed body bytes: **4839.6 GB** logical across ranks.
- Example placement used for reference: TP32 DP1 EP32 PP4, 8 ranks/node; per-rank PP stage bytes: 38.1GB, 38.0GB, 38.0GB, 38.1GB.
- GDN/gated-GQA hybrid: GDN key heads16/dim128 and independent value heads/dim; full attention uses gated Q projection, partial rotary0.25, o_proj on Q width.
- TP8 local gated-QKV width is 4608 (Q4096+K256+V256); GDN packed qkvz local4608, ba32, conv2560, recurrent state per rank 16*128*128 FP32. All92 layers MoE (E512/topK10/I2048/sharedI2048+gate).
- Optional MTP: layers=1; plain forward excludes it. Draft/verify work is selected only by explicit `runtime.metadata.draft_mode`.

## Quantization and cache/state representation

- `QuantSpec`: bf16_qwen; weight format bf16, activation bf16, expert format bf16, block (0, 0), scale fp32, KV bf16, index bf16.
- Packed FP4 uses rows*ceil(cols/2) data plus E8M0 rows*ceil(cols/32); FP8 blocks include their declared scale shape/dtype. Source and runtime formats are separate; unquantized exclusions never downgrade native FP32 categories.
- KV/cache storage is counted per owner before consumer reads; per-request state is an endpoint, not one copy per token. Per-rank placement uses TP/EP/ETP and node-local Engram sharding.

## Native execution profile boundaries

- Qwen vision uses LayerNorm/GELU and merger LayerNorm -> [4608,4608] -> GELU -> [5120,4608], plus the block MLP path. Media positions expand in RequestSpec before scheduling; the tower runs once on the first request chunk.

## Optimizer and asynchronous-preparation assumptions

- Ops carry `semantic_family`, `weight_format`, `activation_format`, `compute_format`, `quant_category`, packed weight bytes, and owner/cache keys. Fusion must consume those records rather than infer from names.
- Shared long-KV/index producers are stored once; every consumer retains its own read. Cross-stage consumers require an explicit transfer. Grouped output keeps the group axis; MLA low-rank q_a/kv_a are replicated while q_b/kv_b output heads are sharded once.
- Prepare-eligible work is exposed through `mindie_sim.a3_catalog_prepare.catalog_prepare_candidates` with known-at/lifetime/owner/consumer fields. Model-load layout/scale/ring-capacity work can be lifted; request-token Engram ID metadata and supplied-grid vision position tables are request-arrival work. Future tokens/QKV/routing/acceptance are never marked early. Missing CPU/H2D rates are measurement gaps.

## Reproduce

```bash
python -m mindie_sim predict --input examples/model-reference/cases/qwen3.8-2.4t-a95b.json
python -m mindie_sim should  --input examples/model-reference/cases/qwen3.8-2.4t-a95b.json
python -m mindie_sim search  --input examples/model-reference/cases/qwen3.8-2.4t-a95b.json --search-spec examples/model-reference/cases/qwen3.8-2.4t-a95b.search.json
```

The case uses the full archived model dimensions and a declared A3 topology; it does not shrink the architecture. Request lengths are representative and can be changed without changing the model card.

## Scope

- Structural/analytic implementation coverage, not measured hardware performance. No A3/NPU execution or heldout data was used in this reference.
- Known boundary: GLM shared-index owner remains derived/conservative; raw image/video resize/crop rules require processor data; host CPU/H2D service rates are explicit inputs. Missing A3 calibration yields analytic estimates plus measurement gaps.

