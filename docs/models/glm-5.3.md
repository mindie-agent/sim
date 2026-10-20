# glm-5.3 — model reference card

## Identity and provenance

- Catalog key: `glm-5.3`.
- Source config: `models/glm-5.3/config.json` in the read-only se-tools checkout at `ae8779fdd5a602bd3a4358bca93b2161d134a47b`.
- Archived HF attribution: `zai-org/GLM-5.3` @ `187fb9fff6319062325ff825627ef6db084d9bc6`.
- Provenance is not a runtime admission gate; structurally valid newer configs of the same families are accepted.

## Structure and representation

- Body layers: **78** (native text body; optional draft/MTP is outside this count).
- Residual width: **6144**; vocab: 154880; attention families: mla_dsa=78.
- FFN families: dense=3, moe=75; routed experts: 256, top-k: 8, shared: 1.
- Parameter count (logical, all resident body/vision/Engram tensors): **743.38B**; packed body bytes: **745.6 GB** logical across ranks.
- Example placement used for reference: TP16 DP1 EP16 PP2, 8 ranks/node; per-rank PP stage bytes: 23.2GB, 25.0GB.
- MLA: q_rank2048, kv_rank512, qk_nope192, rope64, v_head256; body78, dense first3, MoE75, full indexers21, shared57, separate MTP1.
- Shared-index owner is a derived nearest-full candidate with explicit `derived_conservative_nearest_full` status; it is not asserted as a native cache alias. Shared layers still pay their own selection reads.

## Quantization and cache/state representation

- `QuantSpec`: fp8_dynamic; weight format fp8_e4m3, activation fp8_e4m3, expert format fp8_e4m3, block (128, 128), scale fp32, KV bf16, index bf16.
- Packed FP4 uses rows*ceil(cols/2) data plus E8M0 rows*ceil(cols/32); FP8 blocks include their declared scale shape/dtype. Source and runtime formats are separate; unquantized exclusions never downgrade native FP32 categories.
- KV/cache storage is counted per owner before consumer reads; per-request state is an endpoint, not one copy per token. Per-rank placement uses TP/EP/ETP and node-local Engram sharding.

## Native execution profile boundaries

- MTP is one native optional layer. Shared-index ownership is nearest-full conservative, not a native alias assertion; full index K planes are BF16 and shared layers pay their own selection reads.

## Optimizer and asynchronous-preparation assumptions

- Ops carry `semantic_family`, `weight_format`, `activation_format`, `compute_format`, `quant_category`, packed weight bytes, and owner/cache keys. Fusion must consume those records rather than infer from names.
- Shared long-KV/index producers are stored once; every consumer retains its own read. Cross-stage consumers require an explicit transfer. Grouped output keeps the group axis; MLA low-rank q_a/kv_a are replicated while q_b/kv_b output heads are sharded once.
- Prepare-eligible work is exposed through `mindie_sim.a3_catalog_prepare.catalog_prepare_candidates` with known-at/lifetime/owner/consumer fields. Model-load layout/scale/ring-capacity work can be lifted; request-token Engram ID metadata and supplied-grid vision position tables are request-arrival work. Future tokens/QKV/routing/acceptance are never marked early. Missing CPU/H2D rates are measurement gaps.

## Reproduce

```bash
python -m mindie_sim predict --input examples/model-reference/cases/glm-5.3.json
python -m mindie_sim should  --input examples/model-reference/cases/glm-5.3.json
python -m mindie_sim search  --input examples/model-reference/cases/glm-5.3.json --search-spec examples/model-reference/cases/glm-5.3.search.json
```

The case uses the full archived model dimensions and a declared A3 topology; it does not shrink the architecture. Request lengths are representative and can be changed without changing the model card.

## Scope

- Structural/analytic implementation coverage, not measured hardware performance. No A3/NPU execution or heldout data was used in this reference.
- Known boundary: GLM shared-index owner remains derived/conservative; raw image/video resize/crop rules require processor data; host CPU/H2D service rates are explicit inputs. Missing A3 calibration yields analytic estimates plus measurement gaps.

