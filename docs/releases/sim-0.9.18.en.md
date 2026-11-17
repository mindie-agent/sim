# mindie-sim 0.9.18: Clean-Core Mechanism-First Spine, Unified A2/A3/A5 Physical Recipes, and `device-tool` Governance

[中文](sim-0.9.18.md)

`mindie-sim v0.9.18` completes the Clean-Core mechanism-first architecture across Ascend A2 (`DAV_2201`), Ascend A3 (`DAV_2201` dual-Die + CloudMatrix 384), and Ascend A5 (`DAV_3510` + Atlas 950 SuperPoD):

- **Single Canonical Causal Spine & Zero `ResourceEnvelope` Fallbacks**: Unifies single-operator evaluation, subgraphs, pure-model forward graphs, 4D efficiency breakdown, SLA hardware scaling, and PD/topology Pareto sweeps onto a single discrete-event `runtime.Engine` with one `EvaluationContext` per evaluation, banning bypass estimators and `ResourceEnvelope` peak-division fallbacks (`ModelingError` on unmodeled mechanisms).
- **Unified Class A–E `PhysicalRecipe` Registry & Exact A2/A3/A5 Pipe Conservation**: Consolidates `TiledFlashAndPagedAttentionRecipe` (Class A), `TiledSparseIndexerRecipe` (Class B), `ChunkedLinearAttentionRecipe` (Class C), `AIVTiledVectorAndGatherRecipe` (Class D), and `TopologyCollectiveRecipe` (Class E) alongside source-pinned A2/A3 (`paged_mla_arch22`, `w4a8`) and A5 `DAV_3510` (`L0C = 256 KiB`, `l0c->ub` direct pipe, `K_8` UB mesh) `PlannedRecipe` builders, conserving split `mte2_aic` / `mte2_aiv` bytes and enforcing per-Die shared `L2` ownership (`buffer_units.l2 == 1`).
- **Independent 37+ Model Catalog Conservation & Differential Literal Gates**: Enforces independent parameter/weight/KV ledger reconciliation (`scripts/verify_m1.py catalog-meta --require-models 37`) and AST-based differential magic-number checks (`scripts/check_no_magic_numbers.py`).
- **A2/A3/A5 Master Calibration Guide, Docs Archive & `device-tool` Governance**: Ships bilingual contributor & custom-Attention calibration playbooks (`CONTRIBUTING{.en}.md`), archives historical design notes under `docs/archive/`, and aligns PR validation with `--profile device-tool`.
