# `mindie-sim` Documentation Index

[Repository Home (`../README.md`)](../README.md) · [English Home (`../README.en.md`)](../README.en.md) · [Historical Wave & Audit Archive (`archive/README.md`)](archive/README.md)

This directory holds the active architecture, mechanism-first modeling, operator-family, and verification specifications for `mindie-sim`.

## 1. Core Architecture, Modeling & Verification Guides

| Guide | Scope |
| --- | --- |
| [`architecture.md`](architecture.md) | 5-layer causal modeling architecture (physical resources -> operator families -> blocks/subgraphs -> pure-model costs -> serving constraints) |
| [`inputs.md`](inputs.md) | Input case schemas, hardware topology declarations, and workload parameterization |
| [`composable-models.md`](composable-models.md) | Deterministic `compose` synthesis and reusable block/layer composition |
| [`resource-services.md`](resource-services.md) | Hardware resource services, memory hierarchy bandwidths, and overlap arbitration |
| [`operator-mechanisms.md`](operator-mechanisms.md) | Mechanism-first operator cost equations across Cube, Vector, MTE, and HCCL |
| [`operator-extension.md`](operator-extension.md) · [`extending.md`](extending.md) | Adding new operator families, model blocks, and hardware SKU profiles |
| [`modeling-fidelity.md`](modeling-fidelity.md) | Independent fidelity levels and measured calibration contracts |
| [`profiler-structure.md`](profiler-structure.md) | `mindie-model-structure/1` export and 3-state alignment with `mindie-profiler` |
| [`deployment-aliases.md`](deployment-aliases.md) | Canonical deployment aliases across A2 (910B), A3 (910C dual-Die), and A5 (950) |
| [`verification.md`](verification.md) · [`accuracy-regression.md`](accuracy-regression.md) · [`model-scoreboard.md`](model-scoreboard.md) | Accuracy regression gates, leave-one-out generalization checks, and multi-regime scoreboard |

Additional active mechanism notes (`call-analysis.md`, `graph-analysis.md`, `graph-runtime-diagnosis.md`, `invocation-costs.md`, `model-convergence.md`, `observable-service-regimes.md`, `observations.md`, `studies.md`, `migration.md`) remain directly in `docs/`.

## 2. Canonical Operator & Release Directories

- [`operators/`](operators/): Active canonical operator-family specifications (attention, MLA/paged-MLA, GDN/KDA recurrent state, MoE routing/grouped GEMM, MC2/HCCL collectives, and MXFP4/MXFP8/W4A8 quantization).
- [`releases/`](releases/): Immutable `sim-*` version release notes.
- [`archive/`](archive/README.md): Frozen dated wave notes (`*-20261029.md`, `*-20261030.md`), one-off issue notes, and historical point-in-time audit/review/migration records.
