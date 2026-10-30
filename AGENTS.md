# MindIE Sim

This repository owns the `mindie_sim` runtime and `mindie-sim` CLI. Product-level
entry documentation is in the sibling `mindie-agent` repository.

Read `docs/architecture.md` before changing the engine or semantic boundaries.
Read `docs/observations.md` before importing observations or changing calibration.
Read `docs/modeling-fidelity.md` when choosing operator detail or refinement work.

- Keep one causal execution engine for all hardware and deployments. Vendor names
  select input adapters, never performance branches in the execution loop.
- Derive work and dependencies from structure. No serving floors, model-specific
  latency multipliers, body-total fits or measurement-presence admission checks.
- Keep calibration separate from validation. Heldout measurements never enter a
  training lookup. Preserve original boundary, units, source and implementation.
- Extend block/recipe registries. A new model adapter produces existing structural
  records; a new mechanism supplies its own builder or executable recipe.
- Prefer the simplest resource/phase/tile model that preserves relevant work,
  boundaries and constraints. Instruction-level detail is optional and needs a
  material cost, trend or ranking counterexample; nonzero timing error alone is
  not a reason to recreate a kernel/compiler. Calibration is a separate axis.
- Keep core modules cohesive and under 800 lines when practical. Include [Role],
  [Invariants], [Timeouts & Locks] and [Side-Effect Policy] in their first 30 lines.
- Keep live execution memory bounded by active work. Stream large captures; never
  whole-read files over 50 MiB. Routine runs target at most 512 MiB process RAM.
- Simulation must not load weights, initialize device contexts or start remote work.
  Device measurements require resource/VRAM prechecks and explicit bounded work.
- Tests must establish behavior or independent mathematical invariants, not merely
  type checks, static strings, documentation prose or a fabricated external schema.
- Use focused tests during development and the complete maintained suite at delivery.
  Label catalog smoke, CPU benchmarks, primitive measurements and service accuracy
  separately. Never report configured CI as completed CI.
- Preserve unrelated work and external historical raw captures. Keep active test
  fixtures and source identities; do not restore retired snapshots or campaign outputs.
- Maintain one current schema and runtime. Do not restore deleted A3 prediction
  paths, migration shims or old-baseline fallbacks; see `docs/migration.md`.
- Use LF line endings. Prefer the GitHub connector when GitHub operations are needed.
