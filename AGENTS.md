# MindIE Sim (`sim`) — Agent Architecture Guide

Standalone repository (`https://github.com/mindie-agent/sim`) for `mindie_sim` (`mindie-sim` CLI), providing 4-layer Ascend NPU simulation, multi-SKU capability resolution, `D2D`/`H2D`/`D2H`/`RH2D`/`RD2H` transfer bounds, periodic microstage pipeline solving, Phase 0 subgraph fusion triage, and whole-network parallel decomposition.

See `mindie-agent/docs/simulation-system.md` for architecture and invariants.

## Host resources

Large model, long decode, calibration sweeps and parallel predictive jobs run
on a remote Linux server via remote-dev. Keep desktop checks to small unit or
component cases. A `host_capacity_requires_remote` error must be acted on by
moving execution remote; do not bypass it, reduce the declared experiment just
to evade it, or fan the same work out across local agents.

Prediction detail is bounded by default. Use `--trace-detail summary` for
metric-only jobs; use `--trace-detail full` only when complete per-op evidence
is needed on a server. Check `trace_capture` before treating op traces as a
complete inventory. Physical-calibration totals and metrics include all steps.
