/-
  MindIESimSpec: Lean 4 Formal Verification Kernel for mindie-sim v0.9.15.
  Covers five pure algebraic contracts aligned with ASTRA_ISSUE34_SPEC.md:
    1. FourClock         — Ordered critical-span max-plus attribution algebra (§12.2, §12.4)
    2. PrefetchAndLedger — Pass 2.5 MLAPO static prefetch clamp & half-open interval ledger
    3. ShardingAndLayout — Subgraph local TP, EPLB redundant slots & Ascend FRACTAL_NZ tiles (§6, §9)
    4. RecoverSSM        — Seven-family speculative checkpoint rollback = prefix replay (§7)
    5. AllToAllRounds    — Logarithmic vs ring AllToAll startup scaling & steps bounds (§4)
-/
import MindIESimSpec.FourClock
import MindIESimSpec.PrefetchAndLedger
import MindIESimSpec.ShardingAndLayout
import MindIESimSpec.RecoverSSM
import MindIESimSpec.AllToAllRounds
