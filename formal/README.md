# MindIESimSpec — Lean 4 Formal Verification Kernel (`mindie-sim`)

`formal/` contains **`MindIESimSpec`**, a lightweight, zero-runtime-overhead **Lean 4** (`v4.16.0`, zero `Mathlib` dependency) formal verification kernel for the core algebraic invariants of `mindie-sim`.

---

## 1. Division of Responsibility / 职责边界

| Verification Layer | Scope | Mechanism |
|---|---|---|
| **Lean 4 Kernel (`formal/`)** | Universal algebraic & structural invariants across all 7 spec modules: ordered 4-clock max-plus conservation (`Int`), Pass 2.5 prefetch byte conservation & interval ledger non-double-charging, `ceilDiv` / Ascend `FRACTAL_NZ` / Subgraph-TP / EPLB layout isomorphism, 7-family `RecoverSSM` rollback-vs-replay state identity, `AllToAll` `⌈log₂ G⌉` vs `G - 1` round bounds, pure physical critical-path roofline monotonicity + direct `K_W` mesh `(W - 1) * bPeer` capacity scaling, and convex-mixture Farkas LP dual + Direct-MAC reload trace + checked Event-DAG critical path + 5-level certified performance ladder (`OptimalityCertificate`). | Constructive proofs + `omega` + structural induction; compile-time `#print axioms` gate (`AxiomCheck.lean`) forbidding `sorryAx` and custom axioms. |
| **Differential Tier-0 Pytest (`tests/runtime/test_formal_lean_locks.py`)** | Fast (`< 1.0s`) randomized property verification (`250` trials/domain) binding all 7 proved `.lean` modules directly to live `mindie_sim` Python modules, plus comment-stripped `sorry`/`admit`/`axiom` static audit. | Runs in standard CPU CI with zero Lean runtime dependency. |
| **Empirical Calibration (`mindie-profiler` & Live NPU Tiers)** | Hardware instruction service rates, HBM/MTE2 effective bandwidth knees, HCCL link framing overheads, and Ascend A2/A3/A5 operator execution timings. | Measured hardware traces and held-out validation campaigns (never invented by formal axioms). |

**简体中文说明**：`MindIESimSpec` 仅对模拟器内核的**纯数学守恒律、单调性与状态等价性**给出机器可检验的形式化证明（零运行时开销、零 `Mathlib` 依赖、冷启动构建 `< 2s`）；真实硬件速率、指令吞吐与算子耗时仍严格由 `mindie-profiler` 实测标定驱动。

---

## 2. Module Mapping / 模块映射

| Lean 4 Module | Proved Invariants | Paired `mindie_sim` Modules & Test Locks |
|---|---|---|
| `MindIESimSpec/FourClock.lean` | `Ordered` preservation under `rooflineClocks`, `join2`, `joinClocks` (including `List.Perm` stream-order invariance), and `advanceClocks`; `Int` bucket non-negativity & exact conservation `cube + vector + memory + comm = allT`; zero exposed memory when `0 ≤ M ≤ C`; branch-winner flip oracle. | `mindie_sim/prediction/exposed_breakdown.py`, `tests/runtime/test_formal_lean_locks.py` |
| `MindIESimSpec/PrefetchAndLedger.lean` | Signed `Int` Pass 2.5 clamp `prefetchBytes` bounds, monotonicity, and `(W - P) + P = W` conservation; half-open `[start, finish)` disjoint non-double-charging (`≤ max`) and static `[0, T)` + transient `[a, b)` decomposition. | `mindie_sim/prediction/weight_prefetch.py`, `mindie_sim/prediction/runtime/ledger.py` |
| `MindIESimSpec/ShardingAndLayout.lean` | `ceilDiv` coverage (`a ≤ ceilDiv a b * b < a + b`), positivity, exact divisibility; Ascend `FRACTAL_NZ` lower bound (`≥ d * n * elem`) and zero-padding divisibility (`c0 ∣ d ∧ 16 ∣ n`); Subgraph-TP & EPLB (`E_phys = ceilDiv E ep + R`) Frontend/Ledger layout isomorphism and strict `R > 0` byte growth. | `mindie_sim/catalog/weight_terms.py`, `mindie_sim/catalog/expert_slots.py`, `mindie_sim/catalog/projection_placement.py`, `mindie_sim/catalog/cache_format.py` |
| `MindIESimSpec/RecoverSSM.lean` | Polymorphic `(K + 1)`-checkpoint rollback = `1`-slot base + `draft.take acceptedLen` prefix replay across all 7 linear families (`kda`, `gdn`, `mamba`, `gla`, `deltanet`, `lightning`, `retnet`); strict HBM residency bound `S + replayLog < (K + 1) * S` and KDA `64x` record ratio. | `mindie_sim/catalog/cache_pools.py`, `mindie_sim/catalog/speculative_state.py`, `mindie_sim/prediction/recover_ssm_work.py` |
| `MindIESimSpec/AllToAllRounds.lean` | Structural `ceilLog2` (`g ≤ 2 ^ ceilLog2 g`), `1 ≤ ceilLog2 g ≤ g - 1` (`g ≥ 2`), strict `ceilLog2 g < g - 1` (`g ≥ 4`), `128`-rank (`7` vs `127`), and `resolveSteps` legality (`1 ≤ s ≤ g - 1` for `g ≥ 2`, rejecting `steps = g`). | `mindie_sim/prediction/collective_startup.py`, `mindie_sim/prediction/topology.py` |
| `MindIESimSpec/RooflineAndMesh.lean` | Inductive series-parallel DAG `criticalPath` non-negativity and monotonicity, pure physical step bound `0 ≤ ideal1x ≤ predicted`, direct `K_W` mesh per-rank bandwidth `(W - 1) * bPeer`, and directed-link capacity partition `W * (W - 1) * bPeer = W * perRankBw`. | `mindie_sim/prediction/roofline_graph.py`, `mindie_sim/prediction/efficiency/`, `mindie_sim/backends/ascend.py` |
| `MindIESimSpec/OptimalityCertificate.lean` | Convex-mixture Farkas LP dual lower bound (`farkas_dual_le_any_feasible_schedule`), Direct-MAC bilinear epoch & multi-epoch reload trace bounds (`epoch_bilinear_pair_cap`, `capacity_reload_trace_bound`), checked `EventDagSpec` critical-path bound (`max_plus_critical_path_bound`), and unified 5-level certified ladder (`0 ≤ L*_Lean ≤ A*_DSL ≤ A*_Framework ≤ min(M_Mainline, U_Unmerge)` + 4-bucket telescoping conservation). | `mindie_sim/optimization/dual_bound.py`, `mindie_sim/optimization/reload_bounds.py`, `mindie_sim/optimization/ladder.py`, `tests/runtime/test_formal_lean_locks.py` |
| `AxiomCheck.lean` | Compile-time `#guard_msgs in #print axioms` check verifying core theorems use only Lean 4 standard axioms (`[]`, `[propext]`, `[propext, Quot.sound]`, or `[propext, Classical.choice, Quot.sound]`). | `tests/runtime/test_formal_lean_locks.py` |

---

## 3. Building Locally / 本地验证

```bash
cd formal
lake build
```

For isolated builds without touching `$HOME/.elan`:

```bash
export ELAN_HOME=/path/to/isolated/elan
export PATH="$ELAN_HOME/bin:$PATH"
cd formal && lake clean && lake build
```
