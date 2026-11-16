# MindIESimSpec — Lean 4 Formal Verification Kernel (`mindie-sim v0.9.17`)

`formal/` contains **`MindIESimSpec`**, a lightweight, zero-runtime-overhead **Lean 4** (`v4.16.0`, zero `Mathlib` dependency) formal verification kernel for the core algebraic invariants of `mindie-sim`.

---

## 1. Division of Responsibility / 职责边界

| Verification Layer | Scope | Mechanism |
|---|---|---|
| **Lean 4 Kernel (`formal/`)** | Universal algebraic & structural invariants: ordered 4-clock max-plus conservation (`Int`), Pass 2.5 prefetch byte conservation & interval ledger non-double-charging, `ceilDiv` / Ascend `FRACTAL_NZ` / Subgraph-TP / EPLB layout isomorphism, 7-family `RecoverSSM` rollback-vs-replay state identity, and `AllToAll` `⌈log₂ G⌉` vs `G - 1` round bounds. | Constructive proofs + `omega` + structural induction; compile-time `#print axioms` gate (`AxiomCheck.lean`) forbidding `sorryAx` and custom axioms. |
| **Differential Tier-0 Pytest (`tests/test_formal_lean_invariants.py`)** | Fast (`< 1.0s`) randomized property verification (`250` trials/domain) binding the proved equations directly to `mindie_sim` Python modules, plus comment-stripped `sorry`/`admit`/`axiom` static audit. | Runs in standard CPU CI with zero Lean runtime dependency. |
| **Empirical Calibration (`mindie-profiler` & Live NPU Tiers)** | Hardware instruction service rates, HBM/MTE2 effective bandwidth knees, HCCL link framing overheads, and Ascend A2/A3/A5 operator execution timings. | Measured hardware traces and held-out validation campaigns (never invented by formal axioms). |

**简体中文说明**：`MindIESimSpec` 仅对模拟器内核的**纯数学守恒律、单调性与状态等价性**给出机器可检验的形式化证明（零运行时开销、零 `Mathlib` 依赖、冷启动构建 `< 2s`）；真实硬件速率、指令吞吐与算子耗时仍严格由 `mindie-profiler` 实测标定驱动。

---

## 2. Module Mapping / 模块映射

| Lean 4 Module | Proved Invariants | Paired `mindie_sim` Modules & Spec |
|---|---|---|
| `MindIESimSpec/FourClock.lean` | `Ordered` preservation under `rooflineClocks`, `join2`, `joinClocks` (including `List.Perm` stream-order invariance), and `advanceClocks`; `Int` bucket non-negativity & exact conservation `cube + vector + memory + comm = allT`; zero exposed memory when `0 ≤ M ≤ C`; all §12.4 oracles + branch-winner flip. | `ASTRA_ISSUE34_SPEC.md` §12.2, §12.4 (`prediction/exposed_breakdown.py`) |
| `MindIESimSpec/PrefetchAndLedger.lean` | Signed `Int` Pass 2.5 clamp `prefetchBytes` bounds, monotonicity, and `(W - P) + P = W` conservation; half-open `[start, finish)` disjoint non-double-charging (`≤ max`) and static `[0, T)` + transient `[a, b)` decomposition. | `mindie_sim/prediction/weight_prefetch.py`, `mindie_sim/prediction/runtime/ledger.py`, `mindie_sim/memory_ledger_types.py` |
| `MindIESimSpec/ShardingAndLayout.lean` | `ceilDiv` coverage (`a ≤ ceilDiv a b * b < a + b`), positivity, exact divisibility; Ascend `FRACTAL_NZ` lower bound (`≥ d * n * elem`) and zero-padding divisibility (`c0 ∣ d ∧ 16 ∣ n`); Subgraph-TP & EPLB (`E_phys = ceilDiv E ep + R`) Frontend/Ledger layout isomorphism and strict `R > 0` byte growth. | `mindie_sim/catalog/weight_terms.py`, `mindie_sim/kv_layout_physics.py`, `ASTRA_ISSUE34_SPEC.md` §6, §9 |
| `MindIESimSpec/RecoverSSM.lean` | Polymorphic `(K + 1)`-checkpoint rollback = `1`-slot base + `draft.take acceptedLen` prefix replay across all 7 linear families (`kda`, `gdn`, `mamba`, `gla`, `deltanet`, `lightning`, `retnet`); strict HBM residency bound `S + replayLog < (K + 1) * S` and KDA `64x` record ratio. | `mindie_sim/linear_state_physics.py`, `mindie_sim/prediction/recover_ssm_work.py`, `ASTRA_ISSUE34_SPEC.md` §7 |
| `MindIESimSpec/AllToAllRounds.lean` | Structural `ceilLog2` (`g ≤ 2 ^ ceilLog2 g`), `1 ≤ ceilLog2 g ≤ g - 1` (`g ≥ 2`), strict `ceilLog2 g < g - 1` (`g ≥ 4`), EP128 (`7` vs `127`), and `resolveSteps` legality (`1 ≤ s ≤ g - 1` for `g ≥ 2`, rejecting `steps = g`). | `mindie_sim/prediction/topology.py`, `ASTRA_ISSUE34_SPEC.md` §4 |
| `AxiomCheck.lean` | Compile-time `#guard_msgs in #print axioms` check verifying core theorems use only Lean 4 standard axioms (`[]`, `[propext, Quot.sound]`, or `[propext, Classical.choice, Quot.sound]`). | CI gate (`.github/workflows/formal.yml`) |

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
