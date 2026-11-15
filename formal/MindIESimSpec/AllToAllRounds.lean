/-
  MindIESimSpec.AllToAllRounds
  Collective startup round scaling (`ceilLog2` vs `ringRounds`) and `steps` override legality
  (`mindie_sim/prediction/topology.py`, ASTRA_ISSUE34_SPEC.md §4).
-/
namespace MindIESimSpec.AllToAllRounds

/-- Structural fuel-bounded halving helper for `⌈log₂ g⌉`. -/
def ceilLog2Aux : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, g =>
    if g ≤ 1 then 0 else 1 + ceilLog2Aux fuel ((g + 1) / 2)

/-- Integer ceiling base-2 logarithm `⌈log₂ g⌉` (`0` for `g ≤ 1`).
    Defined by structural recursion so concrete instances kernel-reduce via `rfl`/`decide`. -/
def ceilLog2 (g : Nat) : Nat :=
  ceilLog2Aux g g

/-- Legacy / ring peer-startup round count `g - 1`. -/
def ringRounds (g : Nat) : Nat :=
  g - 1

private theorem ceilLog2Aux_congr :
    ∀ (f₁ f₂ g : Nat), g ≤ f₁ → g ≤ f₂ → ceilLog2Aux f₁ g = ceilLog2Aux f₂ g
  | 0, 0, _, _, _ => rfl
  | 0, _ + 1, g, hf₁, _ => by
    have hg0 : g = 0 := by omega
    subst hg0
    rfl
  | _ + 1, 0, g, _, hf₂ => by
    have hg0 : g = 0 := by omega
    subst hg0
    rfl
  | f₁ + 1, f₂ + 1, g, hf₁, hf₂ => by
    unfold ceilLog2Aux
    split
    · rfl
    · have hm₁ : (g + 1) / 2 ≤ f₁ := by omega
      have hm₂ : (g + 1) / 2 ≤ f₂ := by omega
      rw [ceilLog2Aux_congr f₁ f₂ ((g + 1) / 2) hm₁ hm₂]

/-- Exact halving recurrence for `ceilLog2`. -/
theorem ceilLog2_step (g : Nat) :
    ceilLog2 g = if g ≤ 1 then 0 else 1 + ceilLog2 ((g + 1) / 2) := by
  cases g with
  | zero => rfl
  | succ k =>
    unfold ceilLog2
    simp only [ceilLog2Aux]
    split
    · rfl
    · have hm_k : (k + 1 + 1) / 2 ≤ k := by omega
      have hm_self : (k + 1 + 1) / 2 ≤ (k + 1 + 1) / 2 := Nat.le_refl _
      rw [ceilLog2Aux_congr k ((k + 1 + 1) / 2) ((k + 1 + 1) / 2) hm_k hm_self]

/-- Power-of-two upper bound soundness: `g ≤ 2 ^ ceilLog2 g` for all `g ≥ 1`. -/
theorem le_two_pow_ceilLog2 (g : Nat) (hg : 1 ≤ g) :
    g ≤ 2 ^ ceilLog2 g := by
  induction g using Nat.strongRecOn with
  | ind n ih =>
    rw [ceilLog2_step]
    split
    · have hn1 : n = 1 := by omega
      subst hn1
      decide
    · have hm_lt : (n + 1) / 2 < n := by omega
      have hm_ge : 1 ≤ (n + 1) / 2 := by omega
      have hrec := ih ((n + 1) / 2) hm_lt hm_ge
      have hpow : 2 ^ (1 + ceilLog2 ((n + 1) / 2)) = 2 * 2 ^ ceilLog2 ((n + 1) / 2) := by
        rw [Nat.pow_add, Nat.pow_one, Nat.mul_comm]
      rw [hpow]
      omega

/-- Positivity for non-trivial groups: `1 ≤ ceilLog2 g` for all `g ≥ 2`. -/
theorem ceilLog2_pos {g : Nat} (hg : 2 ≤ g) :
    1 ≤ ceilLog2 g := by
  rw [ceilLog2_step]
  split <;> omega

/-- Non-strict upper bound across all multi-rank groups (`g ≥ 2`): `ceilLog2 g ≤ g - 1`. -/
theorem ceilLog2_le_ringRounds {g : Nat} (hg : 2 ≤ g) :
    ceilLog2 g ≤ ringRounds g := by
  induction g using Nat.strongRecOn with
  | ind n ih =>
    by_cases hn3 : n ≤ 3
    · have : n = 2 ∨ n = 3 := by omega
      rcases this with rfl | rfl <;> decide
    · rw [ceilLog2_step]
      unfold ringRounds
      split
      · omega
      · have hm_lt : (n + 1) / 2 < n := by omega
        have hm_ge : 2 ≤ (n + 1) / 2 := by omega
        have hrec := ih ((n + 1) / 2) hm_lt hm_ge
        unfold ringRounds at hrec
        omega

/-- Strict asymptotic separation for all `g ≥ 4`: logarithmic startup strictly beats `g - 1`. -/
theorem ceilLog2_lt_ringRounds {g : Nat} (hg : 4 ≤ g) :
    ceilLog2 g < ringRounds g := by
  rw [ceilLog2_step]
  unfold ringRounds
  split
  · omega
  · have hm_ge : 2 ≤ (g + 1) / 2 := by omega
    have hle := ceilLog2_le_ringRounds hm_ge
    unfold ringRounds at hle
    omega

/-- Concrete EP128 oracle (`ASTRA_ISSUE34_SPEC.md` §4.1): 7 log2 rounds vs 127 sequential rounds. -/
theorem ep128_log2_vs_ring :
    ceilLog2 128 = 7 ∧ ringRounds 128 = 127 := by
  decide

/-! ### Override Legality (`resolveSteps`, ASTRA_ISSUE34_SPEC.md §4.2) -/

/-- Legal `steps` range invariant: `0` when `g ≤ 1`, and `1 ≤ s ≤ g - 1` when `g ≥ 2`. -/
def LegalSteps (g s : Nat) : Prop :=
  if g ≤ 1 then s = 0 else 1 ≤ s ∧ s ≤ g - 1

/-- Validate and resolve AllToAll startup `steps` from `defaultSteps` and optional `override?`. -/
def resolveSteps (defaultSteps : Nat) (override? : Option Nat) (g : Nat) : Except String Nat :=
  if g ≤ 1 then
    let cand := override?.getD defaultSteps
    if cand = 0 then
      .ok 0
    else
      .error s!"communication_round_count_invalid:{cand}"
  else
    let cand := override?.getD defaultSteps
    if 1 ≤ cand ∧ cand ≤ g - 1 then
      .ok cand
    else
      .error s!"communication_round_count_invalid:{cand}"

/-- Soundness of `resolveSteps`: every `.ok s` satisfies `LegalSteps g s` (`1 ≤ s ≤ g - 1` for `g ≥ 2`,
    and `s = 0` for `g ≤ 1`). -/
theorem resolveSteps_ok_legal {defaultSteps : Nat} {override? : Option Nat} {g s : Nat}
    (hok : resolveSteps defaultSteps override? g = .ok s) :
    LegalSteps g s := by
  unfold resolveSteps LegalSteps at *
  dsimp only at hok
  by_cases hg1 : g ≤ 1
  · rw [if_pos hg1] at hok ⊢
    by_cases h0 : override?.getD defaultSteps = 0
    · rw [if_pos h0] at hok
      injection hok with hs
      exact hs.symm
    · rw [if_neg h0] at hok
      contradiction
  · rw [if_neg hg1] at hok ⊢
    by_cases hrange : 1 ≤ override?.getD defaultSteps ∧ override?.getD defaultSteps ≤ g - 1
    · rw [if_pos hrange] at hok
      injection hok with hs
      subst hs
      exact hrange
    · rw [if_neg hrange] at hok
      contradiction

/-- Rejects the baseline G=4 bug (`steps = g` must never succeed for `g ≥ 2`). -/
theorem resolveSteps_rejects_group_size (defaultSteps g : Nat) (hg : 2 ≤ g) :
    ∀ s, resolveSteps defaultSteps (some g) g ≠ .ok s := by
  intro s hok
  have hleg := resolveSteps_ok_legal hok
  unfold LegalSteps at hleg
  have hnot1 : ¬ (g ≤ 1) := by omega
  rw [if_neg hnot1] at hleg
  unfold resolveSteps at hok
  rw [if_neg hnot1] at hok
  dsimp only [Option.getD_some] at hok
  have hnot_range : ¬ (1 ≤ g ∧ g ≤ g - 1) := by omega
  rw [if_neg hnot_range] at hok
  contradiction

/-- Default `ceilLog2 g` without override always resolves cleanly to `.ok (ceilLog2 g)`. -/
theorem resolveSteps_ceilLog2_ok (g : Nat) :
    resolveSteps (ceilLog2 g) none g = .ok (ceilLog2 g) := by
  unfold resolveSteps
  dsimp only [Option.getD_none]
  by_cases hg1 : g ≤ 1
  · rw [if_pos hg1]
    have hz : ceilLog2 g = 0 := by rw [ceilLog2_step, if_pos hg1]
    simp [hz]
  · rw [if_neg hg1]
    have hg2 : 2 ≤ g := by omega
    have hpos := ceilLog2_pos hg2
    have hle := ceilLog2_le_ringRounds hg2
    unfold ringRounds at hle
    rw [if_pos ⟨hpos, hle⟩]

end MindIESimSpec.AllToAllRounds
