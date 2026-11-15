/-
  MindIESimSpec.RecoverSSM
  Two-stage speculative `RecoverSSM` state equivalence and HBM residency bound
  (`mindie_sim/linear_state_physics.py`, `mindie_sim/prediction/recover_ssm_work.py`,
   ASTRA_ISSUE34_SPEC.md §7).
-/
namespace MindIESimSpec.RecoverSSM

/-- Seven linear-attention families unified under `RecoverSSM` (`LINEAR_STATE_KINDS`). -/
inductive LinearFamily
  | kda
  | gdn
  | mamba
  | gla
  | deltanet
  | lightning
  | retnet
  deriving DecidableEq, Repr

/-! ### Part 1: Polymorphic State Rollback vs Replay Equivalence -/

/-- Build `(K + 1)` prefix checkpoints `[s_0, s_1, ..., s_K]` for deterministic transition `step`
    across arbitrary state type `S` and token type `Tok`. -/
def buildCheckpoints {S Tok : Type _} (step : S → Tok → S) (s0 : S) : List Tok → List S
  | [] => [s0]
  | tok :: rest => s0 :: buildCheckpoints step (step s0 tok) rest

/-- Standard speculative rollback (`recover_ssm = false`): index directly into the `(K + 1)`
    materialized state checkpoints at `acceptedLen`. -/
def rollbackCheckpoint {S Tok : Type _} (step : S → Tok → S) (s0 : S)
    (draft : List Tok) (acceptedLen : Nat) : S :=
  (buildCheckpoints step s0 draft).getD acceptedLen s0

/-- Two-stage `RecoverSSM` commit (`recover_ssm = true`): retain only base state `s0`
    and replay the accepted prefix `draft.take acceptedLen` through `step`. -/
def recoverSSMReplay {S Tok : Type _} (step : S → Tok → S) (s0 : S)
    (draft : List Tok) (acceptedLen : Nat) : S :=
  (draft.take acceptedLen).foldl step s0

/-- `buildCheckpoints` materializes exactly `draft.length + 1` full recurrent state slots. -/
theorem buildCheckpoints_length {S Tok : Type _} (step : S → Tok → S) (s0 : S)
    (draft : List Tok) :
    (buildCheckpoints step s0 draft).length = draft.length + 1 := by
  induction draft generalizing s0 with
  | nil => rfl
  | cons tok rest ih =>
    simp only [buildCheckpoints, List.length_cons, ih]

private theorem getD_eq_of_lt {α : Type _} (l : List α) (i : Nat) (d₁ d₂ : α)
    (hi : i < l.length) :
    l.getD i d₁ = l.getD i d₂ := by
  induction l generalizing i with
  | nil =>
    simp at hi
  | cons x xs ih =>
    cases i with
    | zero => rfl
    | succ n =>
      simp only [List.length_cons] at hi
      simp only [List.getD_cons_succ]
      exact ih n (by omega)

/-- Core `RecoverSSM` state-equivalence theorem:
    for ANY state space `S` (covering all 7 linear families), ANY token type `Tok`,
    ANY deterministic transition `step`, ANY base state `s0`, ANY draft sequence `draft`,
    and ANY `acceptedLen ≤ draft.length`, indexing the `(K + 1)` checkpoint table produces
    the **exact** same state as replaying the accepted prefix from `s0`. -/
theorem rollbackCheckpoint_eq_recoverSSMReplay {S Tok : Type _}
    (step : S → Tok → S) (s0 : S) (draft : List Tok) (acceptedLen : Nat)
    (hlen : acceptedLen ≤ draft.length) :
    rollbackCheckpoint step s0 draft acceptedLen =
      recoverSSMReplay step s0 draft acceptedLen := by
  induction draft generalizing s0 acceptedLen with
  | nil =>
    have h0 : acceptedLen = 0 := by simpa using hlen
    subst h0
    rfl
  | cons tok rest ih =>
    cases acceptedLen with
    | zero => rfl
    | succ n =>
      simp only [List.length_cons] at hlen
      have hn : n ≤ rest.length := by omega
      unfold rollbackCheckpoint recoverSSMReplay
      simp only [buildCheckpoints, List.getD_cons_succ, List.take_succ_cons, List.foldl_cons]
      have hlt : n < (buildCheckpoints step (step s0 tok) rest).length := by
        rw [buildCheckpoints_length]
        omega
      rw [getD_eq_of_lt _ n s0 (step s0 tok) hlt]
      exact ih (step s0 tok) n hn

/-! ### Part 2: HBM Residency Strict Capacity Bound (`Int`) -/

/-- Strict HBM capacity saving over `Int`:
    when `k ≥ 1`, `stateBytes > 0`, and compact replay log `replayLogBytes < k * stateBytes`,
    `RecoverSSM` residency `stateBytes + replayLogBytes` is strictly smaller than
    the `(k + 1)`-checkpoint residency `(k + 1) * stateBytes`. -/
theorem recoverSSM_bytes_strict_lt_checkpoints
    (k stateBytes replayLogBytes : Int)
    (_hk : 1 ≤ k) (_hs : 0 < stateBytes) (_hr0 : 0 ≤ replayLogBytes)
    (hr_lt : replayLogBytes < k * stateBytes) :
    stateBytes + replayLogBytes < (k + 1) * stateBytes := by
  have hexpand : (k + 1) * stateBytes = k * stateBytes + stateBytes := by
    rw [Int.add_mul, Int.one_mul]
  omega

/-- Concrete KDA geometry (`Dk = Dv = 128`, FP32 state, FP32 correction + BF16 `k/g` record):
    each token's replay record is `1/64` of one recurrent state slot (`linear_state_physics.py`),
    so `(k + 1)` replay records strictly beat `(k + 1)` state checkpoints for all `1 ≤ k ≤ 62`. -/
theorem kda_symmetric128_recoverSSM_saves_hbm (heads k : Int)
    (hh : 0 < heads) (hk1 : 1 ≤ k) (hk62 : k ≤ 62) :
    let stateBytes := heads * 128 * 128 * 4
    let recordBytesPerTok := heads * (4 * 128 + 2 * 128 * 2)
    let replayLogBytes := (k + 1) * recordBytesPerTok
    stateBytes = 64 * recordBytesPerTok ∧
      stateBytes + replayLogBytes < (k + 1) * stateBytes := by
  dsimp
  refine ⟨by omega, ?_⟩
  have hunit : 65536 + (k + 1) * 1024 < (k + 1) * 65536 := by omega
  have hscaled : heads * (65536 + (k + 1) * 1024) < heads * ((k + 1) * 65536) :=
    Int.mul_lt_mul_of_pos_left hunit hh
  have hlhs : heads * 128 * 128 * 4 + (k + 1) * (heads * 1024) =
      heads * (65536 + (k + 1) * 1024) := by
    have h1 : heads * 128 * 128 * 4 = heads * 65536 := by omega
    rw [h1, Int.mul_add, ← Int.mul_assoc (k + 1) heads 1024,
        Int.mul_comm (k + 1) heads, Int.mul_assoc heads (k + 1) 1024]
  have hrhs : (k + 1) * (heads * 128 * 128 * 4) = heads * ((k + 1) * 65536) := by
    have h1 : heads * 128 * 128 * 4 = heads * 65536 := by omega
    rw [h1, ← Int.mul_assoc (k + 1) heads 65536,
        Int.mul_comm (k + 1) heads, Int.mul_assoc heads (k + 1) 65536]
  rw [hlhs, hrhs]
  exact hscaled

end MindIESimSpec.RecoverSSM
