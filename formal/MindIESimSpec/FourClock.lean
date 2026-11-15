/-
  MindIESimSpec.FourClock
  Ordered critical-span max-plus attribution algebra (ASTRA_ISSUE34_SPEC.md §12.2 & §12.4).
  Models ExposedClocks over Int so bucket non-negativity and telescoping conservation
  are genuinely proved without Nat truncated-subtraction masking.
-/
namespace MindIESimSpec.FourClock

/-- Four counterfactual critical-path clocks for the realized execution DAG:
    * `allT`         : full execution including communication, memory, vector/control, and cube
    * `withoutCommT` : execution with communication service set to 0
    * `computeT`     : execution with communication and memory set to 0 (cube + vector/control)
    * `cubeT`        : execution with communication, memory, and vector/control set to 0 (cube only) -/
structure ExposedClocks where
  allT         : Int
  withoutCommT : Int
  computeT     : Int
  cubeT        : Int
  deriving DecidableEq, Repr

/-- Monotonicity invariant: `0 ≤ cubeT ≤ computeT ≤ withoutCommT ≤ allT`. -/
def Ordered (c : ExposedClocks) : Prop :=
  0 ≤ c.cubeT ∧ c.cubeT ≤ c.computeT ∧ c.computeT ≤ c.withoutCommT ∧ c.withoutCommT ≤ c.allT

/-- Identity clock vector for empty predecessor sets. -/
def zeroClocks : ExposedClocks :=
  { allT := 0, withoutCommT := 0, computeT := 0, cubeT := 0 }

/-- Single concurrent Roofline service interval with service demands `(c, v, m, n)`. -/
def rooflineClocks (c v m n : Int) : ExposedClocks :=
  { allT         := max (max (max c v) m) n
    withoutCommT := max (max c v) m
    computeT     := max c v
    cubeT        := c }

/-- Componentwise max-join across two predecessor clock vectors. -/
def join2 (a b : ExposedClocks) : ExposedClocks :=
  { allT         := max a.allT b.allT
    withoutCommT := max a.withoutCommT b.withoutCommT
    computeT     := max a.computeT b.computeT
    cubeT        := max a.cubeT b.cubeT }

/-- Fold componentwise max-join across an arbitrary predecessor list. -/
def joinClocks (preds : List ExposedClocks) : ExposedClocks :=
  preds.foldr join2 zeroClocks

/-- Componentwise serial addition of predecessor join and node duration. -/
def addClocks (a b : ExposedClocks) : ExposedClocks :=
  { allT         := a.allT + b.allT
    withoutCommT := a.withoutCommT + b.withoutCommT
    computeT     := a.computeT + b.computeT
    cubeT        := a.cubeT + b.cubeT }

/-- Max-plus DAG step: join all predecessors componentwise, then add local node duration. -/
def advanceClocks (preds : List ExposedClocks) (dur : ExposedClocks) : ExposedClocks :=
  addClocks (joinClocks preds) dur

/-- Exposed four-bucket latency attribution over Int (no truncated Nat subtraction). -/
structure ExposedBreakdown where
  cube   : Int
  vector : Int
  memory : Int
  comm   : Int
  deriving DecidableEq, Repr

/-- Terminal telescoping projection from ordered four-clocks to exposed buckets. -/
def finishBreakdown (c : ExposedClocks) : ExposedBreakdown :=
  { cube   := c.cubeT
    vector := c.computeT - c.cubeT
    memory := c.withoutCommT - c.computeT
    comm   := c.allT - c.withoutCommT }

@[ext]
theorem ExposedClocks.ext {a b : ExposedClocks}
    (h1 : a.allT = b.allT)
    (h2 : a.withoutCommT = b.withoutCommT)
    (h3 : a.computeT = b.computeT)
    (h4 : a.cubeT = b.cubeT) : a = b := by
  cases a; cases b; simp_all

theorem zeroClocks_ordered : Ordered zeroClocks := by
  dsimp [Ordered, zeroClocks]
  omega

theorem rooflineClocks_ordered {c v m n : Int}
    (hc : 0 ≤ c) (_hv : 0 ≤ v) (_hm : 0 ≤ m) (_hn : 0 ≤ n) :
    Ordered (rooflineClocks c v m n) := by
  dsimp [Ordered, rooflineClocks]
  omega

theorem join2_ordered {a b : ExposedClocks} (ha : Ordered a) (hb : Ordered b) :
    Ordered (join2 a b) := by
  dsimp [Ordered, join2] at *
  omega

theorem join2_comm (a b : ExposedClocks) : join2 a b = join2 b a := by
  apply ExposedClocks.ext <;> dsimp [join2] <;> omega

theorem join2_assoc (a b c : ExposedClocks) :
    join2 (join2 a b) c = join2 a (join2 b c) := by
  apply ExposedClocks.ext <;> dsimp [join2] <;> omega

theorem join2_idem (a : ExposedClocks) : join2 a a = a := by
  apply ExposedClocks.ext <;> dsimp [join2] <;> omega

theorem joinClocks_ordered (preds : List ExposedClocks)
    (h : ∀ p ∈ preds, Ordered p) : Ordered (joinClocks preds) := by
  induction preds with
  | nil => exact zeroClocks_ordered
  | cons x xs ih =>
    simp only [joinClocks, List.foldr_cons]
    apply join2_ordered
    · exact h x (List.Mem.head xs)
    · exact ih (fun p hp => h p (List.Mem.tail x hp))

/-- Permutation invariance: stream / predecessor arrival list ordering never alters `joinClocks`. -/
theorem joinClocks_perm {l₁ l₂ : List ExposedClocks} (hp : List.Perm l₁ l₂) :
    joinClocks l₁ = joinClocks l₂ := by
  induction hp with
  | nil => rfl
  | cons x _ ih =>
    simp only [joinClocks] at ih
    simp only [joinClocks, List.foldr_cons, ih]
  | swap x y l =>
    simp only [joinClocks, List.foldr_cons]
    rw [← join2_assoc, join2_comm y x, join2_assoc]
  | trans _ _ ih₁ ih₂ =>
    exact ih₁.trans ih₂

private theorem foldr_join2_acc (x acc : ExposedClocks) (xs : List ExposedClocks) :
    xs.foldr join2 (join2 acc x) = join2 x (xs.foldr join2 acc) := by
  induction xs with
  | nil => exact join2_comm acc x
  | cons y ys ih =>
    simp only [List.foldr_cons, ih]
    rw [← join2_assoc, join2_comm y x, join2_assoc]

/-- Left-fold and right-fold reductions coincide identically for all predecessor lists. -/
theorem joinClocks_eq_foldl (preds : List ExposedClocks) :
    preds.foldl join2 zeroClocks = joinClocks preds := by
  unfold joinClocks
  suffices ∀ acc : ExposedClocks, preds.foldl join2 acc = preds.foldr join2 acc from
    this zeroClocks
  intro acc
  induction preds generalizing acc with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldl_cons, List.foldr_cons]
    rw [ih (join2 acc x), foldr_join2_acc]

theorem addClocks_ordered {a b : ExposedClocks} (ha : Ordered a) (hb : Ordered b) :
    Ordered (addClocks a b) := by
  dsimp [Ordered, addClocks] at *
  omega

/-- Advancing across any list of ordered predecessors by an ordered node duration preserves `Ordered`. -/
theorem advanceClocks_ordered (preds : List ExposedClocks) (dur : ExposedClocks)
    (hp : ∀ p ∈ preds, Ordered p) (hd : Ordered dur) :
    Ordered (advanceClocks preds dur) := by
  unfold advanceClocks
  exact addClocks_ordered (joinClocks_ordered preds hp) hd

/-- Non-negativity and exact telescoping conservation over Int:
    every bucket is `≥ 0` and `cube + vector + memory + comm = c.allT` identically. -/
theorem finishBreakdown_nonneg_and_conserves (c : ExposedClocks) (hc : Ordered c) :
    let b := finishBreakdown c
    0 ≤ b.cube ∧ 0 ≤ b.vector ∧ 0 ≤ b.memory ∧ 0 ≤ b.comm ∧
      b.cube + b.vector + b.memory + b.comm = c.allT := by
  dsimp [Ordered] at hc
  dsimp [finishBreakdown]
  omega

/-- Hidden double-buffered MTE2 (`0 ≤ m ≤ c`) incurs zero exposed memory latency. -/
theorem doubleBuffer_hidden_memory_zero_exposed (c m : Int) (hm0 : 0 ≤ m) (hmc : m ≤ c) :
    let b := finishBreakdown (rooflineClocks c 0 m 0)
    b.memory = 0 ∧ b.cube = c ∧ b.vector = 0 ∧ b.comm = 0 := by
  dsimp [finishBreakdown, rooflineClocks]
  refine ⟨?_, rfl, ?_, ?_⟩ <;> omega

/-! ### Concrete Attribution Oracles (ASTRA_ISSUE34_SPEC.md §12.4) -/

/-- §12.4 Row 1: One double-buffer Roofline op `C=10, M=4` -> `T=10, (10, 0, 0, 0)`. -/
theorem oracle_roofline_c10_m4 :
    let clk := rooflineClocks 10 0 4 0
    clk.allT = 10 ∧ finishBreakdown clk = ⟨10, 0, 0, 0⟩ := by
  decide

/-- §12.4 Row 2: One memory-bound Roofline op `C=4, M=10` -> `T=10, (4, 0, 6, 0)`. -/
theorem oracle_roofline_c4_m10 :
    let clk := rooflineClocks 4 0 10 0
    clk.allT = 10 ∧ finishBreakdown clk = ⟨4, 0, 6, 0⟩ := by
  decide

/-- §12.4 Row 3: Serial `Cube 4` then `Vector 3` -> `T=7, (4, 3, 0, 0)`. -/
theorem oracle_serial_cube4_then_vector3 :
    let n1 := advanceClocks [] (rooflineClocks 4 0 0 0)
    let n2 := advanceClocks [n1] (rooflineClocks 0 3 0 0)
    n2.allT = 7 ∧ finishBreakdown n2 = ⟨4, 3, 0, 0⟩ := by
  decide

/-- §12.4 Row 4: Concurrent `Cube 4` and `Vector 7` -> `T=7, (4, 3, 0, 0)`. -/
theorem oracle_concurrent_cube4_and_vector7 :
    let clk := joinClocks [rooflineClocks 4 0 0 0, rooflineClocks 0 7 0 0]
    clk.allT = 7 ∧ finishBreakdown clk = ⟨4, 3, 0, 0⟩ := by
  decide

/-- §12.4 Row 5: Concurrent `Comm 3` in parallel with `(C=5, M=7)` -> `T=7, (5, 0, 2, 0)`. -/
theorem oracle_concurrent_comm3_and_cm57 :
    let clk := joinClocks [rooflineClocks 0 0 0 3, rooflineClocks 5 0 7 0]
    clk.allT = 7 ∧ finishBreakdown clk = ⟨5, 0, 2, 0⟩ := by
  decide

/-- §12.4 Row 6: Concurrent `Comm 10` in parallel with `(C=5, M=7)` -> `T=10, (5, 0, 2, 3)`. -/
theorem oracle_concurrent_comm10_and_cm57 :
    let clk := joinClocks [rooflineClocks 0 0 0 10, rooflineClocks 5 0 7 0]
    clk.allT = 10 ∧ finishBreakdown clk = ⟨5, 0, 2, 3⟩ := by
  decide

/-- §12.4 Row 7: Serial `Comm 3` before `(C=5, M=7)` -> `T=10, (5, 0, 2, 3)`. -/
theorem oracle_serial_comm3_before_cm57 :
    let n1 := advanceClocks [] (rooflineClocks 0 0 0 3)
    let n2 := advanceClocks [n1] (rooflineClocks 5 0 7 0)
    n2.allT = 10 ∧ finishBreakdown n2 = ⟨5, 0, 2, 3⟩ := by
  decide

/-- §12.4 Row 8: Serial host launch/control `2` then `(C=5, M=3)` -> `T=7, (5, 2, 0, 0)`. -/
theorem oracle_serial_control2_then_cm53 :
    let n1 := advanceClocks [] (rooflineClocks 0 2 0 0)
    let n2 := advanceClocks [n1] (rooflineClocks 5 0 3 0)
    n2.allT = 7 ∧ finishBreakdown n2 = ⟨5, 2, 0, 0⟩ := by
  decide

/-- §12.4 Branch-winner flip under `join2`:
    Branch A (`Cube 2 -> Comm 8`, `allT = 10`) wins total wall time over
    Branch B (`Cube 5 -> Mem 3`, `allT = 8`), but Branch B dominates `withoutCommT` (`8 > 2`)
    and `computeT` (`5 > 2`). Componentwise `join2` yields `(5, 0, 3, 2)` rather than `(2, 0, 0, 8)`. -/
theorem oracle_branch_winner_flip :
    let branchA := advanceClocks [rooflineClocks 2 0 0 0] (rooflineClocks 0 0 0 8)
    let branchB := advanceClocks [rooflineClocks 5 0 0 0] (rooflineClocks 0 0 3 0)
    branchA.allT > branchB.allT ∧
      branchB.withoutCommT > branchA.withoutCommT ∧
      join2 branchA branchB = ⟨10, 8, 5, 5⟩ ∧
      finishBreakdown (join2 branchA branchB) = ⟨5, 0, 3, 2⟩ ∧
      finishBreakdown branchA ≠ finishBreakdown (join2 branchA branchB) := by
  decide

end MindIESimSpec.FourClock
