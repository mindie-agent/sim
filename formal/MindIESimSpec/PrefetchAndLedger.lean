/-
  MindIESimSpec.PrefetchAndLedger
  Pass 2.5 static weight prefetch clamping (`mindie_sim/prediction/weight_prefetch.py`)
  and half-open interval memory ledger (`mindie_sim/prediction/runtime/ledger.py`,
  `mindie_sim/memory_ledger_types.py`).
-/
namespace MindIESimSpec.PrefetchAndLedger

/-! ### Part 1: Pass 2.5 MLAPO Static Weight Prefetch Clamp over `Int` -/

/-- Bytes prefetched into a shared on-chip pool with signed headroom `cFree : Int`.
    Defined over `Int` so negative/zero pool headroom cannot be hidden by `Nat` truncation. -/
def prefetchBytes (w cFree : Int) : Int :=
  if cFree ≤ 0 then 0 else min w cFree

theorem prefetchBytes_nonneg {w : Int} (hw : 0 ≤ w) (cFree : Int) :
    0 ≤ prefetchBytes w cFree := by
  unfold prefetchBytes
  split <;> omega

theorem prefetchBytes_le_weight {w : Int} (hw : 0 ≤ w) (cFree : Int) :
    prefetchBytes w cFree ≤ w := by
  unfold prefetchBytes
  split <;> omega

theorem prefetchBytes_le_free {w cFree : Int} (_hw : 0 ≤ w) (hc : 0 ≤ cFree) :
    prefetchBytes w cFree ≤ cFree := by
  unfold prefetchBytes
  split <;> omega

/-- Exact byte conservation: remaining foreground HBM read plus prefetched bytes equals `w`. -/
theorem prefetchBytes_conserves (w cFree : Int) :
    (w - prefetchBytes w cFree) + prefetchBytes w cFree = w := by
  omega

theorem prefetchBytes_mono_weight {w₁ w₂ : Int}
    (_hw₁ : 0 ≤ w₁) (hw : w₁ ≤ w₂) (cFree : Int) :
    prefetchBytes w₁ cFree ≤ prefetchBytes w₂ cFree := by
  unfold prefetchBytes
  split <;> omega

theorem prefetchBytes_mono_free {w c₁ c₂ : Int}
    (hw : 0 ≤ w) (hc : c₁ ≤ c₂) :
    prefetchBytes w c₁ ≤ prefetchBytes w c₂ := by
  unfold prefetchBytes
  split <;> split <;> omega

/-! ### Part 2: Half-Open Interval Memory Ledger `[start, finish)` -/

/-- One half-open temporal or address-space allocation `[start, finish)` owning `bytes`. -/
structure Alloc where
  start  : Nat
  finish : Nat
  bytes  : Nat
  deriving DecidableEq, Repr

/-- Non-empty half-open interval validity (`start < finish`). -/
def Alloc.Valid (a : Alloc) : Prop :=
  a.start < a.finish

/-- Bytes owned by allocation `a` that are live at instant `t`. -/
def Alloc.activeAt (a : Alloc) (t : Nat) : Nat :=
  if a.start ≤ t ∧ t < a.finish then a.bytes else 0

/-- Total live bytes across a list of allocations at instant `t`. -/
def activeBytesAt (allocs : List Alloc) (t : Nat) : Nat :=
  match allocs with
  | [] => 0
  | a :: rest => a.activeAt t + activeBytesAt rest t

/-- Disjoint-lifetime allocations (`a1.finish ≤ a2.start`) never overlap at any instant `t`,
    so `activeBytesAt [a1, a2] t ≤ max a1.bytes a2.bytes` (zero double-charging of transients). -/
theorem disjoint_allocs_no_double_charge (a1 a2 : Alloc) (hdisj : a1.finish ≤ a2.start) (t : Nat) :
    activeBytesAt [a1, a2] t ≤ max a1.bytes a2.bytes := by
  simp only [activeBytesAt, Alloc.activeAt]
  split <;> split <;> omega

/-- Static + transient decomposition across arbitrary transient list `trs`:
    if `s` spans `[0, T)`, its liveness splits cleanly from `activeBytesAt trs t`. -/
theorem static_cons_transients_decomp (s : Alloc) (trs : List Alloc) (T t : Nat)
    (hs0 : s.start = 0) (hsT : s.finish = T) :
    activeBytesAt (s :: trs) t = (if t < T then s.bytes else 0) + activeBytesAt trs t := by
  simp only [activeBytesAt, Alloc.activeAt, hs0, hsT, Nat.zero_le, true_and]

/-- Static + transient peak coincidence: if `s` is live on `[0, T)` (`T > 0`) and `tr` is live
    on `[a, b)` with `a < b ≤ T`, then at `t = a` both are simultaneously live and sum exactly. -/
theorem static_plus_transient_at_start (s tr : Alloc) (T a b : Nat)
    (_hT : 0 < T) (hs0 : s.start = 0) (hsT : s.finish = T)
    (htra : tr.start = a) (htrb : tr.finish = b)
    (hab : a < b) (hbT : b ≤ T) :
    activeBytesAt [s, tr] a = s.bytes + tr.bytes := by
  rw [static_cons_transients_decomp s [tr] T a hs0 hsT]
  simp only [activeBytesAt, Alloc.activeAt, htra, htrb]
  have haT : a < T := Nat.lt_of_lt_of_le hab hbT
  split <;> split <;> omega

end MindIESimSpec.PrefetchAndLedger
