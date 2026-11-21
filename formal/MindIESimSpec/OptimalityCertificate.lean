/-
  MindIESimSpec.OptimalityCertificate
  Lean 4 Formal Verification of the 5-Level Performance Ladder & Dual Lower Bounds:
    1. Farkas LP Dual Soundness over Convex Tile/Mode Mixtures (`farkas_dual_le_any_feasible_schedule`)
    2. Direct-MAC Bilinear Epoch & Multi-Epoch Reload Bounds (`epoch_bilinear_pair_cap`,
       `capacity_reload_trace_bound`)
    3. Checked Dependent-Event DAG Critical Path Soundness (`max_plus_critical_path_bound`)
    4. Unified 5-Level Certified Ladder & Telescoping Gap (`certified_ladder_chain`)
  Toolchain: leanprover/lean4:v4.16.0 (Init core only; zero Mathlib, zero sorry/axiom).
-/
namespace MindIESimSpec.OptimalityCertificate

universe u v

/-- Local pairwise list relation (avoids external Batteries/Mathlib `List.Forall₂`). -/
inductive Rel2 {α : Type u} {β : Type v} (p : α → β → Prop) : List α → List β → Prop where
  | nil : Rel2 p [] []
  | cons {x : α} {y : β} {xs : List α} {ys : List β} :
      p x y → Rel2 p xs ys → Rel2 p (x :: xs) (y :: ys)

/-! ### Part 1: Integer Linear Algebra & Convex-Mixture Farkas LP Dual -/

def sumInt : List Int → Int
  | [] => 0
  | x :: xs => x + sumInt xs

def dotInt : List Int → List Int → Int
  | [], _ => 0
  | _, [] => 0
  | a :: as, b :: bs => a * b + dotInt as bs

def addVec : List Int → List Int → List Int
  | [], _ => []
  | _, [] => []
  | a :: as, b :: bs => (a + b) :: addVec as bs

def sumVecs (numR : Nat) : List (List Int) → List Int
  | [] => List.replicate numR 0
  | v :: vs => addVec v (sumVecs numR vs)

structure ResourceChoiceProgram where
  numResources : Nat
  globalWork   : List Int
  siteModes    : List (List (List Int))
  deriving DecidableEq, Repr

structure FarkasCertificate where
  scaleD       : Int
  lambdaNum    : List Int
  muNum        : List Int
  lowerBoundNs : Int
  deriving DecidableEq, Repr

def checkSiteModes (numR : Nat) (lam : List Int) (mu : Int) (modes : List (List Int)) : Bool :=
  match modes with
  | [] => false
  | ms => ms.all (fun m => decide (m.length = numR) && decide (mu ≤ dotInt lam m))

def checkAllSites (numR : Nat) (lam : List Int) : List Int → List (List (List Int)) → Bool
  | [], [] => true
  | mu :: mus, modes :: rest =>
      checkSiteModes numR lam mu modes && checkAllSites numR lam mus rest
  | _, _ => false

def checkFarkasDual (prog : ResourceChoiceProgram) (cert : FarkasCertificate) : Bool :=
  decide (0 < cert.scaleD) &&
  decide (prog.globalWork.length = prog.numResources) &&
  decide (cert.lambdaNum.length = prog.numResources) &&
  cert.lambdaNum.all (fun l => decide (0 ≤ l)) &&
  decide (sumInt cert.lambdaNum ≤ cert.scaleD) &&
  checkAllSites prog.numResources cert.lambdaNum cert.muNum prog.siteModes &&
  decide (cert.lowerBoundNs * cert.scaleD ≤ dotInt cert.lambdaNum prog.globalWork + sumInt cert.muNum)

/-- Realized R-vector `w` at one site dominates the convex hull of `modes` under any `lam ≥ 0`
    (covers both integral single-mode selection and fractional/multi-tile mode mixtures!). -/
def DominatesSiteHull (numR : Nat) (w : List Int) (modes : List (List Int)) : Prop :=
  w.length = numR ∧
  ∀ lam : List Int, lam.length = numR → (∀ l ∈ lam, 0 ≤ l) →
    ∃ m ∈ modes, m.length = numR ∧ dotInt lam m ≤ dotInt lam w

def FeasibleSchedule (prog : ResourceChoiceProgram) (siteLoads : List (List Int)) (wallNs : Int) : Prop :=
  0 ≤ wallNs ∧
  Rel2 (DominatesSiteHull prog.numResources) siteLoads prog.siteModes ∧
  Rel2 (· ≤ ·)
    (addVec prog.globalWork (sumVecs prog.numResources siteLoads))
    (List.replicate prog.numResources wallNs)

private theorem selectedLengths {n : Nat} {cs : List (List Int)} {os : List (List (List Int))}
    (h : Rel2 (DominatesSiteHull n) cs os) : ∀ w ∈ cs, w.length = n := by
  induction h with
  | nil => exact List.forall_mem_nil _
  | cons hw _ ih => exact List.forall_mem_cons.mpr ⟨hw.1, ih⟩

theorem dotInt_replicate_zero : ∀ lam : List Int, dotInt lam (List.replicate lam.length 0) = 0
  | [] => rfl
  | _ :: as => by
    simp only [List.length_cons, List.replicate_succ, dotInt, Int.mul_zero, Int.zero_add]
    exact dotInt_replicate_zero as

theorem dotInt_addVec : ∀ (lam xs ys : List Int),
    lam.length = xs.length → xs.length = ys.length →
    dotInt lam (addVec xs ys) = dotInt lam xs + dotInt lam ys
  | [], [], [], _, _ => rfl
  | [], _ :: _, _, h1, _ => by simp at h1
  | _ :: _, [], _, h1, _ => by simp at h1
  | _ :: _, _ :: _, [], _, h2 => by simp at h2
  | a :: as, x :: xs, y :: ys, h1, h2 => by
    simp only [List.length_cons, Nat.succ.injEq] at h1 h2
    simp only [addVec, dotInt]
    rw [Int.mul_add, dotInt_addVec as xs ys h1 h2]
    omega

theorem dotInt_mono_right : ∀ (lam xs ys : List Int),
    (∀ l ∈ lam, 0 ≤ l) → Rel2 (· ≤ ·) xs ys → dotInt lam xs ≤ dotInt lam ys
  | [], _, _, _, _ => by simp [dotInt]
  | _ :: _, [], [], _, Rel2.nil => by simp [dotInt]
  | l :: ls, x :: xs, y :: ys, hpos, Rel2.cons hxy hrest => by
    simp only [dotInt]
    have hl : 0 ≤ l := hpos l (List.Mem.head ls)
    have hmul : l * x ≤ l * y := Int.mul_le_mul_of_nonneg_left hxy hl
    have htail : dotInt ls xs ≤ dotInt ls ys :=
      dotInt_mono_right ls xs ys (fun z hz => hpos z (List.Mem.tail l hz)) hrest
    omega

theorem dotInt_replicate_const : ∀ (lam : List Int) (t : Int),
    dotInt lam (List.replicate lam.length t) = sumInt lam * t
  | [], _ => by simp [dotInt, sumInt]
  | l :: ls, t => by
    simp only [List.length_cons, List.replicate_succ, dotInt, sumInt, Int.add_mul]
    rw [dotInt_replicate_const ls t]

theorem addVec_length : ∀ (xs ys : List Int),
    xs.length = ys.length → (addVec xs ys).length = xs.length
  | [], [], _ => rfl
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | _ :: xs, _ :: ys, h => by
    simp only [List.length_cons, Nat.succ.injEq] at h
    simp [addVec, addVec_length xs ys h]

theorem sumVecs_length (numR : Nat) : ∀ (vs : List (List Int)),
    (∀ v ∈ vs, v.length = numR) → (sumVecs numR vs).length = numR
  | [], _ => by simp [sumVecs]
  | v :: vs, h => by
    simp only [sumVecs]
    have hv : v.length = numR := h v (List.Mem.head vs)
    have hvs : (sumVecs numR vs).length = numR :=
      sumVecs_length numR vs (fun u hu => h u (List.Mem.tail v hu))
    rw [addVec_length v (sumVecs numR vs) (hv.trans hvs.symm), hv]

theorem sumMu_le_dot_sumVecs {numR : Nat} {lam : List Int}
    (hlamLen : lam.length = numR) (hlamPos : ∀ l ∈ lam, 0 ≤ l)
    {siteLoads : List (List Int)} {siteModes : List (List (List Int))}
    (hrel : Rel2 (DominatesSiteHull numR) siteLoads siteModes) :
    ∀ mus : List Int,
      checkAllSites numR lam mus siteModes = true →
      sumInt mus ≤ dotInt lam (sumVecs numR siteLoads) := by
  induction hrel with
  | nil =>
    intro mus hchk
    cases mus with
    | nil =>
      simp only [sumInt, sumVecs]
      rw [← hlamLen, dotInt_replicate_zero]
      exact Int.le_refl 0
    | cons _ _ => simp [checkAllSites] at hchk
  | @cons w modes ws restModes hw htail ih =>
    intro mus hchk
    cases mus with
    | nil => simp [checkAllSites] at hchk
    | cons mu musTail =>
      simp only [checkAllSites, Bool.and_eq_true] at hchk
      rcases hchk with ⟨hsite, hrestChk⟩
      rcases hw.2 lam hlamLen hlamPos with ⟨m, hmMem, _, hmDot⟩
      have hmuLeM : mu ≤ dotInt lam m := by
        cases modes with
        | nil => cases hmMem
        | cons _ _ =>
          simp only [checkSiteModes, List.all_eq_true, Bool.and_eq_true, decide_eq_true_eq] at hsite
          exact (hsite m hmMem).2
      have hih : sumInt musTail ≤ dotInt lam (sumVecs numR ws) := ih musTail hrestChk
      have hsumLen : (sumVecs numR ws).length = numR :=
        sumVecs_length numR ws (selectedLengths htail)
      simp only [sumInt, sumVecs]
      rw [dotInt_addVec lam w (sumVecs numR ws) (hlamLen.trans hw.1.symm) (hw.1.trans hsumLen.symm)]
      omega

/-- Core Theorem 1: Any verified Farkas certificate lower-bounds every feasible
    single-mode or convex tile-mixed schedule. -/
theorem farkas_dual_le_any_feasible_schedule
    (prog : ResourceChoiceProgram) (cert : FarkasCertificate)
    (siteLoads : List (List Int)) (wallNs : Int)
    (hcert : checkFarkasDual prog cert = true)
    (hfeas : FeasibleSchedule prog siteLoads wallNs) :
    cert.lowerBoundNs ≤ wallNs := by
  simp only [checkFarkasDual, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at hcert
  rcases hcert with ⟨⟨⟨⟨⟨⟨hdPos, hgLen⟩, hlLen⟩, hlNonneg⟩, hlSum⟩, hsites⟩, hobj⟩
  rcases hfeas with ⟨hwNonneg, hdom, hloads⟩
  have hsumLen : (sumVecs prog.numResources siteLoads).length = prog.numResources :=
    sumVecs_length prog.numResources siteLoads (selectedLengths hdom)
  have hmuBound : sumInt cert.muNum ≤ dotInt cert.lambdaNum (sumVecs prog.numResources siteLoads) :=
    sumMu_le_dot_sumVecs hlLen hlNonneg hdom cert.muNum hsites
  have hsplit : dotInt cert.lambdaNum (addVec prog.globalWork (sumVecs prog.numResources siteLoads)) =
      dotInt cert.lambdaNum prog.globalWork + dotInt cert.lambdaNum (sumVecs prog.numResources siteLoads) :=
    dotInt_addVec cert.lambdaNum prog.globalWork (sumVecs prog.numResources siteLoads)
      (hlLen.trans hgLen.symm) (hgLen.trans hsumLen.symm)
  have hmono : dotInt cert.lambdaNum (addVec prog.globalWork (sumVecs prog.numResources siteLoads)) ≤
      dotInt cert.lambdaNum (List.replicate prog.numResources wallNs) :=
    dotInt_mono_right cert.lambdaNum _ _ hlNonneg hloads
  have hrep : dotInt cert.lambdaNum (List.replicate prog.numResources wallNs) =
      sumInt cert.lambdaNum * wallNs := by
    rw [← hlLen, dotInt_replicate_const]
  have hcap : sumInt cert.lambdaNum * wallNs ≤ cert.scaleD * wallNs :=
    Int.mul_le_mul_of_nonneg_right hlSum hwNonneg
  have hmulBound : cert.lowerBoundNs * cert.scaleD ≤ wallNs * cert.scaleD := by
    rw [Int.mul_comm wallNs cert.scaleD]
    omega
  by_cases hle : cert.lowerBoundNs ≤ wallNs
  · exact hle
  · have hlt : wallNs < cert.lowerBoundNs := by omega
    have hbad := Int.mul_lt_mul_of_pos_right hlt hdPos
    omega

/-! ### Part 2: Direct-MAC Bilinear Epoch & Multi-Epoch Trace Capacity Bound -/

theorem epoch_bilinear_pair_cap
    (aElem bElem bitsA bitsB mBits : Int)
    (ha : 0 ≤ aElem) (hb : 0 ≤ bElem) (hba : 0 ≤ bitsA) (hbb : 0 ≤ bitsB)
    (hcap : bitsA * aElem + bitsB * bElem ≤ mBits) :
    4 * bitsA * bitsB * (aElem * bElem) ≤ mBits * mBits := by
  let u : Int := bitsA * aElem
  let v : Int := bitsB * bElem
  change u + v ≤ mBits at hcap
  have h0 : 0 ≤ u + v := by
    have hu : 0 ≤ u := Int.mul_nonneg hba ha
    have hv : 0 ≤ v := Int.mul_nonneg hbb hb
    omega
  have hs : 0 ≤ (u - v) * (u - v) := by
    by_cases h : 0 ≤ u - v
    · exact Int.mul_nonneg h h
    · have hn : u - v ≤ 0 := by omega
      exact Int.mul_nonneg_of_nonpos_of_nonpos hn hn
  have hfour : 4 * (u * v) ≤ (u + v) * (u + v) := by
    have hc : v * u = u * v := Int.mul_comm v u
    simp only [Int.sub_mul, Int.mul_sub] at hs
    simp only [Int.add_mul, Int.mul_add]
    omega
  calc
    4 * bitsA * bitsB * (aElem * bElem) = 4 * (u * v) := by
      dsimp only [u, v]
      ac_rfl
    _ ≤ (u + v) * (u + v) := hfour
    _ ≤ mBits * mBits := Int.mul_le_mul hcap hcap h0 (Int.le_trans h0 hcap)

structure EpochAccount where
  completePairs : Int
  splitOutPairs : Int
  epochPairCap  : Int
  deriving DecidableEq, Repr

def sumEpochCaps : List EpochAccount → Int
  | [] => 0
  | e :: es => e.epochPairCap + sumEpochCaps es

def ValidEpochTrace : List EpochAccount → Int → Prop
  | [], totalPairs => totalPairs ≤ 0
  | e :: es, totalPairs =>
      e.completePairs + e.splitOutPairs ≤ e.epochPairCap ∧
      ValidEpochTrace es (totalPairs - (e.completePairs + e.splitOutPairs))

/-- Core Theorem 2: Any multi-epoch trace whose per-epoch completed + split-survivor pairs
    are bounded by `epochPairCap` satisfies `totalPairs ≤ sumEpochCaps`. -/
theorem capacity_reload_trace_bound :
    ∀ (trace : List EpochAccount) (totalPairs : Int),
      ValidEpochTrace trace totalPairs → totalPairs ≤ sumEpochCaps trace
  | [], totalPairs, hv => by simp [ValidEpochTrace, sumEpochCaps] at *; omega
  | e :: es, totalPairs, ⟨hcap, hrest⟩ => by
    have ih := capacity_reload_trace_bound es (totalPairs - (e.completePairs + e.splitOutPairs)) hrest
    simp only [sumEpochCaps]
    omega

/-! ### Part 3: Checked Dependent-Event DAG Critical Path -/

structure DagEdge where
  src    : Nat
  dst    : Nat
  wireNs : Int
  deriving DecidableEq, Repr

structure EventDagSpec where
  nodeQuantumNs : List Int
  edges         : List DagEdge
  deriving DecidableEq, Repr

private def peelDag (outgoing : Array (List Nat)) : Nat → List Nat → Array Nat → Bool
  | 0, _, _ => true
  | _ + 1, [], _ => false
  | fuel + 1, src :: ready, indegree =>
      let (ready, indegree) := outgoing[src]!.foldl (fun (todo, counts) dst =>
        let remaining := counts[dst]! - 1
        (if remaining = 0 then dst :: todo else todo, counts.set! dst remaining)) (ready, indegree)
      peelDag outgoing fuel ready indegree

/-- Check graph geometry, then peel one zero-indegree node per step. -/
def checkDagAcyclic (dag : EventDagSpec) : Bool := Id.run do
  let n := dag.nodeQuantumNs.length
  if n = 0 || dag.nodeQuantumNs.any (fun q => q < 0) then return false
  let mut outgoing : Array (List Nat) := Array.mkArray n []
  let mut indegree := Array.mkArray n 0
  for e in dag.edges do
    if e.src ≥ n || e.dst ≥ n || e.src = e.dst || e.wireNs < 0 then return false
    outgoing := outgoing.modify e.src (e.dst :: ·)
    indegree := indegree.modify e.dst (· + 1)
  let ready := (List.range n).filter (fun i => indegree[i]! = 0)
  return peelDag outgoing n ready indegree

def lookupFloor (xs : List Int) (idx : Nat) : Option Int :=
  xs[idx]?

def lookupEdge (es : List DagEdge) (idx : Nat) : Option DagEdge :=
  es[idx]?

def nextHeadMatches (dag : EventDagSpec) (dst : Nat) (rest : List Nat) (sinkId : Nat) : Bool :=
  match rest with
  | [] => decide (dst = sinkId)
  | nextIdx :: _ =>
      match lookupEdge dag.edges nextIdx with
      | some eNext => decide (dst = eNext.src)
      | none => false

private def evalPath (dag : EventDagSpec) (sinkId : Nat) : List Nat → List Nat → Option Int
  | [], visited =>
      match lookupFloor dag.nodeQuantumNs sinkId with
      | some q => if !visited.contains sinkId && decide (0 ≤ q) then some q else none
      | none => none
  | eIdx :: rest, visited =>
      match lookupEdge dag.edges eIdx with
      | none => none
      | some e =>
        if !visited.contains e.src && decide (e.src ≠ e.dst) &&
           decide (0 ≤ e.wireNs) && nextHeadMatches dag e.dst rest sinkId then
          match lookupFloor dag.nodeQuantumNs e.src, evalPath dag sinkId rest (e.src :: visited) with
          | some qSrc, some tailNs => if 0 ≤ qSrc then some (qSrc + e.wireNs + tailNs) else none
          | _, _ => none
        else none

/-- Recompute the critical-path floor for a connected, nonrepeating path. -/
def evalCheckedPath (dag : EventDagSpec) (edgeIdxs : List Nat) (sinkId : Nat) : Option Int :=
  evalPath dag sinkId edgeIdxs []

def ValidDagSchedule (dag : EventDagSpec) (startNs : Nat → Int) (wallNs : Int) : Prop :=
  (∀ i q, lookupFloor dag.nodeQuantumNs i = some q → 0 ≤ startNs i ∧ startNs i + q ≤ wallNs) ∧
  (∀ idx e qSrc, lookupEdge dag.edges idx = some e →
    lookupFloor dag.nodeQuantumNs e.src = some qSrc →
    startNs e.src + qSrc + e.wireNs ≤ startNs e.dst)

def pathHeadNode (dag : EventDagSpec) (edgeIdxs : List Nat) (sinkId : Nat) : Nat :=
  match edgeIdxs with
  | [] => sinkId
  | eIdx :: _ =>
      match lookupEdge dag.edges eIdx with
      | some e => e.src
      | none => sinkId

private theorem evalPath_bound
    (dag : EventDagSpec) (startNs : Nat → Int) (wallNs : Int) (sinkId : Nat)
    (hsched : ValidDagSchedule dag startNs wallNs) :
    ∀ (edgeIdxs visited : List Nat) (floorNs : Int),
      evalPath dag sinkId edgeIdxs visited = some floorNs →
      0 ≤ startNs (pathHeadNode dag edgeIdxs sinkId) ∧
      startNs (pathHeadNode dag edgeIdxs sinkId) + floorNs ≤ wallNs
  | [], _, floorNs, heval => by
    simp only [evalPath] at heval
    cases hq : lookupFloor dag.nodeQuantumNs sinkId with
    | none => simp [hq] at heval
    | some q =>
      simp only [hq] at heval
      split at heval
      · injection heval with hEq
        subst hEq
        exact hsched.1 sinkId q hq
      · contradiction
  | eIdx :: rest, visited, floorNs, heval => by
    simp only [evalPath] at heval
    cases he : lookupEdge dag.edges eIdx with
    | none => simp [he] at heval
    | some e =>
      simp only [he, Bool.and_eq_true, decide_eq_true_eq] at heval
      split at heval
      · rename_i hcond
        have hlink := hcond.2
        cases hq : lookupFloor dag.nodeQuantumNs e.src with
        | none => simp [hq] at heval
        | some qSrc =>
          cases htl : evalPath dag sinkId rest (e.src :: visited) with
          | none => simp [hq, htl] at heval
          | some tailNs =>
            simp only [hq, htl] at heval
            split at heval
            · injection heval with hEq
              subst hEq
              have ⟨h0Src, _⟩ := hsched.1 e.src qSrc hq
              have hedge := hsched.2 eIdx e qSrc he hq
              have ⟨_, ihBound⟩ := evalPath_bound dag startNs wallNs sinkId hsched rest (e.src :: visited) tailNs htl
              have hheadEq : pathHeadNode dag rest sinkId = e.dst := by
                cases rest with
                | nil =>
                  simp only [nextHeadMatches, decide_eq_true_eq] at hlink
                  simp [pathHeadNode, hlink]
                | cons nextIdx _ =>
                  simp only [nextHeadMatches] at hlink
                  cases hnext : lookupEdge dag.edges nextIdx with
                  | none => simp [hnext] at hlink
                  | some eNext =>
                    simp only [hnext, decide_eq_true_eq] at hlink
                    simp [pathHeadNode, hnext, hlink]
              rw [hheadEq] at ihBound
              simp only [pathHeadNode, he]
              omega
            · contradiction
      · contradiction

/-- Core Theorem 3: Every checked path bounds the wall time of a valid schedule. -/
theorem max_plus_critical_path_bound
    (dag : EventDagSpec) (startNs : Nat → Int) (wallNs : Int)
    (hsched : ValidDagSchedule dag startNs wallNs)
    (edgeIdxs : List Nat) (sinkId : Nat) (floorNs : Int)
    (heval : evalCheckedPath dag edgeIdxs sinkId = some floorNs) :
    0 ≤ startNs (pathHeadNode dag edgeIdxs sinkId) ∧
    startNs (pathHeadNode dag edgeIdxs sinkId) + floorNs ≤ wallNs :=
  evalPath_bound dag startNs wallNs sinkId hsched edgeIdxs [] floorNs heval

/-! ### Part 4: Unified 5-Level Certified Performance Ladder -/

structure OptimalityCertificate where
  farkas      : FarkasCertificate
  pathEdges   : List Nat
  sinkNodeId  : Nat
  lStarLeanNs : Int
  deriving DecidableEq, Repr

def checkOptimalityCertificate
    (prog : ResourceChoiceProgram) (dag : EventDagSpec) (cert : OptimalityCertificate) : Bool :=
  checkDagAcyclic dag &&
    match evalCheckedPath dag cert.pathEdges cert.sinkNodeId with
    | none => false
    | some pathNs =>
        checkFarkasDual prog cert.farkas &&
        decide (0 ≤ cert.lStarLeanNs) &&
        decide (cert.lStarLeanNs = max cert.farkas.lowerBoundNs pathNs)

structure FiveLevelLadder where
  lStarLean      : Int
  aStarDsl       : Int
  aStarFramework : Int
  mMainline      : Int
  uUnmerge       : Int
  deriving DecidableEq, Repr

/-- Core Theorem 4: Verified `OptimalityCertificate` + constructive seed inclusion proves
    `0 ≤ L*_Lean ≤ A*_DSL ≤ A*_Framework ≤ min(M_Mainline, U_Unmerge)` and telescoping gap conservation. -/
theorem certified_ladder_chain
    (prog : ResourceChoiceProgram) (dag : EventDagSpec) (cert : OptimalityCertificate)
    (dslLoads : List (List Int)) (startDsl : Nat → Int) (ladder : FiveLevelLadder)
    (hchk : checkOptimalityCertificate prog dag cert = true)
    (hlStar : ladder.lStarLean = cert.lStarLeanNs)
    (hfeasDsl : FeasibleSchedule prog dslLoads ladder.aStarDsl)
    (hdagDsl : ValidDagSchedule dag startDsl ladder.aStarDsl)
    (hDslLeFw : ladder.aStarDsl ≤ ladder.aStarFramework)
    (hFwLeSeed : ladder.aStarFramework ≤ min ladder.mMainline ladder.uUnmerge) :
    0 ≤ ladder.lStarLean ∧
    ladder.lStarLean ≤ ladder.aStarDsl ∧
    ladder.aStarDsl ≤ ladder.aStarFramework ∧
    ladder.aStarFramework ≤ min ladder.mMainline ladder.uUnmerge ∧
    0 ≤ ladder.aStarDsl - ladder.lStarLean ∧
    (ladder.uUnmerge - ladder.lStarLean) =
      (ladder.uUnmerge - ladder.mMainline) +
      (ladder.mMainline - ladder.aStarFramework) +
      (ladder.aStarFramework - ladder.aStarDsl) +
      (ladder.aStarDsl - ladder.lStarLean) := by
  simp only [checkOptimalityCertificate, Bool.and_eq_true] at hchk
  have hpath := hchk.2
  cases hp : evalCheckedPath dag cert.pathEdges cert.sinkNodeId with
  | none => simp [hp] at hpath
  | some pathNs =>
    simp only [hp, Bool.and_eq_true, decide_eq_true_eq] at hpath
    rcases hpath with ⟨⟨hfChk, hlPos⟩, hlEq⟩
    have hfBound : cert.farkas.lowerBoundNs ≤ ladder.aStarDsl :=
      farkas_dual_le_any_feasible_schedule prog cert.farkas dslLoads ladder.aStarDsl hfChk hfeasDsl
    have ⟨h0Head, hpSum⟩ :=
      max_plus_critical_path_bound dag startDsl ladder.aStarDsl hdagDsl cert.pathEdges cert.sinkNodeId pathNs hp
    omega

end MindIESimSpec.OptimalityCertificate
