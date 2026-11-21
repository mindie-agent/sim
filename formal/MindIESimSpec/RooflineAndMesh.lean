/-
  MindIESimSpec.RooflineAndMesh
  Formal verification of pure physical roofline & direct UB mesh invariants:
    1. Critical-path roofline monotonicity (zero 0.80x target heuristic):
       `0 ≤ Physical_Ideal_1x ≤ Predicted`.
    2. Direct `K_W` point-to-point UB mesh scaling:
       per-rank aggregate egress capacity equals `(W - 1) * bPeer`.
-/
namespace MindIESimSpec.RooflineAndMesh

/-- Inductive series-parallel DAG expression over node indices. -/
inductive DagExpr where
  | empty : DagExpr
  | node  : Nat → DagExpr
  | seq   : DagExpr → DagExpr → DagExpr
  | par   : DagExpr → DagExpr → DagExpr
  deriving DecidableEq, Repr

/-- Critical-path length of `DagExpr` under node weight assignment `w`. -/
def criticalPath (w : Nat → Int) : DagExpr → Int
  | .empty    => 0
  | .node i   => w i
  | .seq a b  => criticalPath w a + criticalPath w b
  | .par a b  => max (criticalPath w a) (criticalPath w b)

theorem criticalPath_nonneg (w : Nat → Int) (hw : ∀ i, 0 ≤ w i) :
    ∀ g : DagExpr, 0 ≤ criticalPath w g := by
  intro g
  induction g with
  | empty => simp [criticalPath]
  | node i => simpa [criticalPath] using hw i
  | seq a b iha ihb =>
    dsimp [criticalPath]
    omega
  | par a b iha ihb =>
    dsimp [criticalPath]
    omega

theorem criticalPath_mono (w₁ w₂ : Nat → Int) (hle : ∀ i, w₁ i ≤ w₂ i) :
    ∀ g : DagExpr, criticalPath w₁ g ≤ criticalPath w₂ g := by
  intro g
  induction g with
  | empty => simp [criticalPath]
  | node i => simpa [criticalPath] using hle i
  | seq a b iha ihb =>
    dsimp [criticalPath]
    omega
  | par a b iha ihb =>
    dsimp [criticalPath]
    omega

/-- Combined DAG critical-path and whole-device pipe conservation bound. -/
def stepRooflineBound (cp pipeBound : Int) : Int :=
  max cp pipeBound

/-- Core physical roofline theorem: the 1.0x ideal roofline lower-bounds the
    predicted step latency whenever per-node ideal bounds are nonnegative and
    dominated by predicted weights and the whole-device pipe cut is dominated
    by the predicted span. -/
theorem physical_ideal_le_predicted
    (wIdeal wPred : Nat → Int)
    (pipeIdeal tPred : Int)
    (g : DagExpr)
    (hNonneg : ∀ i, 0 ≤ wIdeal i)
    (hNode   : ∀ i, wIdeal i ≤ wPred i)
    (hPipe   : pipeIdeal ≤ tPred)
    (hPredCP : criticalPath wPred g ≤ tPred) :
    let ideal1x := stepRooflineBound (criticalPath wIdeal g) pipeIdeal
    0 ≤ ideal1x ∧ ideal1x ≤ tPred := by
  have hcp0 : 0 ≤ criticalPath wIdeal g := criticalPath_nonneg wIdeal hNonneg g
  have hcp1 : criticalPath wIdeal g ≤ criticalPath wPred g :=
    criticalPath_mono wIdeal wPred hNode g
  dsimp [stepRooflineBound]
  omega

/-- Per-rank aggregate unidirectional egress bandwidth on a direct `K_W` mesh. -/
def kMeshPerRankBandwidth (w : Nat) (bPeer : Nat) : Nat :=
  (w - 1) * bPeer

/-- Total directed peer links in a `K_W` mesh equals `W * (W - 1)`. -/
def kMeshTotalDirectedLinks (w : Nat) : Nat :=
  w * (w - 1)

theorem kMesh_capacity_partition (w bPeer : Nat) :
    kMeshTotalDirectedLinks w * bPeer = w * kMeshPerRankBandwidth w bPeer := by
  dsimp [kMeshTotalDirectedLinks, kMeshPerRankBandwidth]
  exact Nat.mul_assoc w (w - 1) bPeer

theorem kMesh_w2_w4_w8_scaling (bPeer : Nat) :
    kMeshPerRankBandwidth 2 bPeer = 1 * bPeer ∧
    kMeshPerRankBandwidth 4 bPeer = 3 * bPeer ∧
    kMeshPerRankBandwidth 8 bPeer = 7 * bPeer := by
  dsimp [kMeshPerRankBandwidth]
  omega

end MindIESimSpec.RooflineAndMesh
