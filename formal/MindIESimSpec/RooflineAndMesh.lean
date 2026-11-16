/-
  MindIESimSpec.RooflineAndMesh
  Formal verification of Milestone M2 invariants:
    1. Critical-path roofline monotonicity:
       `0 ≤ Physical_Ideal_1x ≤ Target_Roofline_Eta ≤ Predicted`.
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

/-- Core M2 Theorem 1: the target roofline lies between the 1.0x ideal and the
    physical prediction whenever per-node and whole-device pipe bounds are
    monotone. -/
theorem ideal_le_target_le_predicted
    (wIdeal wTarget wPred : Nat → Int)
    (pipeIdeal pipeTarget tPred : Int)
    (g : DagExpr)
    (hNonneg : ∀ i, 0 ≤ wIdeal i)
    (hNode1  : ∀ i, wIdeal i ≤ wTarget i)
    (hNode2  : ∀ i, wTarget i ≤ wPred i)
    (hPipe1  : pipeIdeal ≤ pipeTarget)
    (hPipe2  : pipeIdeal ≤ tPred)
    (hPredCP : criticalPath wPred g ≤ tPred) :
    let ideal1x   := stepRooflineBound (criticalPath wIdeal g) pipeIdeal
    let targetEta := min tPred (stepRooflineBound (criticalPath wTarget g) pipeTarget)
    0 ≤ ideal1x ∧ ideal1x ≤ targetEta ∧ targetEta ≤ tPred := by
  have hcp0 : 0 ≤ criticalPath wIdeal g := criticalPath_nonneg wIdeal hNonneg g
  have hcp1 : criticalPath wIdeal g ≤ criticalPath wTarget g :=
    criticalPath_mono wIdeal wTarget hNode1 g
  have hcp2 : criticalPath wTarget g ≤ criticalPath wPred g :=
    criticalPath_mono wTarget wPred hNode2 g
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
