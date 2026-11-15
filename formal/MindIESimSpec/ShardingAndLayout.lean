/-
  MindIESimSpec.ShardingAndLayout
  Subgraph local TP, EPLB redundant experts, and Ascend FRACTAL_NZ physical layout algebra
  (`mindie_sim/catalog/weight_terms.py`, `mindie_sim/kv_layout_physics.py`,
   ASTRA_ISSUE34_SPEC.md §6 & §9).
-/
namespace MindIESimSpec.ShardingAndLayout

/-! ### Part 1: Ceiling Division Algebra (`ceilDiv`) -/

/-- Exact integer ceiling division matching `mindie_sim.catalog.weight_terms._ceil_div`. -/
def ceilDiv (a b : Nat) : Nat :=
  (a + b - 1) / b

/-- Sharding coverage: `ceilDiv a b * b` never drops logical rows, columns, or experts. -/
theorem ceilDiv_mul_ge (a : Nat) {b : Nat} (hb : 0 < b) :
    a ≤ ceilDiv a b * b := by
  unfold ceilDiv
  have hdiv : b * ((a + b - 1) / b) + (a + b - 1) % b = a + b - 1 :=
    Nat.div_add_mod (a + b - 1) b
  have hmod : (a + b - 1) % b < b := Nat.mod_lt (a + b - 1) hb
  have hcomm : ((a + b - 1) / b) * b = b * ((a + b - 1) / b) :=
    Nat.mul_comm _ b
  omega

/-- Strict padding upper bound: padding added by `ceilDiv` is strictly less than one shard/tile `b`. -/
theorem ceilDiv_mul_lt (a : Nat) {b : Nat} (hb : 0 < b) :
    ceilDiv a b * b < a + b := by
  unfold ceilDiv
  have hdiv : b * ((a + b - 1) / b) + (a + b - 1) % b = a + b - 1 :=
    Nat.div_add_mod (a + b - 1) b
  have hcomm : ((a + b - 1) / b) * b = b * ((a + b - 1) / b) :=
    Nat.mul_comm _ b
  omega

/-- Non-empty shard positivity: any positive dimension has positive sharded extent. -/
theorem ceilDiv_pos {a b : Nat} (ha : 0 < a) (hb : 0 < b) :
    0 < ceilDiv a b := by
  have hcov := ceilDiv_mul_ge a hb
  cases hq : ceilDiv a b with
  | zero =>
    rw [hq, Nat.zero_mul] at hcov
    omega
  | succ k =>
    omega

/-- Exact divisibility: when `b ∣ a`, `ceilDiv a b * b = a` with zero tile/shard padding. -/
theorem ceilDiv_mul_eq_of_dvd {a b : Nat} (hb : 0 < b) (hdvd : b ∣ a) :
    ceilDiv a b * b = a := by
  rcases hdvd with ⟨k, rfl⟩
  have hge : b * k ≤ ceilDiv (b * k) b * b := ceilDiv_mul_ge (b * k) hb
  have hlt : ceilDiv (b * k) b * b < b * k + b := ceilDiv_mul_lt (b * k) hb
  have hge' : k * b ≤ ceilDiv (b * k) b * b := by
    rw [Nat.mul_comm k b]; exact hge
  have hlt' : ceilDiv (b * k) b * b < (k + 1) * b := by
    calc ceilDiv (b * k) b * b < b * k + b := hlt
    _ = k * b + 1 * b := by rw [Nat.mul_comm b k, Nat.one_mul]
    _ = (k + 1) * b := by rw [← Nat.add_mul]
  have hk_le : k ≤ ceilDiv (b * k) b :=
    Nat.le_of_not_lt fun hlt_q => by
      have hmul : ceilDiv (b * k) b * b < k * b := Nat.mul_lt_mul_of_pos_right hlt_q hb
      omega
  have hq_lt : ceilDiv (b * k) b < k + 1 :=
    Nat.lt_of_not_le fun hge_q => by
      have hmul : (k + 1) * b ≤ ceilDiv (b * k) b * b := Nat.mul_le_mul_right b hge_q
      omega
  have hq_eq : ceilDiv (b * k) b = k := by omega
  rw [hq_eq, Nat.mul_comm k b]

/-! ### Part 2: Ascend `FRACTAL_NZ` Physical Tile Layout -/

/-- Physical bytes of a 2D matrix `(d, n)` formatted as Ascend `FRACTAL_NZ`
    with inner tile `c0` (e.g. `16` for BF16/FP16, `32` for INT8/FP8) and outer 16-row cube tile. -/
def fractalNzBytes (d n c0 elem : Nat) : Nat :=
  ceilDiv d c0 * ceilDiv n 16 * 16 * c0 * elem

private theorem fractalNzBytes_reassoc (d n c0 elem : Nat) :
    fractalNzBytes d n c0 elem = (ceilDiv d c0 * c0) * (ceilDiv n 16 * 16) * elem := by
  unfold fractalNzBytes
  calc ceilDiv d c0 * ceilDiv n 16 * 16 * c0 * elem
    _ = ceilDiv d c0 * (ceilDiv n 16 * 16) * c0 * elem := by rw [Nat.mul_assoc (ceilDiv d c0)]
    _ = ceilDiv d c0 * ((ceilDiv n 16 * 16) * c0) * elem := by rw [Nat.mul_assoc (ceilDiv d c0)]
    _ = ceilDiv d c0 * (c0 * (ceilDiv n 16 * 16)) * elem := by rw [Nat.mul_comm (ceilDiv n 16 * 16) c0]
    _ = (ceilDiv d c0 * c0) * (ceilDiv n 16 * 16) * elem := by rw [← Nat.mul_assoc (ceilDiv d c0) c0]

/-- `FRACTAL_NZ` never under-allocates logical tensor bytes `d * n * elem`. -/
theorem fractalNzBytes_ge_logical (d n : Nat) {c0 elem : Nat}
    (hc0 : 0 < c0) (_helem : 0 < elem) :
    d * n * elem ≤ fractalNzBytes d n c0 elem := by
  rw [fractalNzBytes_reassoc]
  have hd : d ≤ ceilDiv d c0 * c0 := ceilDiv_mul_ge d hc0
  have hn : n ≤ ceilDiv n 16 * 16 := ceilDiv_mul_ge n (by decide)
  have hdn : d * n ≤ (ceilDiv d c0 * c0) * (ceilDiv n 16 * 16) :=
    Nat.mul_le_mul hd hn
  exact Nat.mul_le_mul_right elem hdn

/-- Exact tile divisibility (`c0 ∣ d` and `16 ∣ n`) yields zero `FRACTAL_NZ` layout padding. -/
theorem fractalNzBytes_eq_logical_of_dvd (d n : Nat) {c0 elem : Nat}
    (hc0 : 0 < c0) (_helem : 0 < elem) (hd : c0 ∣ d) (hn : 16 ∣ n) :
    fractalNzBytes d n c0 elem = d * n * elem := by
  rw [fractalNzBytes_reassoc]
  rw [ceilDiv_mul_eq_of_dvd hc0 hd, ceilDiv_mul_eq_of_dvd (by decide) hn]

/-! ### Part 3: Frontend vs Static-Weight-Ledger Isomorphism (Subgraph TP & EPLB Redundant Slots) -/

/-- Partition axis for local dense-subgraph TP (`o_proj_tp`, `linear_proj_tp`, `lm_head_tp`). -/
inductive SubgraphPartition
  | rowShard
  | colShard
  deriving DecidableEq, Repr

/-- Canonical local matrix shape under subgraph `localTp ≥ 1` (`ASTRA_ISSUE34_SPEC.md` §6.1). -/
def placedMatrixShape (rows cols localTp : Nat) (part : SubgraphPartition) : Nat × Nat :=
  match part with
  | .rowShard => (rows, ceilDiv cols localTp)
  | .colShard => (ceilDiv rows localTp, cols)

/-- Canonical physical shard layout operator shared by static weight ledger and frontend cold read. -/
def shardLayoutBytes (shape : Nat × Nat) (c0 elem : Nat) : Nat :=
  fractalNzBytes shape.1 shape.2 c0 elem

/-- Per-rank static weight ledger bytes for a local-TP subgraph matrix. -/
def staticLedgerSubgraphBytes (rows cols localTp : Nat) (part : SubgraphPartition)
    (c0 elem : Nat) : Nat :=
  shardLayoutBytes (placedMatrixShape rows cols localTp part) c0 elem

/-- Frontend single-token cold weight bytes parameterized by a layout operator `frontendOp`. -/
def frontendColdSubgraphBytes (rows cols localTp : Nat) (part : SubgraphPartition)
    (frontendOp : Nat × Nat → Nat) : Nat :=
  frontendOp (placedMatrixShape rows cols localTp part)

/-- Isomorphism theorem: per-rank static weight bytes equal Frontend full-layer single-token
    cold weight bytes iff Frontend applies an operator agreeing with `shardLayoutBytes` on the placed shape. -/
theorem staticLedger_eq_frontendCold_iff (rows cols localTp : Nat) (part : SubgraphPartition)
    (c0 elem : Nat) (frontendOp : Nat × Nat → Nat) :
    staticLedgerSubgraphBytes rows cols localTp part c0 elem =
      frontendColdSubgraphBytes rows cols localTp part frontendOp ↔
    frontendOp (placedMatrixShape rows cols localTp part) =
      shardLayoutBytes (placedMatrixShape rows cols localTp part) c0 elem := by
  unfold staticLedgerSubgraphBytes frontendColdSubgraphBytes
  constructor <;> intro h <;> exact h.symm

/-- Direct corollary: sharing `shardLayoutBytes` gives exact Frontend/Ledger byte identity. -/
theorem staticLedger_eq_frontendCold_canonical (rows cols localTp : Nat)
    (part : SubgraphPartition) (c0 elem : Nat) :
    staticLedgerSubgraphBytes rows cols localTp part c0 elem =
      frontendColdSubgraphBytes rows cols localTp part (fun s => shardLayoutBytes s c0 elem) :=
  rfl

/-- Physical routed-expert slots per EP rank (`E_phys = ceilDiv E_routed ep + redundantPerRank`). -/
def physicalExpertSlots (eRouted ep redundantPerRank : Nat) : Nat :=
  ceilDiv eRouted ep + redundantPerRank

/-- Resident routed-expert bytes per rank from physical slot count and per-slot bytes. -/
def expertBytes (eRouted ep redundantPerRank perSlotBytes : Nat) : Nat :=
  physicalExpertSlots eRouted ep redundantPerRank * perSlotBytes

theorem expertBytes_decomp (eRouted ep redundantPerRank perSlotBytes : Nat) :
    expertBytes eRouted ep redundantPerRank perSlotBytes =
      expertBytes eRouted ep 0 perSlotBytes + redundantPerRank * perSlotBytes := by
  unfold expertBytes physicalExpertSlots
  rw [Nat.add_zero, Nat.add_mul]

/-- Positive `redundantPerRank` strictly increases per-rank resident expert bytes (`ASTRA_ISSUE34_SPEC.md` §9.1). -/
theorem expertBytes_strict_gt_zero (eRouted ep : Nat) {redundantPerRank perSlotBytes : Nat}
    (hr : 0 < redundantPerRank) (hw : 0 < perSlotBytes) :
    expertBytes eRouted ep redundantPerRank perSlotBytes >
      expertBytes eRouted ep 0 perSlotBytes := by
  rw [expertBytes_decomp]
  have hmul : 0 < redundantPerRank * perSlotBytes := Nat.mul_pos hr hw
  omega

/-- EPLB Frontend saturated single-pass cold read equals static expert ledger iff both use `shardLayoutBytes`. -/
theorem expertLedger_eq_frontendSaturated_iff (eRouted ep redundantPerRank : Nat)
    (slotShape : Nat × Nat) (c0 elem : Nat) (frontendSlotBytes : Nat)
    (hslots : 0 < physicalExpertSlots eRouted ep redundantPerRank) :
    expertBytes eRouted ep redundantPerRank (shardLayoutBytes slotShape c0 elem) =
      physicalExpertSlots eRouted ep redundantPerRank * frontendSlotBytes ↔
    frontendSlotBytes = shardLayoutBytes slotShape c0 elem := by
  unfold expertBytes
  constructor
  · intro h
    exact Nat.eq_of_mul_eq_mul_left hslots h.symm
  · intro h
    rw [h]

end MindIESimSpec.ShardingAndLayout
