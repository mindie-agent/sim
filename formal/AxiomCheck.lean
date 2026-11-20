/-
  AxiomCheck: Compile-time `#print axioms` audit for `MindIESimSpec`.
  `#guard_msgs` fails `lake build` if any theorem's axiom footprint deviates from
  standard Lean 4 foundations (`[]`, `[propext, Quot.sound]`, or
  `[propext, Classical.choice, Quot.sound]`).
-/
import MindIESimSpec

/-- info: 'MindIESimSpec.FourClock.rooflineClocks_ordered' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms MindIESimSpec.FourClock.rooflineClocks_ordered

/-- info: 'MindIESimSpec.FourClock.joinClocks_perm' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms MindIESimSpec.FourClock.joinClocks_perm

/-- info: 'MindIESimSpec.FourClock.advanceClocks_ordered' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms MindIESimSpec.FourClock.advanceClocks_ordered

/--
info: 'MindIESimSpec.FourClock.finishBreakdown_nonneg_and_conserves' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
-/
#guard_msgs in
#print axioms MindIESimSpec.FourClock.finishBreakdown_nonneg_and_conserves

/-- info: 'MindIESimSpec.FourClock.doubleBuffer_hidden_memory_zero_exposed' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms MindIESimSpec.FourClock.doubleBuffer_hidden_memory_zero_exposed

/-- info: 'MindIESimSpec.FourClock.oracle_branch_winner_flip' does not depend on any axioms -/
#guard_msgs in
#print axioms MindIESimSpec.FourClock.oracle_branch_winner_flip

/-- info: 'MindIESimSpec.PrefetchAndLedger.prefetchBytes_conserves' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms MindIESimSpec.PrefetchAndLedger.prefetchBytes_conserves

/-- info: 'MindIESimSpec.PrefetchAndLedger.disjoint_allocs_no_double_charge' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms MindIESimSpec.PrefetchAndLedger.disjoint_allocs_no_double_charge

/--
info: 'MindIESimSpec.PrefetchAndLedger.static_plus_transient_at_start' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
-/
#guard_msgs in
#print axioms MindIESimSpec.PrefetchAndLedger.static_plus_transient_at_start

/-- info: 'MindIESimSpec.ShardingAndLayout.ceilDiv_mul_ge' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms MindIESimSpec.ShardingAndLayout.ceilDiv_mul_ge

/-- info: 'MindIESimSpec.ShardingAndLayout.ceilDiv_mul_eq_of_dvd' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms MindIESimSpec.ShardingAndLayout.ceilDiv_mul_eq_of_dvd

/-- info: 'MindIESimSpec.ShardingAndLayout.fractalNzBytes_ge_logical' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms MindIESimSpec.ShardingAndLayout.fractalNzBytes_ge_logical

/-- info: 'MindIESimSpec.ShardingAndLayout.fractalNzBytes_eq_logical_of_dvd' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms MindIESimSpec.ShardingAndLayout.fractalNzBytes_eq_logical_of_dvd

/-- info: 'MindIESimSpec.ShardingAndLayout.staticLedger_eq_frontendCold_iff' does not depend on any axioms -/
#guard_msgs in
#print axioms MindIESimSpec.ShardingAndLayout.staticLedger_eq_frontendCold_iff

/-- info: 'MindIESimSpec.ShardingAndLayout.expertBytes_strict_gt_zero' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms MindIESimSpec.ShardingAndLayout.expertBytes_strict_gt_zero

/-- info: 'MindIESimSpec.RecoverSSM.rollbackCheckpoint_eq_recoverSSMReplay' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms MindIESimSpec.RecoverSSM.rollbackCheckpoint_eq_recoverSSMReplay

/-- info: 'MindIESimSpec.RecoverSSM.recoverSSM_bytes_strict_lt_checkpoints' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms MindIESimSpec.RecoverSSM.recoverSSM_bytes_strict_lt_checkpoints

/-- info: 'MindIESimSpec.AllToAllRounds.ceilLog2_lt_ringRounds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms MindIESimSpec.AllToAllRounds.ceilLog2_lt_ringRounds

/-- info: 'MindIESimSpec.AllToAllRounds.ep128_log2_vs_ring' does not depend on any axioms -/
#guard_msgs in
#print axioms MindIESimSpec.AllToAllRounds.ep128_log2_vs_ring

/-- info: 'MindIESimSpec.AllToAllRounds.resolveSteps_ok_legal' does not depend on any axioms -/
#guard_msgs in
#print axioms MindIESimSpec.AllToAllRounds.resolveSteps_ok_legal

/--
info: 'MindIESimSpec.RooflineAndMesh.ideal_le_target_le_predicted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
-/
#guard_msgs in
#print axioms MindIESimSpec.RooflineAndMesh.ideal_le_target_le_predicted

/-- info: 'MindIESimSpec.RooflineAndMesh.kMesh_capacity_partition' does not depend on any axioms -/
#guard_msgs in
#print axioms MindIESimSpec.RooflineAndMesh.kMesh_capacity_partition
