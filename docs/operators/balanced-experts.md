# Forced-balance expert execution

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

`runtime.routing = {"policy": "balanced"}` declares an integer load-balancing
scenario. Ordinary `target_graphs`, `graph_costs`, serving `predict` and
`predict(scope="body")` use the same frontend, recipes and Engine. The scenario
is independent of calibration and does not require observed router assignments.

## Computation and representative rank

Let T be the number of router rows in one EP cohort, K the top-k count, and E
its experts. The scenario assigns `q, r = divmod(T*K, E)`, then gives the first r
expert identities q+1 rows and the others q rows. Strided EP placement preserves
all expert identities, including empty banks. TP copies do not increase T and
independent DP cohorts do not share assignments. Graph-capture padded rows are
included in T; metadata retains both valid and padded local rows.

The ordinary frontend represents expert work using rank zero's strided
histogram. Under this q/r law, rank zero has the maximum assigned rows and at
least as many active banks as any other rank. `rank_assignment_rows` retains all
rank totals and `rank_timing_assumption="heaviest_strided_rank_upper_bound"`
identifies the conservative compute representation. This is not an exact graph
of every rank's distinct matrix completion time or a measured routing trace.

Each `balanced_grouped_gate_up` or `balanced_grouped_down` is one grouped
invocation. Zero-row experts read no weight bank. Equal positive row counts
continue through the selected matrix recipe unchanged. Mixed counts are
partitioned into their positive integer row classes, usually q and q+1; the
selected mapped, steady, phase, hierarchical or tiled matrix recipe executes
each class. Activation work and row scales partition by row count; independent
weight banks and column scales partition by active expert count. No expert row
padding, average-expert matrix or invented token IDs are introduced.

The row classes execute as serial internal phases under one startup/minimum
boundary. Mapping diagnostics expose
`serial_row_classes_with_selected_matrix_owner_schedule` and
`declared_analytical_schedule_not_vendor_dispatch`. Each class keeps the chosen
recipe's owner tiling and pipeline. This practical schedule supports arbitrary
integer balanced populations without claiming a vendor scheduler's exact
cross-class packing. `program_kind="matmul"` stays intact so activation
quantization and its payload/scale producer dependencies remain connected.

PCP partition and chunked MC2 execution remain outside this declared scenario;
EP across DP requires a complete cohort. Restricted router groups no longer
reject forced counts: router scores, group selection and weights still execute,
then the forced assignment scenario overrides the expert IDs. Explicit source
planners retain their typed domains and are not silently replaced by this
analytical composition when they require actual token IDs.

## Communication and completion

Outgoing assignment payload is partitioned across `min(TP, EP)` local expert
ranks, accounting for sequence shards when present. Separate ETP groups still
need their own copies. The all-to-all volume is the total remote useful payload
per physical source rank; topology does not divide it by TP again.

`uniform_peer_payload` remains an explicit communication assumption. Balanced
expert counts alone do not determine an exact source/destination matrix.
Fractional peer payload is an analytical traffic distribution, not fabricated
integer routing. Source populations can differ, including an idle DP source;
the combine references `reverse_of_collective` so the common collective engine
uses the transpose of the dispatch payload relationship.

Select `runtime.kernels.families.alltoall.kind="pairwise_collective"` for causal
rounds. One complete physical cohort executes each collective with rank input
readiness and send/receive dependencies. Per-replica completion spans must not
be summed as whole-cohort wall time. The representative-rank compute envelope
and uniform-peer communication are distinct approximations; the latter does
not establish a measured source-to-expert permutation.

## Source boundary for forced assignments and graph padding

The independently retained vLLM-Ascend tree at
`ae923905948f360f4b7ab06df52d0789a0c9316e` provides the structural evidence:

- `vllm_ascend/ops/fused_moe/force_eplb.py`, SHA256
  `bd85c5f499f04a6e15aa23fca8b653d1c527cd35651c80da4504efae8cdd5f2a`:
  `build_force_eplb_topk` prepares capture-sized tables;
  `get_force_eplb_topk` uses `topk_ids.shape[0]`, including router padding.
- `vllm_ascend/ops/fused_moe/routed_experts.py`, SHA256
  `aaea89c2d83ebbb4f19f0162ef3076c33e359de0017355f9fb6d831c64d716fb`:
  forced IDs replace `_select_experts` output after group-limited selection.

The retained source uses contiguous rank expert ranges and a source-rank-shifted
round-robin ID table. Sim's q/r strided scenario is an ideal balancing declaration,
not that exact token-ID table. This local source revision is not established as
the deployed 0.30 binary; no source-to-capture identity is inferred.

## Quantized grouped calls

The historical `torch_npu.npu_grouped_matmul` adapter has a separate typed
contract: INT8 activations, INT8 FRACTAL_NZ weights, FP32 row/column scales,
INT32 accumulation and BF16 output. Scale reads and four logical output
operations follow the complete K reduction; group metadata is fetched outside
matrix operands. Catalog INT8 includes channel scale/offset pairs, while
FP4/FP8 can have K-axis scales. Their declared representation work stays intact;
they are not silently converted to the historical no-offset epilogue.

## Verification and remaining accuracy work

`tests/test_grouped_integer_rows.py` executes batches 1, 2, 4, 8, 16, 32, 64,
128 at q=1,4,5,8 with restricted router groups. Other controls cover mixed rows
under five matrix recipes, packed INT8 weights/scales, one invocation startup,
input readiness, padded cross-DP cohorts and idle sources.

Full 93-layer Kimi K3 batch128/q4/context2048 controls, TP8/EP8 with W8A8 and
steady matrix programs, now complete on both A3 and A5 profiles. Their declared
FLOPs/HBM/link work agrees and both leave zero live tasks. Predictions are
632.200ms and 169.986ms respectively; these are mechanism execution controls,
not matched-hardware accuracy results. The
retained report
and bounded producer
run with the ordinary `programs.matmul=steady_matmul` declaration and no
balanced-specific recipe overrides. Rank-specific compute graphs and a measured
source/destination routing matrix remain future precision refinements.
