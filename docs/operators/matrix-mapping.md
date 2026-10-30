# Finite matrix mappings

The explicit `hierarchical_matmul` and `mapped_matmul` recipes separate private
refill tiles from matrix operand tiles on the existing Engine. They address a
material counterexample: after correcting private-copy accounting, four dominant
16K Qwen3.8 matrix families contributed a conditional +780.105 ms residual against
retained tensor-contract medians. That comparison does not establish the actual
binary's tiling or whole-model accuracy. Rates and observations are unchanged.

## Memory boundaries and execution

`TiledMatmul` remains the fixed-mapping control and default. The new recipes are
explicit runtime choices:

```json
{
  "programs": {
    "matmul": {"kind": "mapped_matmul"}
  }
}
```

This object goes in `runtime.kernels`. It applies to ordinary dense and equal
batched matrices. An explicitly bound source planner keeps priority.

The hierarchical program has these boundaries:

1. Shared HBM reads retain the existing explicitly ideal reuse assumption.
2. Each physical owner refills its own L1 panel. A larger M/N refill tile can
   reuse these operands among smaller products on that owner.
3. Each micro product still moves A and B through the shared MTE1 pipe and
   occupies private L0A/B buffers. L1 reuse never deletes these transfers.
4. Its accumulator remains in L0C across K panels and is released after the
   corresponding output store.

`RefillPipeline` gives refill producers and consumers independent finite
windows. The next panel can load while all micro products of the current
panel execute. The operand/compute streams continue across K boundaries, so
each K panel does not introduce a new micro-pipeline drain. Both windows use
the same Engine, with physical service, allocation constraints and contention.
Only live panels and consumer steps are retained.

For an independent control, each of two L1 panels takes 3 ms to load and 4 ms
to compute. With two refill slots and sufficiently fast inner copies, completion
is 11 ms: load 0 at 0..3, load 1 at 3..6, compute at 3..7 and 7..11.
A micro-only lookahead window incorrectly delayed the second refill and gave
13 ms. This counterexample is retained in `tests/test_matrix_mapping.py`.

## Candidate selection

`mapped_matmul` screens eight declared candidates: M/N of 128 or 256, and K of
64 or 128, with 128x128 micro products and two buffer slots at each level.
An explicit `candidates` list of `[M, N, K]` triples replaces that set.
Single fixed mappings can instead use `hierarchical_matmul`.

Screening uses the maximum actual panel footprint, including tails, for every
declared L1/L0A/L0B/L0C pool. Missing pool declarations remain explicitly
unscreened; they do not establish infinite physical capacity. Surviving candidates
are ranked by resource lower bounds using each owner's actual assigned tails,
not an assumed fully occupied device. This includes MTE1, MTE2, matrix work and
stores; HBM remains shared. Equal bounds prefer larger K panels, then smaller
residency. The selected program is subsequently executed by the Engine.

The selection bound is not a prediction of its complete runtime and is not
proof of an optimal mapping. It omits pipeline fill/drain and invocation
contention. It is a finite analytical choice, not a reconstruction of runtime
dispatch or a parameter fitted to observations.

`parameters.matrix_mappings` in graph/serving reports and each operator's
`basis.matrix_mappings` contain the input shape, group count, selected tiles,
candidate bounds and capacity exclusions. At most 32 distinct records are
retained per CostModel; omitted invocation counts remain visible.

## Independent matrices and representations

`gemm_op(weight_count=G)` retains `matrix_groups=G` when G equal independent
matrices contribute rows to the aggregate M dimension. MLA head projections
and grouped output projections declare this axis. The new recipes assign the
G independent tile grids across owners while keeping each B matrix and
accumulator in its own domain. They do not allocate all G weights into one
matrix operand buffer.

MoE probability moments and aggregate expert histograms do not by themselves
declare this equal-batch geometry. Their ordinary analytical path has a separate
`grouped_gemm` family binding, retaining the existing approximate recipe.
Selecting `programs.matmul` does not silently invent per-expert geometry for
that path. Integer-routing source grouped planners remain available through
their explicit bindings. General unequal grouped mapping is still unfinished.

Packed payloads and scales remain included in private refill service. Inner
transfer and residency retain the larger of the proportionally assigned
payload/scales and compute-width operand footprint. This is a stated
representation bound, not a reconstruction of scale instructions, packing
alignment or fused dequantization latency.

An explicitly declared ordinary `mte2` role retains priority. When only split
read pipes are declared, hierarchical/mapped matrices bind operand reads to
the existing `mte2_aic` resource and final output-scale reads to `mte2_aiv`.
No new capacity or rate is created. Mapping bounds combine demands by physical
resource and unit, so operand and scale reads contend when they share a pipe.
`read_role_bindings` and missing transfer roles are visible in mapping records.
The fixed `TiledMatmul` control retains its earlier generic-role abstraction.

Declared `row_column_scale` epilogues separate their scale bytes from K-panel
operands. The complete K reduction precedes scale fetch, logical vector
transformation and BF16 output store; there is no invented HBM accumulator
spill. This contract does not apply to K-block scales. It does not reconstruct
the native L0C/Fixpipe/UB transfer schedule.

One conservative overlap limit remains: the final K panel is retained until its
epilogue/store completes. A one-slot, two-output control with 1 ms matrix work
and 10 ms epilogue per output finishes at 22 ms; releasing that L1 panel after
matrix consumption could finish at 21 ms when sufficient accumulator storage
is available. This is a documented lifetime abstraction, not a fitted delay.

## Reproduction and evidence

```sh
.venv/bin/python -m mindie_sim study examples/studies/matrix-mapping.json
.venv/bin/python tools/model_convergence.py --help
.venv/bin/python -m pytest -q tests/test_matrix_mapping.py
```

The study compares fixed mapping, an explicit fixed hierarchy and finite
selection on unchanged workload and hardware inputs. Multi-model results and
remaining concentration are recorded in `docs/model-convergence.md`.
Independent source review and nominal resource arithmetic are in
`matrix-mapping-review.md`; tests establish work, lifetime, dependency and
ownership invariants, not measured accuracy.

At revision `76df99d`, the four dominant 16K Qwen3.8 families were rerun
through `target_operators` and `operator_costs` with only the explicit mapping
override. All source hashes were stable across the run. The hardware rates,
primitive parameters, shapes, dtypes and 224 occurrences were unchanged.

| Family | Fixed mapping ms/call | Hierarchical selection ms/call | Retained kernel median ms/call | Selected difference |
| --- | ---: | ---: | ---: | ---: |
| Dense gate/up | 11.925185 | 7.038386 | 6.770202 | +3.96% |
| Linear QKVZ | 10.762600 | 6.490428 | 6.375468 | +1.80% |
| Dense down | 5.906295 | 3.418840 | 3.392483 | +0.78% |
| Linear output | 4.034762 | 2.426274 | 2.394767 | +1.32% |

The repeated conditional residual across these four families changes from
+780.105 ms to +25.881 ms. This is a useful check of the missing mapping
mechanism, with no residual fit, but still only a tensor-contract comparison
against retained medians. It does not prove the actual binary uses the chosen
tile or establish independent heldout or whole-model accuracy. The complete
input hashes, selected candidates, service volumes and observed source lines
are retained in `data/verification/matrix-refinement-20261029.json`.
