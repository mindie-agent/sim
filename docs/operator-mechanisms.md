# Operator mechanism investigation

This investigation starts with operator semantics, tensor representations,
implementation stages and the sum of a target model's isolated device programs.
Serving throughput and TTFT/TPOT constraints follow after these costs are
understood. Neither a passing test suite nor a close body total establishes
mechanism correctness. The replacement is not yet ready to retire the old checkout.

## Repairs implemented on 2026-10-28

| Mechanism | Previous omission | Current execution |
| --- | --- | --- |
| Activation quantization | Packed GEMM consumed payload/scales with no explicit producer | Shared conversions produce payload and scales from the declared activation format and group geometry |
| Norm and residual fusion | All semantic operators retained intermediate traffic | Opt-in graph passes remove only dead intermediates; live residual output is retained |
| Matrix tail waves | A small compute wave could use all distributed copy units | Compute and declared distributed transfer service are limited to participating units |
| GDN scalar gates | Channel width was confused with per-head scalar gates | Decay/beta use `(tokens, value_heads)`; output gating consumes the independent z projection |
| GDN decode | The recurrence consumed cube work | A finite vector program executes state decay, two dot products, delta, update and checkpoint stores |
| Batched recurrence | Independent sequences became separate kernel launches | One launch schedules independent sequence/head chains with distinct lengths and persistent state |
| Tensor materialization | Packed slices and output-gate reshape were treated as free | Shape/stride/offset metadata determines views and required payload copies |
| Final norm | Only selected logits rows were normalized | An explicit software policy normalizes all forward rows before selection, or declares selection-first execution |
| Scalar output transfer | All stores were treated as bulk DMA | A finite address-dependent issue/response queue executes the compiled scalar output path |
| Independent tiled units | Tile aggregation imposed implicit global synchronization | Units with different ownership/address phases retain independent instruction streams |
| Native GDN selection | Mixed prefill bypassed native kernels and used analytical chunk work | Explicit source binding selects native FN convolution and Prepare/H/O; mixed decode remains separately labeled |
| Multi-output identity | Equal-shaped Q/K exports shared one copy-cache key | Producer output ports retain distinct storage identities |
| Actual/padded boundaries | Padded output rows were charged as native results | Core work and copies use actual rows; outer zero-fill and normalization retain padding |
| Native register work | Prefix sum and normalization were generic lane counts | Source calls, masks, payload and independent physical capacity bounds are retained |
| Nested input dependency | Initial prefetch could precede its parent's ready event | Lazy issue starts after readiness; delayed-input verification catches the violation |
| Source planner selection | An unbound source op could silently use an envelope, including zero work | Source kernels require an exact planner binding; source calls also require real kernel expansion |
| Source fusion boundary | Generic norm fusion could delete a selected source implementation | Generic passes preserve source nodes; a source fusion requires its own planner |
| Service-mode ownership | A mode mapped to a different physical pipe could borrow idle units | Ownership resolves work roles onto the actual physical pipe, including register issue and UB copy |
| Native snapshot reuse | Aggregated reads/writes hid reuse of a single temporary UB row | Each history slice executes read, write, then reuse; shared HBM contention remains visible |
| First native input | Restore incorrectly waited for the current token's copy | History precedes independent Restore and first-input load; compute and pair prefetch wait for both |
| A5 vector units | 17 active examples overrode element operations with an old byte-based number | Examples inherit the declared FP32-lane convention; typed instruction service keeps its own units |

No model-name timing branch, efficiency multiplier, serving floor or fitted
per-sequence overhead was added. The retired GDN per-sequence fitted parameter
was removed from the active example metadata and retained under its migration
record. Simulation still works without measurements.

`RecurrentVector` has `7*q*heads*dk*dv` elementary vector work for its five
arithmetic stages. The two dot products each include their reductions in the work
count; this is not a claim that every operation has the same instruction latency.
State is loaded once per sequence/head and checkpointed after each query token.
For 48 heads and 64 declared vector units, one sequence occupies one partial
wave; four sequences occupy three full waves. The implementation preserves this
geometry without scaling a measured single-sequence duration.

Tensor copies currently account for selected read/write bytes and their
dependencies. Their default resource envelope does not reconstruct native
transaction, instruction or synchronization costs. The address-map oracle uses
NumPy views/reshapes on small arrays; it checks actual addresses, including
single-row views and gaps between token rows.

## Sources and comparison boundaries

The original native CSV is streamed from:

```
/Volumes/data/mindie-sim-data/a5/2026-10-22/qwen38-c4-16k/raw/prof/
```

| Capture | Selected lines, including header as line 1 | Geometry |
| --- | --- | --- |
| `p16384_c1_o32/.../kernel_details.csv` | 45-2315 | First target prefill body, 16384 tokens |
| `p16384_c1_o32/.../kernel_details.csv` | 2559-4125 | First target decode body, one sequence with four verify rows |
| `p16384_c4_o32/.../kernel_details.csv` | 16131-17745 | First target decode body with 16 rows and four sequences |
| `p16384_c4_o256/.../kernel_details.csv` | 16131-17745 | Independent capture of the same four-sequence geometry |

Each selected body has 64 FFN-down projections, 48 GDN recurrences/projections
and 16 full-attention blocks. The four-sequence files were compared after the
mechanism changes; they never supply prediction parameters. The o256 file is
over 50 MiB and is read line by line.

Windows run from the first target norm through the last FFN-down projection.
The analytical body includes a terminal residual that the captured software
fuses with the final norm outside this window. Therefore sums are diagnostic
work ledgers, not exact-boundary model accuracy scores. No client TTFT/TPOT is
used in this comparison.

Matrix comparisons check native shape, dtype, weight layout and activation/weight
scale geometry. Recurrence rows expose total tokens and sequence count, but not
the values of per-sequence length arrays. Slice offsets are also absent from the
CSV. Reports preserve these limits rather than inferring them from timings.
The 96 BA slice calls share one observed shape contract, so the individual alpha
and beta calls are not distinguished by the CSV.

Four source files were read from image
`sha256:c3d00a196905159dbf0b97588d34f085086c2708712fa0d1c8fedd807f0e48a4`,
which matches the original capture's image record. Their original text, paths
and SHA-256 values are retained in
`data/observations/sources/gdn-image-c3d00a196905/`. The source includes a
different prefill dispatch from the local latest checkout; it selects the
FLA implementation on the captured hardware. Image source identifies the Python
path more precisely. The subsequent compiler/native experiment also retains
the GDN gating binary and disassembly, an independent scalar-store primitive,
and address/queue validation. See `docs/scalar-transfers.md`; this does not
establish binary-level coverage for the remaining operators.

The golden fixture keeps 38 original CSV rows with source line numbers and hashes.
The baseline in `operator-baseline-81b376b.json` was generated before these repairs
through the same isolated-program runner. It is a comparison artifact, never
an input to prediction.

## Earlier operator snapshot

All table values are microseconds per kernel/program. Raw values are medians
within the selected target-body window. Matrix geometry is `(M,N,K)`. This
snapshot predates the rate-unit correction and typed rolling program below;
its predictions, including gating and recurrence, are not current results.

| Operator | Before | Earlier repaired snapshot | Raw |
| --- | ---: | ---: | ---: |
| BF16 `(4,96,5120)` | 0.298 | 8.205 | 7.378 |
| BF16 `(4,5120,6144)` | 15.761 | 25.203 | 30.098 |
| MXFP8 `(4,5120,17408)` | 23.012 | 36.808 | 32.743 |
| MXFP8 `(4,5120,6144)` | 8.132 | 12.999 | 13.434 |
| BF16 `(16384,16384,5120)` | 6355.272 | 6355.272 | 6375.468 |
| MXFP8 `(16384,5120,17408)` | 3394.081 | 3394.081 | 3392.483 |
| GDN vector recurrence, one sequence of four | 6.597, wrong cube mechanism | 6.395 | 10.595 |
| GDN vector recurrence, four sequences of four | Not compared at this geometry | 19.185 | 22.054; independent capture 21.974 |
| GDN gating `(16384,48)` | 1.966, memory envelope | 582.938 | 673.704 |

The gating "before" value is the intermediate corrected-head resource envelope
from this investigation, not a shape-matched row in revision `81b376b`; its
machine-readable baseline `before_us` is therefore null.

The recurrence change is justified by execution resources and state flow, not
by whether a wrong cube formula happened to be numerically close. Some matrix
predictions improve substantially; others remain biased or overpredict.
Default 128-cubed GEMM tiles and idealized reuse are still assumptions.

Before native prefill integration, diagnostic body sums were 1464.383 ms for prefill, 10.792 ms for single-sequence
decode, and 13.665 ms for four-sequence decode. Corresponding selected raw kernel
sums are 1714.543, 18.045 and 21.839 ms (21.824 ms in the independent capture).
The boundary difference described above still applies. These sums show that
operator-level agreement on selected matrices is insufficient for replacement.

## Native prefill integration

The selected historical source is now executed through native FN convolution,
Prepare, H and token-first O. The caller is explicitly the noninterleaved
QKVZ/BA, PCP=1, BF16, FP32-state, non-spec branch. A mixed batch runs FN and QKV
packing once over all actual rows, slices its decode prefix and prefill suffix,
then concatenates outputs before copying the actual prefix into padded storage.
The analytical decode recurrence is not relabeled as native.

The exact source-derived regression reports are retained in
`data/verification/native-convolution-review/` and the GDN frontend review.
Per-unit allocations, separate AIC/AIV reads, MTE1, Fixpipe, source overlap,
output-port identity and delayed readiness are covered independently of the
authors' campaign tests.

Latest bounded raw replay after source boundaries, rate units, typed rolling
work and independent dependency/ownership repairs:
`data/verification/gdn-native-20261028-reviewed-rolling.json`.

| Kernel | Predicted us | Observed us | Difference |
| --- | ---: | ---: | ---: |
| Native FN convolution, row 52 | 373.734 | 336.261 | +11.144% |
| Prepare, row 65 | 1228.873 | 1541.022 | -20.256% |
| H, row 66 | 683.174 | 810.911 | -15.752% |
| O, row 67 | 356.700 | 512.657 | -30.421% |

The four-family isolated sum is 126.839 ms across 48 occurrences of each
family. The current complete-model isolated sum is 1544.207 ms for 1956
operator occurrences, retained in
`data/verification/qwen38-native-20261028-model-reviewed-rolling.json`.
The earlier 113.013 ms family sum and 1530.330 ms complete-model sum
remain in the integrated-caller/model-prefill reports; those used the old
vector alias and rolling work. Complete-model results must use the same
hardware declaration and source programs as the current operator replay.
These are sums of isolated device programs, not overlap-aware latency, TTFT,
TPOT or an exact-boundary comparison with the earlier captured body window.

The ABI matches do not observe sequence-length values or all dispatch flags.
FN cache capacity rows are not treated as active rows. Source controls and raw
measurements support separate conclusions. The discrepancies above remain open.

## Register work and independent counterexamples

The old active A5 examples carried `vector_ops_per_s=27033600000000`,
overriding the profile's `rates.vector=6758400000000`. The former derived
from bytes per vector instruction; the latter uses 64 FP32 lanes per AIV,
64 AIVs and a nominal 1.65 GHz clock. Removing the obsolete alias is a unit
correction, not a fitted reduction of machine efficiency. It is still a
generic single-instruction convention, not a universal limit on all vector
instructions. The provenance and independent old/new controls are retained
in `shared-mechanism-review/review-a5-vector-rate-provenance-20261028.json`.

Native rolling and restore now use separate instruction calls, active lanes,
full unmasked UB loads and masked UB stores. Widths 2/3/4 with SiLU require
18/24/30 issue slots per register-vector token; restore requires 7/15/25.
Channels 16 and 64 issue the same number of loads, while 80 requires two
register vectors. Issue slots, register writes, UB accesses/stores, arithmetic
lanes and special-function lanes impose distinct physical capacity bounds.
Source weight loads remain present; no unproved compiler hoisting is assumed.

The compiler controls retain the image's shim and compiler-driver source.
The declared math path uses default FTZ and intrinsic Exp/Div. The packaged
ELF's exact macro expansion and issue schedule have not been observed.
The current compiler and packaged ELF share a Clang revision but have
different build dates. Ordinary disassembly still contains unknown and
placeholder opcodes; it cannot establish load elimination, register bundles
or latency. Setup casts/zeros and the generic bias path remain analytical.

Independent review reproduced two failures after the earlier suite passed:
a remapped issue pipe made one 110-slot block take 110, 55 and 13.75 ms as
idle owners were added, and a first-token dependency added 1 ms of false
serialization. The corrected real execution takes 110 ms for all three
owner counts. The delayed-input load/restore control takes 6.427083 ms and
matches the independent source DAG with identical serviced work. The original
failure reports, scripts and subsequent rechecks remain in
`data/verification/native-convolution-review/`.
The final review adds 54 independent mixed-mode/owner contention controls and
18 asymmetric first-load/restore controls. For 7, 127 and 1023 tokens, the
reviewed single-block program keeps 26 live tasks at peak and finishes with
no live tasks, programs or allocations. This establishes bounded execution
for those controls, not a compiled occupancy or latency result.

## Parallel source extensions

The latest DSH wave returned online tiled attention, CUDA normalization and
dynamic per-token INT8 quantization. Root integration preserves their explicit
source dispatch, per-CTA ownership, shared HBM and typed instruction/reduction
work. Independent controls cover attention's CC100/D256 branch, early-return
metadata, online rescaling and pre-PV cast; normalization's vector/generic and
partial-warp reductions; and both quantization conversion passes.

Their final construction reports are
`data/verification/operators/*-source-boundaries-20261028.json`.
Grouped GEMM, MLA decode, index/top-k and MoE align/combine also remain explicit
source extensions. Multi-kernel wrappers expand into actual launches and a
zero-work join. The shared public-path review checks 13 adapters, including
default-path rejection, exact positive bindings and isolated/iteration work
agreement. The original default path could return zero cost for CUDA norm;
that failure is preserved in the review reports.

These extensions are not silently selected for every model of the same
semantic family. Source compatibility and frontend integration still need
to be established for each runtime implementation. Synthetic controls and
source acceptance do not establish measured GPU/NPU accuracy.

The final maintained suite after these repairs passed 742 tests, including
the native 16K bounded-memory case. The native extension's 20 final campaign
rows and the three latest parallel extension reports also pass. These are
regression and source-contract checks; the open measured differences above
remain acceptance work.

## Unresolved mechanisms

1. In the earlier rate configuration, the compiled scalar store path and
   finite address queue predicted large GDN gating at 582.938 us versus
   673.704 us, leaving -13.47%. This is a historical result. The scalar-pipe
   median is 636.146 us and vector-pipe median is 31.144 us. Shared transfer
   response, vector transcendental expansion and synchronization remain open.
   A native replay of the unchanged binary gives 647.665 us; it does not enter
   primitive calibration or replace the historical capture.
2. Earlier generic small quantization and fused norm predictions were roughly
   0.02-0.21 us versus 4-5 us observed. The new CUDA source extensions do not
   represent those native Ascend kernels. Their reductions, issue, memory
   response and synchronization need an appropriate phase/service model or
   primitive calibration; instruction-level source programs are not mandatory.
   A blanket minimum kernel time would conceal the mechanism.
3. GDN prefill uses native source programs and source caller materializations.
   Its remaining discrepancies may involve memory service, issue or dependency
   latency; the current comparisons do not identify which parameter is missing.
   Aggregate lane work still covers some vector functions. The typed register
   projection is a capacity envelope, not a
   compiled schedule. Native speculative mixed, other projection layouts, zero
   length metadata, PCP and BF16-state paths remain unimplemented. KDA decode
   was not repaired by this GDN investigation.
4. Full attention and many elementwise operators still use resource envelopes.
   Check their repeated cost and heldout trends. Add phases for a demonstrated
   missing serialization, reuse or launch effect; an envelope does not need to
   become a native instruction program solely to qualify as a model.
5. GPU memory-copy validation remains limited to the prior RTX 4090 primitive.
   It does not establish GPU GEMM or attention accuracy.

The 2026-10-28 architecture review changes the follow-up priority; see
`modeling-fidelity.md`. Keep structural repairs, then rank repeated costs and
conditional residuals across prefill/decode workloads. Prefer reusable phase/tile
models and independently calibrated services; reserve compiled instruction detail
for a demonstrated material failure. Check heldout shapes, bottleneck trends and
configuration ranking before escalating detail. Complete block/model work and
serving constraints still follow, without requiring zero error on every operator.
Historical files are preserved, but full mechanism/data absorption and
old-version retirement remain unfinished.

## Reproduction

```sh
.venv/bin/python -m mindie_sim operators examples/cases/qwen38-a5.json \
  --phase prefill --tokens 16384 --sequences 1 --scope model \
  -o /tmp/qwen38-target-prefill.json

PYTHONPATH=. .venv/bin/python tools/compare_operators.py examples/cases/qwen38-a5.json \
  /Volumes/data/mindie-sim-data/a5/2026-10-22/qwen38-c4-16k/raw/prof/p16384_c1_o32/ASCEND_PROFILER_OUTPUT/kernel_details.csv \
  --phase prefill --tokens 16384 --sequences 1 --start-line 52 --end-line 67 \
  --family causal_conv1d_fn --family gdn_prepare --family gdn_state --family gdn_output \
  -o /tmp/qwen38-native-prefill.json

.venv/bin/mindie-sim operators examples/cases/qwen38-a5.json \
  --phase decode --tokens 4 --sequences 4 --context 16400 --scope model \
  -o /tmp/qwen38-target-model.json

.venv/bin/python tools/compare_operators.py examples/cases/qwen38-a5.json \
  /Volumes/data/mindie-sim-data/a5/2026-10-22/qwen38-c4-16k/raw/prof/p16384_c4_o32/ASCEND_PROFILER_OUTPUT/kernel_details.csv \
  --phase decode --tokens 4 --sequences 4 --context 16400 \
  --start-line 16131 --end-line 17745 -o /tmp/qwen38-operators-c4.json
```
