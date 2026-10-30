# Qwen decode mechanism review, 2026-10-29

This review prioritizes the large decode residuals without changing resource
rates or importing kernel durations into prediction. The reproducible compact
record is `data/verification/qwen-decode-mechanisms-20261029.json`.

## Attention ownership changes the Qwen3.8 result

The existing resource envelope lets attention consume aggregate device supply.
For the inspected single-request GQA geometry, that permits work owned by four
KV heads to borrow supply from 32 matrix owners. The native CSV's 32 AIC / 64 AIV
launch is not evidence that all those owners process matrix tiles.

The current source-compatible branch is described in
`docs/operators/attention-decode.md`. The host source merges query/head rows
when the grouped query extent is small, partitions complete outer query tiles,
and suppresses `SplitS2` while `pfaMergeFlag` is true. It still declares the
platform block count at launch. Thus the selected analytic declaration uses
four active owners, resident queries and one unsplit KV stream per owner.

The first interpretation, "32 launched AIC means eight KV partitions for each
of four KV heads," was rejected by this source counterexample. It is retained
only as a sensitivity candidate, never selected by proximity to measured time.

The DSH implementation and independent corrections execute a bounded
Q/K/V -> QK -> local softmax -> PV -> online accumulation pipeline on the common
Engine. Review corrected AIC/AIV read-pipe ownership, probability-operand MTE1
traffic, online numerator rescaling, and a split-reduction buffer that originally
read every partial before reserving enough storage. Split reduction now streams
one partial at a time.

The ordinary frontend explicitly selects
`runtime.kernels.recipes.flash_attention_decode` for target decode and
incremental draft attention. Prefill and draft cache preparation retain their
own recipes. Full iteration acceptance caught the draft phase being omitted by
the first selector; a complete 19-call frontend test now prevents that omission.

| Boundary | Envelope | Staged mechanism | Retained observation |
| --- | ---: | ---: | ---: |
| Qwen3.8 19 attention calls, isolated sum | 0.319250 ms | 2.920071 ms | 2.714835 ms |
| Complete modeled device iteration | 14.716206 ms | 17.317027 ms | Not used to choose the recipe |

The attention residual changes from -2.395585 ms to +0.205236 ms, or +7.56%
relative to the retained native attention sum. No parameter or multiplier was
fit. A declared split sweep also retains the much lower split-eight prediction.

These are conditional comparisons. The native sampler windows contain 17 calls
with query length four and two with query length one. The current public MTP
iteration contains 16 target calls with query length four and three incremental
draft calls with query length one. The first draft's query extent therefore
remains different. Actual sequence-length array values and historical binary
tiling identity are also unavailable. The declaration uses context 16384 and an
analytic 128-row key tile, not a claim to reproduce the source's 256-row macro
tile and inner operand scheduling for query length four.

The complete iteration and a public `mindie-sim graph` invocation both finish
with no live tasks/programs or allocations. The public graph preserves its
7 ms input readiness and executes all 16 target attention calls through the new
recipe. Sixteen focused tests cover request geometry, draft selection, causal
tails, traffic, split reduction, ownership and bounded live state.

## Qwen3.5 representation must be resolved before fitting timing

The A3 CSV records BF16 ND input and weight operands for ordinary attention,
GDN and shared-expert projections. The case's W8A8 preset currently assigns
INT8 to those categories. Routed expert operations remain INT8, so changing the
whole model to BF16 would be equally wrong.

The original A3 checkpoint quantization description is not retained locally.
An independent A5 campaign report describes INT8 body projections, while the A3
capture provenance has conflicting checkpoint directory sizes. Consequently,
the global preset and checkpoint facts were not changed. A separate conditional
case uses the existing explicit `model.quant.exclusions` representation for
`attn`, `gdn` and `shared_expert` while preserving INT8 routed experts.

On the same complete iteration that conditional representation increases the
isolated operator sum from 3.468963 ms to 3.763358 ms. The matrix-family subtotal
listed in the evidence rises from 0.780858 ms to 1.076303 ms; that subtotal is
explicitly enumerated and differs from the scoreboard's narrower family
mapping. This repairs an operand-contract mismatch but does not explain the
remaining multi-millisecond native activity residual.

For the two-token, top-eight, 256-expert uniform probability model, expected
active experts are 15.75 and the maximum is 16. The captured matrix dimensions
are already aligned to the relevant NZ storage tiles. Neither this 1.6%
active-expert difference nor an invented NZ padding factor explains an
approximately eightfold grouped-kernel residual.

The observed small grouped gate/down kernels spend roughly 32-35 microseconds
in reported scalar activity, while matrix activity is below one microsecond
and MTE2 activity is around three microseconds. These overlapping profiler
counters are diagnostic observations, not additive latency components or
independently identified resource rates. Scalar/control scheduling remains
unresolved. Small norm and quant kernels likewise need real invocation/control
mechanisms; no 40-microsecond total floor was added.

Historical observations retain their original unassigned role. None of this
review establishes heldout accuracy, serving TTFT/TPOT, or permission to remove
the old implementation and its raw data.
