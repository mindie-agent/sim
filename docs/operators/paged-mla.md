# Dense paged MLA decode

Select `runtime.kernels.lowerings.paged_mla_attention` with implementation
`ascend_paged_mla_decode_arch35` and options `{"kv_splits": 1}`. The split
count is explicit; its default is one. Neither observed latency nor a model
name selects a split. This is a source-compatible phase model, not proof that
the capture loaded this binary or host tiling result.

For the pinned q1/no-mask host algorithm, select options
`{"dispatch": "source_split_core_v2"}`. This computes owner segments and FD
head partitions from current contexts, physical heads and compute-unit count.
It preserves the source's M64/KV128 tiles and `fdTolerance=9`; integer host
cost scores choose ownership and never supply elapsed-time or resource rates.
The planner specializes `SplitCoreV2` at source revision
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8`. Eighteen retained original-C++
controls cover uneven contexts, tails, core counts and graph capacities.

Active sequence count and native query storage are distinct. Existing graph
capture padding supplies native capacity (B6→8 and B12→16 in the six 128K
source cases). Padded Q/O storage is exported in native tensor views while
inactive sequences have zero KV length and add no attention tiles. An explicit
`native_query_capacity` can declare a caller's capacity without increasing
active contexts. Existing caller head-pad copies keep their full capacity.

The core binds one kernel across independent q=1 requests, BF16 latent512 plus
rope64, one shared KV head, CP/PCP1 and physical query heads at most64. Each
request keeps its own context. Its actual KV tiles are partitioned into the
declared split count and scheduled across a shared AIC owner grid. This does
not concatenate request contexts or submit B serial kernels. Nontrivial
split partials are merged behind the producer barrier; sequence merges on
the same owner execute serially, with one partial resident at a time.

The retained arch35 kernel fixes M64/KV128. Its three L1 KV/P slots and resident
query consume 516096 bytes per owner. L0A/B require 65536 bytes each, L0C262144,
and each paired AIV has241920 bytes of no-mask/no-LSE UB allocations. These
capacities are checked against the declared hardware, never enlarged to make
a case run. They remain live through split reduction. The A5 implementation
is not bound to the A3 cases.

The Cube loads latent KV and rope directly from paged GM. K_nope is also V:
there is one HBM load, followed by separate real MTE1 operand transfers for
QK and PV. Scores and partial products travel through Fixpipe to UB. Softmax
writes BF16 probabilities into the consumed rope region of the same L1 ring;
this MTE3 transfer is not another HBM transaction. M64 probability L0A loading
is preserved even when actual matrix rows are fewer. Both UB double rings
wait for their corresponding consumer before being reused.

This selector owns only the attention core. An existing
`vllm_ascend_mla_head_padding` caller retains its native prepared weights,
strided query materialization and F.pad nodes. The core adds none of those
again. With a generic logical-head prolog it adds only the necessary head
padding; the leading-head NBSD output slice is a view. Logical value expansion
and shared KV capacity do not expand to physical query head count.

Known limits remain visible in the program metadata: without a `dispatch`
declaration, equal tile splits and round-robin owner placement remain the
declared policy. Inner L0 instruction scheduling, cross-core flag latency, page-translation latency and detailed
Vector instruction issue are not reconstructed. Matrix/softmax stages retain
their actual physical services and finite queues without an empirical floor.

`tests/test_paged_mla.py` checks independent HBM ledgers for B1/B32,
4096/16384 contexts and splits1/4; tail-page/context separation; input readiness;
one batched call; caller-copy nonduplication; shared owner utilization;
physical-rate response; two-owner split-merge lifetime and cleanup. Source
file identities are pinned in `prediction/paged_mla.py`. Complete-body
scenario evidence is kept separately from these mechanism controls.

Three full 93-layer A5 source scenarios with the explicit `kv_splits=1`
control completed at runtime `d85c7f83`:

| Batch / context | Body ms | Local wall s | Peak MiB | Engine tasks |
|---|---:|---:|---:|---:|
| 1 / 4096 | 24.40238 | 8.52 | 147.2 | 526419 |
| 32 / 4096 | 27.66017 | 14.27 | 152.1 | 1013635 |
| 32 / 16384 | 30.37815 | 21.50 | 151.1 | 1972099 |

All used a90-second budget, unchanged start/end package hashes and zero live
tasks/programs. The preceding source scenario without this paged core predicts
23.50149/27.91168ms for the two4K cases; this mechanism raises the first and
slightly lowers the second. Neither direction establishes measured accuracy.

The separate CPU-only `paged-mla-split-oracle-20261030` producer compiles the
retained CANN `SplitCore` scheduling algorithm. It stubs logging and records a
single explicit `std::max<uint64_t>` portability adjustment; no scheduling
arithmetic or observed timings change. For the declared B1/4K and B1/16K
geometry it produces32AIC split segments and16AIV head reductions. B32 uses
32AIC and noFD. Intermediate batches have uneven segments crossing sequence
boundaries. Thus the uniform one-split control is not claimed to reproduce
the retained host's automatic dispatch. The oracle report preserves exact
endpoints and FD head ranges for a subsequent explicit metadata binding.

The optional `dispatch` declaration now accepts that metadata directly:
`contexts`, `heads`, `aic_owner_segments` (each AIC's ordered sequence, start
tile, tile count triples), and `aiv_merge_segments` (each AIV's sequence,
start head, head count). Validation requires an exact, gap-free partition of
every KV tile and of every split output's heads. It rejects a reused dispatch
declaration for a different context. Multiple sequences on one AIC execute
in source order; each AIV reduces its own head range after the shared barrier.
The default equal-split policy remains available when `dispatch` is absent.

`kimi-source-paged-host/` contains12 additional conditional variants generated
from the retained CPU oracle. Both preceding source variants remain unchanged.
This oracle fixes source host scheduling, not capture-binary identity or the
unknown actual contexts. Tests execute the B1/B4/B32 host declarations, including
one AIC crossing sequence boundaries and16 independent one-head FD writers.
An independent external-contention counterexample for H33–64 additionally led
to splitting each Vector/read/write phase into distinct AIV subblocks, rather
than applying a total-rate cap that could borrow an idle sibling. The initial
fixed-M32 partition was subsequently corrected as described below.

Two complete 93-layer source-host variants ran at `cd50952b` with unchanged
start/end package SHA256
`468e6f6e01991eeb12a10ec40b46f394d1a83b0c50cfe14a85fd652a1846b840`:

| Batch / context | Body ms | Attention core elapsed sum ms | Wall / CPU s | Peak MiB | Engine tasks |
|---|---:|---:|---:|---:|---:|
| 1 / 4096 | 23.58040 | 0.11679 | 11.05 / 10.68 | 141.5 | 576771 |
| 32 / 4096 | 27.66017 | 0.96039 | 15.62 / 15.25 | 152.5 | 1013635 |

Both finished within the unchanged 90-second budget with zero live tasks,
programs or allocations. Each attention sum contains 24 core calls; caller
projections, copies and output expansion are separate. The complete graph is
measured by `body_forward_ms`; family timeline sums are not additive.

Compared with the preceding one-split control, native host metadata decreases
the B1 body from 24.40238 to 23.58040 ms and leaves B32 unchanged. This follows
the source's 32-way B1 KV partition and unsplit B32 assignment. It does not
establish capture dispatch identity or improved measured accuracy. Input and
full-body execution receipts are in `data/verification/kimi-source-host-body-20261030*`;
the existing producer reproduces them with `--host-dispatch`.

Independent review exercised H16/32/33/48/64 against an external AIV0 contender:
the fixed subblock implementation matched the separate owner-service oracle.
The 16 focused tests also cover the explicit dispatch contract. That review
did not independently compile the C++ host oracle and therefore provides no
additional capture-binary identity claim.


## Actual-M Vector partition correction (2026-10-30)

The retained kernel `InitTask` at lines 575–603 computes actualM from
`actual_query_rows * query_heads_per_KV_head`, capped by M64, then partitions
it as `ceil(actualM/2)` and `floor(actualM/2)`. The capacity M64 does not imply
that the first AIV owns all rows up to32. For these q=1 cases, physical16 heads
therefore require8+8 rows. MTE3 probability and output transfers use the same
row ownership. Fixpipe rounds only its transfer dimensions; it does not
change the Vector valid-row partition. FD owner metadata remains independent.

Nine H1/2/16/31/32/33/48/63/64 controls place an external contender on AIV0 and
check the exact separate-owner service bound. The20 focused tests pass.
All12 actual-context single-kernel comparisons preserve declared physical
work and zero live tasks/buffers. Source-compatible estimates decrease slightly:
B16/context4104 changes21.6309→21.2843us; B16/context16392 changes78.2555→77.9089us.
The retained means are47.6727us and142.0950us. This correction therefore does
not close the observed residual and is not presented as accuracy progress.

The compact PMU receipts show B16 uses32AIC/64AIV, consistent with the declared
host owner count. B1 actually uses17AIC/34AIV at4K and26AIC/52AIV at16K, whereas
the retained CPU-host declaration uses32AIC. The B1 scenario is therefore a
source-compatible dispatch alternative, not capture-exact dispatch. B16/16K
has mean MTE2 97.15us, MTE1 55.38us, MAC44.85us, and AIC scalar70.89us inside
an observed142.10us kernel. These counters overlap and cannot be added as a
residual floor. C1-C1-C2 source ordering and inner L0 instruction scheduling
remain separate modeling gaps. Evidence and a local producer are retained as
`data/verification/kimi-paged-mla-owner-20261030*`.


## Source C1-C1-C2 command sequence (2026-10-30)

`ExecuteTask` at kernel lines493–509 first invokes C1(i), then C2(i−1), with
PRELOAD_N=2. The AIV driver analogously issues V1(i) before V2(i−1). In the
Cube block, each C1/C2 call waits for `M_FIX` before submitting its Fixpipe;
the next call cannot begin a new operand load while the preceding MMA is
still executing. Treating QK and PV as independent streams permitted an
unsupported overlap and understated the source program's physical span.

The finite stage DAG now has one prologue iteration and a final C2/V2 drain.
Cube command order is C1(0), C1(1), C2(0), C1(2), C2(1), …, C2(last).
AIV, Fixpipe and directional transfers retain independent resources; no
global all-pipeline barrier was added. Three L1 slots, both modulo-two UB
releases, immutable service amounts and fixed owner buffers remain. The
individual L0 double-buffered sub-instruction overlap inside C1/C2 is still
represented by aggregate operand/compute phases, not a claimed instruction
trace.

All12 conditional FIA single-call controls preserve exactly the declared
physical services. Examples in microseconds:

| Batch / context | Half-owner control | Ordered source | Retained mean | Signed relative gap |
|---|---:|---:|---:|---:|
| 1 / 4104 | 4.544 | 4.603 | 16.075 | −71.36% |
| 8 / 16392 | 40.795 | 71.851 | 79.395 | −9.50% |
| 16 / 4104 | 21.284 | 37.416 | 47.673 | −21.52% |
| 16 / 16392 | 77.909 | 145.473 | 142.095 | +2.38% |
| 32 / 4104 | 39.694 | 72.974 | 74.792 | −2.43% |
| 32 / 16392 | 152.943 | 289.089 | 264.091 | +9.47% |

These are conditional source/reference comparisons, not heldout validation
or complete-body performance. Small-batch dispatch identity remains different
from the retained CPU host declaration. The52 focused checks include five
actual Engine traces for prologue/flush and K tails, and nine independent
copy-plus-MMA duration oracles where all other services are negligible.
Twenty-four additional real A5 resource controls at0.5/1/2× HBM, AIC, AIV,
and matrix supply remain strictly responsive and release all resources.
Neither launch parameters nor hardware bandwidths were fitted.

Reproduce the12 same-tree comparisons with:

```sh
python data/verification/kimi-paged-mla-owner-20261030.producer.py \
  /tmp/kimi-paged-mla-order.json 15ba9e7e source_half_only source_c1c1c2
python data/verification/kimi-paged-mla-order-20261030.rates.py \
  /tmp/kimi-paged-mla-rates.json
```


The subsequent complete-body campaign is pinned to `90ae9d2e`, based on
`96b4a043` plus only the actual-M ownership and command-sequence corrections.
All12 unchanged `kimi-source-complete` inputs completed all93 layers within
the existing60-second per-case limit. The runtime package hash before and
after every worker was
`4e122d795d10e137b2bb560417423904926baad4b7aa7db5b32c805453bed05e`.
Every result has zero live tasks/programs and released allocations. The
largest local elapsed time was 34.67s; peak RSS stayed below 159.98MiB.

| Batch | Body ms, context4104 | Body ms, context16392 |
|---:|---:|---:|
| 1 | 23.391264 | 23.553350 |
| 2 | 23.515499 | 23.835845 |
| 4 | 23.752987 | 24.384306 |
| 8 | 24.198061 | 25.458716 |
| 16 | 25.616089 | 28.209447 |
| 32 | 28.543760 | 33.730514 |

These are complete graph predictions, not native operator activity sums or
an assertion of body accuracy. They retain the preceding source scenario's
communication, matrix and KDA declarations. A subsequent TP/communication
change must use a different runtime identity. The full compact receipt and
60-second subprocess producer are `kimi-paged-mla-body-20261030*`.

## Source SplitCoreV2 128K control

The six current cases in `examples/cases/kimi-a5-paged-host-20261030/` bind the
geometry planner explicitly. Their MLA core errors at B1/2/4/6/8/12 are
−12.38%, −0.96%, +3.50%, +5.69%, +6.23% and +7.72% against the retained native
means. These are source-compatible controls with unobserved binary identity;
they do not establish whole-Body accuracy or a trained heldout model.

The complete 93-layer B1 Body finishes in 21.02 seconds at 102.0 MB peak RSS,
predicting 27.01558 ms against 32.48482 ms native mean (−16.84%, fails 15%).
B12 reaches the unchanged 90-second execution budget at 102.7 MB peak RSS and
remains failed; no complete prediction is reported. The other four complete
Body geometries were not rerun in this control. Both successful primitive and
failed complete-Body outcomes are retained under
`data/verification/kimi-a5-paged-host-20261030/`.

Reproduce with the adjacent `.producer.py` using `prepare`, `core`, or
`body --batch 1` / `body --batch 12`. Core comparison requires the retained
native-family ledger via `--native-ledger`; Body execution needs only the
committed input cases. The compiler oracle is `tools/paged_mla_host_v2_oracle.cpp`
linked against the pinned original `split_core_v2.cpp` and header. Its recorded
portability adaptation only specifies `std::max<uint64_t>` explicitly.
