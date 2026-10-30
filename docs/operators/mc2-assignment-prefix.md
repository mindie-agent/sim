# MC2 sender assignment-prefix work

The arch35 full-mesh sender scans preceding expert IDs for each assignment after
the first token. This is independent of the count-core scan of every expert.
The new work is bound to the existing Vector resource on the ordinary Engine;
no launch floor, effective-throughput multiplier, or observed residual is added.

Source: `cann/ops-transformer` revision
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8`, dispatch file
`mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_v2_a5_full_mesh.h`.
The repository revision and current code are inspectable at
`/Users/maoxx241/code/refs/ops-transformer`. A match to the captured binary has
not been established by this source reference.

## Source arithmetic and owner contract

`SendToMoeExpert`, lines 652–663, assigns expanded index `i` to AIV
`i % moeUsedAivNum_` and calls `CalTokenSendExpertCnt(expertId, i, ...)`.
Lines 778–804 return immediately for `i < top_k`. Otherwise Duplicate, Sub,
Abs and Mins each process `i` elements, followed by ReduceSum with `i-1`
additions. The resource ledger therefore adds `5*i-1` Vector element operations.

For complete, integer, valid assignment grids `S = source_tokens * top_k`,
the total is `sum(5*i-1, i=top_k..S-1)`. For owner `u` and `P` payload AIVs:

- `first = top_k + (u - top_k) % P`
- `count = max(0, (S - 1 - first) // P + 1)`
- `sum_indices = count * (2*first + (count-1)*P) // 2`
- `owner_vector_work = 5*sum_indices - count`

This takes O(AIV) construction and stage count, including tails and inactive
owners. Per-owner prefix work runs before that owner's read/quantization stage.
The program retains its existing whole-phase owner ordering; it does not infer
fine-grained per-assignment UB window overlap.

`CalAndSendCntByRankAndExpert`, lines 897–923, independently calls
`CalExpertCnt` once per expert over the complete ID array. Its existing
`E*(5*S-1)` work remains unchanged, as do payload HBM bytes, framing,
quantization and successful-poll work. In particular, adding sender scans must
not add a second copy of the original expert-count work or quantization.

If a caller supplies fractional token counts or filtered assignment totals,
the retained ID indices are unknown. The new scan alone is marked unresolved;
existing analytic work continues. A total count is never treated as invented
"first N" retained IDs.

## Independent scalar/control service still required

The selected subset now has an explicit hardware-service binding in
[MC2 source-operation service supply](mc2-instruction-services.md). The table
below describes the default without declared instruction services; unknown
polling, address/control and event operations remain unpriced.

The source defines additional fixed work, but source operation counts are not
measured issue times. These counts must not be converted using GDN's
`scalar_store_issue`: that parameter describes a BF16 UB-to-GM store path and
finite queue, not UB scalar reads/writes or scalar arithmetic.

| Source | Fixed work for the stated no-mask/no-elastic branch | Pricing boundary |
|---|---|---|
| Sender 655, 799; `FillTriple` 472–478 | S expert-ID UB reads, max(S-K,0) reduction-result UB reads, 3S triple UB stores | Scalar UB service unpriced |
| `CalAndSendCntByRankAndExpert` 897–923 | E count-result reads and E status stores | Existing Vector count scan remains priced |
| `CalRecvAndSetFlag` 1126–1162 | C prefix reads, E count reads, E source-level additions and E prefix stores | Scalar arithmetic unpriced; metadata now records E additions, not E-C |
| `WaitDispatch` 1087–1122; `GetCumSum` 1024–1083; `WaitCumSumFlag` 1221–1246 | One successful scalar status test per participating owner, plus explicit control stores | Existing successful-poll Vector/HBM work remains; retry counts are unknown |
| `CheckDataArriveWithFlag` 1299–1316 | Successful payload flag reduction and scalar result test per actual receive batch | Batch count needs source/expert cell counts and native UB capacity |
| Combine `ExpertAlltoAllDispatchCopyAdd` 855–896 | Three expandIdx UB reads per returned row | Scalar address/issue service unpriced |
| Combine `CheckPackedFlagRangeArrive` 958–993 | One UB word read and SFF per 64-bit successful flag mask | CompareScalar and flag transfer already priced |
| Combine `ProcessMoeExpertsLoop` 1385–1402 | One expert-weight scalar read per output token/slot when present | Existing weighted Vector arithmetic remains priced |

The combine file is
`mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_mte.h`.
Unknown failed polls depend on arrivals and cannot be derived from a body
residual. Additional scalar/control rates require independent physical service
declarations. Missing rates do not disable analytic simulation.

## Actual ordinary caller phase controls

The retained report is
`data/verification/mc2-assignment-prefix-20261030.json`; its adjacent producer
constructs complete ordinary body graphs, selects their real dispatch sender
phase, and runs that phase on the Engine using each case's unchanged A5 rates.
The counterfactual removes only this newly accounted prefix work and leaves
owners, dependencies, payload bytes and quantization unchanged.

| Caller | Local source tokens | Added Vector operations/call | Sender phase delta (µs) |
|---|---:|---:|---:|
| Qwen3.5 c4, TP token partition | 4 | 2,316 | 0.00141 |
| Qwen3.5 c64 | 64 | 653,436 | 0.12179 |
| Qwen3.5 c128 | 128 | 2,617,724 | 0.46582 |
| Kimi c1, SP local rows | 1 | 0 | 0 |
| Kimi c32, SP local rows | 4 | 2,316 | 0.00141 |
| V4.1 c1, symmetric DP8 | 1 | 0 | 0 |
| V4.1 c64, symmetric DP8 | 8 | 5,523 | 0.00217 |

These small increments do not explain the multi-millisecond MC2 body residual.
They are isolated phase costs, not additive body improvements or measured
accuracy. Explicit Vector rate sweeps preserve work and produce nonincreasing
phase durations; inputs released at 7 ms receive no service before release.
All seven probes finish with zero live tasks/programs.

Focused verification: **95 tests passed**, covering integer tails, tiny grids,
filtered/fractional unresolved identities, owner arithmetic, late readiness,
actual INT8/MX producers and ordinary TP caller paths. A 32,768-assignment
control still completes in fewer than 256 Engine tasks.
