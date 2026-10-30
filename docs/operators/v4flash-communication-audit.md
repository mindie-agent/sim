# V4Flash A3 TP8/EP8 communication audit

This is a descriptive audit of one retained rank, not a V4.1 result, a heldout
accuracy claim, or an inferred HCCL algorithm. Raw activity sums, interval
unions and isolated predicted program sums retain separate boundaries.

## Capture and original comparison

The input is DeepSeek V4 Flash (`deepseek-v4-flash-0731`), A3 TP8/EP8/DP1,
4,096-token prefill, one sequence, context zero, 43 target layers plus MTP1.
The capture is
`/Volumes/data/mindie-sim-data/a3/2026-10-23/v4f-prefill/raw/pre4096.kernel_details.csv`,
SHA256 `b56b7439213cc9f76eaae63347393d4add6ae5c985bf28a6a7737f5783c81788`.
It contains 4,064 positive rows, 46 columns, 1,391,917 bytes and only Device_id8.
The request window is 575.235407 ms and the all-activity sum is 609.513832 ms.
This single request is not a repeated-trial median or a whole-cohort timeline.

The original scoreboard compared 149.270651 ms of typed-HCCL activity against
39.644353469 ms of isolated predicted communication programs. The difference
109.626298 ms was neither a whole-model gap nor a communication critical path.
The audit report records the input file hashes so later scoreboard corrections
do not silently change the historical comparison.

## Projected views, wrappers and interval arithmetic

| Class | Rows | Sum ms | Interval union ms |
| --- | ---: | ---: | ---: |
| Named `hcom_*` | 312 | 111.032451 | 111.032451 |
| `Name=AivKernel, Type=hcom_*` projections | 178 | 38.238200 | 38.238200 |
| AICPU control/wrappers | 134 | 49.203160 | 49.203160 |
| All three classes | 624 | 198.473811 | 129.606120 |

There are zero exact full-record duplicates. All 178 projected rows have
one-to-one matching named rows on device, type, core, start and end. Names,
stream IDs and task IDs differ. This supports **projection candidates**, not
proven physical-event identity: the exported CSV has no producer/correlation
join. The raw population remains intact. A canonical named-HCCL view and an
all-typed-HCCL union can be reported separately; timestamp agreement must not
become general identity deduplication.

The original family sum is exactly 111.032451 + 38.238200 = 149.270651 ms.
All typed-HCCL intervals have union 111.032451 ms. This union does not require
claiming the matching rows are physically identical. AICPU intersects named
communication by 30.629491 ms; its outside-union portion is 18.573669 ms.
Adding native and wrapper sums would count overlap. The 129.606120 ms union
includes control and therefore is a different boundary from named-HCCL time.

Named communication intersects non-communication, non-wrapper activity by
2.439153 ms. Its remaining 108.593298 ms is occupancy without that overlap,
not a dependency-derived critical path. No causal timeline is reconstructed
from these interval sums.

The accepted audit first merges overlapping intervals within each set before
intersection. Projection matching includes device identity and requires one
named row and one projection per key; unmatched and ambiguous rows stay
visible. Wrapper time outside native uses wrapper union, not wrapper sum.
The dedicated capture audit caps input at 100,000 streamed rows.

## Physical family counts

| Family | Named count / ms | Original prediction count / ms |
| --- | ---: | ---: |
| allreduce | 90 / 57.195819 | 88 / 29.444197 |
| allgather | 90 / 21.445021 | 2 / 0.185951 |
| alltoallv | 132 / 32.391611 | 88 / 10.014206 |
| total | 312 / 111.032451 | 178 / 39.644353 |

The named-only difference is 71.388098 ms. No portion of this residual is used
as a rate, startup constant or floor. In the allgather population, logical name
suffix503 has 46 calls/15.640371 ms and suffix703 has 44/5.804650 ms. Native
CSV Stream ID is N/A: these suffixes are name groups, not inferred device streams.

A five-millisecond time-gap clustering groups the 132 alltoallv calls into
44 clusters of three. Ordinal sums are 3.009098, 6.039040 and 23.343473 ms.
These are descriptive time clusters; neither layers nor payload roles are
proven by ordinal position. In particular the third call cannot be assigned a
larger tensor merely because it lasts longer.

## Source-supported protocol

The official independently retained source is vllm-ascend revision
`8fd0bc4885cb52e58791aaa8efacc571b9bd115d`, file
`vllm_ascend/ops/fused_moe/token_dispatcher.py`, SHA256
`1dcfd8181cb2d34c584d8336c3379dbeefba75c0bf664f924063701b93295cbb`.
`TokenDispatcherWithAll2AllV` supplies a concrete quantized branch:

1. `_preprocess`, lines616-641, constructs per-expert counts and gathers them
   across the EP group; splits also have CPU extraction/control work.
2. `token_dispatch`, lines507-516, dynamically quantizes packed tokens,
   exchanges row scales separately, then waits for that exchange.
3. Lines518-522 exchange the quantized token payload and wait.
4. `token_combine`, lines559-576, exchanges expert output using the forward
   split counts in reversed input/output roles.

This branch depends on `dispatch_with_quant`, its quantization format and the
absence of the LoRA override. It does not establish the capture's loaded binary
or algorithm. The source supports an explicit histogram-gather / scale-exchange /
payload-exchange / reverse-output protocol. It explains a candidate missing
call structure, not the remaining measured time. Source CPU split extraction,
post-permutation and synchronization need their own boundaries if implemented.

A declared INT8 forward payload and BF16 expert output reverse the same peer
**row counts**, with their respective element widths. Transposing INT8 byte
totals unchanged would undercount BF16 return bytes. Scales follow the same
routing rows but remain a separate collective identity.

## Existing prediction's ownership and algorithms

The original case binds `steady_matmul` and native fusion, but no collective
family or recipe. Its 178 communication programs are `ResourceEnvelope`.
Names such as `ring_allreduce` do not select `RingCollective`; `steps` supplies
framing/setup arithmetic without causal round dependencies. Explicit ring or
pairwise scenarios remain distinguishable from the unknown native HCCL choice.

For the declared balanced DP1/TP8/EP8 case, 4,096 x top-k6 = 24,576 assignments
are partitioned across eight source ranks. Each source owns 3,072 rows, of which
384 are local. At BF16 hidden4096 this is 3,145,728 self bytes and 22,020,096
remote bytes per source per layer. Local self traffic is not a link message;
pack/unpack account the declared local tensor work. The original per-rank
7,134,737,344 link bytes reconcile as:

```text
2 * 44 * 58,720,256   attention and MoE/shared allreduce
2 * 44 * 22,020,096   BF16 dispatch and combine
      29,360,128      draft features gather
         226,240      logits gather
```

That balanced case does not exercise an asymmetric inverse peer matrix. The
runtime already has explicit peer-traffic transposition; the selected quantized
protocol must additionally preserve row representation when the return dtype
changes. No measured routing matrix is inferred from the capture.

## Independent acceptance and reproduction

DSH commit `db8cc1d9` supplied the first audit. Independent acceptance corrected
intersection over overlapping inputs, rank/cardinality matching, wrapper-union
subtraction and overclaimed producer identity. Added counterexamples cover each
of those behaviors and retain the exact raw-capture arithmetic. The maintained
tests derive current scoreboard residuals from current inputs instead of freezing
the old misaggregation as correct behavior.

```sh
python3 tools/audit_v4flash_communication.py --out data/verification/v4flash-communication-audit-20261029.json
python3 -m pytest -q tests/test_v4flash_communication_audit.py
```

No shared runtime, scoreboard, raw capture or hardware state was changed by
this audit. Observations remain descriptive; no train/heldout role is inferred.
