# GLM Decode grouped source and physical-work audit

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The two largest retained GLM Decode grouped discrepancies remain device-program
gaps. Removing the old default host submission latency does not close them:
the rerun below has **zero host service**. Active expert weights and the fused
quantization/output paths are already charged. The confirmed omission is the
matrix operand pipeline: the selected W4A8 source programs issue no MTE1 work
and do not establish L0 operand/accumulator lifetimes.

This is a read-only audit of runtime revision
`0061a39f059f231ca687679eef2f1d7f92d0a87e`, not a mechanism fix or an assertion
that this omission explains the entire residual. The retained PMU evidence also
shows substantial scalar/control activity whose causal timing remains unresolved.

## Reproduction and comparison boundary

The compact evidence
contains the streamed native contracts, all PMU aggregates, source hashes, and
the exact prediction resource ledger. Its
producer
retains constant aggregate state while scanning the gzip CSV.

```sh
.venv/bin/python data/verification/glm-msd-final-profile-20261030.producer.py \
  . /tmp/glm-decode-grouped-audit-current-20261030.json
.venv/bin/python data/verification/glm-decode-grouped-audit-20261030.producer.py \
  /Volumes/data/mindie-sim-data/a3/2026-10-23/glm53-attn/raw/kernels/dc4k_b1.csv.gz \
  /tmp/glm-decode-grouped-audit-current-20261030.json \
  data/verification/glm-decode-grouped-audit-20261030.json
```

The original capture is 9,236,949 compressed bytes with SHA256
`ed5e9696949d51c2d1d31ab099a71a2c98722d5719f004fcf90c9ceff036f60c`.
The scan covered 166,204 rows in 0.697 seconds. It selected 5,550 events for each
of the two exact tensor contracts; all have `OP State=static`, `Block Num=20`,
and `Mix Block Num=40`.

The prediction producer uses ordinary `target_graphs` and `graph_costs` with the
explicit source recipes. Its only case patch is `machine.backend="ascend"`.
Runtime content was stable across the 0.539-second execution; both predictions
finished with zero live tasks/programs. No device access occurred.

| Native complete kernel boundary | Selected recipe | Observed mean µs/call | Predicted µs/call | Signed gap µs/call | Relative gap |
|---|---|---:|---:|---:|---:|
| GroupedMatmulSwigluQuantV2 | `ascend_w4a8_msd` | 88.038726 | 15.932216 | −72.106510 | −81.90% |
| GroupedMatmul | `ascend_w4a8_msd_grouped_down` | 52.907483 | 7.263622 | −45.643861 | −86.27% |

Both are in the `>50 µs/call` tier and fail the 15% relative criterion. These
are conditional exact-tensor-contract comparisons: native binary identity and
the actual captured `group_list` values are not recovered. The observation is
an entire fused native kernel, not an unfused GEMM or a sampled inner stage.
The predictions also include their declared fused stages. This capture cannot
validate a full target-body Forward boundary; see
[Decode body evidence](../decode-body-evidence.md).

## Tensor identity and routing

The gate/up capture inputs are:

```text
shapes: 8,6144;8;256;256,8,384,16,64;256,512;256,512
dtypes: INT8;FLOAT;INT64;DT_INT4;UINT64;FLOAT
formats: ND;ND;ND;FRACTAL_NZ;ND;ND
outputs: 8,256;8 (INT8;FLOAT)
```

The down capture inputs are:

```text
shapes: 8,256;256,256,6144;256,6144;256,1,6144;;;;256;8
dtypes: INT8;DT_INT4;FLOAT;UINT64;FLOAT;FLOAT16;FLOAT16;INT64;FLOAT
formats: ND;FRACTAL_NZ;ND;NCL;ND;ND;ND;ND;ND
output: 8,6144 (BF16)
```

The scenario declares 256 experts and the integer histogram
`[1] * 8 + [0] * 248`: logical gate shape `(M,N,K)=(8,512,6144)` and down
`(8,6144,256)`. For one token and ordinary distinct top-8 selection, eight
one-row active experts follow structurally. This is a conditional inference;
the capture does not identify their physical IDs or prove the forced-balanced
scenario. Prefixing the eight active entries is a declared scenario, not a
reconstruction of the captured routing values.

The model already reads every declared active packed weight bank:

| Work per invocation | Gate/up | Down |
|---|---:|---:|
| Active expert packed weights | 12,582,912 B | 6,291,456 B |
| Total modeled HBM service | 12,863,552 B | 7,680,384 B |
| Modeled AIC MTE2 service | 12,713,984 B | 6,709,248 B |
| Modeled AIV MTE2 service | 81,952 B | 674,176 B |
| Modeled MTE1 service | **0 B** | **0 B** |
| Modeled Fixpipe service | 16,384 B | 196,608 B |
| Modeled MTE3 service | 51,232 B | 100,352 B |

Thus “256 experts means 256 full weight banks” would overcount, and “only the
eight input rows were charged” is also incorrect. A scan of 256 metadata entries
is distinct from eight active matrix banks. `_down_pre` already charges the
2,048-byte count vector on each of 40 participating AIVs, including owners with
no input row, plus vector reduction work. The unresolved issue is the actual
scalar/control execution of those scans and synchronization, not absent argument
bytes alone.

## What the retained PMU counters establish

| Mean counter µs | Gate/up | Down |
|---|---:|---:|
| Task duration | 88.038726 | 52.907483 |
| AIC total | 77.892981 | 34.460890 |
| AIC MAC | 1.249895 | 0.623178 |
| AIC scalar | 54.058497 | 24.992925 |
| AIC MTE1 | 7.802860 | 4.005699 |
| AIC MTE2 | 6.413945 | 9.176572 |
| AIC Fixpipe | 1.139035 | 6.302935 |
| AIV total | 83.380904 | 45.984386 |
| AIV vector | 0.238618 | 0.616835 |
| AIV scalar | 24.032541 | 30.579028 |

The counters expose real MTE1 activity absent from the prediction and indicate
that arithmetic throughput alone cannot describe this shape. They overlap and
are not additive stage times. In particular, neither scalar time nor the
task-minus-AIV difference is a justified invocation floor. The capture does not
separate scalar issue, polling, contention and dependencies into independently
calibrated rates. This audit introduces no multiplier, rate fit or residual floor.

## Exact source evidence and the omission

The down kernel is pinned to ops-transformer revision
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8`. The local file
`refs/ops-transformer/gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd.h`
has SHA256 `1851c45726a01ada89a2c2f034021185cd5a477bef963037338e6dba0e02e302`,
matching the runtime source declaration. Its relevant behavior is:

- Lines 259–282 synchronize, scan each group count, skip empty groups, double
  active rows for MSD and assign a rolling output-tile grid to physical owners.
- Lines 326–355 wait for reusable output space, set the matrix shape, submit A/B
  operands and quantization vectors, iterate the matrix program and publish C.
- Lines 342–343 disable weight L2 caching when `blockDimM == 1`.
- Lines 359–439 synchronize the AIC/AIV producer-consumer path. Subsequent
  vector code performs the dequantization/rescale/assist/output work.

The compatible gate sources in the local ops-transformer mirror were also
inspected, but their hashes **do not match** the runtime's pinned vllm-ascend
gate source. They are a structural cross-check only, not proof of native binary
identity or an exact replacement for that pinned gate version.

In the current model, `w4a8_schedule._matrix_tiles` collapses each owner's output
rectangles into aggregate INT4 arithmetic, HBM/MTE2, Fixpipe and vector work.
`w4a8._owner_program` and `w4a8_down._owner_program` emit that aggregate as one
`Stage(work=...)` per owner. They never call the shared matrix hierarchy and
never request MTE1 service or L0 allocations. This bypass is why fixing the
device adapter's MTE1/service-mode scaling cannot by itself affect these source
programs.

The source base/output partition is already present: the gate scenario has 16
output tiles on 20 AIC owners; down has 96 output tiles with the source's CV
pairing. Replacing this with a global dense matrix would erase expert boundaries
and alter the very low-occupancy case under investigation.

## Narrow next mechanism change

Keep the existing source schedules, expert identity, owner assignment and fused
pre/post stages. Replace **only the matrix phase** with the shared matrix core:

1. Expose lazy owner rectangle iterators from the W4A8 schedule, retaining expert
   ID, doubled rows, N/K offsets and tails and output-publication dependencies.
2. Follow the existing `call_grouped_source.py` interface:
   `HierarchicalMatmul.lower(matrix_context, owner_tiles=..., output=...,
   serial_tiles=True)`. Its explicit owner-tile streams already avoid a second
   execution engine. Adapt INT4 packed representation and the source output
   boundary; do not copy that module's INT8 output constants blindly.
3. Account for directional MTE1 service and L0A/L0B/L0C lifetimes using the shared
   matrix machinery. Preserve already charged preprocessing, counts, assist,
   workspace publication, Fixpipe and final output exactly once. Aggregate
   matrix-stage work must be replaced, not added to the new program.
4. Preserve bounded execution with the shared steady regular-K lowering after
   verifying its equivalence on small detailed controls. Do not expand a full
   model into unbounded per-panel Engine tasks.

Acceptance should include physical FLOP/byte conservation, nonzero MTE1 service,
released local buffers, source owner/tail geometry, and an MTE1 half/double scan
that affects a deliberately MTE1-bound control. A noncritical resource need not
change total latency. Recompare the retained native boundaries after that fix;
remaining scalar/handshake/dispatch discrepancies stay visible for a separate
mechanism investigation. No current evidence guarantees this change alone will
bring the 88/53 µs native kernels within 15%.
