# Open-source mechanism absorption

This review on 2026-10-29 inspects pinned official source, then checks particular
mechanisms against independent arithmetic. It does not run external simulators
or establish hardware accuracy. The scope is a reusable physical abstraction,
not a claim that copying more source improves prediction.

## 1. Finite credits, serialization and response are separate

Garnet's `NetworkInterface::scheduleOutputPort` requires both a ready flit and a
credit, decrements that credit, and sends at most one flit on that port per
invocation. Incoming credit events restore capacity. `NetworkLink::wakeup`
schedules arrival using link latency while scheduling its next transmission
independently. This establishes why an outstanding-request window and
propagation cannot generally be collapsed into one average bandwidth.

Inspected source, gem5 revision
`5a29fd6f8c556f11d4308e07ba6019b816921930`:

- [NetworkInterface.cc](https://github.com/gem5/gem5/blob/5a29fd6f8c556f11d4308e07ba6019b816921930/src/mem/ruby/network/garnet/NetworkInterface.cc):
  `scheduleOutputPort`, lines 486-543; incoming credit handling, lines 282-296.
- [NetworkLink.cc](https://github.com/gem5/gem5/blob/5a29fd6f8c556f11d4308e07ba6019b816921930/src/mem/ruby/network/garnet/NetworkLink.cc):
  `wakeup`, lines 82-110.

The adopted model is smaller than Garnet: a transfer invocation has a serial
transmission stream, a finite credit count, and a fixed response delay after
each transaction's service. It uses the common `Engine` for competition with
other invocations. It does not add flit routing, virtual channels or inferred
store-and-forward hops.

For one otherwise idle resource, let transaction service time be `s`, response
delay be `R`, credit count be `O`, and transaction number be `i`. An independent
recurrence is:

```text
service_end[i] = max(service_end[i - 1], response_end[i - O]) + s
response_end[i] = service_end[i] + R
```

The initial absent service/response times are zero. For `N > 0` equal-sized
transactions this gives:

```text
last_response =
    N*s + R + floor((N - 1)/O) * max(0, R - (O - 1)*s)
```

It includes pipeline fill, credit starvation and final response. Five
transactions with `s=1 ms`, `R=4 ms`, `O=2` return at
`5, 6, 10, 11, 15 ms`. The former aggregate formula
`R + bytes / min(bandwidth, O*transaction_bytes/R)` gave `14 ms`.
For one transaction it gave `6 ms`, although that transaction takes `5 ms`.
The error changes sign with transfer length; one multiplier cannot fix it.

Implemented:

- `prediction/transfer.py` holds credits until response and issues work lazily.
- `TransferRecipe` continues to use the ordinary recipe/graph engine.
- `tools/finite_transfer_oracle.py` implements the independent recurrence and
  closed form without importing the simulator.
- `tests/test_transfer_program.py` checks finite responses, contention,
  independent paths and live-state bounds.
- `tests/test_finite_transfer_oracle.py` checks 180 equal-transaction cases,
  with additional finite-tail controls.

The oracle's optional exact tail describes a different wire contract. The
runtime recipe currently rounds the final transaction to `transaction_bytes`.
The existing `QueuedStore.response_ms` is an issue-relative minimum response
span that overlaps its memory service. That separate declared behavior is
preserved; it is not silently reinterpreted as post-service propagation.

Reproduce a finite-credit control:

```sh
.venv/bin/python tools/finite_transfer_oracle.py \
  --payload-bytes 50 --transaction-bytes 10 \
  --bandwidth-bytes-per-ms 10 --outstanding 2 --response-ms 4
```

## 2. Shared-memory reuse does not erase private operand copies

LLMCompass represents L2 and L1 tiles separately. Mapping enumeration considers
their working-set capacities and double buffering; tile execution accounts for
loop-order-dependent reuse. `L2TileSimulator` separates device I/O from its
nested L1 compute simulation. The relevant lesson is to identify the memory
boundary at which reuse removes traffic.

Inspected source, LLMCompass revision
`2e015fd2ee750e6cad8d7152df52551e5b41ef20`:

- [software_model/matmul.py](https://github.com/PrincetonUniversity/LLMCompass/blob/2e015fd2ee750e6cad8d7152df52551e5b41ef20/software_model/matmul.py):
  `Mapping`, lines 217-256; exhaustive capacity screening, lines 314-395;
  tile reuse and double buffering, lines 894-954;
  `L2TileSimulator`, lines 972-1040.

A two-owner BF16 matrix gives a small counterexample: `M=1, N=2, K=1`, with one
`1x1x1` tile on each owner. Shared-cache reuse needs one HBM read of A (2 bytes),
two B elements (4 bytes), and two output writes (4 bytes): 10 bytes total.
Each owner still receives its own A element and B element, so private operand
copy service is 8 bytes. The previous program reported 6 private-copy bytes
because it reused the HBM byte count for the local transfer resource.

Implemented in `prediction/matrix.py`:

- `reuse_inputs` and `reuse_weights` select ideal shared-cache HBM reuse.
- Private input-copy traffic follows each tile's M/N/K operand slice, regardless
  of whether those bytes hit the shared cache.
- Compute work, output stores and explicit per-owner buffer lifetimes remain
  separate from these two input-traffic quantities.

`tests/test_matrix_hierarchy.py` checks the two-owner arithmetic, ragged M/N/K
tiles, packed weights with activation scales, and a bottleneck transition.
With private-copy capacity set to 2 bytes/ms, both cached and uncached variants
take 4.08 ms. With HBM capacity set to 2 bytes/ms, they take 5.04 and 6.04 ms.
The constants are synthetic resource controls, not measured device parameters.

Remaining scope:

- Ideal shared-cache reuse remains a declared mapping assumption. There is no
  inferred L2 hit rate, cache replacement simulation or finite shared-cache
  capacity claim.
- Packed weight and activation-scale bytes retain their declared totals.
  Their placement across K panels uses the existing proportional approximation;
  this does not reconstruct packed block boundaries or compiler reload choices.
- No automatic mapping search is introduced. A fastest candidate is a result
  for the declared mappings, not evidence that a deployed kernel selects it.

## 3. Profile domains and resource identification precede interpolation

Vidur's compute-profile loader filters structural conditions such as head
counts, embedding/MLP dimensions and tensor-parallel degree before fitting.
AISimulate's empirical implementation likewise separates categorical slices
from numeric interpolation; its source explicitly describes utilization as an
effective factor that can exceed one.

Inspected sources:

- Vidur `abae7f63aa857300f5cdc6f5e0d27860cd24721b`,
  [sklearn_execution_time_predictor.py](https://github.com/microsoft/vidur/blob/abae7f63aa857300f5cdc6f5e0d27860cd24721b/vidur/execution_time_predictor/sklearn_execution_time_predictor.py),
  `_load_compute_df` and `_load_attention_df`, lines 103-168.
- AISimulate `17625dc8e6b0ef12b6da43fbf08f19cfd6e8b77e`,
  [util_empirical.rs](https://github.com/ai-dynamo/aisimulate/blob/17625dc8e6b0ef12b6da43fbf08f19cfd6e8b77e/crates/core/src/perfmodel/operators/util_empirical.rs),
  module contract and `UtilGrid`, lines 4-30 and 146-168.

This simulator already retains categorical hardware/runtime/kernel/layout/
owner/boundary identity in `ServiceCurve` and accepts only one explicit resource
phase in `ResourceService`. Those are useful mechanisms to preserve. An
eight-millisecond opaque duration could represent eight milliseconds of memory,
eight milliseconds of compute, or overlapping portions of both. It does not
identify the sharing behavior when another memory consumer arrives.

We therefore do not transfer SOL-relative utilization, model-partitioned timing,
GQA batching fractions or whole-forward residuals into physical rates.
Numeric interpolation remains inside an explicitly identified phase/domain.
Analytical simulation still runs without measurements; calibration coverage
and independent validation remain separate outputs.

Multidimensional service surfaces and automatic dispatch-region selection
remain open work, to be driven by real shape/ranking counterexamples. Adding
those surfaces does not require changing the physical execution engine or
making measurements an admission gate.

## 4. Receive-forward order is separate from aggregate link work

The next implementation wave reviewed NCCL's ring allgather, reduce-scatter
and allreduce at fixed tag `v2.26.6-1`. Initial send, receive-forward steps and
final receive supply a causal structure that an aggregate latency term loses.
Source paths, Git blobs and reviewed ranges are retained in
[collective rounds](operators/collective-rounds.md).

The adopted transport abstraction separates source serialization from message
arrival. The next send waits for prior source service and predecessor delivery,
while propagation can overlap other work. This extends the earlier gem5-inspired
separation of service, response and credits; it does not merge different protocol
contracts or reproduce NCCL channels and reduction kernels.

The ordinary Engine gives 13 ms for a four-rank asymmetric allgather where the
old aggregate envelope gives 33 ms, with identical bytes. Concurrent calls share
the real link resources. Explicit whole-message multihop forwarding is available;
packet credits, cut-through, uneven chunks, reduction compute and cross-DP round
coordination remain outside this recipe. No timing factor or automatic runtime
algorithm selection is inferred from the source.

Historical callback absorption also now follows the same layer separation:
audited arguments build full device work, declared submission composes with it,
and original events remain an independent comparison. See
[complete call analysis](call-analysis.md) for supported domains and actual
remaining differences. An unpaired event does not block analytical simulation.
