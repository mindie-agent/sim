# Frozen A5 HCCL responses on independent long-context captures

The unchanged older HCCL responses improve the new TP4 B4 communication
comparison, but both complete Bodies still miss the 15% criterion: Qwen3.6 is
24.353 ms versus 28.964 ms observed, and Qwen3.8 is 24.601 ms versus 29.240 ms.
No new HCCL coefficient, domain extension, cross-TP substitution or Body-derived
correction is used. These are conditional source candidates; the captured
algorithm and binary identity remain unobserved.

## Frozen boundaries and domains

`data/observations/a5-hccl-service-regimes-20261030/tp4-parameters.json` supplies
the original TP4 BF16 `mesh_twoshot`, Block60, separate-buffer response:

```text
native-call us = 24.619873955981763 + 0.000010624772236973017 * payload_bytes
valid payload: 65,536 .. 2,097,152 bytes
```

Its assignment sources are older Qwen3.5 c4/c128 native-call ledgers; c64 was
excluded from coefficients but previously inspected. The response measures a
deduplicated complete AivKernel. It is an inclusive group completion constraint,
not an added control delay or a bandwidth measurement. The source program still
services its physical work. Native flag/poll/control costs are not individually
identified by this response. The frozen parameter file is present and hashed;
the three historical ledger paths embedded in it are absent in this checkout
and were not independently reverified during this transfer.

`examples/cases/longctx-decode-20261030/qwen36-a5-tp2.json` supplies the unchanged
TP2 OneShot/3 fixed response, 10.832807169117647 us, at exactly 40,960 bytes.
The original archived provenance was hash-verified. It names the older Qwen3.6
c1 capture, 10,880 native calls over 85 complete forwards, and an inclusive
whole-native-call arithmetic mean. The original `in_sample` labels remain
unchanged. The new evaluation relation is recorded separately: these new
128K captures did not assign those coefficients, but have already been examined
and are not blind holdouts.

The declared hardware is `Ascend950DT_9582`; the new deployment's actual SKU
and compiled communication binary are not independently established. Payload
is source-derived BF16 `B * q4 * 5120 * 2`, because native HCCL rows lack tensor
descriptors. New raw captures and retained extraction identities are hashed in
`data/observations/a5-gdn-store-binding-20261030/hccl-frozen-validation.json`.

## Actual low, representative and high batch populations

The producer selects B1, B4 and the largest actual q4 population from the
Profiler manifest's extraction, rather than treating directory concurrency as
batch evidence. These are B1/B4/B24 for TP2 and B1/B4/B8 for TP4.

All native populations contain exactly 128 deduplicated all-reduce calls in
the complete 64-layer Body. TP2 has 3 native blocks throughout. TP4 has **5
blocks at B1**, and 60 at B4/B8. The low-B TP4 control therefore declares the
source-compatible OneShot/5 branch separately; the higher batches retain
TwoShot/60. This agrees with the conditional modern selector analysis in
[`a5-gdn-store-hccl-controls.md`](a5-gdn-store-hccl-controls.md), and is not a
selection based on latency error.

The frozen TP2 rule hits only B1. The TP4 rule hits B4/B8; its low-B native
template/block identity differs and its payload is below the training domain.
No rule is extended to force a hit. Each matching full Body has exactly 128
response queries; unmatched points retain their analytic source program.

The table reports steady native GDN/GQA means separately. The Body observations
and simulations still include every first-layer wait tail and all internal
gaps. Native call means are not added to Body spans. Context is the declared
126,000-token prompt; exact per-step KV increments and all-rank arrivals remain
unobserved.

| Model / TP | B | Selected call us | Native steady GDN / GQA us | Analytic Body ms | Frozen Body ms / error | Observed Body ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q3.6 / 2 | 1 | 10.833 | 10.619 / 10.891 | 22.854 | 23.713 / -13.24% | 27.332 |
| Q3.6 / 2 | 4 | 14.140 | 19.176 / 18.440 | 25.792 | 25.792 / -18.61% | 31.689 |
| Q3.6 / 2 | 24 | 84.839 | 74.589 / 72.182 | timeout | timeout / unavailable | 63.879 |
| Q3.6 / 4 | 1 | 6.890 | 13.915 / 13.931 | 21.070 | 21.070 / -17.65% | 25.585 |
| Q3.6 / 4 | 4 | 26.361 | 25.643 / 26.294 | 21.742 | 24.353 / -15.92% | 28.964 |
| Q3.6 / 4 | 8 | 28.101 | 27.990 / 27.966 | 23.392 | 25.924 / -18.12% | 31.660 |
| Q3.8 / 2 | 1 | 10.833 | 11.546 / 12.078 | 23.387 | 24.245 / -14.77% | 28.447 |
| Q3.8 / 2 | 4 | 14.140 | 20.864 / 22.857 | 26.349 | 26.349 / -19.45% | 32.711 |
| Q3.8 / 2 | 24 | 84.839 | 71.993 / 72.522 | timeout | timeout / unavailable | 64.457 |
| Q3.8 / 4 | 1 | 6.890 | 13.015 / 16.079 | 21.309 | 21.309 / -17.43% | 25.808 |
| Q3.8 / 4 | 4 | 26.361 | 24.703 / 29.886 | 21.990 | 24.601 / -15.86% | 29.240 |
| Q3.8 / 4 | 8 | 28.101 | 27.675 / 30.601 | 23.661 | 26.193 / -17.82% | 31.871 |

All twelve primitive comparisons completed. Ten Body comparisons completed;
two TP2 B24 Bodies exceeded the explicit 90-second budget and have no Body
score. The producer exits nonzero for that incomplete run. The retained final
segment took 485.7 seconds, reusing three rows from an earlier 174.2-second
segment; peak RSS was 113.5 MB. Completed executions drained all live tasks and
programs. Domain misses execute one Body and reuse its identical analytic result;
the report distinguishes this from running both response/control Bodies.

## Existing local-response and caller inventory

`caller-frozen-inventory.json` and its producer inspect existing response
declarations and build finite graphs without running or fitting any Body.

- The old Qwen3.6 convolution point is 7.0079346 us at M4/N5120, kernel width4,
  six history slots and cache-index producer stride4. New B4 is M16, and the
  new declaration uses stride1 with the actual producer stride unobserved. It
  cannot reuse that rule unchanged. The old Qwen3.5 convolution point is
  M16/N2048, whereas the new TP4 width is 2560; it also does not match.
- The old TP2 Q/K L2 point `(1,4,8,128)` at 1.8204685 us and output norm point
  `(96,128)` at 3.4277782 us match the new TP2 B1 source geometry, 48 calls each.
  The old Qwen3.5 TP4 Q/K L2 point `(1,16,4,128)` at 2.0165363 us matches new
  TP4 B4, 48 Q and 48 K calls. It does not match B1/B8. The output norm point
  `(128,128)` does not match the new TP4 B4 `(192,128)`.
- Native helper shapes are N/A, so these matches establish a declared source
  geometry, not a measured native tensor contract. None of these optional local
  response points was added to the HCCL comparison above.
- New captures contain 47 `AddRmsNorm`, 80 fused residual-norm-quant calls and
  one initial `RmsNorm`. Existing passes `activation_quantization`, `norm_quant`,
  `residual_norm` produce those semantic counts. An explicit
  `add_rmsnorm_quant` binding to `vllm_ascend_native_norm` with
  `external_gamma_shift=true, has_bias=false` yields 80
  `AddRmsNormDynamicQuant` calls for Qwen3.6 or 80
  `AddRmsNormDynamicMxQuant` for Qwen3.8, plus 80 separate gamma-shift producers.
  Their output scales are respectively FP32 `[16]` and E8M0 `[16,80,2]` at B4.
  The catalog's weight offset is +1, so that external producer must be preserved.
- The same native selector currently labels unquantized `add_rmsnorm` as
  `AddRmsNormBias` even when bias is absent, while these captures name
  `AddRmsNorm`. The inventory therefore does not bind that family or claim a
  complete native ABI match. It also does not infer a native tiling or latency
  from call-count agreement.

```sh
python3 data/observations/a5-gdn-store-binding-20261030/hccl_frozen.py
python3 data/observations/a5-gdn-store-binding-20261030/caller_inventory.py
# After a separately verified engine performance fix, retry only incomplete
# Bodies with the same 90-second limit and retain their earlier timeout record:
python3 data/observations/a5-gdn-store-binding-20261030/hccl_frozen.py --resume --retry-timeouts
```

## Structurally verified B4 caller candidates

Four explicit cases are now available under
`examples/cases/longctx-decode-20261030/a5-gdn-caller-fused/`. They add only the
three fusion passes and the existing `add_rmsnorm_quant` native binding described
above. `caller_binding.py` generated them and
`caller-binding-validation.json` retains the actual CSV sample locations,
complete INT8/MXFP8 tensor descriptors, matching predicted contracts and call
counts. No convolution, output-norm or other local point-response rule was added;
the HCCL rules remain byte-for-byte equal as JSON declarations.

Each declared fused call has one external gamma dependency, zero remaining
internal weight offset and a residual return. The baseline had no separately
declared gamma producers; the candidate adds exactly 80. Fusion consumes 127
standalone residual calls into 80 quantized and 47 unquantized norm calls,
leaving the final standalone residual once. The native population contains
128 scalar `Add [5120]` calls. The other 48 norm callers retain their semantic
weight offset and are not claimed to have an explicit native gamma producer.

The four new full Body predictions have **not been run**. This stable delivery
contains structural and sampled ABI verification only; it does not turn the
earlier Body errors into new accuracy evidence. `caller-cases.json` is a complete
Profiler input manifest with absolute case paths, original capture/extraction
identities, and both `body` and `operators` scopes. Reproduce through either the
existing Profiler pipeline with that manifest or the bounded local producer:

```sh
python3 data/observations/a5-gdn-store-binding-20261030/caller_binding.py
python3 data/observations/a5-gdn-store-binding-20261030/caller_binding.py --predict
```
