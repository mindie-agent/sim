# A3 regular-query FAI response across contexts

This optional native-call response improves seven in-domain 126K complete Body
controls to six within 15%; three additional flash-decode controls remain on
their analytical program and pass. The remaining Qwen3.5 TP4 B24 Body is
53.303 ms versus 63.494 ms observed, **−16.05%**, and remains a failure.

The response uses no Body timing and no 126K operator timing in its coefficients.
`data/observations/parameters/a3-fai-long-context-20261030/collect.py` streams
three independent deployment capture series into 78 complete q4 native ABI
groups. Actual contexts come from the hashed experiment declarations: 8K means
8,512 here, 24K means 24,512, 64K means 64,512 and 96K means 96,512. These are
prompt lengths, not observed per-call sequence-length values.

`produce.py` fits the per-owner native duration against context using only
shorter-context Qwen3.5 TP4 and Qwen3.6 TP2 observations, grouped by physical
query-head/KV-head ratio. It applies the pinned regular host schedule's
`ceil(batch * local_kv_heads / 20)` owner waves. The resulting rules contain
hardware, dtype, head geometry, q4, batch and context domains; no model name.
The unchanged Qwen3.8 TP4 series tests transfer across a different deployment
and local KV geometry. Historical experiment roles remain `unassigned`;
these are retrospective calibration/validation partitions, not blind holdouts.

The eleven 126,000-token native-call comparisons all pass, with median-relative
errors −5.30% through −0.22%. Over all shorter and longer populations, training
passes 10/12 and separate context/geometry validation passes 14/15. The three
8,512-token misses remain in `operator-validation.json`; none is removed.

The physical discrepancy motivating this experiment is recorded separately in
`data/verification/a3-qwen36-fai-cold-read-audit-20261030.json`: a conditional
two-wave cold-read MTE2 owner bound is 4.888 ms, while the matching native q4
median is 3.598 ms. Page-table values, actual KV aliasing and L2 hit rates were
not captured. Neither the response slope nor PMU clock conversion establishes
a measured DDR bandwidth, cache-hit fraction or DVFS state.

Consequently this is an explicit **joint native kernel service** on a declared
device queue. Its analytical work remains reference-only. It does not inflate
bytes, lower a physical pipe rate, add a Body residual or identify the separate
responses to AIC/AIV/HBM bandwidth changes. Use the analytical program for such
hardware experiments unless independent service measurements are supplied.
These parameters are conditional case inputs, not a hardware default.

The domain is homogeneous q4 BF16/ND, D256, group4 or group6, 20 AIC owners,
regular-source occupancy above 16 tasks, B≤64 and context 8,512..129,024.
129,024 is a declared extrapolation endpoint, not another measured point.
The accurate low-occupancy flash-decode branch, ragged requests, other dtypes,
other widths and other hardware retain their analytical program. A new frontend
contract exposes uniform query/context scalars only when every request agrees;
ragged batches cannot accidentally match a response using just their first row.

`validate_body.py` runs ordinary `predict(scope="body")` with ten complete
40/64-layer controls and a 90-second limit per prediction. Seven controls select
exactly one joint FAI service per GQA layer; three do not select it. All drain
tasks/programs. Nine Bodies pass and the Qwen3.5 B24 miss above is preserved.
The runtime revision and all input/parameter/report hashes accompany the result.

To reproduce from the retained census, without rescanning raw captures:

```sh
python3 data/observations/parameters/a3-fai-long-context-20261030/produce.py
python3 data/observations/parameters/a3-fai-long-context-20261030/validate_body.py \
  --profiler-root /path/to/profiler
```

The five conditional inputs are in
`examples/cases/longctx-decode-20261030/a3-fai-regular/`. They retain the source
cases' other programs and rules; they are not alternative best-of-Body fits.

The retained Body measurements and input hashes above refer to commit
`1c5939cd`. The current packet additionally fixes the original physical resource
rates, BF16/FP32 compute peaks, MTE1 service modes and owner counts in its match
domain. Changing HBM, AIC or AIV supplies on the same SKU therefore selects the
analytical program instead of silently keeping an opaque native duration.
`domain-validation.json` verifies unchanged selection on all ten actual frontend
graphs and forty independent hardware-rate perturbations. These are binding
checks, not new Body accuracy results; parameter values and native timing
predictions are unchanged.
