# Kimi BF16 AttnRes caller binding

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The twelve A5 E16 BF16 declarations now explicitly select `vllm_ascend_attn_res_prefix` through the ordinary frontend. The complete 93-layer caller emits 187 source points: two per layer and one final point. The eight cross-layer bank slots, deferred FFN addition, BF16 round boundaries, SP-local rows and owner limits are supplied by the existing source planner. No runtime, resource rate or invocation parameter changed.

The retained source is image `883670fc`, caller SHA `be9adc792774f919f443caf483eb2e04e4453617465381f47a482dd4736ec63f`; files and kernel hashes are recorded in the existing source oracle. The new captures contain exactly 187 `AttnResFwd` events per accepted body in all twelve inputs. This supports the call-count boundary, but their tensor contracts, scalar flags and loaded kernel library remain unresolved. The comparisons below are conditional family references, not strict tensor cohorts or heldout accuracy.

## Correct the old cost boundary first

The previously quoted 1.312 us is the minimum duration of one small physical program. It is not the cost of every native residual point. The unbound frontend emits 1,488 mixture, normalization and addition operators for 93 layers; at batch 1 their isolated cost sum is 2.012485 ms. The source binding replaces these nodes with 187 complete calls and predicts 1.348694 ms, so this binding does **not** recover 1.8 ms of whole-body runtime. The old numerical proximity to measured 2.052168 ms activity does not prove its decomposed execution boundary. Neither isolated sum is a whole-body critical-path span.

For the selected source path, the first point is 1.312 us and the final point is 10.522 us, with a mean of 7.212 us. Increasing the live bank prefix from zero through eight blocks changes both read work and resident/reload selection. Batches 1–8 pad to eight global SP rows and one row per TP rank; batches 16/32 have two/four local rows on independent AIV owners. Their selected source latency plateau follows owner parallelism, while serviced bytes/arithmetic grow with local rows.

## Pooled family comparison

Values are one device’s full-family isolated sum versus profiler native activity divided by complete body count. Context is an explicit conditional declaration; this residual family does not read KV context. All native observed calls in this family are below 50 us. The mean per-call absolute difference is 3.76–4.19 us, below the 15 us micro-operator tolerance; missing ABI prevents promoting these reference rows into strict accepted cohorts.

| Batch | Declared context | Observed family ms/body | Source family ms | Old decomposed scope ms | Mean absolute gap us/call |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 2.052168 | 1.348694 | 2.012485 | 3.762 |
| 2 | 4096 | 2.056637 | 1.348694 | 2.012485 | 3.786 |
| 4 | 4096 | 2.058713 | 1.348694 | 2.012485 | 3.797 |
| 8 | 4096 | 2.052706 | 1.348694 | 2.012485 | 3.765 |
| 16 | 4096 | 2.092806 | 1.348694 | 2.138582 | 3.979 |
| 32 | 4096 | 2.132355 | 1.348694 | 2.282177 | 4.191 |
| 1 | 16384 | 2.052678 | 1.348694 | 2.012485 | 3.765 |
| 2 | 16384 | 2.056031 | 1.348694 | 2.012485 | 3.783 |
| 4 | 16384 | 2.058516 | 1.348694 | 2.012485 | 3.796 |
| 8 | 16384 | 2.056478 | 1.348694 | 2.012485 | 3.785 |
| 16 | 16384 | 2.090489 | 1.348694 | 2.138582 | 3.967 |
| 32 | 16384 | 2.131787 | 1.348694 | 2.282177 | 4.188 |

The remaining conditional activity difference is 0.703–0.784 ms/body. The source planner currently uses semantic vector lanes and serial owner load/compute/store phases, not register issue, special-function/reduction latency, compiler barriers or the captured binary schedule. This work does not infer one combined fixed floor from that residual. Those limits remain explicit; the next whole-body replay must use the selected source boundary once, never add these values to the old decomposed rows.

## Verification

- 22 affected tests passed, including complete 93-layer BF16 B1/B32 prefix DAGs with late external inputs, all eight retained bank dependencies, SP gather/scatter boundaries, first/final points, resident/reload, no duplicate norms/additions, owner count and resource cleanup.
- All twelve complete residual families were evaluated through public `target_operators` and `operator_costs`, using original complete model declarations. They completed in 15.81 s with unchanged package fingerprints; peak RSS was 123.8 MiB on this host. No full-body simulation or raw capture CSV was rerun.
- Producer and compact report retain all point classes, original report hashes, input hashes, old alternative scope and source limits.

Run from the checkout with its package on `PYTHONPATH`:

```sh
PYTHONPATH=. python data/verification/kimi-bf16-attnres-20261030.producer.py
```
