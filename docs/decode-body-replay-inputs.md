# Decode body replay inputs from the new profiler receipts

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](verification.md).

The compact inventory contains **30 receipts, 37 distinct per-capture geometry
groups and 2,605 measured Forward spans**. It prepares the next stable-runtime
replay; it does not run Sim or report prediction accuracy.

The JSON inventory
retains every receipt hash, source identity, case hash, precision declaration,
parallel layout, geometry group and observed mean/P50. Its
producer
reads compact profiler outputs only and never modifies profiler files or reads
the large original CSVs. All 30 receipt source hashes match their manifest
identities, and all 30 current case hashes match the manifest declarations.
This checks identity agreement, not a new re-hash of the raw captures.

The receipt boundary is `target_body_device_graph`, from first to last retained
body device event. Values below are elapsed spans, not summed family activity.
They come from `auto-forward-*.json`; the independently generated
`decode-forward-replay.json` was still an older eight-case unresolved report at
inspection and must not be mixed with this 30-case manifest.

## Geometry that must reach the ordinary body API

| Model/capture family | Cases / forwards | Verified query tokens per sequence | Observed sequence counts | Declared TP / DP / EP / PP | Declared precision |
|---|---:|---:|---|---|---|
| Qwen3.5 A3 | 6 / 403 | 2 | 1, 2, 4, 8 | 4 / 1 / 1 / 1 | W8A8_DYNAMIC |
| Qwen3.6 A3 | 4 / 128 | 4 | 1, 4 | 4 / 1 / 1 / 1 | W8A8_DYNAMIC |
| Qwen3.8 A5 | 6 / 345 | 4 | 1, 2, 4, 8 | 1 / 1 / 1 / 1 | W8A8_MXFP8 |
| GLM-5.3 A3 | 5 / 346 | 1 | 1, 2, 4, 8 | 8 / 1 / 1 / 1 | w4a8_dynamic |
| Qwen3.6 A5 hold captures | 8 / 1,320 | 4 | 1, 4 | 8 / 1 / 1 / 1 | w8a8_dynamic |
| Kimi K3 A5 | 1 / 63 | 1 | 1 | 8 / 4 / 32 / 1 | catalog default; historical quantization metadata incomplete |

These q values are retained query-row geometry, not acceptance-weighted work.
Qwen3.5 must use q=2 and Qwen3.6/3.8 q=4 for these Forward groups. An assumed
0.8 acceptance cannot reduce the target verify rows.

All **37 groups have `context_tokens=null`**. Client prompt length is retained
as scenario evidence, but it is not exact current KV length: context grows as
requests decode, and buffer capacity is not valid length. A first replay may
use an explicitly declared context scenario or range, carrying that assumption
into the comparison. It must not label that context independently observed.

Qwen3.5 c2/c4/c8 captures contain startup/drain Forward groups with fewer
sequences. The inventory partitions by each Forward's geometry before computing
means or medians. For example, q=2/b=8 in `a3_qwen35_p_4096_c8` has 63 spans and
P50 17.777895 ms; the whole capture's P50 17.763455 ms includes other batches.

## Small first-pass replay set

These are observed body span P50s for the indicated geometry group. They are
reference inputs, with no prediction or pass/fail claim attached.

| Receipt case ID | q / sequences | Matching spans | Observed body P50 ms |
|---|---:|---:|---:|
| a3_qwen35_p_4096_c1 | 2 / 1 | 66 | 14.760555 |
| a3_qwen35_p_4096_c8 | 2 / 8 | 63 | 17.777895 |
| a3_qwen36_p_4096_c1 | 4 / 1 | 15 | 16.513651 |
| a3_qwen36_p_4096_c4 | 4 / 4 | 48 | 20.265935 |
| a5_qwen38_p8192_c1 | 4 / 1 | 39 | 17.162646 |
| a5_qwen38_p8192_c8 | 4 / 8 | 62 | 25.505534 |
| a3_glm53_dc4k_b1 | 1 / 1 | 74 | 38.942772 |
| a3_glm53_dc4k_b8 | 1 / 8 | 61 | 47.148578 |
| a5_qwen36_p4096_c1_hold | 4 / 1 | 182 | 10.624639 |
| a5_qwen36_p126976_c4_hold | 4 / 4 | 176 | 26.650282 |
| a5_kimi_k3_p32768_c1 | 1 / 1 | 63 | 32.533125 |

The receipts do not verify forced load balancing, 100% prefix hit, acceptance
0.8 or DSpark. A declared balanced runtime in an existing Sim case is not
observed routing evidence. Preserve those separate fields in future results.

Kimi has only a single-rank capture, while its declared topology is DP4/EP32.
The other replica workloads and complete cross-DP critical path are not
observed. Its catalog-default precision is also distinct from captured
checkpoint quantization. Those two gaps remain explicit before treating a full
cohort prediction as an exact counterpart to this rank's body span.

No V4 Decode body receipt is included in this manifest. The existing V4 Prefill
or operator-family reports cannot fill that slot. The earlier
[body evidence audit](decode-body-evidence.md) describes the previous available
boundaries; this inventory adds the newly supplied structure-guided body spans
without converting them into the user's fully controlled production scenario.
