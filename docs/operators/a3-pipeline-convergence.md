# A3 source-program convergence, October 30

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The independent copy-pipeline declaration and source bindings are executable
mechanisms. Their accuracy still depends on source dispatch, workspace placement
and the observed boundary. The formal
operator report
retains every selected case, complete tensor contract, input-ready check, source
hash and serviced-resource ledger. Rates and their provenance are documented in
[the profile declaration](a3-pipeline-parameters.md).

## Conditional native-call results

| Declared source call | Predicted us | Observed mean us | Signed gap |
| --- | ---: | ---: | ---: |
| Qwen3.5 prefill, q8192 | 2733.285906 | 2476.503909 | +10.37% |
| Qwen3.6 prefill, q8192 | 4065.730832 | 3678.352412 | +10.53% |
| GLM SFA decode, q1/context4096 | 85.721212 | 90.730718 | -5.52% |
| GLM SFA prefill, q4096 | 25812.476125 | 13199.664513 | +95.55% |
| V4 Flash SCFA, q4096 | 7037.370578 | 3738.518619 | +88.24% |
| V4 Pro SCFA, q1024 | 835.600371 | 1180.346042 | -29.21% |

The first three meet the conditional per-call 15% criterion. They are not
held-out validation or complete-model accuracy. The 21 V4 Flash and 120 V4 Pro
observations cover the selected compressed branch; they are not every attention
event, nor a proven layer-to-event occurrence mapping.

V4 Pro uses a context-zero source candidate. Retained serving metadata specifies
prompt4096 and maximum batched tokens1024, so its observation population may
contain several chunk contexts. Event-to-chunk identity is under investigation;
-29.21% is a candidate-versus-population difference, not proven same-context
accuracy.

All eleven public graph/operator controls finish in 74.307 seconds, with maximum
RSS 51.97 MiB. Delayed input, serviced work, buffer release and retirement checks
pass. The selected attention, profile and Engine files are unchanged during
these controls. The entire package is not unchanged: `native_fusion.py` changes
and `family_situ_mx.py` is added. The report's initial/final file manifests retain
both facts; its `changed_files` list describes modified common files only.

The GLM prefill observation references the original comparison SHA
`4a058d4e59dd082fe17297b209046a89e79470e3f974c35e1c5e59938ee1aced`.
The profiler later regenerated that path. The original identity is retained;
reading the current path does not reproduce those original report bytes.

## Decode dispatch remains unresolved

| Declared decode geometry | Split 1 us | Split 2 us | Split 4 us | Split 8 us | Observed us |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 q2 | 164.055644 | 85.017045 | 44.331136 | 24.399318 | 53.573506 |
| Qwen3.6 q1 | 162.760322 | 84.242784 | 43.798655 | 23.884943 | 54.717000 |
| Qwen3.6 q4 | 174.418220 | 91.211136 | 48.590985 | 28.514318 | 58.191914 |

The full candidate table is retained without selecting a split by latency.
For Qwen3.5 q2, a single producer copies 8,394,752 bytes through a private
52.8 GB/s AIC port, giving a 158.992 us copy lower bound. Splitting changes
ownership and reduction work, not the physical port rate. BlockNum can constrain
native grid geometry but cannot by itself identify active cores or KV splits.
Qwen3.5 q1/q4 have no matching retained observation and remain unscored.

The profiler's independent full-population audit, `decode-hc-pmu-evidence.json`
(SHA-256 `e346fced05097d8eca286c86a2e8d7dc328d47efa833c1dc15f0476c96728250`),
streams 186,701 rows from three retained CSVs. The selected FIA populations are:

| Descriptor / execution state | Events | Block Num / Mix Block Num | Mean us |
| --- | ---: | --- | ---: |
| Qwen3.5 q2, static | 715 | 17 / 34 | 53.573506 |
| Qwen3.6 q1, dynamic | 34 | 17 / 34 | 54.717000 |
| Qwen3.6 q4, dynamic | 15 | 17 / 34 | 58.485533 |
| Qwen3.6 q4, static | 240 | 17 / 34 | 58.173563 |

Qwen3.5 q1 has zero selected events. The 46-column CSV has no BlockDim field.
Qwen3.6 q4's earlier 58.191914 us pooled mean includes both states; dropping
the 15 dynamic events changes the population. No observed block count is
translated into a selected split factor. AIC MAC means of roughly 2-3.5 us,
MTE2 means of 20-22 us and scalar means of 25-28 us provide separate diagnostic
constraints; these overlapping counters are not a sum or timing fit.

## Offchip work, exports and whole-model limits

V4 Flash services 8,417,286,400 HBM bytes per selected SCFA call. At the case's
1.2 TB/s, its 7014.405 us lower bound explains nearly all the 7037.371 us
prediction. V4 Pro services 981,664,000 bytes, with an 818.053 us lower bound
against 835.600 us predicted. The opposing residuals therefore cannot be
resolved by one bandwidth multiplier. Source workspace traffic, cache placement
and query/selected-row geometry need independent evidence.

The cases retain their previous explicit generic MTE2, MTE3 and vector
overrides. They are not bare-profile measurements. The actual source programs
use split AIC/AIV reads and shared HBM.

For non-fusion `PlannedRecipe` nodes, the current structural export's
`physical_work_per_instance` remains the analytic envelope. Execution applies
the source recipe's work correctly. Use the executed service ledger for source
traffic; do not interpret the structural envelope as that ledger. The profiler
confirmed this distinction independently during its isolated replay.

On runtime `26f5adb4`, the added GLM indexed-source offchip full graph completes
in 50.299 seconds and predicts 2790.139446 ms against 1680.733399 ms observed.
The added V4 Flash source-SCFA target-plus-MTP full graph times out at 90 seconds.
Both outcomes are explicit references in the
[main scoreboard](../../data/verification/model-scoreboard-20261029-final.md).
Original primary declarations are preserved.

The V4 source graph contains 21 identical SCFA instances. A completed single
instance uses 137,374 tasks, so approximately 2.88 million task completions plus
isolated-family evaluation are consistent with the whole-worker budget pressure.
The timeout receipt has no stage progress or stack sample; it cannot establish
where execution stopped or diagnose a deadlock. The 176.763 GB aggregate SCFA
HBM work gives only that family's supply bound, not a complete graph span.

## Other remaining large gaps

The source SituMxQuant path models three AIV owners for T1/H768 and predicts
0.170951 us, compared with the previous 0.001946 us semantic estimate. The
q8192 source prediction is 59.645999 us. The retained 6,898-call TaskDuration
mean is 65.134110 us while mean AIV time is 3.824011 us. These have different
exporter boundaries. Their difference does not prove a 61 us STARS overhead,
and pipeline counters cannot be summed into a replacement duration. See the
[source and PMU account](situ-mx-quant.md).

GLM E256/M8 MSD source programs now preserve balanced active experts, rolling
owner assignment, full group-list processing and private operand traffic. With
the final A3 profile, gate/down predict 15.932216/7.263622 us against
88.038726/52.907483 us. Actual group-list values remain unknown. The matrix
phases still omit L1-to-L0 operand service and scalar/flag latency; new MTE1
declarations cannot help until this work is represented. The combined control
includes artificial input-ready time at 7 ms; its elapsed kernel chain is
23.195838 us, not 7.023196 ms. See the
final-profile evidence.

No residual-derived launch floor, utilization factor or cache hit ratio is
introduced. Historical data and the old implementation remain preserved.
