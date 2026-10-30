# Decode body comparison evidence

This 2026-10-30 local audit identifies usable historical observations for the
Decode-first modeling work. No device workload, remote container or new
collection was started. The numerical predictions below belong to their saved
reports and declarations; they are not reruns of the current checkout.

## Current evidence boundary

The inspected retained captures do not establish a complete target-body Forward
duration under the new joint conditions of forced expert balancing, 0.8
speculative acceptance, 100% prefix-cache hit and high concurrency. Analytic
body prediction can run for those explicit conditions; its whole-body accuracy
remains unscored until an independently identified matching Forward is supplied.

| Model / capture | Independent retained workload | Available timing boundary | Body Forward acceptance |
| --- | --- | --- | --- |
| Qwen3.5 A3 | p16384/o128/c1, TP4, MTP1 | 64 clean sampler intervals; full draft/verify cycle plus runtime gaps | No target-body occurrence boundary; realized routing and acceptance unavailable |
| Qwen3.6 A3 | p16384/o32/c1, TP4, MTP3 | 14 clean sampler intervals; full draft/verify cycle plus runtime gaps | Same boundary gap; verify width 4 does not identify all individual calls |
| Qwen3.8 A5 | p16384/o32/c1, TP1, MTP3 | 10 retained sampler windows plus native event cohorts | Same boundary gap; estimated context progression is not an acceptance trace |
| GLM-5.3 A3 | p4096/o256/c1, TP8 | Gated 3-second Decode capture, 166,204 original rows, 5,772 SFA calls | No retained token/step/occurrence IDs; no host/module trace |
| Kimi K3 A5 | p32768/o64/c1, TP8/DP4/EP32 | 286,988-row mixed Prefill/Decode capture, native tensor cohorts | No independently timed complete model step; Python trace belongs to API process |
| V4 Flash / Pro A3 | p4096/o1/c1; TP8/EP8 and TP16/EP16 | Prefill full-request captures; Pro has four 1,024-token chunks | These retained comparison arms provide no Decode body measurement |
| V4.1 A5 | Existing serving metric campaigns | Request/metric summaries, no raw kernel capture in inspected roots | Cannot establish target-body or operator acceptance |

The Qwen inventory explicitly records `prefix_caching: false`. None of these
rows proves the requested 100% prefix-hit execution. The V4 Flash Prefill
prediction declares balanced routing, but that declaration is not measured
Decode routing evidence. Historical transfer-directory names such as
`dp32ep32_8K_4K_bs1536` are useful search hints; their inspected SQLite exports
lack independent request geometry and exact SKU identity, so the directory name
cannot become a high-concurrency acceptance case.

Files named `body-*.json` in the profiler acceptance directory describe
`historical_full_capture_body_contract_replay`. Their `capture_bounds_ns` span
the retained capture; their body tensor matches do not identify a body-call
start and finish. Likewise, a family activity sum counts overlapping native
events and cannot replace the body's elapsed graph span. A communication
wrapper and its internal stream tasks must retain separate physical identities
before any total is formed.

## Reusable Decode operator evidence

These are selected concrete cohorts from saved comparison files, in us/call.
The cohort IDs and source hashes are available in those files. Calls reused
across conditional declarations are not new measurements. All retained
observations keep their original `unassigned` calibration/validation role.

| Declaration / native operator | Calls | Observed mean | Predicted | Current saved result |
| --- | ---: | ---: | ---: | --- |
| Qwen3.8 staged paged q1 / FIA | 24 | 135.013 | 134.640 | Relative tolerance passes |
| Qwen3.8 staged paged q4 / FIA | 187 | 143.916 | 157.259 | Relative tolerance passes |
| Qwen3.8 staged paged q4 / QuantBatchMatmulV3 | 704 | 53.350 | 50.222 | Relative tolerance passes |
| Qwen3.8 staged paged q4 / MatMulV3, GDN projection | 528 | 51.276 | 43.272 | -15.61%; 8.004 us absolute gap, still outside the >50 us tier |
| GLM-5.3 source SFA q1, separate A3 pipeline control | 5,772 | 90.731 | 85.721 | -5.52%, conditional tensor comparison passes |
| Qwen3.5 q2 / FIA, old envelope arm | 715 | 53.574 | 6.999 | Failed; old envelope arm is not the current source selection |
| Qwen3.6 q1 / FIA, old envelope arm | 34 | 54.717 | 6.996 | Failed; original rows are dynamic |
| Qwen3.6 q4 / FIA, old envelope arm | 255 | 58.192 | 7.014 | Failed; 15 dynamic plus 240 static rows |
| GLM-5.3 native-fusion Decode / GroupedMatmulSwigluQuantV2 | 5,550 | 88.039 | 11.472 | Failed; substantial grouped execution gap |
| GLM-5.3 native-fusion Decode / GroupedMatmul | 5,550 | 52.907 | 6.400 | Failed; substantial grouped execution gap |
| Kimi current-checkpoint single-group Decode / SituMxQuant | 6,898 | 65.134 | 0.002 | Old envelope arm fails; device AIV time and full event duration differ |
| Kimi current-checkpoint single-group Decode / GroupedMatmulSituQuant | 7,266 | 91.780 | 7.735 | Reference only; its unsupported execution boundary is not an accuracy pass |

The A3 source SFA result is from
`data/verification/a3-pipeline-convergence-20261030.json`, not from overwriting
the old profiler cohort prediction. Its observed raw source is unchanged, and
its original comparison-report hash remains retained. The saved pipeline
report records selected attention/profile/Engine stability and separate
package changes during the broader campaign.

The Qwen A3 PMU audit records Block/Mix counts 17/34 for these FIA cohorts.
There is no `BlockDim` column. Those counts do not establish `kv_splits`; the
split sensitivity runs remain explicitly declared candidate programs. The
Qwen3.5 q1 descriptor has zero observed FIA events and cannot acquire an
observation from the q2 population.

The retained 35-variant replay summary reports 81/314 passing >50 us cohorts
and 138/163 passing <=50 us cohorts across eight captures. These historical
counts belong to receipt SHA
`adf90344f80c7b075a9ac9751d8073185aa776473a385641ca89f77bfd94ac7f`.
Some individual comparison paths have since been regenerated: use each saved
file's source identity and hash when selecting a current comparison, rather
than merging its values into that older receipt. The summary is not a current
checkout verification or a unique-event pass rate.

## Body probe semantics required for matching

A target-body probe begins with ready hidden activations and existing KV/state
at the declared context. It includes layer attention, FFN, residual work and
their TP/EP/PP/SP communication. Embedding, final logits/sampling, draft
generation, host submission, inter-step waiting and PD handoff are outside
this body boundary. The input cache is assumed resident; a prefix-hit fraction
must not erase Decode KV reads or recurrent-state updates.

`tokens` must state target query rows per sequence. A verify pass after three
draft proposals generally supplies four target rows; an acceptance rate of
0.8 governs committed progress, not a 0.8 multiplier on those rows or their
resource work. DSpark proposal shape and execution domain likewise need an
explicit declaration. A body duration alone does not establish a 5/10/30/50 ms
TPOT operating point: committed-token progress, batching and other iteration
work are additionally required for that service constraint.

Capture padding must match the ordinary frontend. The pre-change
`operator_analysis._target_ops(scope="body")` bypassed `GraphPolicy.select`
and `apply`. A local counterexample with valid Decode rows 1 and capture bucket
8 produced body norm shape `(1,32)` and 160 vector operations, while the
ordinary model path produced `(8,32)` and 1,280 operations. It also let balanced
MoE succeed where the ordinary captured path correctly reported
`balanced_routing_requires_unpadded_tokens`. Root received this executable
counterexample for the public body API work.

The earlier equal-group restriction was subsequently removed in `c60e9c4c`.
Forced balance now retains integer quotient/remainder rows, zero experts and
capture padding across complete DP cohorts. Distinct positive row classes use
the ordinary matrix recipe under one grouped invocation. The timing
representative is the heaviest strided EP rank. Histogram balance still does
not prove token IDs or an observed all-to-all traffic matrix; the peer payload
assumption remains explicit. See [balanced expert mechanisms](operators/balanced-experts.md).

## Source inventory

All profiler paths below are relative to the sibling
`profiler/examples/structure/acceptance-20261029/` directory.

- `model-evidence-inventory.json`: independent workloads, capture roles and
  high-concurrency transfer-export limitations.
- Sim `data/verification/model-scoreboard-20261029-observed-inventory.json`,
  SHA `f3967008155728567c350381c987edaaca22dcfb41774ed3162113fb43690da9`:
  Qwen sampler and prefix declarations.
- `fusion-cohort-a5_qwen38_paged_decode_q1_staged-selected.comparison.json`,
  SHA `5aef3ef40e005aa8fee9939a1fa214f3081a11001c1ba5030ae9f4dbc8db76b9`.
- `fusion-cohort-a5_qwen38_paged_decode_q4_staged-selected.comparison.json`,
  SHA `bdf3ae4ce16d1ccea2efff0430d7eb93993c2cf4c957d595bbf79baaf10cd68e`.
- `fusion-cohort-a3_glm53_decode-selected.comparison.json`,
  SHA `53f12049ec955170d8cf7bea1ae654b81062bb24ceab46e56d5719864eacf569`.
- `fusion-cohort-a5_kimi_k3_decode_checkpoint_eager_single_group-reference.comparison.json`,
  SHA `6138d39d1b86244f113ff470ffd92c9c21ed954ad813f87de5ff1f0aecb547a4`.
- `decode-hc-pmu-evidence.json`,
  SHA `e346fced05097d8eca286c86a2e8d7dc328d47efa833c1dc15f0476c96728250`:
  original static/dynamic and core-counter groups.
- Sim `data/verification/a3-pipeline-convergence-20261030.json`,
  SHA `4aee32a935f98b3db703f33230e544fada1d49aba32e26008feb6b1eb5c20d27`:
  separately executed source programs and explicit split sensitivity.

No raw file larger than 50 MiB was loaded in this audit. Existing compact
reports supply the capture provenance and complete-event aggregates.
