# Model scoreboard mechanism refresh

The October 29, 2026 refresh executes ten complete graph/iteration scenarios and
four Kimi descriptor-family workers. Every worker uses package content SHA256
`5dea59fbb1e1c79548e0d6d83e79aea78d310b026711bdf140459ddc0cff0014`
(189 package files), unchanged before and after execution. The source baseline
is `10c2e531` plus the profiler-owned stable source that was not yet committed.
This is the actual execution identity, not a claim of a clean checkout.

The complete probes take 40.816 seconds together; the four Kimi workers take
1.338 seconds. The maximum observed process RSS is 88.625 MiB. All graph tasks,
programs and allocated buffers are released. These checks establish completed
execution and provenance, not prediction accuracy.

## Complete-call results

| Case | Retained observed span ms | Previous prediction ms | Current prediction ms | Current signed difference |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5 A3, complete MTP1 iteration | 23.718750 | 3.468963 to 3.470165 | 3.468963 to 3.470165 | -85.37% |
| Qwen3.6 A3, complete MTP3 iteration | 34.479625 | 10.764904 to 10.765423 | 10.764904 to 10.765423 | -68.78% |
| Qwen3.8 A5, complete MTP3 iteration | 25.661551 | 14.716206 to 14.716829 | 17.317027 to 17.322150 | -32.52% to -32.50% |
| GLM-5.3 A3, prefill4096 | 1680.733399 | 921.407663 | 1071.284508 | -36.26% |
| V4 Flash A3, prefill4096 plus MTP preparation | 575.235407 | 314.206973 | 299.103055 | -48.00% |

The Qwen ranges are two explicit context endpoint predictions, not a fitted
median or a proven latency bound between those endpoints. Native sampler spans
include runtime gaps; simulated device iterations do not. GLM/V4 raw activity
sums are not substituted for their observed spans. No family cost is added to a
span. These remain conditional descriptive comparisons.

The previous report is retained in Git. Changes include source selections,
representation contracts, and reporting taxonomy. They are not a controlled
one-mechanism timing attribution. In particular, selecting the HC source path
makes the V4 prediction smaller and increases its complete-step residual; this
unfavorable result is retained.

## What actually executes

Qwen3.8 selects `staged_gqa_decode` for target verification and incremental draft
attention, with key tile128, query tile64, KV split1 and two slots. The native
launch geometry does not imply that all32 Cube cores own work. Independent
arch35 query/head merge logic supports four KV-head owners for this declared
geometry. The actual predicted attention sum is 2.920071 to 2.925194 ms, compared
with 2.714835 ms native activity. The native caller has17 q4 and2 q1 attention
calls, while the generic MTP3 iteration has16 q4 and3 q1. Count19 alone does not
establish an identical caller boundary.

GLM explicitly selects source-conditioned W4A8 MSD gate/up and down paths, and
the MLA declaration keeps checkpoint UK/UV operands BF16. The resulting grouped
family is 237.755345 ms versus 192.908112 ms native activity, a +23.25% residual.
The selected source is a coarse resource schedule; exact native overlap, cache
behavior and binary identity remain unresolved. The main case retains the
generic indexed gather model.

V4 explicitly selects `hc_pre_m_k_split_175ef4d4` and
`hc_post_d_split_175ef4d4`. The40MiB cast ring retains its GM/HBM traffic.
The real worker executes172 source HC calls, totaling87.623842 ms isolated.
Native HcPre has88 calls, equivalent to44 layers, whereas the selected HC scope
covers43 target layers. The generic MTP preparation adapter still does not
reproduce its complete HC caller boundary.

## Explicit indexed-attention alternatives

| GLM scenario | Complete predicted graph ms | Attention isolated sum ms |
| --- | ---: | ---: |
| Main case: generic gather | 1071.284508 | 477.451653 |
| Source pipeline: scratch resident | 1299.675084 | 557.413212 |
| Source pipeline: scratch offchip | 2752.860224 | 2010.598351 |

Resident/offchip are declarations, not observed cache-hit classifications. Their
copy/cache rates use an **UNVERIFIED per-core interpretation multiplied by20
AIC** of independently retained current-toolkit metadata. The conditional rate
declarations also change other matrix/copy work. Therefore the total differences
from the main case cannot be attributed solely to attention scratch. Neither
alternative is selected because it is closer to the measured span.

## Observation and reporting corrections

The Qwen3.8 raw ten-window audit finds48
`RecurrentGatedDeltaRule` events per window that the historical classifier put in
`other`. Only those events move to GDN. Corrected medians are computed from
complete per-window family sums:

| Native family | Corrected median activity ms |
| --- | ---: |
| Matrix | 14.116662 |
| Attention | 2.714835 |
| GDN | 1.9492255 |
| Remaining other | 3.975191 |

The old groups and every raw event remain retained. Separate medians are never
added. No complete-row duplicate or hcom/HCCL/AICPU-named event appears in these
ten TP1 windows. Remaining `other` is still not a proven pure-vector population.
Its copies and control operations cannot all be moved into GDN merely from a
shape association.

The report classifier now includes `cast` programs in the vector-associated
family. Previous complete graph and isolated totals already included them;
only the coarse family report omitted them. This accounts for9.437184 ms in the
old GLM vector inventory and32.463913 ms in the old V4 inventory. Unclassified
work is explicitly reported; only zero-cost token commits remain unassigned in
this campaign.

GLM native communication and AICPU control activities remain separate. Exported
native hcom rows lack enough identity to infer logical-call duplicates from
overlap. They are not added together or treated as additional physical link
work. Qwen3.5 vector and Qwen3.6 collective populations contain mixed control
activities, so their coarse physical-family comparisons remain incomparable.

## Remaining large mechanisms

Qwen3.5 grouped compute is still0.444933 ms versus3.862807 ms native activity;
GDN is0.171302 versus1.107093 ms. Qwen3.6 vector-associated work is0.053657 ms
versus2.887361 ms, and its attention/GDN predictions are about75% low. These
remain visible instead of being absorbed into a whole-step multiplier.

V4 communication is39.644353 ms versus149.270651 ms native activity, and
attention is31.390961 versus98.896984 ms. GLM's main attention and matrix
families remain more than50% low. Input/caller identity, low-occupancy work,
copy ownership and native control phases are still consequential. A completed
test suite does not resolve those mechanisms.

Kimi has four descriptor-family reports, but no independently timed complete
model-step boundary. The current-checkpoint declaration is conditional:
the recovered config SHA matches historical config, while the historical
quantization-description SHA is absent. Mixed-capture native activity is not
treated as a model step.

The new stream-boundary audit recovers276 maximal closed runs of four KDA
events. Independent pinned wrapper order and the surrounding operators support
a conditional complete-call comparison. This is not a division of1104 events
by four. The8192-token scenario is the upper endpoint permitted by128 chunks;
captured sequence-length values are absent, leaving a valid8129 to8192 range.
The existing raw1104-launch mixed-stage mean/P50 remains unpaired.

| KDA conditional boundary | Observed activity ms | Fixed baseline ms | Source-state update ms | Current signed gap ms |
| --- | ---: | ---: | ---: | ---: |
| Complete four-stage core,276 calls | 881.997838 | 60.706774 | 195.793222 | -686.204616 |
| Gate/prepare component | 517.895295 | 9.591555 | 9.591555 | -508.303740 |
| Post-WU component | 20.208037 | 7.813988 | 7.813988 | -12.394049 |
| Recurrent-state component | 298.471487 | 29.409695 | 164.496143 | -133.975344 |
| Finalize component | 45.423019 | 13.891535 | 13.891535 | -31.531484 |

The component rows partition the core comparison; they are not additional
populations to add to it. The explicit `vllm_ascend_kda_chunk_state` selection
now executes the arch35 FWD_H owner/slot/chunk program: 12 AIC and 24 AIV,
6.442450944 GFLOPs, 806879232 HBM bytes, FP32 GM state every chunk and BF16 H
snapshots. It predicts 0.596000520 ms/call, versus the old 0.106556866 ms.
Hardware rates and all three other stage predictions are unchanged.

This source-phase update uses development-branch package SHA256
`aed586425bf8990127510ebc1179a4cd890cbbe4acf747428e99d702d1bb6305`
(190 package files), distinct from the retained whole-model baseline above.
Its resource-envelope assumptions remain explicit: no inferred flag/queue
latency, no internal row pipeline reconstruction, unpriced local L1 tail fill,
and aggregate AIV-pair supply for odd V1 tails. The actual 8192-token scenario
has full 64-row chunks. Independent delayed-input, slot-refill and work checks
run on the common Engine; no stage observation supplies a rate.

Prepare remains generic and contributes -508.303740 ms of conditional residual.
The source state residual is still -133.975344 ms. The large KDA residual cannot
be hidden by signed cancellation among the other three descriptor families or
reported as complete Kimi accuracy.

## Reproduction and artifacts

`tools/refresh_model_scoreboard.py` reuses the existing bounded workers, saves
each actual report, and emits the explicit semantic family assignment including
unassigned work. `tools/model_scoreboard.py` assembles retained reports without
rerunning predictions. Their workers receive case and workload geometry, never
observed durations.

The current entry points are:

- `data/verification/model-scoreboard-20261029-final.md`
- `data/verification/model-scoreboard-20261029-final.json`
- `data/verification/model-scoreboard-20261029-final-input.json`
- `data/verification/model-scoreboard-20261029-final-execution.json`
- `data/verification/model-scoreboard-20261029-mechanisms/`
- `data/verification/model-scoreboard-20261029-mechanisms-kimi-family.json`
- `data/verification/model-scoreboard-20261029-kimi-kda-call-alignment.json`
- `data/verification/kda-state-source-alignment-20261029.json`
- `data/verification/kda-state-source-20261029.json`
- `data/verification/qwen38-other-review-20261029.json`

The raw worker reports retain exact case hashes, package hashes, source
selection, physical work, graph boundaries, resource usage and liveness.
The old implementation and all historical raw measurements remain intact.
