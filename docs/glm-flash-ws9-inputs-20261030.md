# Seven GLM Flash WS9 body input contracts

The new `examples/cases/high-concurrency-20261030/glm-flash-ws9/` inputs retain
45 target layers and the resolved A5 TP8/DP1 deployment. EP is disabled; TP8
must not be interpreted as EP8. The startup log at
`.workstreams/hp-serving/glm53-hc/raw/ws9/logs/serve-mtp.log` lines 25 and 67
records forced EPLB, APC enabled, 3 MTP proposals, synthetic marginal acceptance
`[0.8,0.64,0.512]`, and FULL_AND_PIECEWISE graph mode. The input retains the
actual capture bucket list through 512 and max sequence count 256.

The producer derives client concurrency, prompt-token counts, measured
acceptance ratio and median client TPOT from each retained summary. In
particular, the named p4k prompt has 4,104 actual prompt tokens; this is not
silently rounded to 4,096. Zero prefix queries in these summaries leave the
hit ratio unknown, even though APC is enabled. Configured acceptance is not
substituted for the observed ratio, and neither ratio scales target-body work.

| Cell | Client median TPOT (ms) | Body graph input check |
|---|---:|---|
| a5_glm53flash_tp8_c16_p4k | 15.216 | q4 × 16, context 4104 |
| a5_glm53flash_tp8_c16_p8k | 15.558 | q4 × 16, context 8200 |
| a5_glm53flash_tp8_c32_p4k | 20.333 | q4 × 32, context 4104 |
| a5_glm53flash_tp8_c32_p8k | 25.050 | q4 × 32, context 8200 |
| a5_glm53flash_tp8_c4_p8k | 12.933 | q4 × 4, context 8200 |
| a5_glm53flash_tp8_c64_p4k | 29.828 | q4 × 64, context 4104 |
| a5_glm53flash_tp8_c8_p4k | 14.604 | q4 × 8, context 4104 |

All seven cases parse and construct the complete 45-layer target graph. These
are conditional graph checks, not measured body geometry: the native profiler
segmentation must still identify active requests, per-sequence context, graph
padding and accepted-state indices. No whole-body simulation or CSV replay is
duplicated by this producer.

Quantization retains explicit FP8/E8M0 linear/expert representation and native
BF16 KDA/MLA projection defaults from the source adapter. A bounded c8/p4k
landmark independently showed BF16 KDA packed input N3336 at M32/K4096,
BF16 rank128 gate projections, and FP8/E8M0 dense MLP matrices. This confirms
the sampled representation; it is not evidence that every module in the
checkpoint has been inspected. No old observed timing or bandwidth override is
transferred. The model's current checkpoint config bytes remain unresolved.

The placeholder request is not a serving replay. Full request/step/sampling
behavior is outside these `scope="body"` templates. The compact identities and
graph counts are in `data/verification/glm-flash-ws9-inputs-20261030.json`.
