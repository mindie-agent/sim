# Matrix recipe integration on retained native cohorts

The 2026-10-29 diagnostic changes only the explicit ordinary `matmul` recipe
from its fixed-tile default to `steady_matmul`. It regenerates structure mapping
from the same retained captures, then executes `compare_cohorts` on matching
native descriptors. No observation, hardware rate or work multiplier changes.

The executed runtime is `239f443c25b640d8545416ef115cafd64b26d031`, package
SHA-256 `4295bd69826a8127bd2a4adc8e28cc12d2946079eb84867f0872ae1b68c5fe5b`.
Before/after package identities match in both summaries. The fixed reference
predictions are retained previous reports, not fresh fixed-recipe executions.
Their exact file hashes and case origins are preserved in each summary.

| Native cohort family | Calls | Observed activity ms | Fixed reference ms | Explicit steady ms | Steady signed gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 BF16 MatMulV3 | 184 | 458.381475 | 749.773104 | 471.914144 | +2.952% |
| Qwen3.8 FP8 QuantBatchMatmulV3 | 160 | 714.307065 | 1253.382633 | 749.188169 | +4.883% |
| GLM INT8 QuantBatchMatmulV3 | 180 | 46.253634 | 34.289111 | 29.358475 | -36.527% |

All eight Qwen matrix descriptors satisfy the current per-call relative 15%
criterion. All native counts, observations, descriptors and logical physical
work are unchanged. These are isolated predicted call costs multiplied by actual
native call counts; their sum is neither graph span nor whole-model accuracy.
The recipe is an explicit analytical mapping, not identified native tiling.

GLM provides a material counterexample: only two of five descriptors meet the
criterion. The 4096x6144x256 product has 75 calls, observed 242.871 us and
steady prediction 74.263 us. Its signed activity gap is -12.646 ms. The
compatible arch22 per-token quantization source has INT32 GM scratch and
vector-consumer synchronization, but this capture's precise native dispatch
is not established. That mechanism must not be applied to every quantized
matrix by matching a residual.

## Replay

Run from the simulator checkout with its Python environment:

```sh
python -m mindie_sim.cohort_comparison \
  data/verification/matrix-cohort-wave4-20261029/a5_qwen38-steady.case.json \
  data/verification/matrix-cohort-wave4-20261029/a5_qwen38-steady.mapping.json \
  -o /tmp/qwen38-steady-cohorts.json
python -m mindie_sim.cohort_comparison \
  data/verification/matrix-cohort-wave4-20261029/a3_glm53_prefill-steady.case.json \
  data/verification/matrix-cohort-wave4-20261029/a3_glm53_prefill-steady.mapping.json \
  -o /tmp/glm53-steady-cohorts.json
```

The retained `*.comparison.json` files contain the freshly executed results;
`*-summary.json` keeps prior-reference and new-prediction numbers together.
The mappings are scoped to `MatMulV3` and `QuantBatchMatmulV3`; omitted native
families are not implied to have zero cost or to satisfy accuracy criteria.

## Default selection boundary

This evidence supports exposing the already implemented steady mapping in real
Qwen selected cases, with source planners retaining precedence. It does not
establish a universal accuracy advantage for GLM small-K output schedules.
Steady preserves physical work, nominal finite-buffer overlap and input
readiness; external contention remains its declared fluid approximation.
Source-owned schedules and aggregate-expert approximations keep their own
selection domains. No generic default changes in this diagnostic.
