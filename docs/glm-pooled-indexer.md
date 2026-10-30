# GLM pooled index keys and causal token selection

GLM 5.3 Flash declares `index_kpool=4`. Its indexer scans completed pools,
selects at most 512 pool IDs, expands each into four token IDs, and appends
the causal unfinished tail. The output capacity is 2051 indices; the token
KV cache remains uncompressed. At context 131201, each of the six target
verification queries scans its own roughly 32800 completed pools. None of
the target queries is removed by `index_share_for_mtp_iteration`.

The model now includes one shared FP32 hidden cast, a joint FP32 K/head-weight
projection, FP32 normalization and gate projection, FP32 APE, softmax-weighted
pool compression, bounded FP32 K/gate tail storage, pooled BF16 index-cache
storage, pool selection and token-index
expansion. Compression reads the historical tail before its replacement.
The indexed attention program consumes the same causal token counts, including
short contexts and calls crossing a pool boundary. Context partitioning is
rejected until an explicit pooled-cache ownership policy exists.

The source basis is vllm-ascend revision
`8c30e4f491f73cba737a1dacc522894e437cdf3d`, specifically
the model `attention.py`, backend `indexer_kpool.py`,
`sparse_attn_indexer_kpool.py`, `glm5_next_kpool_tail_compress.py`, and
`glm5_next_lightning_indexer.py`. Their content hashes are in the
[compact evidence report](../data/verification/glm-kpool-body-20261030.json).
This establishes causal pooling and tail semantics. It does not establish
the captured `KeyPool`/`PoolKeyIndexer` binary dispatch or owner schedule.

The default score follows the source: cast query and head weights to FP32,
form the head-weighted query, then compute one FP32 matvec against the pooled
keys. It has no per-head ReLU. The latter expression is available only as an
explicit `causal_pool_indexer_analytic` lowering with
`score_mode="per_head_relu"`; it is not treated as equivalent. BF16 WK/weight
and gate parameter storage remains resident alongside the prepared FP32
execution copies. Speculative rollback retention is outside the
contiguous-forward tail model.

FULL graph padding applies to the joint projection, normalization, query
conversion, weighted query and output capacity. For a bucket containing 18
valid target queries and 24 physical rows, these producers keep 24 rows; only
valid query metadata contributes pooled cache updates and causal scans.

The retained A3 TP16 input ran all 45 target layers with q6 and context 131201
for every observed batch population. Input model and runtime declarations,
existing response parameters and hardware rates remained unchanged; no new
timing parameter was fitted.

| Batch | Observed Body mean (ms) | Earlier prediction (ms) | Current prediction (ms) | Relative error | Within 15% |
|---:|---:|---:|---:|---:|:---:|
| 1 | 22.6032 | 23.7043 | 23.0722 | +2.07% | yes |
| 2 | 25.7131 | 28.3920 | 26.3404 | +2.44% | yes |
| 3 | 31.4148 | 33.0531 | 29.5749 | -5.86% | yes |
| 4 | 28.2507 | 37.7729 | 32.8846 | +16.40% | no |
| 7 | 35.7478 | 49.8780 | 40.7453 | +13.98% | yes |
| 8 | 31.5800 | 53.7661 | 43.2170 | +36.85% | no |

The earlier predictions come from the retained prior runtime, so this table
is not an isolated source-toggle experiment. Independent one-layer indexer
and attention controls at B1/B4/B8 reduce the context-131201 operator sum
from 0.3071/0.9211/1.7926 ms to 0.2495/0.4769/0.8339 ms. The 4K B1 control
increases from 0.1402 to 0.2094 ms because the newly represented gate and
cache phases still have a cost.

All six full Body runs completed in 2.2–3.4 seconds each, with a 95.3 MB
process peak and no live tasks or programs left behind. Runtime source hashes
were unchanged throughout the run. Related checks passed 82 tests, with
three preexisting fixture-dependent skips. The remaining B4 and B8 errors
remain failures. Actual per-call context values were not captured: 131201
is the same-run initial prompt, and 131265 is the requested-output upper
bound. These results cover the target Body boundary, not Serving latency.

The initial per-head/BF16 producer result is retained as superseded review
evidence in the compact report. It is excluded from the current table and
acceptance status. The source correction was selected from code semantics,
not from observed error.
