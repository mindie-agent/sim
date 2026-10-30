# GLM pooled score response migration

The Triton source at `8c30e4f491f73cba737a1dacc522894e437cdf3d`,
`vllm_ascend/ops/triton/glm5_next_lightning_indexer.py`, computes a separate
FP32 head-weighted query and then calls
`_glm5_next_lightning_indexer_score_kernel` with that query and the BF16 pooled
cache. Top-k and pool-to-token expansion follow in separate calls. The current
`index_pool_scores` boundary preserves that source split.

Older A5 response declarations matched `index_scores` with BF16 and a
`[tokens,heads,width]` shape. The corrected score boundary has FP32 and
`[tokens,width]`; those old selectors no longer match. Their numerical HBM
feature was also based on **uncompressed** causal token pairs:

```
old feature = causal_token_pairs * (128 * 2 + 4)
causal_token_pairs = sum(q * context + q * (q + 1) / 2)
```

`data/observations/a5-pooled-score-migration-20261030/producer.py` transports the
existing function exactly to `metadata.causal_token_pairs`, retaining the
260-byte scale as a geometry proxy. It does not interpret this proxy as actual
pooled-cache traffic, substitute the newly corrected byte count into an old
slope, or fit any new coefficient. The original rule and source boundary are
retained in each migrated rule's provenance. Query weighting and top-k remain
separate operators. A3 CANN KeyPool responses are unrelated to this migration.

The three explicit cases are in `examples/cases/pooled-score-migration-20261030/`.
Body predictions before/after restoring the selector are:

| Case | Before ms | After ms | Native score predicted / observed us |
| --- | ---: | ---: | ---: |
| A5 C4 P8K, B4 q4 | 18.161 | 18.666 | 48.097 / 48.097 |
| A5 heldout C12 P4K, B2 q4 | 15.201 | 15.490 | 27.627 / 53.706 |
| A5 heldout C24 P12K, B24 q4 | 31.147 | 33.641 | 245.947 / 313.767 |

The two transferred functions still underpredict their native score kernels by
48.6% and 21.6%. Selector migration restores the existing parameters; it does
not establish that the transfer is accurate. The A5 Triton CSV rows report
tensor shapes as `N/A`, so the native boundary is supported by retained symbol
names and source semantics; exact installed binary and tensor ABI remain
unproven. Validation uses the retained complete native call populations within
the selected Body geometry, and adds no new NPU execution.
