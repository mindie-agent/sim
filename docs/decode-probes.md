# Long-context decode probes

The device-body boundary includes every target transformer layer and its causal
communication. It excludes prefill execution, host submission, draft generation,
sampling and inter-step gaps. A successful body execution is a prediction; its
accuracy still needs a matching measured body window.

Use exact token counts. The current long-context campaign declares 126000 input
tokens and 3000 output tokens, not 126 × 1024 and 3 × 1024. Existing captures keep
their actual counts (for example, 131201 input and 64 output).

```python
from mindie_sim import predict
from mindie_sim.prediction.decode_probe import decode_windows, probe_capacity

for window in decode_windows(126000, 3000, query_tokens=4):
    geometry = dict(context=window["context"], tokens=window["tokens"],
                    sequences=[8] * 8)
    capacity = probe_capacity(case, **geometry)
    result = predict(case, scope="body", **geometry)
```

The example represents 64 requests across a complete DP8/EP cohort. Scalar
`sequences=8` also means eight requests **per participant**, not eight globally.
An explicit `[1, 0, 0, 0, 0, 0, 0, 0]` keeps seven idle peers at the expert
collectives. Independent DP cohorts outside the first modeled cohort are listed
as unprobed; their traffic is not silently added to this body result.

`decode_windows` samples the beginning, midpoint and last complete verification
window. With four verification rows the contexts are 126000, 127498 and 128996;
the last KV extent is 129000. With one row the last context is 128999. These are
declared sampling cursors, not a prediction of speculative acceptance or of the
actual sequence of scheduler batches. An observed replay can pass exact output
cursors with `output_offsets`. Client concurrency does not establish a body
batch size; the latter must come from the actual call geometry.

`probe_capacity` reuses the ordinary weight and KV allocation formulas. It
reports per-replica/stage cache, resident bytes and headroom, together with
max-model-length, per-replica sequence and batch-token limits. Body workspace is
derived from the maximum padded query rows, not from the input case's long
prefill request. Padding consumes execution workspace without inventing private
request KV. Ready prefix KV still occupies memory. Page groups, draft caches,
explicit graph pools and other reserves retain their declared ownership.

`predict(scope="body")` includes this capacity report. Capacity violations stay
visible and do not suppress an analytical body prediction or reduce its batch.
The report is a declared resident-geometry check, not a serving admission or
memory-peak guarantee; runtime scratch not declared in the case remains outside
that estimate. No hardware context, weights or remote process is initialized.
