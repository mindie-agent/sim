# Kimi actual-context source composition

The twelve inputs in `examples/cases/high-concurrency-20261030/kimi-source-complete/` retain the corresponding `kimi-source-paged-actual-context/` parent and add only `lowerings.causal_conv_update = "ascend_causal_conv1d_update"` plus provenance metadata. Every parent SHA is pinned in its child and in `data/verification/kimi-source-complete-20261030.json`. The original cases, actual-context CPU host oracle and their receipts are unchanged.

The new selection is explicitly **q1 non-speculative decode**: one active state slot per sequence, three history rows read and written, and one update invocation per KDA layer. SP padding remains outside the active convolution and attention rows. Prepared convolution weights coexist with checkpoint weights through the existing shared selector. Existing KDA, AttnRes, MLA head padding and paged host dispatch declarations are inherited without modification.

Contexts remain 4104 or 16392, from the retained profiler/client geometry reference; per-forward KV growth and captured binary identity are unobserved. “Complete” names this input composition, not complete kernel reconstruction or measured accuracy.

Reproduce input generation and the bounded checks with:

```sh
python data/verification/kimi-source-complete-20261030.producer.py \
  --run-complete -o data/verification/kimi-source-complete-20261030.json
```

The producer executes two four-layer controls (B1/context4104 and B32/context16392), verifies three active convolution calls and one MLA call, source state-write bytes, recurrence dependencies, 7 ms late input readiness, and released resources. The optional single complete B32/context16392 graph has a **60 s** subprocess budget and verifies 69 convolution and 24 MLA calls. Timeouts remain failures; the producer does not extend its budget or modify a program to obtain completion. Numerical results and runtime identity are retained in the JSON receipt.
