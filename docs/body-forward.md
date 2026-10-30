# Decode target-body forward

Both public prediction entry points accept an explicit device-body probe:

```python
from mindie_sim import predict, predict_pd

result = predict(case, scope="body", phase="decode", tokens=4,
                 context=16384, sequences=32)
milliseconds = result["body_forward_ms"]
```

`tokens` is the number of query/verification rows **per sequence**. For one
current token plus three speculative proposals, use four even if the declared
acceptance rate is 0.8. `sequences` is the active batch per participating DP
replica. All participants of the first independent EP cohort execute together.
The report records their identities and total valid query rows. A forced
balanced scenario uses `runtime.routing = {"policy": "balanced"}`. Round-robin
integer counts retain remainder rows and zero experts, including declared
capture padding. Router-group restrictions do not reject this forced policy.
The heaviest strided EP rank is the explicit timing representative; the
uniform peer communication assumption remains visible. Distinct positive row
classes execute inside one invocation using the selected matrix recipe.

The start boundary has hidden-state inputs and context KV ready. The end is
the target transformer body output ready across the cohort. All body attention,
FFN, residual and TP/EP/PP/SP communication remains in the shared Engine.
Declared graph padding and source implementation selection are preserved.
Host enqueue, request scheduling, embedding, final normalization/LM head,
draft generation, sampling, PD handoff and cross-step gaps are excluded.
Architecture-owned final body residuals remain included.

This gives a single forward span at a declared geometry. It does not derive a
steady-state batch from a TPOT target, simulate DSpark draft generation, or
convert acceptance into a latency multiplier. Ordinary `predict(case)` retains
the complete serving path for those downstream constraints. Context readiness
models a cache hit for this probe; it does not remove query attention or cache
updates. The graph probe has no request admission or memory-capacity guarantee;
deployment feasibility is checked through ordinary serving and its memory ledger.

For a PD input, either `predict` or `predict_pd` with `scope="body"` returns
`body_forward_ms = {"prefill": ..., "decode": ...}` and a complete report under
each pool. The same requested geometry is evaluated independently on both pool
configurations. These spans are not added together, and connector bandwidth does
not affect them. Use the decode pool entry for the Decode target-body comparison;
ordinary PD prediction includes actual KV bytes, destination admission and handoff.

```sh
python -m mindie_sim predict case.json --scope body --phase decode \
  --tokens 4 --context 16384 --sequences 32 --event-limit 0 -o body.json
```

Serving geometry comes from its workload. Passing these probe-only geometry
arguments with `scope="serving"` is rejected rather than silently ignored.
`tests/test_body_forward.py` checks host/arrival invariance, HBM response,
capture padding, balanced-routing domain preservation, complete EP/PP traffic,
independent PD pool timing and the ordinary CLI. Measurements require the same
body and query/context/batch boundary before an accuracy claim can be made.

## Default Decode attention mechanisms

The body entry selects the existing staged GQA decode program when independent
copy pipes are declared and the case uses causal, unwindowed floating-point KV
without CP/PCP. Its default KV split count remains one; an explicit
`flash_attention_decode` recipe can declare the actual split count. A3/A5
profiles supply the private pipe capacities. The selection contains no
model-name timing branch or measurement lookup.

BF16 indexed MLA with dense cache, latent/rope dimensions 512/64, 1--16 local
heads and CP/PCP=1 uses the existing compact indexed pipeline with offchip
scratch. Declared directional MTE1 modes are used when present; other devices
retain their generic MTE1 supply. Each existing per-sequence attention
operator retains its query/context geometry and dependencies. This does not
establish the captured runtime's fused-batch launch contract.

Compatible BF16 GDN Decode with FP32 recurrent state and PCP1 now selects the
non-interleaved QKVZ/BA `vllm_ascend_gdn_decode` caller. It retains separate
convolution, gating, Q/K norms and caller-owned copy/fill invocations. Independent
AIC/AIV pipes, the188416B convolution UB footprint, local head ratio and the
speculative width4 convolution domain are checked before installing the default.
Unsupported default domains continue analytically with a visible reason;
explicit lowerings and existing recurrent/gating recipe choices take precedence.
The state core stays analytical unless `recurrent_implementation="arch35_fp32"`
is explicitly selected. Scalar beta-store queue parameters remain independent
declarations; installing the caller does not infer them from an observed body gap.

`default_mechanisms` records these decisions and unsupported domains. Explicit
recipe, family, lowering or supplied RecipeBook choices take precedence. SCFA
has a reviewed explicit prefill implementation; that source is not relabeled as
a Decode implementation. Windowed, quantized-KV and other unsupported default
domains keep their existing analytic programs with a visible reason.

These are physical mechanism defaults, not claims that every attention shape
or hardware configuration was calibrated to 15 percent. In particular, the
A3 single-request KV split dispatch and sparse scratch residency need matching
implementation evidence. Ordinary serving retains its own explicit case recipe
declarations; compare body results only at the reported implementation boundary.

## Capacity selection for automatic Decode attention

Automatic GQA tiles are selected from a finite power-of-two domain using the
declared L1/L0A/L0B/L0C/UB capacities. The capacity check retains resident query
and online state plus both live key-tile windows; a wide head therefore reduces
the selected query/key tiles instead of crashing on the default 128-key tile.
This is an analytical capacity choice, not a claim about vendor dispatch.
The returned `default_mechanisms.gqa_tiles` records it. If even the minimum
tile cannot fit, the original analytical program and the reason remain visible.
Per-operator recipes retain priority over these installed family defaults;
the actual executed programs remain available in the timeline.
