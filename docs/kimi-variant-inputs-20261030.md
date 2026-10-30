# Independent Kimi capture variants

Seventeen new inputs under
`examples/cases/high-concurrency-20261030/kimi-variants/` preserve two genuinely
different checkpoints. They are not placement variants of the old DP4/EP32
MXFP8/MXFP4 case.

| Variant | Captures | Target layers | MLA heads | Experts / top-k | Representation |
|---|---:|---:|---:|---:|---|
| A5 TP8/EP8 p4k and p16k | 6 + 6 | 93 (69 KDA, 24 MLA) | 128 | 16 / 8 | BF16 sampled projections and expert matrices |
| A3 TP8/EP8 four-layer | 5 | 4 (3 KDA, 1 MLA) | 96 | 896 / 16 | W8A8 projections; INT4 routed experts; explicit BF16 exceptions |

The first eleven cases reuse the profiler bundle0406 receipt. Six p16k cases
are the subsequent requested extension; their summaries and identical launch
source are retained, but the earlier p4k topology probe does not constitute a
new p16k body segmentation. New p16k CSV identity is path/stat pending profiler
hashing. No full event extraction or body replay is repeated here.

The original checkpoint config files have not been delivered. Each input uses
an explicit raw HF configuration with independently supported field changes and
lists retained catalog assumptions separately in model metadata. Public output
has `model_coverage="partial"`; this is a real native mechanism limitation,
not a gate requiring measurements before analytic execution. All seventeen
inputs parse and construct their complete declared target graphs through the
current runtime, with a stable package hash retained in the receipt.

## Native shape evidence and precision

A5 router `[1,7168] × [16,7168]` and grouped banks with two experts per EP8
rank support E16. The launch checkpoint explicitly names expert16/top8.
`MlaPrologV3` and FIA expose 16 local query heads, hence 128 at TP8. KDA still
has 12 local heads of dimension128, hence 96 KDA heads. Expert latent width3584
and intermediate3072 are supported by grouped weights. The dense first layer
uses local intermediate4224; its N8448 gate/up matrix corresponds to global
intermediate33792. It must not be relabeled as the MoE shared expert. The
subsequent shared path has N1536 and down K768, corresponding to shared global
intermediate6144 (two experts of3072), a distinct matrix.

A3 router N896, grouped bank112 and dispatch top-k `[1,16]` confirm E896/top16
at EP8. Q-b N2304 and absorbed Q shapes establish 12 logical local MLA heads;
the native FIA call separately pads those heads to16. The input retains96
logical global heads and marks the extra physical FIA work unmodeled. It does
not create128 logical heads to match a padded kernel tensor.

The bounded A3 tensor audit proves BF16 composed BFG and output projections,
INT8 Q/K/V, ordinary FFN/shared/latent projections, and INT4 routed matrices.
These become explicit `kda`/`mla` precision role overrides. The old FP8 matrix
wrapper, scalar-store calibration and measured bandwidth overrides are not
inherited. A5 BF16 input packing and A3 offline-composed BFG remain separate
native-work gaps; the generic analytical projections remain executable.

## Deployment and boundary

Both Kimi variants have retained TP8/DP1/EP8 launch scripts, no MTP and
FULL_DECODE_ONLY intent. Resolved startup logs were not located. Forced balance
is an explicit scenario declared by those scripts, not independently verified
startup state. A5 APC is disabled by script; A3 APC is enabled and its retained
profile cell gives a real93.75% hit ratio. An enabled cache is not described as
100% prefix reuse.

The cases choose device-body eager wrapper semantics because exact graph
capture buckets are unresolved; this does not claim that the captured service
was eager. Source sequence-parallel right padding is explicit, retaining valid
query rows separately from physical padded FFN rows. Native
MoeDistributeDispatch/Combine V2 or V3 events are not assumed to be the generic
AllToAll protocol: its decomposition is a declared analytical approximation.
Input templates retain this gap rather than binding an unrelated source kernel.

Each input's placeholder request is not the serving workload. Conditional q1
checks use client batch and actual prompt-token counts from summary metadata;
profiler must still supply exact per-forward context and active batch for an
accuracy comparison. Client median TPOT stays outside `observed_body_ms`.
