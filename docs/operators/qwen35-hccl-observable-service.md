# TP4 AIV collective response and native GDN audit

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The explicit candidate appends two observable HCCL call responses to the
existing grouped/DSQ/MC2 candidate. All physical programs, buffers, peer
messages, payload bytes and dependencies remain. Each whole HCCL kernel uses
`max(source-program completion, response)` once, with no per-phase invocation.

Parameters and reproducible assignment:
[`tp4-parameters.json`](../../data/observations/a5-hccl-service-regimes-20261030/tp4-parameters.json),
`qwen35-hccl-service-regime-20261030.json`.

## Domain, population and withheld geometry

Rules match Ascend950DT_9582, BF16, four ranks, separate buffers and the existing
explicit source templates: AIV two-shot AllReduce with 60 blocks, and AIV mesh
AllGather with 64 blocks. No model name or layer number is a selector.

The native communication descriptor and exact selected HCCL algorithm are not
present in the capture. Source caller payloads agree with all adjacent observed
BF16 tensors: AllReduce 64 KiB / 1 MiB / 2 MiB; AllGather local contribution
16 KiB / 256 KiB / 512 KiB. This is a **source-compatible conditional response**,
not an independently measured link-bandwidth primitive or recovered dispatch.
No guessed hardware `scale_with` is applied.

AllReduce assignment pools the 78 steady-layer calls per forward across attention
and FFN. The initial attention reduction has a different transient population at
identical payload: means 114 / 256 / 76 μs at the three geometries. It is retained
as a validation failure; there is no initial-layer response or exception.
AllGather uses all 40 calls per forward. Endpoints assign the line; the middle
geometry is withheld from coefficient assignment, not a blind experiment.

| Kernel | Inclusive response, μs, with B = source payload bytes |
|---|---:|
| AllReduce | 24.619874 + 0.0000106247722 B |
| AllGather | 9.941193 + 0.0000385916173 B |

| Middle geometry | Observed mean μs | Response μs | Relative error | Absolute μs |
|---|---:|---:|---:|---:|
| AllReduce steady population | 33.387324 | 35.760755 | +7.11% | 2.373432 |
| AllGather | 24.071144 | 20.057754 | −16.67% | 4.013390 |

AllGather fails the relative 15% criterion and meets the agreed small-call
absolute tolerance. Both results remain visible. Summary/device duplicate rows
and unlinked AICPU wrappers never enter these responses. PMU components overlap
and are not added to native-call duration.

## Actual Engine and complete c128 Body

`qwen35-hccl-service-execution-20261030.json`
retains three actual one-layer HCCL controls on runtime `43a23cb0` plus evidence
commit `e258573a`. Real group buffers remain reserved through completion, input
readiness is 7 ms, and service work is conserved between response and analytic
arms. Each reduced graph charges two AllReduce groups and one AllGather group.

The complete q4/B128/context4096 target Body charges exactly **80 AllReduce and
40 AllGather responses**. It completes in **29.49 s / 110.73 MiB**, with zero live
tasks, programs and allocations. Predicted Body is **24.288207 ms**, compared
with observed mean **28.584774 ms**: **−4.296567 ms / −15.03096%**. This is still
outside 15%, without rounding it into a pass. The same-runtime parent without
these two responses is 22.436580 ms. Body time was never an assignment input.
Actual active KV length and routed expert IDs remain conditional as in the
parent scenario.

## GDN is not a uniform missing-cost family

The same receipt executes source-bound CausalConv1d, gating and recurrent native
operators individually through ordinary frontend/Engine programs. These are
single-call source program estimates, not additive Body error components.

| Native boundary | B4 predicted / observed μs | B64 predicted / observed μs | B128 predicted / observed μs |
|---|---:|---:|---:|
| CausalConv1d | 1.312 / 7.337 | 2.792 / 10.995 | 5.584 / 16.326 |
| fused GDN gating | 11.138 / 5.242 | 42.444 / 6.260 | 42.444 / 6.481 |
| RecurrentGatedDeltaRule | 6.415 / 10.133 | 55.998 / 56.665 | 111.996 / 124.319 |

The recurrent middle/high geometries are already within 15%. Gating predicts
more time than observed and has no capture tensor descriptors; its current
scalar-store engineering extrapolation and compiled branch identity need a
separate review. An inclusive response cannot correct this overestimate and
must not hide it. CausalConv1d retains a small-call shortfall of 6–11 μs; its
shape/state ABI is available, but descriptor/service decomposition is assigned
to the shared instruction-service work. No blanket GDN residual or additional
native response was introduced by this HCCL experiment.
