# Kimi explicit source scenarios

The `kimi-source/` cases derive from the corrected `kimi-variants/` inputs;
the originals remain unchanged. This is source-compatible conditional execution,
not identification of the loaded capture binary, host tile dispatch or group list.
No timing, bandwidth, launch parameter or checkpoint geometry was fitted.

| Inputs | Selected programs | Deliberately unresolved domain |
|---|---|---|
| 12 A5 E16 BF16, 93 layers | `arch35_fullmesh` native MC2, `ascend_bf16_grouped`, `vllm_ascend_recurrent_kda_arch35`; existing AttnRes and logical/physical MLA head caller retained | Dense paged MLA kernel; exact MC2 host dispatch; captured expert histogram |
| 5 A3 E896 W4A8, 4 layers | `vllm_ascend_recurrent_kda_arch22` | Native V2 dispatch/combine, W4A8 source mapping, paged MLA kernel |

The BF16 grouped factory requires integer expert rows, BF16 operands/output and
at least two banks. The A5 E16/EP8 layout supplies two banks per rank. It cannot
be applied to the A3 W4A8 checkpoint. `native_moe_protocol` replaces the old
pack/wire/unpack graph, including its two old wire recipes; it is not a second
communication charge.

KDA consumes the actual source-compatible BF16 gate, FP32 beta and FP32 state.
The compact capture contracts in
`acceptance-20261030-kimi-operators-332f0420` retain these dtypes. The no-MTP
launch has one query token and no accepted-token array. The native program
owns Q/K normalization and gate evaluation, replacing those old graph nodes.
The selected safe-gate scalar branch remains source-conditioned, not binary-proven.

All 17 cases parse with their complete models. Four representative complete
body runs through public `predict(scope="body")` at source base `b11beed3`:

| Scenario | Layers | Conditional body ms | Wall s | Peak MiB |
|---|---:|---:|---:|---:|
| A5 c1, context 4096 | 93 | 23.50149 | 7.20 | 147.5 |
| A5 c32, context 4096 | 93 | 27.91168 | 9.17 | 170.0 |
| A3 c1, context 4096 | 4 | 2.54965 | 0.28 | 39.1 |
| A3 c32, context 4096 | 4 | 5.34863 | 0.33 | 40.5 |

Each run retained unchanged start/end package fingerprints and released all
tasks, programs and allocations. These are whole model body predictions, not
TPOT or measured accuracy claims. Context is the existing conditional prompt
declaration; neither cache backing capacity nor client TPOT determines it.
Family timeline sums include overlapping phases, so they are labeled descriptive
activity, never an additive decomposition of the body span.

Reproduce inputs and bounded serial runs with
`data/verification/kimi-source-body-20261030.producer.py`. The compact manifest
and per-case receipts beside it retain inputs, source selectors, actual program
counts, work, resource service, timing, memory and timeline records.
