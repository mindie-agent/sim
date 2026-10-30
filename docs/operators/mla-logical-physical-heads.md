# MLA logical heads and native decode head padding

The twelve A5 Kimi E16/BF16 TP8 captures have **12 logical heads per rank**, hence 96 global heads. Their native FIA uses 16 physical head lanes. The former input inferred 128 logical heads from FIA alone, making value expansion, output gate and output projection a third too wide.

The retained compact profiler ABIs independently establish this in all twelve captures:

- `FusedInferAttentionScore`: Q `[B,16,1,512]`, RoPE Q `[B,16,1,64]`, shared latent cache head count 1, output `[16,B,1,512]`.
- `TransposeBatchMatMul`: 12 matrices, each `[B,512] × [512,128]`.
- MLA output matrix: logical input width `12 × 128 = 1536`.
- Router: BF16 `[local_rows,7168] × [16,7168]`, **FP32** `[local_rows,16]` output.

`data/verification/kimi-logical-head-contract-20261030.json` retains each source CSV identity and compact ABI receipt. The raw CSV files were not rescanned. Checkpoint config bytes are still unavailable; these are explicit tensor-supported input corrections, not a claim that a particular retained source checkout is the captured binary.

## Explicit generic caller

```json
{"runtime":{"kernels":{"lowerings":{
  "mla_head_padding":"vllm_ascend_mla_head_padding"
}}}}
```

This implementation derives the next power of two from `plan.num_heads / TP`. There is no model-name branch. It covers BF16 native-prolog weights, dense shared latent BF16 KV, single-token decode requests, and the retained-prefill-weight branch. A separately selected MLA decode kernel or prolog is rejected rather than silently rewritten. Prefill keeps its logical geometry; unselected models keep the prior behavior.

The source is `vllm_ascend/attention/mla_v1.py`, SHA256 `d808b7b41203cbcc4458b55b83f35d7db7152891fd87a49b4a1376eca1f64194`:

- Lines 846–849 compute physical power-of-two head count.
- Lines 1090–1124 prepare padded qB and UK weights. Thus qB and q-absorption execute 16 heads in this decode branch.
- Lines 1804–1808 slice prolog outputs back to logical heads. This is a view, with no invented copy or launch.
- Lines 1674–1699 call `q_nope.contiguous()` and then separately `F.pad` q_nope and q_rope. After `[B,16,D][:,:12]`, q_nope needs a real copy only for B>1. Each pad reads logical payload and writes physical payload including zero lanes. q_rope has no additional contiguous copy on this branch.
- Lines 1799–1801 slice NBSD output along its leading head dimension. Lines 973–979 consume the resulting contiguous logical-head view in value expansion. It has no transpose/copy charge.

Projection and attention recipes remain analytic expanded programs. This change does not claim fused MlaProlog timing or a measured FIA dispatch. The extra caller copies are outside the prolog, so a future native prolog plan must not include them again.

## Precision and capacity

`LayerPlan.router_output_dtype` is independent from `router_dtype`. Explicit BF16 operands plus FP32 output retain BF16 matrix arithmetic and weights; output writes and downstream top-k reads use four-byte elements. This is supported by the captured ABI, even though a different retained AscendGateLinear override forces FP32 operands. That override is not asserted as this capture's identity.

The selected native caller retains original prefill parameters while preparing separate qA, kvA and padded qB NZ tensors. Lines 1011–1048 also retain the original kvB parameter and create contiguous logical UK/UV tensors; both copies are included. Padded UK is another allocation only when head padding is nonzero. The weight inventory includes these runtime-only tensors without counting them as checkpoint parameters. For the unexercised single-logical-head edge, UK may alias an ND parent rather than allocate a copy. An original NZ tensor instead requires a full ND parent retained by the UK view. With original layout unspecified, the model remains runnable and explicitly marks coverage partial and prepared-weight capacity as an upper bound, charging that full ND parent. The actual twelve-head cases have noncontiguous UK/UV and do not use this bound. Shared latent KV and recurrent state capacities do not acquire the physical query-head multiplier. Other allocator overhead and temporary load-time peaks remain outside this weight inventory.

The twelve E16 BF16 case files keep their previous provenance and record the superseded FIA-only inference. Old p32k checkpoint cases and A3 four-layer variants are unchanged. `num_key_value_heads` in the inherited HF metadata remains an unconfirmed legacy field; the latent MLA cache geometry explicitly remains one shared head.

## Verification and limits

Focused suite: 44 passed, 3 existing optional-profiler skips. Controls cover logical/physical pairs 3/4, 12/16, 16/16 and 20/32; B1/B2/B32 copy traffic; BF16 router arithmetic with FP32 output; prepared-weight/checkpoint separation; unchanged KV capacity; input readiness and zero live Engine tasks/programs. Every one of the twelve complete 93-layer inputs parses.

Four actual one-layer graph executions compare the prior input and corrected input under the same runtime and declared resource rates. Each first constructs the complete 93-layer graph, then executes complete logical layer 3 with outside dependencies ready at 7 ms. This is a diagnostic layer boundary, not whole-body accuracy:

| Decode batch | Prior input | Corrected input |
|---|---:|---:|
| 1 | 222.012 us | 220.076 us |
| 32 | 290.931 us | 289.144 us |

The newly represented caller copies offset part of the smaller logical output matrices. No duration, multiplicative correction or calibration parameter was taken from the captures. Full-body replay remains the separate profiler acceptance boundary.

Reproduce the source/case receipt and these controls with `PYTHONPATH=. python data/verification/kimi-logical-head-contract-20261030.producer.py --controls`.
