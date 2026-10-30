# Role-specific matrix representations

`model.quant.category_overrides` declares matrix representations by semantic
family or role. It uses the same resolution in operator construction and weight
residency; a projection cannot execute FP8 while its resident weights are counted
as BF16. This is an input declaration, independent of timing observations.

```json
{
  "name": "mixed_checkpoint",
  "weight_format": "bf16",
  "activation_format": "bf16",
  "expert_weight_format": "bf16",
  "weight_block": [1, 32],
  "expert_block": 32,
  "scale_format": "fp8_e8m0",
  "scale_bytes": 1,
  "activation_group_size": 32,
  "activation_scale_format": "fp8_e8m0",
  "activation_scale_pack": 2,
  "category_overrides": {
    "mla.q_a": {
      "weight_format": "fp8_e4m3",
      "activation_format": "fp8_e4m3"
    },
    "shared_expert": {
      "weight_format": "mxfp4",
      "activation_format": "fp8_e4m3"
    }
  }
}
```

Overrides resolve from a family to its more specific dotted role, with explicit
role fields taking precedence. They can set weight and activation formats,
weight block/expert group geometry, scale format/bytes, and activation scale
group/format/packing. A role's `weight_format` applies to that role even when it
is an expert. Other fields inherit the base declaration. This is neither a
model-name dispatch table nor a regular-expression module matcher.

Existing exclusions remain authoritative. Intrinsic FP32 parameter and BF16
control categories retain their existing representations. `mla_absorb` inherits
the `mla` family unless it has its own override. Generic KDA linears expose
`kda.q`, `kda.k`, `kda.v`, `kda.g`, `kda.b`, `kda.f_a`, `kda.f_b`, and `kda.o`;
latent MLA exposes `mla.q_a`, `mla.q_b`, `mla.kv_a`, `mla.g`, and `mla.o`.
Latent MoE transforms use `latent_moe` in both execution and weight terms.
Existing grouped output projections keep their existing category.

GQA and its known-projection-only path expose `attn.qkv` and `attn.o`;
their layer and draft weight terms use those same roles. Both inherit `attn`
unless a child is declared. For an INT8 parent, adding
`"attn.o":{"weight_format":"bf16","activation_format":"bf16"}` keeps
the QKV INT8 payload and quantization producer while giving the output matrix
BF16 operands and resident weights. No model name selects this override.
The retained Qwen 35B TP4 and 397B TP16 controls use the observed output
shapes `(256,2048,1024)` and `(256,4096,512)`, respectively; their 10/15
GQA occurrences per body and QKV shapes remain unchanged. These controls
establish representation consistency, not native MatMulV2/V3 timing accuracy.

FP4 storage no longer erases an explicitly declared FP8 activation. Its input
payload, E8M0 scales and dynamic quantization producer remain in the graph.
The generic FP4 path still expands operands for BF16 computation. It does not
infer an FP4 hardware peak, native conversion latency, or a fused native
instruction sequence. Representation support is distinct from native kernel
support.

The independent current-checkpoint case is
`examples/cases/kimi-k3-a5-current-checkpoint-conditional.json`. Its evidence is
retained under `data/observations/sources/kimi-checkpoint-20261029/`:

| Module role | Weight representation | Activation representation |
| --- | --- | --- |
| KDA q/k/v; MLA q_a/q_b/kv_a | FP8 E4M3, group32 E8M0 scales | FP8 E4M3, group32 E8M0 scales |
| Dense MLP; latent MoE transforms | FP8 E4M3, group32 E8M0 scales | FP8 E4M3, group32 E8M0 scales |
| Routed and shared experts | Packed FP4 E2M1, group32 E8M0 scales | FP8 E4M3, group32 E8M0 scales |
| Gate/output projections and MLA kv_b | BF16 | BF16 |
| Norms, router and LM head | Existing unquantized categories | Existing category contract |

The 5,883-byte model config SHA matches the retained historical config. The
45,168,749-byte quantization description was independently recovered from the
current checkpoint, with SHA256
`f22b76c32dd43f29a34bf15d114b2eb4be1dcd8217a1f89a91919ecf5df0a893`.
No historical hash for that file was retained, so this case is explicitly
`current_checkpoint_conditional`. Neither capture timings nor operator
occurrence counts determine its quantization.

The declaration census
`data/verification/model-scoreboard-20261029-kimi-current-checkpoint-declaration.json`
uses all 93 body layers. It confirms 69 occurrences of each KDA q/k/v role,
24 of each MLA q_a/q_b/kv_a role, 184 latent transforms, and 184 each of
routed/shared expert matrices with the expected activation producers.
Two matrix operators represent a packed gate/up and a down projection;
these operator counts are not counts of checkpoint parameter tensors.
Per-rank resident weights are 60,804,783,856 bytes, including
2,942,193,664 bytes of quantization metadata under TP8/DP4/EP32.
This census establishes representation consistency, not timing accuracy.

The same case inherits the existing A5 profile's shared MTE1 declaration,
13.5168 TB/s, instead of copying the old 4 TB/s off-chip declaration into
MTE1. That separate input correction and its six geometry controls remain in
`data/verification/model-scoreboard-20261029-kimi-mte1-input-audit.json`.
The old case/report values remain available for comparison.
