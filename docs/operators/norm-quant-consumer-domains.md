# Explicit Norm/Quant fusion consumer domains

`runtime.kernels.fusion_domains.norm_quant` limits the existing `norm_quant`
lowering pass by its actual consumers. It does not enable the pass by itself.
For a caller that fuses only Dense FFN input normalization and quantization:

```json
{
  "lowering_passes": ["activation_quantization", "norm_quant", "residual_norm"],
  "kernels": {
    "fusion_domains": {
      "norm_quant": {"consumers": [{"recipe": "dense_gate_up"}]}
    }
  }
}
```

The supported selector fields are `recipe`, `family`, `structure_block` and
`structure_role`. Fields within one selector are conjunctive; separate selectors
are alternatives. Every direct consumer of a shared quantized payload/scales
producer must match. Matching one consumer is not sufficient to fuse a producer
that also serves a disallowed Attention projection. No operator ID, layer index,
model name or timing observation participates in selection.

An absent configuration preserves the existing generic fusion behavior. An empty
consumer list disables this fusion. Source-bound operators, nonexclusive norm
outputs, mismatched shapes, streams and stages remain governed by the existing
fusion contract. The later residual fusion retains its live residual output.
Selected fused records expose `fusion_consumer_domain` metadata.

`tests/test_norm_quant_domains.py` checks the complete GLM78 capture-row input:
existing generic fusion yields 60 `add_rmsnorm_quant` calls; selecting
`dense_gate_up` yields 3 and leaves 152 `add_rmsnorm` calls. It restores 57
ordinary Attention quantizers, alongside the 21 query-norm variants already
present. All 78 native prolog groups remain. Total matrix and vector arithmetic
are unchanged; the split path restores exactly one BF16 intermediate store/read
for each of those 57 M12/H6144 norm outputs. This does not claim that every
remaining generic quantizer already has a one-to-one native capture match.

The same tests verify shared-consumer exclusion, conjunctive structure selectors,
empty-domain behavior and no-config compatibility. The mechanism changes no
resource rate or invocation latency, and source cases opt into it explicitly.
