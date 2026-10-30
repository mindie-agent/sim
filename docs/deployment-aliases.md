# Deployment labels and representation presets

Short catalog names resolve to a fixed existing structural declaration:

| Short label | Canonical catalog key |
| --- | --- |
| `deepseek-v4-flash` | `deepseek-v4-flash-0731` |
| `deepseek-v4-pro` | `deepseek-v4-pro-0813` |
| `deepseek-v4.1` | `deepseek-v4.1-flash` |
| `qwen3.8` | `qwen3.8-27b` |

These labels do not mean the newest upstream release. Catalog labels are
case-insensitive; unrecognized local file paths retain their original spelling.
Loading an alias produces the same layer plans, auxiliary structures, packed
weight accounting and coverage metadata as its canonical key.

Catalog cases may use these quantization aliases:

| Alias | Existing representation | Meaning |
| --- | --- | --- |
| `w8a8` | `w8a8_dynamic` / `W8A8_DYNAMIC` | INT8 body projections and routed experts, dynamic per-token activations, retained FP32 channel-scale policy and native exclusions |
| `w4a8` | `w4a8_dynamic` / `W4A8_DYNAMIC` | Routed expert INT4 weights and INT8 activations; other eligible linears retain INT8 weights and activations |
| `fp8` | `w8a8_mxfp8` / `W8A8_MXFP8` | The explicitly selected Qwen checkpoint policy: E4M3 weights/activations, E8M0 block-32 scales, GDN and MTP exclusions retained |
| `bf16` | `bf16` | No weight quantization; BF16 representation with intrinsic FP32 parameter categories retained |

The `fp8` alias is the user-requested MXFP8 declaration, not a generic alias for
every FP8 scale format or every checkpoint. The normalized `quant` record keeps
the canonical preset name, checkpoint provenance, exclusions and notes. It
does not claim that another model's historical checkpoint used that policy.
The W4A8 declaration likewise describes mixed expert/non-expert representations,
not four-bit weights on every matrix.

For example, the ordinary public prediction entry accepts:

```json
{
  "machine": {"backend": "ascend", "sku": "Ascend950DT_9581", "num_devices": 8},
  "model": {"preset": "qwen3.8", "quantization": "fp8"},
  "workload": {"requests": [{"id": "r", "prompt_tokens": 16, "output_tokens": 2}]},
  "runtime": {"tp": 8, "max_num_batched_tokens": 16}
}
```

Representation and hardware support remain separate. A3 has no declared FP8
matrix peak and the same FP8 case reports
`compute_precision_rate_required:fp8_e4m3`; the alias does not change peaks or
fall back to another precision. A5 has an explicit FP8 peak. Lower-level manual
model mappings still require an explicit `quant` record for FP8 block and scale
geometry; catalog `preset` declarations resolve that geometry through this
registry.

Acceptance uses the ordinary `predict`/`target_graphs` paths. It keeps each
catalog's native hidden, attention and expert dimensions, but limits controls
to one declared layer and a 128-entry LM head to bound local test cost.
DeepSeek Flash/Pro/V4.1, Qwen3.8, GLM5.3, KimiK3 and MiniMaxM3 controls exercise
W8A8, W4A8 and BF16 on A3 and A5. These are representation and execution
contracts, not full-model accuracy or deployed-checkpoint validation. MiniMax's
partial coverage and unmodeled features remain visible in the result.
The 51 dedicated controls pass in 2.55 seconds; the combined SKU, alias,
pipeline, invocation and coverage checks pass 126 tests in 2.80 seconds.
