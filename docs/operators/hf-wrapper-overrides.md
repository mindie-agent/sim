# HF wrapper and catalog override equivalence

`model.hf_config` supplies the native architecture. Explicit wrapper fields
(`quant`, `quantization`, metadata, layer count and layer overrides) use the same
post-adapter override path as catalog cases. A complete `quant` record describes
the actual matrix representation; a quantization preset does not change the
residual/attention arithmetic precision. Explicit coverage metadata stays
structured through public prediction and analysis.

The former HF early return skipped all wrapper overrides. This affected the
17 Kimi variant inputs: their declared BF16 or W4A8 expert representation was
replaced by the source adapter's default MXFP4 representation. Their predictions
at `09811dad` and the subsequent window-only input revision must be re-evaluated;
source CSV extraction and physical body geometry remain reusable. Those earlier
values are not evidence of BF16/W4A8 accuracy or a reason to change hardware rates.

Study candidates preserve native config field spelling. Updating an existing
outer simulator field does not hide the new value below the old outer override.
An explicit new quantization preset replaces a previous complete quant record;
a new complete quant record replaces the previous preset. Study assembles that
resolved wrapper once rather than recursively resurrecting removed declarations.

Real controls cover the retained Kimi BF16 expert bank payload (two local banks,
3584 × 6144 × 2 bytes each), its W8A8 alternative with scale bytes, actual grouped
programs and quantization producers, layer-count changes, graph execution,
coverage propagation, serialization and input immutability.
