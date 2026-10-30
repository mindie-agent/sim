# Generic model quantization preset boundary

A hand-written structural model now resolves registered `quantization` names through the same `QuantSpec` registry as HF and catalog inputs. Previously `w8a8_mxfp8` matched a substring check that changed the residual activation dtype to `fp8`, without loading the preset at all. Normalization then created BF16 weights with FP8 activations and no MX scale geometry; an A5 public graph failed with `compute_precision_rate_required:fp8`.

The corrected boundary keeps residual, norm and attention tensors BF16 by default. Matrix operands come from the declared preset: E4M3 payload, MX32 groups, E8M0 scales and the preset's role exclusions. Explicit QuantSpec wins over presets. Explicit residual/compute dtypes remain unchanged; legacy weight/KV dtype fields override preset defaults, consistently with the catalog wrapper.

Only registered names select predefined geometry. The registered `fp8` alias already denotes the MXFP8 preset across catalog/HF inputs and now has the same meaning in generic input. Unregistered `w8a8_fp8`, `fp8_e4m3` and custom FP8 names still require an explicit QuantSpec; no block size or scale format is guessed.

Validation: 73 focused tests passed. The real CLI control uses the complete one-layer quickstart geometry with 8 experts, top-k 2, TP2/DP2/EP4, four A5 devices, forced-balanced integer routing and explicit `ascend_mxfp8_grouped_nz` down projection. Both `graph --scope body` and `predict --scope body` complete at 0.0542583 ms, with zero live tasks/programs/allocations. This is a functional simulation control, not a hardware accuracy result.

The retained receipt and producer are `data/verification/generic-mxfp8-preset-cli-20261030.{json,producer.py}`. The in-memory tests also compare every generated operator against the same structure with an explicit QuantSpec, check real quantization producers and E8M0 metadata, and preserve explicit dtype precedence.
