# Shared-expert W8A8 INT32 / SwiGLU / INT8 boundary

This mechanism models one verified native caller: the shared expert in
`vllm_ascend/ops/fused_moe/shared_experts.py` that first runs an INT8
`npu_quant_matmul` with `pertoken_scale=None` and `output_dtype=torch.int32`,
then runs a separate `npu_dequant_swiglu_quant` that consumes that INT32 matrix,
the stored FP32 channel scales and the original activation row scales. The
second kernel writes INT8 activation data plus dynamic FP32 row scales for the
ordinary down matrix. It is not a BF16 matrix epilogue and the two native calls
are not merged.

## Selection contract

The narrow selector is exactly:

```json
{"runtime": {"kernels": {"lowerings": {
  "shared_expert": "ascend_shared_gate_up_int32"
}}}}
```

An operator is considered only with the approved structural metadata
`ffn_category='shared_expert'` and `ffn_role='gate_up'`, plus an activation
consumer with `ffn_role='activation'` and a down consumer with
`ffn_role='down'`. The producer must be a W8A8 INT8 matrix that already has a
real activation-scale producer. Op ids and name prefixes are never selection
evidence.

Only the W8A8 + SwiGLU branch is selected. W4A8/W4A8MXFP matrices are left
unselected. A selected SiTU activation raises
`shared_quant_situ_unsupported` rather than reusing the SwiGLU arithmetic.
Nonzero source clamp limits are modeled with the three additional source clamp
operations; no other branch is inferred.

The selector and consumer recipe use the shared implementation registry.
The graph hook runs after `activation_quantization` and before `native_fusion`.
The ordinary dense/shared frontend declares structural category/role metadata;
no operator-name matching or manually injected recipe book is required.

## Producer boundary

`npu_quant_matmul` consumes the original INT8 activation producer, INT8 weights,
and its declared weight-scale ABI. The approved fixture declares the API scale
as BF16 (`scale[512] BF16`); the producer keeps that descriptor and does not
pretend it is FP32. Its actual logical reads are:

* `M*K` INT8 activation bytes;
* `N*K` INT8 weight payload bytes;
* the declared BF16 producer channel-scale bytes (`2*N` in the fixture).

It writes `4*M*N` INT32 output bytes. The activation row scales are not read by
this call (`pertoken_scale=None`); the original quantizer still produces them for the separate
consumer. The matrix export therefore has no per-token-scale input slot. The `matrix_epilogue` row/column scale stage and its vector work are
removed from the producer. The matrix still performs the full `2*M*N*K`
reduction.

## Consumer boundary

The consumer is one source-bound kernel, `ascend_shared_dequant_swiglu_quant`,
with `source_boundary="kernel"`. It depends on the real INT32 matrix and on the
original activation quantize producer (or an equivalent shared scale producer
that returns `activation_scales`). It consumes `[M, 2H]` INT32, FP32 channel
scales `[2H]`, and FP32 row scales `[M]`, and writes `[M,H]` INT8 and FP32 row
scales `[M]`.

Logical GM traffic:

```text
read  = 4*M*(2H) + 4*(2H) + 4*M
write = M*H + 4*M
```

The FP32 channel-scale read is charged once logically. The native kernel
reloads parameters per active core; that tiling is not reconstructed and is
declared as a limitation.

The analytical vector ledger is the sum of the source expressions, not a
duration or instruction count:

```text
dequant cast, channel mul, token mul                 3 * M * (2H)
gate bias add, SwiGLU alpha/exp/add/div/gate mul      6 * M * H
abs, row-max comparisons, factor, Brcb, div, casts    M*H + M*(H-1) + M
                                                      + 8*M + M*H + 3*M*H
optional clamp (gate min/max + act min)              3 * M * H
```

`Brcb` is included as the source's eight-way row-scale broadcast; layout and
copy reshapes are not charged as arithmetic. The implementation is a coarse
three-stage load/compute/store program on the common Engine (`Work` and the
source `PlannedRecipe`). It claims neither native UB tiling/core partition nor
workspace pipeline overlap, and no captured binary identity.

The former SwiGLU operator and the separately inserted down-activation
quantize are removed. Down now has exactly one dependency on the consumer and
reads the consumer's INT8 data and row scales. It therefore performs no second
dynamic quantization; its own matrix epilogue remains the ordinary BF16 output
dequantization of `npu_quant_matmul`.

## Source identity

| Item | Identity |
| --- | --- |
| Caller | `vllm_ascend/ops/fused_moe/shared_experts.py`, lines 305-350, revision `ae923905948f360f4b7ab06df52d0789a0c9316e`, SHA-256 `f1780df8ec2fc290bfeb946e5d48edcb8c1b4aee0f50c1bd9d10291def201e77` |
| Kernel body | `csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant.h`, lines 244-817, same revision, SHA-256 `2e516dc7332449057ea8379eebd61a0b6f2a2917907b7f88d07b6025d3cf72b2` |
| Kernel entry | `csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant.cpp`, lines 159-180, same revision, SHA-256 `bfc9cac8008acdc9c4b9ab3933c7a275bcb362681bd9128bbe6090ceed7456d5` |
| Approved fixture | `profiler/tests/fixtures/mixed_step_body/a3_glm53_prefill.csv` line 2, capture SHA-256 `ded3a03adc7a82b0801acc6a88bb85589b8f3a3f3a4a416a1322ebfb514c614f`; `x[4096,6144] INT8 ND`, `w[16,384,16,32] INT8 NZ`, `scale[512] BF16 ND` -> `[4096,512] INT32 ND`, 75 native events |

## Limitations

* Native chunking, UB tiling, core count and exact vector instruction schedule
  are not reconstructed.
* The producer BF16 scale descriptor is preserved as an ABI fact; it is not
  silently converted to FP32. The consumer's separately stored FP32 scale is a
  distinct tensor and is charged as FP32.
* The logical GM ledger reads the FP32 channel scale once; native cores may
  refetch it.
* Bias, group index, static quant scale/offset and SiTU are absent or rejected;
  no value is invented for them.
* Captured CANN binary identity is not established.
* The source program keeps the original shared stream and declares one
  invocation for the matrix and another for the activation consumer. Original
  and newly generated row scales have distinct producers and dependencies.
* Repeated layers preserve one physical signature per tensor/work contract;
  occurrence references stay in source provenance. No observation determines
  a hardware rate or implementation choice.

## Same-capture producer and consumer acceptance

The retained 2026-10-29 reports in
`data/verification/shared-quant-wave4-20261029/` rescan the original complete
GLM and V4Flash captures and execute the public `compare_cohorts` API. The
selected cases change only `lowerings.shared_expert` from each profiler
reference case. They keep the original hardware declarations, matrix recipes,
observations and call counts. The simulator runtime is revision `b6f06859`;
each summary contains matching package hashes before and after execution,
case/mapping hashes, raw capture identity and bounded scanner statistics.

The consumer contract is the retained seven-input/two-output ABI, including
four absent optional inputs. Its separately stored FP32 channel scale cannot
be substituted with the producer's BF16 channel-scale input. The regression
fixture copies the actual GLM consumer event from the profiler fixture;
its original line, source hash and copied bytes are recorded alongside it.
The contract does not establish scalar values, core tiling or captured binary
identity.

| Capture / complete native boundary | Calls | Observed mean us | Predicted us | Signed cumulative gap ms | Criterion |
| --- | ---: | ---: | ---: | ---: | --- |
| GLM prefill INT32 QuantBatchMatmulV3 | 75 | 75.697253 | 58.682044 | -1.276141 | outside relative 15% |
| GLM prefill DequantSwigluQuant | 75 | 23.719947 | 16.845576 | -0.515578 | within absolute 15 us |
| V4Flash prefill INT32 QuantBatchMatmulV3 | 44 | 69.420318 | 67.728861 | -0.074424 | within relative 15% |
| V4Flash prefill DequantSwigluQuant | 44 | 32.947932 | 10.131394 | -1.003928 | outside absolute 15 us |

GLM's selected two-family event activity is 7.456290 ms observed versus
5.664572 ms predicted (-24.03%). V4Flash is 4.504203 ms versus 3.425851 ms
(-23.94%). These are sums of separate complete event activities, not elapsed
spans, complete shared-expert chains, model totals or serving latency. The
GLM small consumer passes its absolute-error criterion while still having a
negative cumulative residual; the V4 consumer remains the largest gap of
these two selected boundaries. A nearby INT32 matrix prediction does not
establish convergence of its consumer.

The original BF16 matrix boundary cannot serve as a paired baseline for these
INT32 events. The new reports contain the actual newly supported descriptors
without recasting the previous incompatible prediction. No measurements enter
the selected recipe or hardware parameters. Native vector tiling, repeated
parameter traffic, load/compute/store overlap and invocation resource details
remain the stated refinement domains; these residuals alone do not identify
which mechanism caused them.

Reproduce the prediction from a retained case and mapping without rescanning
or modifying the raw capture:

```python
import json
from pathlib import Path
from mindie_sim.cohort_comparison import compare_cohorts

root = Path("data/verification/shared-quant-wave4-20261029")
name = "a3_glm53_prefill"  # or a3_v4f
case = json.loads((root / f"{name}.case.json").read_text())
mapping = json.loads((root / f"{name}.mapping.json").read_text())
report = compare_cohorts(case, mapping)
print(report["windows"][0]["cohorts"])
```
