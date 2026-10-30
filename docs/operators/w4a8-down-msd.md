# Ordinary A8W4 MSD grouped-down

[Source] The caller is `vllm-project/vllm-ascend` revision
`8fd0bc4885cb52e58791aaa8efacc571b9bd115d`,
`vllm_ascend/quantization/methods/w4a8/w4a8.py` (`apply_gmm2`, lines 535-550):
`torch_npu.npu_grouped_matmul` receives INT8 activations, packed INT4 weights,
uint64 bit-pattern FP32 column scales, a nonempty FP32 `bias` tensor used as the
MSD scale-assist matrix, and an FP32 per-token scale, and returns BF16.

The independently selected CANN source scenario uses `ops-transformer` revision
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8`,
`gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp` lines
2153-2180 and 2240-2353. This profile requires `quantGroupNum == 1`,
`K % 64 == 0`, `K <= 18432`, and
`isMSD` is true when `tuningConfig == 0`, `avg_m == 0`, `N/avg_m > 4`, or
`withOffset`, together with a nonempty bias tensor. The caller supplies no
tuning configuration, no runtime weight offset, and a nonempty scale-assist
tensor. In this source revision those arguments select MSD; the GLM shape is
outside the host's two explicit whitelist entries, so it selects
`MSD_API_DEQUANT`. This is not identification of the captured CANN binary.
The absent workspace allowance keeps the ND in-place input packing path;
no extra NZ preprocessing allocation is inferred.

The dequantization body is
`gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd.h` lines
442-516. It casts the two FP16 matrix outputs to FP32, applies the `16x` high-digit
term, adds the low digit, adds the per-channel scale-assist value, multiplies
the per-token scale and casts to BF16. The shared common W4A8 pre pipeline is
`gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd_pre.h`
lines 151-204:
`high = floor(x/16)` and `low = (x & 15) - 8`.

Exact captured CANN binary identity is not established. The metadata carried by
the lowered operator records the semantic source revisions, files, hashes and
line ranges, and states `capture_binary_identity: not_established`.

## Public selection

```json
{
  "runtime": {
    "kernels": {
      "lowerings": {
        "grouped_gemm": {
          "implementation": "ascend_w4a8_msd_grouped_down",
          "recipes": ["grouped_down", "balanced_grouped_down"]
        }
      }
    }
  }
}
```

The ordinary registry binding applies only to those two semantic recipes. The
gate/up `grouped_gate_up` recipe is not selected by this binding. Model names,
architecture names and vendor labels do not select it.

The binding runs after the ordinary `activation_quantization` pass. That
pass owns producer dtype, dependencies and common-subexpression reuse; the
source binding never manufactures a BF16 producer. Omitting the required
pass gives an explicit modeling error. `OpSpec.dtype` and matrix work use
INT4 compute precision, while `metadata.output_format` remains BF16.

`lower_op` rejects non-down recipes, probability-moment routing, non-integer
expert rows, non-INT8 activations, non-per-channel packed INT4 weights,
transposed weights, generic linear bias, runtime weight offsets, non-BF16
output, `N % 64 != 0`, `K % 64 != 0`, and `K > 18432`. It requires an input
quantization producer with one FP32 scale per row. The producer is never folded
into the down operator: with the ordinary down-only binding it remains the
separate `dynamic_quantize` operator; when an independently selected gate/MSD
profile already publishes the same INT8 payload and scales, that producer stays
the gate program's external output and the down operator consumes it.

## Coarse phases

| Phase | Source evidence | Modeled work |
| --- | --- | --- |
| pre | shared common W4A8 pre | read `M*K` INT8; write `M*K` total packed high/low INT4 scratch; vector `6.5*M*K`; MTE2 `M*K`, MTE3 `M*K` |
| matrix | `_msd.h` `MMCompute`/`GetTensorC`, tiling | INT4 product `4*M*N*K`; read `M*K` scratch, `0.5*N*K` packed weights and `8*N` uint64 column scales per **active** group; publish `4*M*N` FP16 scratch; column scale vector equivalent `2*M*N` |
| post dequant | `ComputeDequantAndActivate` | read `4*M*N` FP16 scratch, `4*N` assist per active group and `4*M` per-token scales; vector `6*M*N` |
| output | BF16 cast/store in the post body | BF16 cast `M*N`; write `2*M*N`; MTE3 `2*M*N` |

The matrix publication is Fixpipe/`GetTensorC`, not an AIV MTE3 store. Its
`4*M*N` HBM bytes are retained in every case. An additional `fixpipe` service
record is added only when the execution geometry declares a `fixpipe` resource;
no rate is inferred or invented. The `2*M*N` column-scale application is a
declared vector-equivalent approximation shared with the gate MSD model, not a
claim about dedicated Fixpipe issue throughput.

The four phases execute in source order on one finite `KernelProgram`, each
bound with `RecipeContext.for_phase`. The coarse sequential model does not claim
the native workspace-split overlap, exact chunk boundaries, tile dispatch or
instruction throughput.

## Ledger

For `M` logical rows, `N` outputs, `K` inputs, and `A` active expert groups:

```text
FLOPs (INT4) = 4*M*N*K
HBM  = 3*M*K + 0.5*A*N*K + 12*A*N + 10*M*N + 4*M
MTE2 = 2*M*K + 0.5*A*N*K + 12*A*N + 4*M*N + 4*M
MTE3 = M*K + 2*M*N
vector = 6.5*M*K + 9*M*N
```

Zero-row experts stay in the declared histogram, retain their expert ids and
read no weight bank, encoded column scale or post-dequant assist row. A missing
`int4` compute peak fails with `compute_precision_rate_required:int4`; there is
no INT8 or FP16 fallback.

## Deliberate limitations

- The model is an explicit coarse source-phase ledger, not a reconstruction of
  the native CV pipeline, workspace-split overlapped schedule or exact tiles.
- `SetQuantVector` is charged once as a declared vector-equivalent.
- Matrix publication remains an HBM transfer; Fixpipe service requires a
  declared execution resource.
- The input quantization producer and output BF16 materialization remain
  ordinary graph boundaries; no SwiGLU or output quantization is added.
- A semantic source branch does not establish captured binary identity.

`examples/source-families/w4a8-down-msd-control.json` is a synthetic control
with declared rates, `hidden_size = 64`, `moe_intermediate_size = 64`, explicit
`[3,0,3,2]` routing and both the gate/MSD native-fusion profile and this down
binding enabled. It is a mechanism check, not an accelerator specification or
calibration.
