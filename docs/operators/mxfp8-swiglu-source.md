# Explicit MXFP8 grouped gate/up, SwiGLU and MX quantization

The optional `ascend_mxfp8_grouped_swiglu_quant` program replaces exactly the
ordinary grouped gate/up, SwiGLU and output-quantization nodes with one native
kernel. The down projection stays independent. It accepts integer expert rows,
E4M3 input/weights/output, E8M0 MX32 scales, NZ weights, no bias, and widths
divisible by 64. It uses the same resource Engine and rates as other programs.

```json
{
  "runtime": {
    "kernels": {
      "lowerings": {
        "native_fusion": {
          "implementation": "declared_native_fusion",
          "options": {
            "roles": ["grouped_gate_up_activation_quant"],
            "program_profiles": {
              "grouped_gate_up_activation_quant": "ascend_mxfp8_grouped_swiglu_quant"
            },
            "program_options": {
              "grouped_gate_up_activation_quant": {"l1_k": 512}
            }
          }
        }
      }
    }
  }
}
```

`l1_k` is an explicit tiling declaration, not a value chosen from measured
latency. `compact: false` retains the expanded finite K-panel control.

The pinned CANN `ops-transformer` revision is
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8`; primary SHA256s are retained in the
program's `SOURCE`. `kernel_gmm_swiglu_mxquant.h:363–413` performs two independent
matrix calls for the gate/up halves, delivering **FP32 directly to two UB
tensors**. It then invokes `BlockEpilogueSwigluQuant`, where lines677–718 split
valid rows as `ceil(M/2)` and `floor(M/2)` across the two fixed AIVs, perform
SwiGLU plus MX quantization, and write FP8 data and E8M0 scales. There is no
intermediate BF16/FP32 HBM store or reread. All three control processors read
the int64 expert group list; scalar latency/cache hits are not inferred.

The source scheduler retains fixed base M (host capped at128), N256, K128 and
the rolling owner cursor across banks. This fused caller does not invoke the
down program's per-bank M rebalance or last-bank tail split. The same source
MX/NZ transfer abstraction supplies payload/scale reads and L0 aligned operand
service. Matmul API internal dispatch beyond the declared K geometry remains
conditional. BF16 direct-output grouped and this FP32-to-AIV path are separate.

Finite per-AIC L1/L0 allocations and the fixed 184,832-byte UB allocation per
AIV (at base M128) live through the kernel. K-panel compression reuses
`compact_refill`; it preserves work and owners with a fluid interior. **Output
tiles are conservatively serial:** the source's PIPE_FIX wait is broadened to
the next complete tile, so possible next-tile L1/Cube work ahead of the previous
AIV epilogue is not represented. Vector work is a source lane inventory, not a
register issue-cycle reconstruction. These limits are included in every plan's
metadata. Neither this program nor tensor compatibility establishes captured
binary/host tiling identity.

The focused suite passed **40 tests in3.54s**, including existing MXFP8-down and
native fusion paths. New controls cover the real complete45-layer frontend
(42 expert calls), exact preservation of the other ordinary nodes, ownership
across zero/uneven banks, both matrix products, local FP32 delivery, unequal
AIV rows, quantized output/scale stores, late inputs, freed buffers, HBM/AIC/AIV
scaling and detailed/compact work conservation. Additional members or externally
visible intermediates cannot disappear inside the selected three-node program.

The initial real c32/q4 operator control predicts0.165203ms with the explicit
L1K512 declaration. This is a source-program result, **not an accuracy pass**.
No multiplier, invocation floor, resource-rate change or measured residual was
introduced, and original production cases are unchanged.


## Retained ordinary-frontend controls

The committed `data/verification/mxfp8-swiglu-source-20261030.json` and its
producer use the original c16/c32/c64 inputs, q4 and context4096, with the explicit
source selection above. Each full frontend contains 42 selected expert calls.
The isolated Engine control starts only after an external input at 7 ms.

| Batch | Fused duration (us) | Expanded tasks | Compact tasks | Compact wall (s) |
|---:|---:|---:|---:|---:|
| 16 | 163.5503 | 65,449 | 8,425 | 0.0994 |
| 32 | 165.2034 | 65,641 | 8,617 | 0.1066 |
| 64 | 168.4841 | 65,641 | 8,617 | 0.1009 |

For these controls, physical service is conserved and expanded/compact duration
agrees to numerical precision. This does not establish equivalence under arbitrary
external contention. The complete, unmodified c32 expert layer has 87 operators,
0.664059 ms simulated span, 12,674 tasks and 0.2037 s wall time; predecessor
outputs are declared ready at zero, so this is a layer control, not a full-body
measurement. All seven executions finish with zero live tasks, programs and
allocations. The serial producer takes 2.436 s wall / 2.394 s CPU and 59.22 MiB
peak RSS on the recorded host; package source hashes match before and after.

Independent review additionally checked six geometries against independent
resource formulas, including unequal/zero banks and fixed AIV ownership, and
confirmed that the remaining 3,463 ordinary frontend nodes survive the 42
fusions with only required dependency redirects. The reviewer accepted runtime
`907d86f0` with the PIPE_FIX serialization limitation above. These controls do not
establish captured group-list values, captured L1 tiling or a measured error pass.
