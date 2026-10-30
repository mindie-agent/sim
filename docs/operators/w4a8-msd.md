# A8W4 MSD grouped gate/up + SwiGLU + output quant

[Source] `vllm-project/vllm-ascend` revision
`8fd0bc4885cb52e58791aaa8efacc571b9bd115d`. The selected public profile is
`program_profiles: {"grouped_gate_up_activation_quant": "ascend_w4a8_msd"}`.
It is explicit opt-in; no model, architecture or vendor-name inference selects
it.

The source evidence is the Key2 `A8W4_MSD` branch of
`GroupedMatmulSwigluQuantV2`, not the `DT_INT8` NZ 5D fused tiling branch:

| Phase | Source | Modeled work |
| --- | --- | --- |
| pre | `grouped_matmul_swiglu_quant_v2_a8w4_msd_pre.h` | INT8 activation read `M*K`; high `floor(x/16)` (cast, multiply, cast), low halfword mask + cast/add/cast; packed high/low INT4 scratch write `M*K`; vector `6.5*M*K` |
| matrix | `_mid.h`, `_pipeline.h` | double-M INT4 product `4*M*N*K`; packed high/low read `M*K`; `0.5*N*K` packed weights and `8*N` uint64-encoded column scales per **active** group; two FP16 outputs `4*M*N`; column scale application `2*M*N` once |
| post dequant | `_post.h` | FP16 scratch read `4*M*N`; assist `4*N` per active group and per-token scale `4*M` total; cast `2*M*N` plus `MN` high*16, `MN` low add, `MN` assist, `MN` row scale |
| SwiGLU | `_post.h` | `H=N/2` outputs per row; explicit expression ledger `5*M*H` (exp, add, reciprocal, gate multiply, up multiply) |
| output quant | `_post.h`, `CastFp32ToInt8Template` | `Abs M*H`, max `M*(H-1)`, `2*M` scale/reciprocal, `M*H` multiply, two casts `2*M*H`; INT8 output `M*H` and FP32 scales `4*M` |

The phases execute in source order on one finite `KernelProgram`. Each phase is
bound with `RecipeContext.for_phase`, so the enclosing kernel's startup and
completion constraint are not repeated. Pre/post intermediate values never
become additional HBM traffic. The INT4 matrix phase declares precision
`int4`; a case without an explicit `int4` compute peak fails with
`compute_precision_rate_required:int4`. There is no INT8 or FP16 fallback.

The matrix's two FP16 workspaces are published by `GetTensorC`/Fixpipe,
not by the AIV store pipe. Their `4*M*N` HBM bytes remain in all cases;
additional Fixpipe service is charged only when that resource is declared.
Pre scratch and final INT8 output use MTE3. Missing Fixpipe service is
reported as a fidelity limitation, not filled with an invented bandwidth.
When a case declares only split read pipes, matrix operands use its existing
`mte2_aic` and pre/post phases use `mte2_aiv`. A declared generic `mte2`
keeps its existing semantics. No extra capacity or alias resource is created.

## Supported domain

- INT8 activations, packed INT4 per-channel weights, uint64 bit-pattern column
  scales, FP32 assist and per-token scales.
- Gate/up grouped matrix, SwiGLU and dynamic INT8 output quantization as one
  complete selected group.
- No bias beyond the MSD assist matrix, no per-group weight quantization, no
  transposed weights, no SwiGLU clip (`swiglu_limit=0` initial domain).
- Explicit integer `expert_rows` from declared routing or forced balanced
  counts. Probability-moment histograms and fractional rows are rejected.
  Zero-row groups are retained in the histogram and read no weight or scale
  bytes.
- NZ source geometry requires `N` divisible by 64 and `K` divisible by 16.
  Input and output quantization have one FP32 scale per row. Any externally
  consumed intermediate gate/activation value prevents this complete fusion:
  the selected native kernel does not publish that value.

## Configuration

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
              "grouped_gate_up_activation_quant": "ascend_w4a8_msd"
            }
          }
        }
      }
    }
  }
}
```

The fused operator keeps the group key, semantic member ids, external inputs,
external outputs, requests, stage/phase/stream and structure-group evidence.
Its declared work is exactly the phase ledger; member recipe work and the old
traffic-profile savings are not added. Other selected roles retain the normal
`declared_native_fusion` member-program behavior.

`examples/source-families/w4a8-msd-control.json` is a synthetic control with
declared rates and explicit routing. It is a mechanism check, not an
accelerator specification or calibration.

## Deliberate limitations

- The coarse sequential phase model does not claim the source's exact
  workspace-split pipeline, component cache behavior or overlap.
- `SetQuantVector` column dequant is charged once as a declared vector-equivalent
  `2*M*N`; no dedicated Fixpipe rate is invented.
- Kernel construction, instruction counts and the exact native tiling schedule
  are not reconstructed. A semantic source branch does not establish capture
  binary identity.

Independent acceptance exercises ordinary graph and predict entrypoints,
an uneven `[3,0,3,2]` histogram, forced balanced counts, delayed inputs,
live-outs, scale geometry and exact matrix-publication accounting. The
synthetic control's INT4 peak is independently declared and is deliberately
different from INT8; no rate is inferred from a duration.
