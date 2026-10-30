# Prepared MX matrix representation and AIC output

`ascend_prepared_mx32_matrix` is an explicit ordinary-frontend selection for
`q_b`, `grouped_output_a`, and `grouped_output_b`. It replaces their runtime
weight-scale view with one E8M0 byte per 32 K elements, and publishes BF16
outputs through AIC Fixpipe to GM. It retains the shared finite matrix owner
and K-panel pipeline. No resource rate, invocation floor or measured residual
is an input.

```json
{
  "runtime": {
    "kernels": {
      "lowerings": {
        "mx_matrix": {
          "implementation": "ascend_prepared_mx32_matrix",
          "options": {
            "projection_recipes": ["q_b", "grouped_output_a", "grouped_output_b"],
            "scale_storage": "replace_checkpoint_scales"
          }
        }
      }
    }
  }
}
```

The domain is bias-free E4M3 operands, E8M0 activation scales with group 32,
BF16 output, K divisible by 64 and N divisible by 32. An already source-bound
matrix or another matrix epilogue cannot be silently replaced. Activation
quantization remains a separate producer. Grouped output permutations remain
native tensor strides; this binding does not invent a GM transpose copy.

## Checkpoint and runtime residency

The V4.1 input declares checkpoint scales on a 32 × 32 grid. Captured matrix
ABIs instead contain runtime scales such as `[32768,20,2]` for a
`[32768,1280]` weight and `[8,64,1024,2]` for eight `[4096,1024]`
matrices. These are different storage contracts. `WeightTerm.runtime_quant`
changes only selected post-load resident terms; `source=True` keeps the
checkpoint contract. Removing the selector also removes the derived runtime
view. Parsing and serializing a case preserves this distinction.

`scale_storage=replace_checkpoint_scales` explicitly declares that the old
scale tensor is replaced after preparation. Preparation time is outside the
body boundary. The data payload does not change. The selector does not add a
second model-weight allocation or change the global catalog quantization.

The retained `AscendW8A8MXFP8DSDynamicLinearMethod` replaces parameter `.data`,
but its scale-expansion implementation hardcodes the 128 × 128 checkpoint
case. It does **not** establish how the captured 32 × 32 checkpoint binary
prepared its scales. Therefore the replacement policy is a conditional runtime
declaration, not a claim that this retained preparation function is the captured
binary. This uncertainty is evidence metadata, not missing modeled structure.

## Source boundaries and limitations

- `w8a8_mxfp8.py:493–555`, SHA256
  `9f01afd72f299e3617d4e89059840102fd7df93358f82f372daede7a723aaa16`:
  prepared weight/scale parameter replacement and grouped output weight view.
- `dsa_v1.py:1464–1487`, SHA256
  `3fb4e94672e6adb931cb2e2a164201000822fe89250fd3bfd3585ba397acf52e`:
  separate dynamic MX quantization, strided grouped matrix, BF16 output.
- `quant_batch_matmul_mx.py:1560–1627`, SHA256
  `37c9d7f034ae249d2d8fcfe5b310ec6197a6df711736e2ab04ae7e2cdb3ed74e`:
  independent MX matrix source loads scales into L1/L0 and uses Fixpipe from
  L0C to GM. It supplies the physical output boundary, not captured dispatch.

The matrix mapping is still the common finite analytical mapping. Scale loads
share the operand-load ledger; separate MX scale issue instructions and ND
stride transaction granules are not modeled. No captured host tile selection
or active-core count is inferred from latency. The selected program is not
advertised as an exact native planned kernel.

## Verification

`tests/test_prepared_mx_matrix.py` builds complete 40-layer ordinary graphs at
B1 and B16. Only the three selected matrices per layer change; all other
operators, dependencies and FLOPs remain identical. Six actual Engine controls
check late inputs, weight/scale traffic, exclusive Fixpipe output and cleanup.
The resource scans vary HBM, MTE1, MTE2_AIC, Fixpipe and Cube independently and
retain legitimate plateaus; dedicated bottleneck controls respond strictly.

`data/verification/v41-prepared-mx-matrix-20261030.json` retains these controls
and checkpoint/runtime bytes. Its paired durations measure this mechanism
change only. They are not a full-body accuracy score or captured host-dispatch
validation. In particular the previously reported 18.9–21.1 us TQBM bucket is
`grouped_output_a`, not `q_absorb`; V4.1 uses the latent-MQA path.

The six paired controls completed in 1.015 s, with 62.8 MiB peak RSS on the
recorded macOS process. At B1 the three predictions changed from
10.513 / 8.411 / 16.049 us to 10.830 / 8.665 / 16.532 us. At B16 they changed
from 11.131 / 8.638 / 17.075 us to 11.499 / 8.881 / 17.558 us. Correcting
scales changes packed weight traffic by about 3%; this does not explain the
entire observed TQBM residual. Further dispatch/transaction behavior remains
unproven, so no extra latency is added to close it.
