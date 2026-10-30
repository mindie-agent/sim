# A5 small-M BF16 matrix call services

Three native `MatMulV3` tensor contracts have measured small-M behavior that
exceeds the current source-only schedule. The reusable parameters in
`examples/service-regimes/a5-small-m-bf16-matmul-20261030.json` use physical
hardware, BF16 input/weights, output dtype, N/K, one matrix bank, and local
**M ∈ [1,4]**. Runtime selection contains no model, case, layer or role names.

| N × K | Output | Assigned complete-call service |
|---|---|---:|
| 3584 × 7168 | BF16 | 27.136 µs |
| 7168 × 3584 | BF16 | 22.648 µs |
| 16 × 7168 | FP32 | 7.5055 µs |

Only the c32 p4k native operator medians assign the three values. The response is
an inclusive small-M call plateau: the source program and physical contention
still run; the larger of physical and observed response determines completion.
No Body error enters assignment. The reference response is expressed as an
effective service per weight byte, with declared inverse HBM-rate scaling. This
keeps hardware bandwidth scans active for weight-dominated small matrices.
It is an engineering scaling assumption, not a separately measured HBM pipe
bandwidth or a recovered row slope. The ordinary Cube invocation bound and
source stages still apply; they can dominate when HBM becomes faster.

All 12 c1/c2/c4/c8/c16/c32 × p4k/p16k input geometries were checked through the
ordinary frontend and real isolated Engine. Of **36 native tensor-contract
comparisons, 36 are within 15%**, with maximum absolute relative error **13.95%**
(the M1 latent-down geometry). The p16k c32 errors are 0.00%, −0.0044%, and
−0.259%. The other 11 populations were not used to assign parameters, but retain
historical `unassigned` status rather than being called pristine heldout data.

The metadata contracts are taken from the profiler's complete-Body matrix
partitions, with original capture/cohort identities retained in the receipt.
The rule is explicitly conditional on that native tensor boundary; it does not
claim the analytic tile dispatch reproduces the captured binary.

Three directed controls verify once-only full-call application with late input,
preserved HBM bytes, an M5 analytic-domain fallback, and a severe physical memory
slowdown exceeding the assigned response, and a 0.5×/1×/2× HBM sweep with
unchanged physical work. Parent model inputs remain unchanged.
The rules are a composable parameter artifact; the main task combines them with
MC2/HCCL rules for one full-model replay instead of rerunning each composition
independently.

```sh
python data/verification/a5-small-m-bf16-matmul-regimes-20261030.producer.py \
  data/verification/a5-small-m-bf16-matmul-regimes-20261030.json
python -m pytest -q tests/test_a5_small_m_matrix_regimes.py
```
