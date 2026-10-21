# Synthetic operator-plan examples

`mixed.json` is a model-independent Vector -> BF16 GEMM -> Vector chain.
All machine and profile rates are synthetic arithmetic fixtures. It does not
describe an Ascend SKU, a measured kernel or an end-to-end request.

Run from the repository root:

```bash
python -m mindie_sim operator-plan --input examples/operator-plans/mixed.json --output /tmp/operator-plans-mixed.json
```

The JSON declares a one-device machine with two compute cores. The first
Vector operator transforms the GEMM's 2,560-element A tensor; the last transforms
the 1,536-element output. Each Vector instruction dependency is explicit,
including BF16/FP32 casts. The GEMM has `(M,N,K) = (32,48,80)`, tile sizes
`(16,16,32)` and an L1 K-block of 64.

The expected selected plans have 30, 54 and 18 stages, respectively. With the
declared 1 us device launch per operator, isolated times are 11.752 us,
37.1472 us and 6.120 us. As the three operators form a serial dependency
chain, their complete device timeline is 55.0192 us. These values are regression
references for this input, not latency predictions for a real NPU.

Several quantities can be checked without running the estimator:

* Logical GEMM work is `2*32*48*80 = 245,760` FLOPs.
* The three K tiles issue `3*32 = 96` K lanes, including the final 16 valid
  lanes in a full 32-wide command. Issued work is therefore
  `2*32*48*96 = 294,912` FLOPs. The difference records the declared tail policy.
* Six output tiles each read `2*(16+16)*80 = 5,120` input bytes. In the explicit
  no-cache-credit scenario this is 30,720 HBM read bytes. Output is
  `2*32*48 = 3,072` bytes, giving 33,792 HBM bytes in the selected GEMM plan.
  The baseline's 15,872 bytes represent a different aggregate traffic model;
  those bytes are not added to the selected plan.
* L1 requires `2*(16+16)*64*2 = 8,192` bytes. L0A and L0B each require
  `2*16*32*2 = 2,048` bytes. Two FP32 accumulators require
  `2*16*16*4 = 2,048` L0C bytes. Each matches its declared capacity.

The input enables strict coverage so a refused operator fails visibly. Remove
strict mode only when deliberately exploring mixed plan/legacy fallback.
See [the execution-plan guide](../../docs/operator-execution-plans.md) for the
accounting, provenance and validation boundaries.
