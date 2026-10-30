# Allocated versus valid rows in arch35 grouped dequant/SwiGLU

The existing `grouped_int8_caller` keeps the separate INT32 `GroupedMatmul` and `DequantSwigluQuant` calls. Its new optional program selects the retained arch35 full-load DSQ kernel:

```json
{"implementation":"ascend_grouped_int32_dequant_swiglu",
 "options":{"dequant_program":"arch35_full_load","input_capacity_rows":2048}}
```

The option changes only DSQ execution. Without it, the previous analytical program remains the control. The down-projection's existing matrix epilogue remains; this change does not add another dequantization to it. There is one DSQ invocation, not an invocation per group, tile, vector scope or physical owner.

`input_capacity_rows` describes the physical input/output allocation, not valid work. The native tensor ABI uses that capacity, while `OpSpec.local_shape`, scale-producer validity and all tensor reads/writes follow the explicit integer expert counts. No capacity row is turned into an expert assignment. The current MC2 receiver metadata explicitly describes valid rows and is not used to guess allocation capacity.

## Source boundary

The pinned CANN `ops-nn` revision is `955f8b33985b6d72989ebff341bb9c2c564a51bc`:

- [Host tiling](https://gitcode.com/cann/ops-nn/blob/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling_arch35.cpp), lines 756–768 and 939–1022; SHA-256 `0ff6b530a54192ff335c38abe10c019abc32ae8b96560be981de56b74738e278`.
- [Full-load kernel](https://gitcode.com/cann/ops-nn/blob/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_swiglu_quant/op_kernel/arch35/dequant_swiglu_quant.h), lines 196–223, 240–843; SHA-256 `3bf3fedad839383cee0e644dd3e07c0313e7abe9246092eb4ba3b1c3e997add0`.

The scope is INT32 input, FP32 column and row scales, one-dimensional INT64 group counts, no bias/smooth scale, normal SwiGLU, and INT8 output with FP32 row scales; output width is at most 5120 and must fit the declared UB. This is an explicit source choice, not identification of the captured packaged binary. The older vLLM source snapshot references the arch35 header from its entry file but does not include that directory; its generic base kernel is not substituted for arch35.

The host uses `G>=32 && capacity/G<=16` to select group-wise ownership. In that branch, each group is assigned to its source AIV. Otherwise every group reuses the same contiguous row partition beginning at owner zero. Block count also depends on allocation capacity and finite UB tile rows. Q35's retained DSQ tensor shapes are capacity 128/2048/4096 with 64 groups, while the declared balanced source-complete inputs contain 32/512/1024 valid rows. These are different facts. Accordingly c4 uses the special branch; c64/c128 reuse 8/16 active owners across all 64 groups.

The pinned special branch continues on a zero group before advancing its owner cursor. An earlier zero followed by a positive group at that same owner's next cursor would drop work. This source domain hole is rejected explicitly, while the analytical program remains available; the planner never silently discards that later bank.

## Physical execution

Each launched AIV reserves the host-derived UB footprint: two input/row-scale buffers, two INT8 output buffers, one FP32 scale buffer, one FP32 intermediate, and the column-scale storage. The 1 KiB host reserve does not become extra tensor traffic. The common Engine executes a lazy two-tile window per owner:

1. Read that group's column scales once, then each tile's valid INT32 input and FP32 row scales.
2. Execute dequantization and SiLU/gate, then row reduction.
3. Publish FP32 scales while the final INT8 quantization can execute.
4. Publish INT8 output on the same ordered MTE3 pipe. The next reduction/compute must respect the single scale slot; a new group's column scales cannot replace the previous group's still-consumed scales.

The source register inventory preserves instruction calls, active lanes and UB payloads. It uses the retained arch3510 issue, UB port, ALU, special and reduction capacities already in the hardware declaration. INT32→FP32 and FP32→INT16 use the documented 32-lane throughput, distinct from 64-lane FP32 arithmetic; this is not an empirical efficiency multiplier. The generic vector arithmetic envelope is replaced, not added again. UB stores are separate from GM/MTE3 writes.

The scalar `GetValue` descriptor bytes and call count are retained, but scalar control issue latency is not inferred from an unrelated scalar-store rate. Descriptor reads are a payload lower-bound stage, not a cycle-level model of each scalar branch. The retained source's `inScaleQueue.FreeTensor` appears inside the UB loop despite allocation outside it; multi-tile execution conservatively retains those column scales. This queue-lifetime approximation is explicit. Input slots are retained through a tile's completion in the bounded common pipeline, rather than reconstructing compiler issue bundles.

## Evidence and remaining gap

`data/verification/grouped-dequant-arch35-20261030.json` contains three real ordinary-frontend source controls, five independent resource scans and retained selected-forward PMU summaries. The controls execute the complete caller transformation, then time only DSQ; they do not rerun or claim whole-body accuracy.

Initial single-call predictions are 1.312 / 12.846 / 12.846 µs for c4/c64/c128, versus retained observations 7.299 / 54.031 / 56.560 µs. The high-batch gap remains. PMU AIV activity is approximately 53.76 / 56.29 µs, so treating the entire gap as host submission would be unsupported. Scalar, vector and DMA PMU activities overlap and must not be added. No measured PMU time, per-group constant or residual floor is inserted into the program.

Independent controls check all valid-bank counts, source branch thresholds, irregular tails, input readiness, no live buffers, directional resource sensitivity, and a hand-derived finite pipeline: one group's three one-row tiles finish in `8 + 3*12 + 5 = 49 ms` under explicit 1 B/ms DMA rates; three separate groups require `3*(8+12)+5 = 65 ms`. This tests column-scale reuse and overlap without using hardware observations.
