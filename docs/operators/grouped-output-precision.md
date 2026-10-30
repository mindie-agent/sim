# Grouped output projection precision

`mla.o_a` declares the first grouped output projection independently. It inherits
`mla` when there is no child override. The ordinary matrix operands, activation
quantizer, resident weight term and optional prepared-MX validation all use this
same category. `q_b` and the second output projection retain their own categories.
No precision is selected from a model name or observed duration.

The retained A3 V4 Flash c4 contract identifies `TransposeBatchMatMul` with local
`G=1, M=8, N=1024, K=4096`, BF16 input/weight/output. The former all-W8A8 case
instead declared INT8 operands and 4,202,496 resident weight bytes, including
metadata. An explicit `mla.o_a` BF16 override declares the observed 8,388,608-byte
weight payload and removes only this matrix's activation quantizer. The matrix
was already present in the graph; its measured activity must not be added as a
second operator.

Evidence:

- Profiler retained contract: `acceptance-20261030-a3-borderline/v4-actual/body-operator-v4flash-a3-tp8-c4-new17.actual.json`, SHA256 `39543999f00be6c6371296413e32e7eab528e739ab294eac672b140c55782a5f`.
- Independent caller: `vllm_ascend/attention/dsa_v1.py:1558-1574`, SHA256 `3fb4e94672e6adb931cb2e2a164201000822fe89250fd3bfd3585ba397acf52e`. Its default output branch passes BF16 `wo_a.weight` to `npu_transpose_batchmatmul` without scale; other explicitly selected output branches remain distinct.

These establish an explicit tensor-compatible input, not the captured binary's
precise matrix schedule. No measured timing enters the change. Tests exercise
TP8 and TP4 group axes, parent/child precision precedence, capacity/roundtrip,
prepared-MX incompatibility, and an actual late-input graph with zero live tasks.
The projection, existing mixed-precision and prepared-MX tests total 25 passes.
