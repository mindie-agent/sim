# Ordinary integer matrix output

Per-row dynamic W8A8 matrices now retain their output work in every matrix
recipe. The INT32 result is cast to FP32, multiplied by row and column scales,
and cast to a BF16/FP16 output. FP32 output omits the last cast. This gives four
or three logical vector operations per output, after the complete K reduction.
Scale transfers occur at that boundary, rather than repeating for every K
panel. Fractional routing expectations scale row metadata and output work
without inventing an integer routing realization.

The caller contract was inspected in vllm-ascend
`8fd0bc4885cb52e58791aaa8efacc571b9bd115d`,
`vllm_ascend/quantization/methods/w8a8_dynamic.py`, around line 124:
`npu_quant_matmul` receives weight scales and per-token scales. The retained
source SHA-256 is
`a67601ee596d9aa4ae72ea3638f67ff7236cd8df884564bcd19f33653a1e0910`.
The existing QuantSpec scale-and-offset storage traffic remains declared;
storage of an offset does not establish offset arithmetic in this caller.
Source-bound programs replace generic work with their own physical work and
do not inherit another copy of this epilogue.

The default finite mapping domain now adds M=16/32/64, N=16/32, K=64
candidates when N <= 32. Inner matrix tiles fit those extents, exposing
independent work to more owners for narrow projections. Explicit candidate
lists remain exact. This is a declared mapping search, not evidence of the
deployed compiler's dispatch.

Independent acceptance used 60 actual Engine executions across five recipes,
BF16/FP32 output, group counts and tail geometries. FLOPs, output work, scale
traffic, input readiness and final liveness were checked independently.
Ordinary probability-MoE operators, graph and predict paths also completed for
all five recipes with fractional expected rows. These are mechanism checks,
not measured kernel accuracy.

W4A8 grouped correction-bias work remains outside this change. A native
comparison requiring that work must expose the missing epilogue and retain its
reference-only status instead of treating the bare matrix as the complete call.
