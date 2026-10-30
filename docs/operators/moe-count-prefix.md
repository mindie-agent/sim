# INT8 fused expert count-list conversion

`group_list_prefix: vllm_ascend_group_list_prefix` explicitly selects the caller
that converts an INT64 count list to an inclusive prefix list before fused
gate/up + SwiGLU + output quantization. It is used with the
`grouped_gate_up_activation_quant` native fusion role. The down GEMM retains the
original count-list input.

The retained vllm-ascend source at
`8c30e4f491f73cba737a1dacc522894e437cdf3d` calls `convert_group_list_type` in
`quantization/methods/w8a8/w8a8_dynamic.py:408-415`; the conversion in
`ops/fused_moe/moe_utils.py:205-214` is `group_list.cumsum(dim=0)`. The list
contains every local expert, including zero-count experts. The selected local
caller requires the packing producer and weight-bank expert counts to agree.

The separate `expert_group_list_cumsum` graph node reads and writes the INT64
list and makes the fused matrix wait for its result. It is not absorbed into
the matrix's source phases or repeated once per expert. This preserves the
AI_CPU Cumsum boundary observed in the retained A3 captures.

The uncalibrated reference prices the memory traffic and keeps integer-add
service explicitly unpriced. A whole-kernel observable response may supply its
inclusive device duration independently; no AIV throughput, host submission
time or complete-body residual is invented. The caller source revision is not
proof of the captured binary's identity.

Focused verification covers zero-count descriptors, one conversion per layer,
fusion dependencies, unchanged down-list contract, delayed input readiness,
inclusive service selection, and the real graph engine releasing all tasks.
