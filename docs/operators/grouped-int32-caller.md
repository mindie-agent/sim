# Nonfused INT8 grouped caller

The explicit `grouped_int8_caller: ascend_grouped_int32_dequant_swiglu`
lowering connects the existing nonfused W8A8 path to its real accumulator
boundary: INT8 grouped GEMM writes INT32; DequantSwigluQuant reads that tensor,
the original per-token FP32 scale and per-expert FP32 channel scales, then
produces INT8 payload and FP32 row scales for grouped down. Matrix epilogue
work and the former BF16 activation intermediate are removed. The down matrix
keeps its ordinary output epilogue. Checkpoint storage is unchanged.

This is selected by source semantics and explicit integer expert rows, not a
model name. The source is pinned vllm-ascend `175ef4d4`; the working copy of
`moe_mlp.py` has unrelated conflict edits, so the retained Git object was read.
The source caller at lines326–361 passes no scale input to the INT32 GMM and
passes group-index values to the activation. Independent native CSV samples
also show INT32 `[M,1024]` followed by FP32 `[64,1024]` channel scales,
FP32 `[M]` token scales and INT64 `[64]` groups. Captured allocated M is not
evidence that every row is active. These new cases keep forced-balanced active
rows and leave captured group-list contents unknown.

The consumer uses analytical phases per physical AIV. Each group repartitions
its active rows, each participating owner loads the expert's channel scale,
and all owners traverse the group metadata. Input and output resource service
is pinned to AIV pipes. Logical dequantization, normal SwiGLU and dynamic INT8
work is retained. This coarse program does not reconstruct the deployed host
tiler, finite UB overlap, compiler instruction expansion or scalar descriptor
latency; it introduces no measured floor. GMM timing still uses the selected
ordinary matrix schedule.

The transform preserves the original scale producer and rejects a chain with
another live consumer rather than deleting its output. Tests cover unequal
integer groups, inactive experts, unchanged nonmember operators, renamed
operator identities, repeated lowering, late inputs and hardware-rate changes.
The producer in `data/verification/grouped-int32-caller-20261030.producer.py`
records three full-TP/EP one-layer controls. It does not assert a full-model
accuracy result. The selected caller currently retains the ordinary receiver
quantization producer; quantized native MC2 is a separate caller boundary.
