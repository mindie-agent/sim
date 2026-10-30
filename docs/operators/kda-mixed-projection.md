# KDA mixed input projections

`runtime.kernels.lowerings.kda_input_projection =
"vllm_ascend_kda_mixed_projection"` selects the retained Kimi caller's two
projection groups. The ordinary semantic projection path remains available
when this implementation is not selected.

The source contract is `vllm-ascend` revision
`8c30e4f491f73cba737a1dacc522894e437cdf3d`, `vllm_ascend/ops/kimi_kda.py`:
`_KDAFusedBFGLinear`, `_run_overlapped_qkv_bfg` and `_postprocess_bfg`.
The retained A3 captures agree with its operand shapes and dtypes; their
compiled binary identity has not been established.

For hidden width 7168 and 96 KDA heads of width 128:

| Group | TP8 local output | TP16 local output | Operands |
| --- | ---: | ---: | --- |
| Q, K, V | 4608 | 2304 | INT8 activation and weights |
| beta, F, output gate | 3084 | 1542 | BF16 activation and weights |

F is composed as `F_b.float() @ F_a.float()` at load time and stored in BF16.
The source retains both factors. Capacity therefore adds the composed matrix
without removing checkpoint factors: 22,020,096 bytes per KDA layer at TP8,
or 11,010,048 bytes at TP16. Online execution uses one hidden-to-F slice in
the BFG GEMM; it does not execute either low-rank factor.

The source starts BFG and input quantization on separate streams. QKV waits
for both. Beta conversion and sigmoid also wait for both first-stage
branches and can overlap QKV. Convolution waits for the QKV and beta join.
Split F and output-gate tensors retain their BFG strides. A multi-row beta
slice is packed before its FP32 conversion; an explicit consumer layout
contract copies only its selected gate payload.

The source fixture retains TP8 batch 2 and TP16 batch 1 ABI rows without
durations. The verification producer reads bounded first KDA blocks from
52 existing long-context captures: 46 A3 TP8/TP16 blocks match this contract,
and six A5 BF16 deployments are excluded. This is a shape census, not an
all-forward ABI or timing acceptance claim. Seven prepared TP16 extraction
references have no complete forward, so those entries explicitly identify
their native-prefix scope.

`data/verification/kimi-mixed-projection-source-control-20261030.json` also
records first-layer Engine controls, resident bytes, work and stream
dependencies. TP8 batch 2 changes from 0.406825 ms to 0.373266 ms; TP16 batch 1
changes from 0.426149 ms to 0.340757 ms. These are analytical before/after
controls with unchanged hardware rates, not comparisons with measured
layer duration. No duration, body multiplier or shape-specific timing table
enters the implementation. The declared quantization scale/offset residency
remains conditional because the native BF16 scale argument alone does not
identify every prepared resident buffer.
