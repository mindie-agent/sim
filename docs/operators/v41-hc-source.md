# V4.1 explicit HC source control

`examples/source-families/v41-hc-source-control.json` selects the existing
`hc_pre_m_k_split_175ef4d4` and `hc_post_d_split_175ef4d4` profiles through the
ordinary `native_fusion` lowering. It preserves the retained 40-layer model,
TP1/DP8/EP8, quantization, hardware rates and balanced routing. No duration
parameter is introduced.

The producer `data/verification/v41-hc-source-20261030.producer.py` constructs
the complete eight-rank input graph, then prices only layer-zero attention/FFN
HcPre/HcPost kernels. It is not a whole-body prediction. Both local batch sizes
start at a delayed 7 ms input; no resource is served early and all tasks finish.

| Local batch | Kernel | Observed mean us | Source prediction us | Signed us |
|---:|---|---:|---:|---:|
| 1 | HcPre | 10.587 | 1.578 | -9.009 |
| 1 | HcPost | 4.074 | 2.998 | -1.076 |
| 8 | HcPre | 11.689 | 1.678 | -10.011 |
| 8 | HcPost | 4.724 | 2.998 | -1.726 |

These are conditional tensor-boundary comparisons to whole-capture mean
activity. The source revision is retained but the capture binary identity is
not established. All four absolute differences are within the declared 15 us
small-kernel tolerance; that does not establish heldout accuracy or justify
fitting a launch floor. In particular the larger HcPre gap remains visible.

HcPre includes BF16-to-FP32 workspace, shared L1 X reads for HF32 projection and
FP32 Gram, coefficient normalization and mix output. HcPost includes the
source D-split AIV ownership, coefficient reads and residual output. Current
A5 hardware declares distinct HF32 and FP32 matrix peaks, so this source path
requires no synthetic compute-rate override.
