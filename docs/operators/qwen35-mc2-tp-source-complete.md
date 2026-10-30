# Qwen3.5 MC2 TP complete caller comparison

This conditional composition retains each parent case's hardware, GDN source declaration, W8A8 grouped INT32 caller, TP4/EP4 geometry and 40 layers. It adds the explicit MC2 TP token caller and native per-token INT8 transport, with FP32 row scales. The declared peer window is four, retained from the preceding wrapper control; it admits all three remote peers and is not an independently observed queue-depth calibration.

The full-M router and shared matrices remain unchanged. Selection and MC2 source rows are M/4, and finalize gathers the combined token rows before the final shared/routed add. The c128 case explicitly selects four output TensorMove calls per layer; c4 and c64 retain views. Existing parent reports are preserved.

## Complete Body results

| c | Query rows | Observed mean ms | Predicted ms | Signed gap ms | Relative error | Wall s | Peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 16 | 11.551 | 4.706 | -6.846 | -59.26% | 5.02 | 99.97 |
| 64 | 256 | 21.957 | 11.723 | -10.233 | -46.61% | 13.06 | 103.88 |
| 128 | 512 | 28.585 | 16.622 | -11.963 | -41.85% | 20.95 | 110.38 |

All three computations completed within the 60-second individual budgets, with zero live tasks/programs. None meets the 15% Body criterion. Source correctness and restored calls do not establish sufficient timing accuracy. The observed values are the existing geometry-selected complete Body population, with first-layer waits retained; they are not TPOT or unprofiled step timing. No parameter was fitted to these residuals.

## Structure and producer checks

- Every complete graph has 40 native DispatchV3, 40 CombineV3, 40 source HCCL AllGather and 80 source HCCL AllReduce calls. The latter comprise attention and shared-expert reductions, not an additional routed reduction.
- INT8 payload and FP32 per-row input scales have a single MC2 receiver producer. No duplicate expert-up quantizer is introduced. The INT32 matrix consumes the payload; DequantSwigluQuant consumes the received scale explicitly.
- The expert-down input quantizer remains physically present within DequantSwigluQuant, which produces both INT8 values and FP32 row scales for down GEMM.
- All 80 expert matrix anchors preserve parent FLOPs, vector work and HBM/MTE2/MTE3 bytes. Full router/shared rows and global expert counts remain unchanged.
- c128 has exactly 160 post-gather TensorMove calls. Each copies one local rank slice; c4/c64 have none.
- 24 focused tests for wrapper, typed receiver and per-token INT8 composition passed.

## Predicted component intervals

These are diagnostic sums of caller-visible execution intervals. They are not additive explanations of the Body critical-path residual, and the GDN column includes convolution, gating and recurrence only (not projections, norms or materialization).

| c | Grouped GEMM ms | DequantSwigluQuant ms | MC2 ms | HCCL AR+AG ms | GDN cores ms |
|---:|---:|---:|---:|---:|---:|
| 4 | 1.025 | 0.268 | 0.225 | 0.446 | 0.566 |
| 64 | 2.130 | 0.536 | 0.610 | 1.916 | 3.037 |
| 128 | 2.223 | 0.536 | 1.067 | 3.107 | 4.801 |

The report retains separate observed collective device activity, excluding duplicate collective summaries and unrelated AICPU wrapper entries. Remaining source control/synchronization services, exact binary dispatch and active context lengths are not inferred from the gap. The declared context stays 4096.

Evidence: `data/verification/qwen35-mc2-tp-source-complete-20261030.json`; rerun with its adjacent `.producer.py`. Cases: `examples/cases/high-concurrency-20261030/qwen35-mc2-tp-source-complete-conditional/`.
