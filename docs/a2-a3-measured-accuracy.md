# A2/A3 measured-accuracy round (2026-10-21)

This round built first-class **Ascend A2 (`Ascend910B4`)** predictive support and a
fitted **physical-calibration** layer for A2 and A3 (`Ascend910_9362`), collected new
real measurements on both standing machines, and closed with a predeclared, frozen,
held-out evaluation. **A5 was deliberately not measured**; only its interfaces were
audited (`.workstreams/a5-audit/report.md` in the working checkout).

The round is reported honestly: **both SKU verdicts are `rejected`** against the
predeclared gates. The fitted layer is strong where it exists; the failures are
concentrated in families that had no applied batched fit and in protocol/declaration
mismatches. No measured data is bundled in this repository.

## 1. What was built

### 1.1 A2 as a first-class target

* `sku_profile` clock precedence: caller override → live `npu-smi` rated clock → CANN
  `.ini` declared clock, with the declared value never overwritten and every conflict
  recorded as a discrepancy.
* Evidence-corrected catalog: `Ascend910_9362` = 20 AIC / 40 AIV per die and 168 MiB
  L2 (live `.ini` + `torch_npu`); `Ascend910B4` peak derived from the observed
  1650 MHz rated clock, not the `.ini` 1500 MHz.
* The fabricated all-family sustained-clock band (1250–1300 MHz) is gone; an
  unmeasured band stays empty. A2's observed sustained band is 1600–1650 MHz.
* `infer_arch_family` no longer maps CANN 9.1's `NpuArch=2201` (also used by A2)
  to A3 by itself.
* `MachineSpec` fixtures derive from `DeviceCapabilityProfile` + an `npu-smi`
  snapshot; the old A3 literal fixture builder (arch, 2 dies, 64 GiB, 1.2 TB/s,
  196 Gbps) is deleted.
* Links come from `comm.default_transfer_channels`: A2 has only `d2d_intra_node`
  (HCCS mesh, no SIO); A3 keeps SIO pairs + HCCS_SW. Mesh vs switched semantics are
  declared, and an op resolves its own fabric scope instead of `links[0]`.

### 1.2 Physical calibration layer (`mindie_sim/a3_physical_calibration.py`)

A versioned, provenance-bearing artifact fitted only on **train** observations:

* `t = launch + max(compute, memory)` with a wave-quantization term using the
  machine's own AIC count, fitted block tiles, an optional two-level
  (HBM/L2) memory blend, and fitted `n`-alignment / `m`-tail penalties where the
  train-only CV supports them.
* Per `(hardware_key, family, recipe, protocol, timing_scope)` parameters; timing
  scopes are never merged (A2 batched and per-call launch intercepts differ ~9×).
* Every artifact carries sample counts, train/CV WAPE, per-band breakdown,
  candidate-model comparison, source runs, package trees and a heldout-overlap guard.
* `PredictiveCostModel` consumes it through `physical_calibration`; the CLI exposes
  `predict --physical-calibration PATH --timing-scope SCOPE`. A declared family
  transfer map bridges model-graph recipes to measured runner recipes (projections →
  `gemm`, `rmsnorm`, `attention_softmax` → `softmax`, `add`, `copy`, `swiglu`,
  `quantize`); attention QK/PV, embedding and sampling are never mapped onto GEMM.

### 1.3 Structural corrections in the analytic path

* Wave quantization uses `die.aic_count` (24 only as a labelled fallback).
* Launch is **additive**: `estimated_ms = launch + max(service)`; the old 20 µs
  max-composed floor that produced 247 launch-floor predictions is removed.
* Invented multiplicative floors (`k/(k+128)`, `max(0.02)`, vector
  `0.5+0.5·ops/4096`, memory `0.65+0.35·B/2MiB`) are deleted; demands are raw work
  over resource ceilings.
* Attention PV contracts `kv` (`(q,dim,kv)`), not `dim` like QK.
* Exact `machine_key` joins; an unbound observation can no longer calibrate every
  machine (A3 residuals cannot leak to A2).

### 1.4 Curve and acceptance tooling

* Curves key hardware series + full protocol identity; A3 curves never answer A2
  queries; manifest split is authoritative (a result row cannot promote itself into
  train); Huber/3-point segments, monotone publication, local transition resolution
  and guarded 2-D coverage.
* `tools/accuracy/`: unified observation dataset, geometry-only prediction freeze,
  single-pass scorer (WAPE/coverage/APE quantiles/bootstrap/per-family/per-regime),
  heldout-order generator with a pre-freeze guard. Round-1 reproduction:
  analytic 22.5072 %, calibrated 14.5274 %, primary curve-first 13.4850 %.

## 2. Measurement campaigns (real hardware, train/heldout predeclared)

| machine | train measured | blinded test-set-2 heldout | notes |
| --- | ---: | ---: | --- |
| A2 `173.125.1.2` (8×910B4, NPU 7) | **344 conditions / 30,270 raw samples** | **66 / 5,940** | GEMM 195, rmsnorm 34, swiglu 14, add 14, copy 21, softmax 22, quantize 16, attention 23 |
| A3 `192.168.13.165` (Phy-ID 1, package 0 die 1) | **212 conditions / 12,720 samples** + 45 attention rows | **80 / 4,800** | GEMM 169 across all 18 gap (N,K) families, native brackets, softmax/copy/rmsnorm/quant, attention/RoPE |
| A2+A3 HCCS/SIO collectives | 7 completed runs / 5 placements | — | A2 HCCS α≈57 µs/20.2 GB/s; A3 SIO α≈71 µs/115.6 GB/s; AICPU (A3) vs FFTS+ AI-core (A2) recorded, never relabelled |

Protocol: 3 s continuous pre-soak per trial, warmup 10, 30 event-timed samples,
3 trials (A2) / 2 trials (A3 gap-fill), batched amortized timing for sub-2 ms kernels
and per-call timing for long kernels, in-window `npu-smi` clock attestation, frozen
runtime package hashed per run. Failures are retained, not deleted.

Key physical facts measured (not assumed):

* A2 sustained clock under load 1600–1650 MHz; launch overhead ≈60–75 µs per call vs
  ≈6 µs amortized batched; 4096³ bf16 0.586 ms = 234 TFLOP/s; 512 MiB copy
  633.6 GB/s; warm/flushed ratio 0.94–1.02 (no timing-relevant L2 residency effect).
* A3 holds 1650 MHz under load; 205.7 TFLOP/s soak; per-call dispatch ≈24.5 µs;
  warm calls L2-served up to B≈128 MiB (2.1–4.5× faster than flushed), consistent
  with 168 MiB L2.

## 3. Fitted accuracy on train (train-only CV, no heldout)

| artifact | family | scope | train WAPE | CV WAPE |
| --- | --- | --- | ---: | ---: |
| A2 | gemm | batched | 9.63 % | **9.71 %** |
| A2 | add / rmsnorm / swiglu / softmax / quantize | batched | 2.3–4.7 % | **2.4–4.0 %** |
| A2 | copy | batched | 21.03 % | 21.12 % (low confidence) |
| A3 | gemm | per-call | 14.95 % | **16.05 %** |
| A3 | gemm | batched | 13.44 % | **16.10 %** |
| A3 | rmsnorm / softmax | batched | 6.0 / 4.1 % | **6.1 / 7.0 %** |
| A3 | copy / add | per-call/batched | 35.0 / 17.6 % | low confidence |

`wave_max` beat flat-additive on every family; the `n`-alignment penalty was adopted
on A3 and the `m`-tail on A2; a shared-efficiency/per-scope-launch variant was
rejected by CV.

## 4. Frozen held-out result (test set 2)

Predeclared before measurement; predictions frozen before any outcome was read;
scored once against the final dataset, with a value-only identity check
(0 value differences) between the pre-measurement freeze and the post-measurement
rebound. The scorer never calls a predictor.

| metric | A2 (`Ascend910B4`) | A3 (`Ascend910_9362`) |
| --- | ---: | ---: |
| declared / numeric | 66 / **63** | 80 / **73** |
| coverage | **0.9545** | **0.9125** |
| WAPE | **22.80 %** (95 % CI 11.25–39.02) | **45.72 %** (33.00–60.59) |
| median APE | 14.46 % | 39.28 % |
| p90 APE | 182.61 % | 174.23 % |
| primary path | physical 46 / analytic 20 | curve 10 / physical 28 / analytic 42 |
| protocol exclusions | 3 (long-kernel per-call branch) | 7 (5 GEMM + 2 softmax) |

Per-family WAPE:

| family | A2 | A3 |
| --- | ---: | ---: |
| gemm | **4.78 %** | **16.87 %** |
| rmsnorm | 16.71 % | 24.14 % |
| swiglu | 13.13 % | 32.19 % |
| copy | 45.74 % | 70.80 % |
| attention_softmax | 58.74 % | 53.89 % |
| add | 104.69 % | 141.92 % |

Bands — A2: launch 17.91 % (n=34), mid 35.27 % (26), large 3.65 % (3, insufficient n).
A3: launch 43.81 % (46), mid 46.16 % (27), large n=0 (all large conditions fell in the
per-call protocol exclusions).

**Combined: 29.67 % duration-weighted WAPE over 136/146 conditions, coverage 0.9315.**

### Verdict

Both SKUs are **rejected** against the predeclared gates.

* **A2**: G1 coverage passes; G2/G3 fail on the mid band (35.3 % vs 12 %), G4 on the
  tail, G5/G6 on copy and add, G7 on paired improvement, G12 on softmax numeric count.
  The error is concentrated in 20 analytic-primary conditions (copy + fp32 softmax)
  that carry 57.5 % of the absolute error, plus a physical add over-prediction on
  L2-resident large-element adds.
* **A3**: G1–G7 and G12 fail. 42 conditions are analytic-primary (add, copy, swiglu,
  attention QK, RoPE, quantize) because no applied batched fit exists for them; the
  analytic launch floor over-predicts small batched kernels 2–3×. 7 long-kernel rows
  are protocol exclusions because the declaration itself predeclares per-call timing.

Where the fitted layer applies it is the strongest component: A2 GEMM 4.78 %,
A2 rmsnorm 16.71 %, A2 swiglu 13.13 %, A3 GEMM 16.87 %, A3 rmsnorm 24.14 %.

## 5. Highest-leverage next round

1. Fit/apply batched-scope physical rows for A3 `add`, `swiglu`, `attention_qk`,
   `rope`, `quantize`, and make `copy` non-low-confidence (or replace it with a
   memory-bound model). These are the 42 analytic-primary conditions.
2. A2: apply the copy fit or replace it with a copy-bandwidth model; fit fp32
   softmax; add an L2-residency term so large-element `add` is not over-predicted.
3. Fix the declaration/attestation mismatches (A3 rmsnorm `preallocated`, test-2
   copy dtype, long-kernel per-call scope) and predeclare the per-scope freeze.
4. Measure the round-2 repeat/noise floor and the paired cross-SKU transfer subset
   so G8/G10/G11 become evaluable.
5. A5 remains deferred; the A5 audit's seven code blockers must be fixed before its
   first measurement (fp8 peaks/bytes, UB_MESH placement, exact-soc reason,
   acceptance-schema/assembler single-SKU assumptions).

## 6. Evidence locations (not bundled)

* A2 train/heldout data: `/Volumes/data/mindie-sim-data/a2/2026-10-21/`
* A3 round-2 data: `/Volumes/data/mindie-sim-data/a3/2026-10-20/accuracy-round-20261021/`
* Fitted artifacts: `.workstreams/model-physical/{a3,a2-910b4}-physical-calibration.json`
* Curve refit: `/Volumes/data/mindie-sim-data/a3/2026-10-20/curve-refit-20261021-v2/`
* Freezes: `.workstreams/acceptance/freeze/{a2,a3}-test2*/`
* Scores: `.workstreams/acceptance/score/{a2,a3}-test2/`
* Datasets: `.workstreams/dataset-{a2,a3}-final/`
* Comm report: `.workstreams/comm-campaign/report.md`
* A5 audit: `.workstreams/a5-audit/report.md`

The `.workstreams/` tree is working evidence and is git-ignored; the measured raw data
lives outside the repository. SHA-256 receipts bind every freeze to its predictions,
dataset, declarations and source commit.
