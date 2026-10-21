# A5 physical-calibration readiness (A5 hardware available, measurements still absent)

Status: the **calibration layer is code-ready for A5 and data-blocked**. Four A5 machine
blocks exist and load, the fp8 byte-width regression is fixed, and no A5 artifact is
published because A5 has **zero timed measurements** today. This note records the evidence,
the fixes, the remaining blockers (with owners) and the minimum measurement set.

## 1. A5 machine blocks (declared, not measured)

| file (`.workstreams/model-physical/`) | SKU | dtype | AIC/AIV | clock | peak_flops | HBM peak | L2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `machine-a5-950dt-9581-bf16.json` | Ascend950DT_9581 | bf16 | 32/64 | 1650 MHz | 4.325 376e14 | 4.0 TB/s | 128 MiB |
| `machine-a5-950dt-9581-fp8.json` | Ascend950DT_9581 | fp8 | 32/64 | 1650 MHz | **8.650 752e14** | 4.0 TB/s | 128 MiB |
| `machine-a5-950pr-9579-bf16.json` | Ascend950PR_9579 | bf16 | 32/64 | 1650 MHz | 4.325 376e14 | 1.6 TB/s | 128 MiB |
| `machine-a5-950pr-9579-fp8.json` | Ascend950PR_9579 | fp8 | 32/64 | 1650 MHz | **8.650 752e14** | 1.6 TB/s | 128 MiB |

All four load through `MachineLimits.from_mapping` (the loader re-derives
`aic_count x ops_per_cycle x sustained_clock` and rejects a contradicting declared peak).
Sources per field are in `analysis/a5-machine-blocks.md`; every block carries an explicit
`measured: false` evidence note. `Ascend950DT_9582` (96 GiB) uses the same 9581 block except
the HBM capacity, which is **not** a `MachineLimits` field, so it does not enter a fit.

One artifact = one compute dtype: A5 needs a **bf16 artifact and an fp8 artifact**, because
`ops_per_cycle` differs (8192 vs 16384) and the fitter rejects a dtype outside the fit.

## 2. Fix landed: declared dtype byte widths (this layer)

`DTYPE_BYTES` knew only bf16/fp16/fp32 and eight byte sites used
`DTYPE_BYTES.get(dtype, 2.0)`. An A5 fp8 GEMM (`m=4096, n=14336, k=7168`) was therefore
priced at **32.9 GB instead of 16.4 GB** in both fitting and prediction, with the fitted
`eta_memory` absorbing the 2× error invisibly. `a3_types.py:54-63` already declared
`fp8/fp8_e4m3/fp8_e5m2/int8 = 1.0`, `int4 = 0.5` — the calibration layer was a regression
against its own package.

Now: the table mirrors `a3_types._DTYPE_BYTES` (a unit test asserts equality; the module stays
stdlib-only), `dtype_itemsize()` raises
`dtype_byte_width_undeclared:<dtype>` instead of guessing, the four observation readers count
undeclared dtypes in `skipped["dtype"]`, `Observation.from_mapping` rejects them (counted by
the fitter), and a hand-built artifact with an undeclared fitted dtype is refused at
prediction. Published A2/A3 artifacts are unchanged (all rows are bf16).

## 3. Evidence inventory: A5 timed measurement = 0

| evidence class | what exists |
| --- | --- |
| data roots | `/Volumes/data/mindie-sim-data/` has only `a2/`, `a3/`; no `a5/` |
| A5 `npu-smi` capture | none (`tests/fixtures/mindie_sim/` has A2/A3 five-file sets only) |
| A5 platform config | `Ascend950DT_9581/9582.ini` copied from the **A3 host's** CANN install — "A5 reference configuration, not a live A5 board" (`tests/fixtures/mindie_sim/README.md:10`); no `9579.ini` |
| roofline constants | `A5_REFERENCE_PEAKS` (fp8 864e12, bf16 432e12) labelled "generic cannsim roofline, not a board-certified peak" |
| workstream | `.workstreams/a5-audit/report.md` (read-only, in-memory probes only) |

A5 therefore has **declared capability**, not measurements. The fitter needs per-family train
conditions plus a launch measurement per timing scope; a launch is never transferred across
SKUs, so no A5 artifact can be published from A3 data.

**Minimum set** (details and the ranked 40-condition table in
`analysis/a5-measurement-plan.md`): one `npu-smi`/clock attestation, one HBM-peak measurement,
one per-scope launch/dispatch measurement, then ≥6 train conditions per
`(family, recipe, protocol, timing_scope)` for `gemm`, `attention_fused` (GQA + MLA,
prefill + decode), `moe_gate_up`, `moe_down` and `grouped_matmul`, at bf16 and fp8.

## 4. Remaining A5 blockers outside this layer (verified at HEAD, ranked)

| id | file:line | consequence | owner layer |
| --- | --- | --- | --- |
| **B-1/P5** | `a3_types.py:529-540`, `:638-683` | **silent**: `peak_flops_for("fp8_e4m3")` returns the bf16 peak, so an A5 fp8 5120³ GEMM is priced **0.6406 ms instead of 0.3303 ms** | `a3_types` |
| **P6** | `a3_types.py:505-515`, `:563-635` | `compute_peaks` never comes from `DeviceCapabilityProfile.cube_spec`, so `{"sku": "Ascend950DT_9582"}` yields `{}` | `a3_types` |
| **B-4** | `a3_types.py:686-700` | two vector-peak derivations differ by exactly 2× (`2.703 36e13` vs `5.406 72e13`), so A5 vector families depend on which machine builder ran | `a3_types` |
| **P15** | `a3_operator_campaign_conditions.py:170-187` | `DEFAULT_MACHINE` is an A3 literal with `fp8 == bf16`; a campaign declaring only an A5 SKU silently gets an A3 machine under an A5 key | campaign |
| **B-5** | `sku_profile.py:620-664` | an exact-but-unregistered SKU is reported as `ambiguous_sku` with a wrong reason (no wrong number reaches a cost model) | catalog |
| **D2/D10** | `.workstreams/accuracy-protocol/acceptance-schema.json:85-96`, `tools/accuracy/score_frozen.py:66-74` | the acceptance schema cannot express an fp8 peak (bf16-only block, defaults to `1.0`) and the frozen limitation list is A2/A3-only | acceptance tooling |
| **B-2** | `a3_cost.py`, `a3_communication.py` | **FIXED**: an A5 UB-Mesh collective records `uncalibrated_comm_placement` instead of borrowing an HCCS curve | — |

## 5. What is modeled / unmodeled for A5 today

* **Modeled**: none by measurement. The families and formulas are hardware-agnostic and will
  fit A5 unchanged once observations exist (`MODEL_KINDS` already covers attention and grouped
  matmul; `bf16` and `fp8` byte widths are declared).
* **Unmodeled, with reasons**: everything, because there is no A5 train observation. An A5
  prediction must stay on the analytic/curve path or return `unmodeled_family` — it must never
  borrow an A3 row (the artifact's hardware key is a hard filter).

## 6. Reproduce / publish (once A5 data lands)

```bash
# bf16 artifact
python3 -m mindie_sim.a3_physical_calibration fit \
  --observations-campaign <a5-run>/results.json \
  --machine .workstreams/model-physical/machine-a5-950dt-9581-bf16.json \
  --hardware-key Ascend950DT_9581 --arch-family A5 --compute-dtype bf16 \
  --heldout-guard <a5-split-declaration> \
  --output .workstreams/model-physical/a5-950dt-9581-bf16-physical-calibration.json

# fp8 artifact: same rows with --compute-dtype fp8_e4m3 and the -fp8 machine block.
```

Guards to keep: train-only rows, the declared heldout guard asserted disjoint, one artifact per
compute dtype, and the launch fitted from A5's own dispatch measurement (never transferred).
