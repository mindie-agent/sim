# A5 physical calibration: bf16 artifact published, fp8 pending

Status: the **A5 bf16 artifact is published** from a real 950DT campaign
(`Ascend950DT_9582`, rack06os5, 96 physical conditions / 117 train observations /
5,616 device-event samples, held-out guard clean). fp8 is still unmeasured: the
fp8 probe is prepared but no fp8 fit exists yet. This note records the campaign,
the remaining blockers and the minimum set; the sections below that say "not
measured" describe the state before the campaign and are kept for provenance.

### Published A5 bf16 rows (train-only 5-fold CV / held-out WAPE)

| family | CV | held-out |
| --- | ---: | ---: |
| gemm | **0.59 %** | 0.41 % |
| copy | **0.28 %** | 1.26 % |
| moe_down | 1.54 % | 0.75 % |
| grouped_matmul | 1.87 % | 0.76 % |
| add (batched / per-call) | 2.58 / 4.67 % | 8.27 / 9.90 % |
| moe_gate_up | 3.14 % | 2.05 % |
| attention absorbed MLA (device / block) | **8.29 / 8.71 %** | 2.29 / 2.28 % |
| attention GQA (device / block) | 18.18 / 18.34 % (`low_confidence`) | — |

Artifact: `.workstreams/model-physical/a5-950dt-9582-bf16-physical-calibration.json`
sha256 `89661048efbd09d97487dcef81e5905a05ee1adbc5b3a072d4883661844f2be6`.
Measured physics: dense bf16 430.2 TFLOPS = **99.46 %** of the declared
4.325376e14 peak; copy **3.33 TB/s** (η_m 0.833 of 4.0 TB/s); dispatch
**29.39 µs per-call vs 6.74 µs batched** (22.65 µs submission tax). The A5
fitter adopted a topology split (GQA vs shared-KV/MLA) on CV. GQA *decode* is
over-predicted (0.995 vs 0.300 ms at kv=65536) because the flat-additive form
has no decode-latency term; flagged `low_confidence`. Environment deviation:
rack06 runs firmware 9.2.0.101.235 / software 25.1.rc3 / torch_npu 2.10.0.post6
with the `megamoe-c00588804:ready` image (the `v0.30.0rc2-a5` image exists only
on rack08), and `--init` is unavailable there.

## 1. A5 machine blocks (declared, not measured)

| file (`.workstreams/model-physical/`) | SKU | dtype | AIC/AIV | clock | peak_flops | HBM peak | L2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `machine-a5-950dt-9581-bf16.json` | Ascend950DT_9581 | bf16 | 32/64 | 1650 MHz | 4.325 376e14 | 4.0 TB/s | 128 MiB |
| `machine-a5-950dt-9581-fp8.json` | Ascend950DT_9581 | fp8 | 32/64 | 1650 MHz | **8.650 752e14** | 4.0 TB/s | 128 MiB |
| `machine-a5-950pr-9579-bf16.json` | Ascend950PR_9579 | bf16 | 32/64 | 1650 MHz | 4.325 376e14 | 1.6 TB/s | 128 MiB |
| `machine-a5-950pr-9579-fp8.json` | Ascend950PR_9579 | fp8 | 32/64 | 1650 MHz | **8.650 752e14** | 1.6 TB/s | 128 MiB |

All four load through `MachineLimits.from_mapping` (the loader re-derives
`aic_count x ops_per_cycle x sustained_clock` and the vector peak, and rejects a declared
value that contradicts either).
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
| **B-1/P5** | `a3_types.MachineSpec.peak_flops_for` | **FIXED**: the peak now comes from the declared `compute_peaks` (fp8 = 2× bf16 on A5), and a dtype with no declared peak is `machine_compute_peak_undeclared:<dtype>` instead of the bf16 base. A 5120³ fp8 GEMM on the A5 fp8 block is **0.3303 ms**, not 0.6406 ms | — |
| **P6** | `a3_types._profile_defaults` | **FIXED**: `compute_peaks` is derived from `profile.cube_spec.ops_per_cycle_per_aic × AIC × clock` (or an explicit caller map), so `{"sku": "Ascend950DT_9582"}` yields all six declared peaks instead of `{}` | — |
| **B-4** | `a3_types._profile_vector_ops_per_s` | **FIXED**: one convention, `aiv × simd_width × clock` (A5 `64 × 256 × 1.65 GHz = 2.70336e13`); the 2× measurement nominal is deleted, and `MachineLimits` refuses a block whose declared vector peak contradicts its components | — |
| **P15** | `a3_operator_campaign_machine.resolve_campaign_machine` | **FIXED**: a campaign that names a non-A3 SKU derives the machine from that SKU's profile (A5 gets single-die, 4 TB/s, fp8 2× bf16); a declaration whose family or cube peak contradicts the SKU is refused. An A3 SKU keeps the tuned A3 baseline | — |
| **B-5** | `sku_profile.resolve_device_capability_profile` | **FIXED**: an exact-but-unregistered SKU reports `sku_not_registered:<soc>` (provenance `unregistered_exact_soc`) instead of the family-token reason; a bare family/model prefix keeps `ambiguous_soc_sku_...` | — |
| **D2/D10** | `.workstreams/accuracy-protocol/acceptance-schema.json:85-96`, `tools/accuracy/score_frozen.py:66-74` | the acceptance schema cannot express an fp8 peak (bf16-only block, defaults to `1.0`) and the frozen limitation list is A2/A3-only | acceptance tooling |
| **B-2** | `a3_cost.py`, `a3_communication.py` | **FIXED**: an A5 UB-Mesh collective records `uncalibrated_comm_placement` instead of borrowing an HCCS curve | — |

The five fixed rows are covered by `tests/test_a5_code_blockers.py`. A2/A3 per-die
capabilities and bf16 prices are unchanged by those fixes; the only A2/A3 number that moved
is the *measurement-environment* A3 vector peak, which now equals the A3 machine block
(`8.448e12`) instead of twice it (`1.6896e13`), so ingest and fit use one engine.

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
