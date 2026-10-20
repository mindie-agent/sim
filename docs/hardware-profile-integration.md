# Hardware profile / thermal integration contract

Status: integration in progress (this document is the API contract; verification
results are appended before hand-off).

## Provenance

- Destination repository: `/Volumes/data/code/mindie-sim-a3-modeling`
  (standalone `sim` worktree; shared base verified at
  `d3d04567c0bfd85a61ac405eebb94d439a763afe`).
- Incoming source: `/Users/maoxx241/code/sim` (`mindie-release` worktree),
  read-only for this lane, Antigravity session
  `7168ae6c-5e44-4f5c-ad19-1d9d60962bdf`.
- Incoming files inspected: `mindie_sim/hardware.py`,
  `mindie_sim/sku_profile.py`, `mindie_sim/thermal.py`,
  `mindie_sim/hardware_probe.py`, plus the source export/CLI/test diff.
- Independent review evidence:
  `/tmp/mindie-a3-dsh.DNIfZY/review-antigravity-thermal.md` / `.json` and
  `/tmp/mindie-a3-dsh.DNIfZY/review-antigravity-probe-root.json`.
- Source hashes at inspection:
  - `hardware.py` `838bef183732e05090fa38ef3ffb37577d9d73eeb35ab59f04ef43bf0e238327`
  - `sku_profile.py` `4937719ec6be083c108482530f0816a114520cd4664ca851a66fc311586f704d`
  - `thermal.py` `205cff1a3da817b2523f7e99fd66c29c604af6d5d36f197bae35ba998a8dc73b`
  - `hardware_probe.py` `f00a79840f24ad51f0ea27e885962b32e7ac89a56d915cc62ad046456853d502`
- The imported A5 C8 FlashMLA Golden workload is an **explicit fixture**, not a
  new A5 board measurement. A5 measurement remains TODO and is to be collected
  later by the user.

## Scope

This lane owns only:

- `mindie_sim/hardware.py`
- `mindie_sim/hardware_probe.py` (new)
- `mindie_sim/sku_profile.py`
- `mindie_sim/thermal.py`
- `tests/test_modeling_inputs.py`
- new dedicated hardware/probe tests
- this document

Shared files (`mindie_sim/__init__.py`, `mindie_sim/__main__.py`,
`tests/test_mindie_sim.py`, `a3_*.py`) are **not edited here**. The exact
shared integration patch is delivered separately at
`/tmp/mindie-a3-dsh.DNIfZY/hardware-shared-integration.patch`; the core lane
applies it.

## Cube capability contract

Canonical source: `sku_profile.CubeSpec.ops_per_cycle_per_aic`.

- Unit: integer Cube ops per cycle **per AIC**.
- It is the only stored capability map. The incoming unpublished
  `CubeSpec.throughput` field is **not** added: a second per-die format would
  create contradictory capability sources.
- Normalized accessor (new, unit-explicit):

  ```python
  profile.cube_spec.ops_per_cycle_per_die("bf16")  # -> total per-die ops/cycle or None
  profile.cube_spec.ops_per_cycle_per_die("fp8")   # resolves fp8/fp8_e4m3/fp8_e5m2
  ```

  It returns `None` when the dtype is not declared. No A5 fallback, no
  multiplication by an assumed count, no implicit dtype conversion.
- Custom `sku_profile` / `cube_spec.ops_per_cycle_per_aic` overrides still
  update the canonical map and therefore remain the source of truth.
- Unsupported dtype + non-zero work is analytically `unmodeled`; it never
  borrows A5 peaks.

### A3 integration instructions (not applied by this lane)

`a3_types._profile_defaults` currently does:

```python
ops = cube.ops_per_cycle_per_aic
bf16 = ops.get("bf16", ops.get("fp16"))
...
"cube_flops_per_s": float(cube.aic_count_per_die) * float(bf16) * hz,
```

The core lane should switch to the normalized accessor:

```python
bf16_ops_per_die = cube.ops_per_cycle_per_die("bf16")
if bf16_ops_per_die is None:
    raise A3InputError("machine_sku_cube_capability_missing")
...
"cube_flops_per_s": bf16_ops_per_die * hz,
```

`MachineSpec.compute_peaks` stays the explicit user override and keeps
precedence in `peak_flops_for`. Do not add a second `throughput` field or a
new fallback chain. The normalized accessor does not alias `fp16` to `bf16`;
standard catalog profiles declare `bf16` explicitly, and custom profiles that
need BF16 must declare it.

## Thermal contract

`mindie_sim.thermal.thermal_bounds` is a clock/reference-bound estimator, not a
temperature or power evolution model.

### Inputs

| Input | Unit / meaning |
| --- | --- |
| `fp8_flops`, `bf16_flops` | Aggregate FLOPs. Optional; `0` is explicit zero work. |
| `fp8_ops_per_cycle`, `bf16_ops_per_cycle` | **Total per-die** Cube ops/cycle. Required if the corresponding FLOPs are positive and no explicit cycle basis is supplied. |
| `cube_reference_cycles` | Explicit aggregate reference Cube cycles. Takes precedence over FLOPs and is independently sufficient. |
| `wall_cycles`, `counted_cycles` | Aggregate PMU cycles W/C. Both or neither. |
| `measurement_mhz` | Explicit caller-supplied clock for the W/C pair. Optional; no clock is inferred. |
| `sustained_mhz_range` | Two explicit frequencies. Default is the documented A5 reference range. |
| `short_burst_mhz_range` | Two explicit frequencies. Default is the documented A5 reference range. |
| `extra_frequencies_mhz` | Additional explicit frequencies. Default empty. |
| `target_us` | Optional target latency. |
| `vector_cycles`, `fix_cycles` | Optional PMU components, only used in `required_mhz_for_target`. |
| `workload_origin` | Optional free-form provenance label (the A5 fixture sets `A5_C8_FLASHMLA_GOLDEN_FIXTURE`). |
| `cycle_source` | Optional provenance label for the Cube peak source. |

There is **no implicit A5 FLOPs/peak injection**. Calling
`thermal_bounds()` with no workload, no cycles and no W/C pair returns
`status="unmodeled"`.

### Frequency contract

1. `frequencies_mhz` is the sorted union of the supplied sustained range,
   short-burst range, extra frequencies and `measurement_mhz`. Every time value
   in the report is indexed to one of these values or is explicitly `null`.
2. `cube_reference_cycles` is a frequency-independent aggregate cycle count.
   `reference_mhz_for_cycles` is therefore intentionally `null`. No hidden
   1650 MHz conversion is performed.
3. `reference_lower_bounds_us[f] = cube_reference_cycles / f` for each listed
   frequency. `sustained_reference_us` and `short_burst_reference_us` are the
   same arithmetic at the range endpoints.
4. `clock_table` rows carry `frequency_mhz`, and the W/C/gap times for that row
   are `cycles / frequency_mhz`.
5. `sensitivity.frequencies_mhz` is exactly the report's `frequencies_mhz`.
   There is no separate 1250/1280/1300 default inside `thermal_bounds`.
6. Top-level `wall_us` / `counted_us` / `gap_us` are populated **only** when
   `measurement_mhz` is explicitly supplied; they then equal the clock-table
   row at that frequency. Without it they are `null`, even when W/C are present.
   This prevents a fictitious measured clock.
7. `ideal_hardware_lower_bound_us` is the minimum reference lower bound (the
   fastest listed frequency); `ideal_hardware_lower_bound_mhz` records that
   frequency.
8. `reference_time_us` / `cycles_to_us` remain frequency-parameterized
   arithmetic.

### Status values

| Status | Meaning |
| --- | --- |
| `ok` | A cycle basis **and** a complete W/C pair were supplied. |
| `partial_reference_only` | A cycle basis exists; no complete W/C pair. |
| `partial_measurement_only` | A complete W/C pair exists; no cycle basis. |
| `unmodeled` | No cycle basis and no complete W/C pair, or FLOPs are present without a declared per-cycle peak. `unmodeled_reason` is set. |

`unsupported_dtypes` is an annotation list (for example `["fp8"]`). It does not
by itself invalidate an explicit cycle basis or a PMU-only report: explicit
cycles and a complete W/C pair are independent evidence.

### Errors

Stable `HardwareError` reasons are used for invalid input, including:

- `thermal_clock_range_invalid` for non-finite or non-positive range
  endpoints or a wrong-length range (endpoint order is normalized);
- `thermal_frequency_invalid` for a non-finite/non-positive extra or
  measurement frequency;
- `thermal_cycles_pair_incomplete` when only one of W/C is supplied;
- `counted_cycles_exceed_wall_cycles`;
- `cube_reference_flops_invalid`, `cube_reference_peak_invalid`,
  `cube_reference_cycles_invalid`, `thermal_target_invalid`;
- `wall_cycles_invalid`, `counted_cycles_invalid`, `vector_cycles_invalid`,
  `fix_cycles_invalid`.

NaN/Inf are rejected before any result object is built. There is no
`status="unmodeled"` result containing a NaN frequency.

### Result keys

```text
status, unmodeled_reason, workload_origin, cycle_basis, unsupported_dtypes,
cube_reference_cycles, reference_cube_cycles_per_die,
reference_mhz_for_cycles (always null),
frequencies_mhz, measurement_mhz,
reference_lower_bounds_us,
sustained_mhz_range, sustained_reference_us,
short_burst_mhz_range, short_burst_reference_us,
target_us, required_mhz_for_target,
sensitivity, clock_table, gap_sensitivity,
wall_cycles, counted_cycles, wall_us, counted_us, gap_us,
ideal_hardware_lower_bound_us, ideal_hardware_lower_bound_mhz,
time_basis, assumptions
```

`clock_table` row shape:

```text
{frequency_mhz, reference_lower_bound_us, wall_us, counted_us, gap_us}
```

`time_basis` states the basis of each group (reference cycles, clock table,
sensitivity, top-level times).

### A5 C8 FlashMLA Golden fixture

`thermal.A5_FLASHMLA_C8_GOLDEN_WORKLOAD` is an explicit, origin-labeled fixture
containing FLOPs, its own total per-die Cube peaks
(864 TFLOP/s FP8, 432 TFLOP/s BF16 at 1650 MHz reference), target, W/C and
vector/fix cycles. It is usable through:

```python
thermal_bounds(**A5_FLASHMLA_C8_GOLDEN_WORKLOAD)
```

It is not a default and is not a new A5 measurement. The fixture has no
`measurement_mhz`, so its top-level W/C times are null.

## `hardware_report` thermal contract

- `hardware_report` always includes a `thermal` object, including for
  ambiguous SKUs and for reports with no workload. A report with no thermal
  inputs returns `unmodeled` with `missing_workload_or_cycle_inputs`.
- Cube peaks are resolved from the canonical SKU profile via
  `ops_per_cycle_per_die`. `peak_sources` records `caller_explicit` or
  `sku_profile.cube_spec.ops_per_cycle_per_die`.
- A positive FLOPs dtype with no SKU peak and no explicit cycle basis returns
  `unmodeled` / `unsupported_dtype_or_missing_sku_cube_throughput` (with
  `unsupported_dtypes`). It does not borrow A5.
- If explicit `cube_reference_cycles` is supplied, it is respected even when
  an unrelated dtype is unsupported; the unsupported dtype is only annotated.
- If a complete W/C pair is supplied without a cycle basis, the result is
  `partial_measurement_only` rather than unmodeled.
- Direct thermal inputs may be passed as keyword arguments or via the
  `workload` mapping (`fp8_flops`/`flops_fp8`, `bf16_flops`/`flops_bf16`,
  `target_us`, `wall_cycles`, `counted_cycles`, `vector_cycles`, `fix_cycles`,
  `cube_reference_cycles`, `fp8_ops_per_cycle`, `bf16_ops_per_cycle`,
  `measurement_mhz`, `workload_origin`).
- `ini_text=` is supported as an alternative to `config=`; supplying both is
  `hardware_config_and_ini_text_conflict`.

## npu-smi probe contract

`mindie_sim/hardware_probe.py` is a parser extraction of the existing
`hardware.py` code, re-exported from `hardware` for compatibility.

- `parse_npu_smi_info`, `parse_npu_smi_common`, `parse_npu_smi_board`,
  `parse_npu_smi_snapshot`, `load_npu_smi_snapshot` keep their names.
- Directory loading reads only direct `*.txt` children of the explicit path.
  It identifies the info/common/board captures by actually running the table
  parser, not by an anchored multiline regex. Multiple candidates for one role
  raise a stable `*_ambiguous` error instead of silently mixing captures.
- `parse_npu_smi_common` flushes the previous chip before updating the NPU ID,
  so a multi-NPU capture keeps each chip with its own NPU. The MCU block
  (`Chip Name: mcu`) does not overwrite the last chip's fields.
- A3 dual-die `Phy-ID` 0..15 and 16 device rows are preserved.
- No arbitrary recursive path scan, no NPU execution, no file writes, and no
  subprocesses. The donor parser remains intact in the read-only source tree.

## Shared integration patch

`/tmp/mindie-a3-dsh.DNIfZY/hardware-shared-integration.patch` contains only:

1. `mindie_sim/__init__.py`: export `A5_FLASHMLA_C8_GOLDEN_WORKLOAD` and
   `unmodeled_thermal_bounds`.
2. `mindie_sim/__main__.py`: explicit `--golden-a5-flashmla`,
   `--fp8-flops`, `--bf16-flops`, `--target-us`, `--wall-cycles`,
   `--counted-cycles`, `--vector-cycles`, `--fix-cycles`,
   `--cube-reference-cycles`, `--fp8-ops-per-cycle`,
   `--bf16-ops-per-cycle`, `--measurement-mhz`, `--short-burst-mhz` flags and
   forwarding.
3. `tests/test_mindie_sim.py`: the existing thermal test uses the explicit
   fixture; targeted regression tests for unmodeled no-workload, explicit
   cycles with unsupported FP8, and A2 capability do not borrow A5.

It is generated against the current destination shared files and checked with
`git apply --check`. `a3_*.py` is not in the patch; the A3 change above is an
instruction for the core lane.

## Verification / evidence status

### Implemented and verified in this lane

- `CubeSpec.ops_per_cycle_per_die(dtype)` normalizes the canonical per-AIC map
  to total per-die ops/cycle. The incoming `CubeSpec.throughput` field is
  omitted; `profile.to_dict()["cube_spec"]` has no `throughput` key. Custom
  `ops_per_cycle_per_aic` overrides and unsupported `None` results are tested.
- `thermal_bounds` never injects the A5 Golden workload. The explicit fixture
  is usable both directly and through `hardware_report`, and carries
  `workload_origin="A5_C8_FLASHMLA_GOLDEN_FIXTURE"`.
- The custom `900:1100`/W=1000/C=800 case now reports top-level time only when
  `measurement_mhz` is supplied; at 1000 MHz it is 1.0/0.8/0.2 us. Without it,
  top-level times are null and the clock table carries the per-frequency
  values. Sensitivity uses the same frequency list.
- Non-finite/zero clock inputs and incomplete W/C pairs raise stable
  `HardwareError` reasons; no NaN frequency appears in any result.
- A2 FP8 with an explicit `cube_reference_cycles=1000` is
  `partial_reference_only` with cycles 1000 and `unsupported_dtypes=["fp8"]`;
  the same FP8 input without explicit cycles is `unmodeled`. A2 BF16 uses the
  A2 profile peak (20 AIC x 8192), not A5.
- `hardware_probe.load_npu_smi_snapshot` loads a three-file A3 directory
  capture (info/common/board), returns 16 devices, `phy_ids == range(16)`,
  two chips and `Product Name == IT22HMDA_4_S`.
- Multi-NPU common parsing keeps each chip with its declaring NPU and the MCU
  tail no longer overwrites the last chip temperature.
- Directory role detection uses the actual parsers; multiple info/common/board
  candidates raise `npu_smi_*_ambiguous`, empty directories raise
  `npu_smi_info_missing`, and missing paths raise `npu_smi_path_missing`.

### Test evidence

Run in the destination worktree (shared `tests/test_mindie_sim.py` still has
the pre-integration fixture call and is expected to fail until the patch is
applied):

```text
python -m pytest -q tests/test_hardware_profile_integration.py tests/test_modeling_inputs.py
49 passed

python -m pytest -q
320 passed, 4 skipped, 1 failed
# the single failure is tests/test_mindie_sim.py::test_cube_reference_cycles_and_sustained_reference_lower_bounds
# because the shared test still calls hw.thermal_bounds() with no explicit fixture.
```

The exact shared patch was then applied in a disposable copy at
`/tmp/mindie-a3-dsh.DNIfZY/shared-patch-verify-final`:

```text
git apply --check /tmp/mindie-a3-dsh.DNIfZY/hardware-shared-integration.patch   # OK
python -m pytest -q
323 passed, 4 skipped
```

These full-suite totals are a snapshot; other lanes continue to add tests to the
same worktree. The owned targeted suite (49 tests), the shared-file hashes and a
fresh `git apply --check` were re-verified after the last in-lane edit.

```text
python -m mindie_sim hardware --ini .../Ascend950DT_9581.ini --golden-a5-flashmla
# thermal.status=ok, cycles=442918.5024, origin=A5_C8_FLASHMLA_GOLDEN_FIXTURE, wall_us=null

python -m mindie_sim hardware --ini .../Ascend950DT_9581.ini \
  --wall-cycles 1000 --counted-cycles 800 --cube-reference-cycles 1000 \
  --measurement-mhz 1000 --sustained-mhz 900:1000 --short-burst-mhz 1050:1100
# status=ok, wall/count/gap=1.0/0.8/0.2 us,
# frequencies and sensitivity frequencies = [900, 1000, 1050, 1100]

python -m mindie_sim hardware --ini .../Ascend950DT_9581.ini
# thermal.status=unmodeled, reason=missing_workload_or_cycle_inputs, cycles=null
```

### NPU evidence

None from this lane. No NPU execution, no new measurement, no source checkout,
commit or release. The A3 fixture is a captured `npu-smi` snapshot; the A5
Golden workload is imported fixture data. A5 measurement remains TODO and is to
be collected later by the user.

### Residual follow-ups

- Core lane: apply the shared patch and optionally switch
  `a3_types._profile_defaults` to `ops_per_cycle_per_die("bf16")`; the
  compatibility test in `tests/test_hardware_profile_integration.py` proves the
  numerical result is unchanged for A2/A5 profiles.
- A5 board measurement and a measured-clock provenance capture remain future
  work; the imported Golden fixture must not be cited as either.
