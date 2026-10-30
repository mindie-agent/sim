# Ascend SKU pipeline profiles (A3 expansion and A5 arch3510)

This note records the tightly scoped profile expansion in
`mindie_sim/backends/profiles/ascend.json`. All values are nominal analytical
supply. They are not measured instruction peaks, not SKU measurements and carry
no accuracy claim. Only profile data, focused tests and this note are in scope;
the execution Engine, device schema, materialization, recipes and model/quant
aliases are unchanged.

## A3 capacity and host policy

The existing `Ascend910_9391`, `Ascend910_9392`, `Ascend910_9381`,
`Ascend910_9382` and `Ascend910_9372` declarations keep their own:

- `memory_bytes`, HBM rate and `l2_capacity_mib`;
- buffers, buffer units, compute units and links;
- configured clock, AIC/AIV counts, kernel invocation minima and provenance;
- model/transfer channels and all other invocation constraints.

The only device additions are the pipeline resources, service modes and the A3
precision peaks. `Ascend910_9362` keeps its already-reviewed pipeline declaration
and sources; only its precision peaks and root launch policy changed.

Every one of the 15 profiles now declares `device.launch_us: 0`. Host
submission is undeclared, so zero is the device-path baseline, not a measured
host launch. The shared `assumptions.launch_us` text says that
`runtime.submission.kernel_launch_ms` or an explicit `device.launch_us` can
declare host submission, and that an empty-task duration is not used as a host
parameter. No profile substitutes an empty-task measurement.

## A3 pipeline derivation

Source declaration clock and core counts are used directly. For AIC owners
`N_aic`, AIV owners `N_aiv` and clock `f`:

```text
mte1      = 128 B/cycle/AIC * N_aic * f       generic undifferentiated A/B supply
mte1_l0a  = 256 B/cycle/AIC * N_aic * f       same physical mte1 resource/owners
mte1_l0b  = 128 B/cycle/AIC * N_aic * f       same physical mte1 resource/owners
mte2_aic  =  32 B/cycle/AIC * N_aic * f       AIC local read path
mte2_aiv  =  32 B/cycle/AIV * N_aiv * f       9362-inherited analytical assumption
mte3      =  64 B/cycle/AIV * N_aiv * f       AIV-owned UB->GM
fixpipe   = 128 B/cycle/AIC * N_aic * f       generic GM destination caveat retained
```

| SKU | MHz | AIC/AIV | mte1 | mte1_l0a | mte1_l0b | mte2_aic | mte2_aiv | mte3 | fixpipe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9391 | 1850 | 24/48 | 5.6832 | 11.3664 | 5.6832 | 1.4208 | 2.8416 | 5.6832 | 5.6832 |
| 9392 | 1800 | 24/48 | 5.5296 | 11.0592 | 5.5296 | 1.3824 | 2.7648 | 5.5296 | 5.5296 |
| 9381 | 1800 | 20/40 | 4.6080 | 9.2160 | 4.6080 | 1.1520 | 2.3040 | 4.6080 | 4.6080 |
| 9382 | 1650 | 20/40 | 4.2240 | 8.4480 | 4.2240 | 1.0560 | 2.1120 | 4.2240 | 4.2240 |
| 9372 | 1650 | 20/40 | 4.2240 | 8.4480 | 4.2240 | 1.0560 | 2.1120 | 4.2240 | 4.2240 |
| 9362 (preserved) | 1650 | 20/40 | 4.2240 | 8.4480 | 4.2240 | 1.0560 | 2.1120 | 4.2240 | 4.2240 |

Rates in the table are `1e12 B/s`. The five new profiles record this derivation
under `source.pipeline`, including the exact clock, owner counts, byte/cycle
figures and sources. Their provenance distinguishes the two inputs explicitly:

- `architecture_inference`: A3 `npu_arch_2201` instruction-table inference shared
  with the existing `Ascend910_9362` field definitions; it is not a measurement
  of the SKU.
- `sku_parameters`: clock and AIC/AIV are read from that SKU's
  `source.declaration`; capacity, HBM, links, buffers and invocation constraints
  are retained from the existing profile.
- `rates`: recomputed for each SKU; not copied from `9362`.
- `sku_measurement`: `none`; `source.declaration` is configuration metadata, not
  performance measurement.

There is no generic third MTE2 pipe and no separate `mte1_a`/`mte1_b` resource.
Both directional modes name `resource: mte1` and the same owner set. The 32
B/cycle/AIV read value remains the explicit per-owner analytical mapping
inherited from `9362`.

### A3 precision peaks

The direct arch2201 source table 1 in
`cube_instruction_theoretical_perf_summary.md` gives `int4` cube_k 64, BF16 16,
FP32 4 and HF32 8. Therefore, for every A3 profile:

```text
int4 = 4 * bf16
fp32 = bf16 / 4
hf32 = bf16 / 2
```

The six previous A3 `fp32` values were `bf16 / 2`, which is the HF32 rate. Those
values are moved to `hf32`, and `fp32` becomes the native-table `bf16 / 4`.
`int4` is added at the direct `cubek64`-versus-16 ratio. The existing `bf16`,
`fp16` and `int8` peaks are unchanged.
The INT4 peak requires an explicitly INT4-by-INT4 matrix program. An ordinary
W4A8 representation does not automatically execute at that peak: its INT8
activation/compute path remains separate from a source-selected MSD program.

## A5 arch3510 expansion

`Ascend950DT_9581` and `Ascend950PR_9579` now carry the same arch3510 physical
resources and service modes as `Ascend950DT_9582`, derived per core from 32 AIC,
64 AIV and 1.65 GHz:

| Resource | Formula | Rate (B/s) | Owners |
| --- | --- | ---: | ---: |
| `mte1` | 256 B/cycle/AIC * 32 * 1.65 GHz | 13.5168e12 | 32 AIC |
| `mte2_aic` | 80 B/cycle/AIC * 32 * 1.65 GHz | 4.224e12 | 32 AIC |
| `mte2_aiv` | 40 B/cycle/AIV * 64 * 1.65 GHz | 4.224e12 | 64 AIV |
| `mte3` | 128 B/cycle/AIV * 64 * 1.65 GHz | 13.5168e12 | 64 AIV |
| `fixpipe` | 128 B/cycle/AIC * 32 * 1.65 GHz | 6.7584e12 | 32 AIC |
| `reg_data_writes` | 2/cycle/AIV * 64 * 1.65 GHz | 211.2e9 | 64 AIV |
| `reg_ub_accesses` | 2/cycle/AIV * 64 * 1.65 GHz | 211.2e9 | 64 AIV |
| `reg_ub_stores` | 1/cycle/AIV * 64 * 1.65 GHz | 105.6e9 | 64 AIV |
| `reg_alu_lanes` | 128/cycle/AIV * 64 * 1.65 GHz | 13.5168e12 | 64 AIV |
| `reg_special_lanes` | 32/cycle/AIV * 64 * 1.65 GHz | 3.3792e12 | 64 AIV |
| `reg_reduce_lanes` | 64/cycle/AIV * 64 * 1.65 GHz | 6.7584e12 | 64 AIV |
| `reg_shuffle_lanes` | 64/cycle/AIV * 64 * 1.65 GHz | 6.7584e12 | 64 AIV |

Service modes `ub_copy` at `256 * 64 * 1.65 GHz = 27.0336e12` and
`reg_issue_slots` at `2 * 64 * 1.65 GHz = 211.2e9` both bind to `vector`, as
declared by `9582`. Buffer sizes/units and the register owner sets are also the
same arch3510 declaration.

Capacity, HBM and links are not copied:

| SKU | Memory | HBM | Intra-node link |
| --- | ---: | ---: | --- |
| 9581 | 144 GiB | 4.0e12 B/s | UB mesh 392 GB/s, 1.8 us |
| 9582 (preserved) | 96 GiB | 4.0e12 B/s | UB mesh 392 GB/s, 1.8 us |
| 9579 | 64 GiB | 1.6e12 B/s | UB mesh 392 GB/s, 1.8 us |

The source pointer
`data/observations/sources/arch3510-physical-services/manifest.json` is present on
all three profiles.

### A5 precision peaks

Direct arch3510 table 2 gives native FP32 cube_k 1, HF32 cube_k 8 and BF16 cube_k
16. Therefore:

```text
fp32 = bf16 / 16
hf32 = bf16 / 2
```

`9582` already had those values and they are preserved. `9581` and `9579` had
the HF32 value in `fp32`; their `fp32` is corrected to `bf16 / 16` and `hf32`
is set explicitly to `bf16 / 2`.

## Provenance

Pinned `asc-devkit` revision
`56500ea0dc219380f50f4a17ce73ef5264f830fc`:

- `cube_instruction_theoretical_perf_summary.md`, SHA-256
  `3a249d10fb6b794849d15f66208e43fa27ccd5edc6b635418911ebf11020f728`.
  Table 1 (lines 40-52) covers A3/910b: int4 cube_k 64, int8 32, half/BF16 16,
  float 4, HF32 8. Table 2 (lines 54-78) covers arch3510: BF16 16, HF32 8,
  native FP32 1. Table 3 gives MTE1 direction rates; table 7 gives Fixpipe
  L0C-to-L1 BF16/FP16 at 128 B/cycle and INT8 output at 64 B/cycle.
- `npu_arch_2201.md`, SHA-256
  `d16edc43d897cc4eba425802e0b7d1f61b1bc76fb50768f336d458ddd4927994`.
- `npu_arch_3510.md`, local source copy SHA-256
  `3a69176f4d4eab9e3883755e92598fb9defb3d51dc117de0bd94bb1acaa03caa`.
- arch3510 source manifest, SHA-256
  `15d78fc4e78ae4221c97dcbff451ef78dd95c1850a0406f80d2e22f7984fc902`.
- arch3510 vector/register performance table, SHA-256
  `9bad332c6f0251c19d3282a9ad8a428cde09b36a9274cacafb9e7f8fe2146973`.
- `Ascend950DT_9582.ini`, SHA-256
  `b8e7d61780056889e795639504530ecacb2b759ad1602e929b863d837fdf0099`.

The existing `9362` CANN 9.2.0 ini evidence is preserved: SHA-256
`ffd0e8d061db7fa8b3877bcbaf8151edac77abcf3a50df95b42b21e2332a9ca3`, with its
20 AIC/40 AIV, 64 GiB, `ddr_read` 32 and `ub_to_ddr` 64 declarations. Its 1500
MHz `cube_freq` is still not substituted for the configured 1650 MHz.

## Unproven assumptions and limits

- Every rate is nominal analytical supply, not a measured SKU peak.
- A3 `mte2_aiv` 32 B/cycle/AIV is inherited from `9362` as an analytical
  per-owner mapping of the platform read scalar, not an instruction measurement.
- A3 Fixpipe 128 B/cycle for generic GM destinations is an assumption; the
  direct source establishes L0C-to-L1 BF16/FP16 at 128 B/cycle and INT8 output
  at 64 B/cycle.
- The A3 rates use each profile's configured clock. They do not use a platform
  ini clock for the newly expanded SKUs, and no SKU clock was measured here.
- A5 `mte2_aic` 80 and `mte2_aiv` 40 B/cycle are nominal memory-path
  declarations, not measured MTE2 instruction rates. Shared L2 behavior is not
  characterized.
- A5 `ub_copy` and register lane capacities are nominal resource declarations;
  they are not a cycle-accurate instruction scheduler. Masked-lane service,
  dependency latency, compiler pairing and predicate-write ports remain
  uncharacterized.
- `launch_us: 0` removes an undeclared nominal host cost from the device path.
  It is not evidence that host submission is free. The stated
  `runtime.submission.kernel_launch_ms`/`device.launch_us` declaration path is a
  policy statement; no host-launch measurement is included.
- No A2 peaks, capacities or links were changed.

## Focused checks

`tests/test_sku_pipeline_profiles.py` parses the profiles through `Hardware` and
runs explicit `Work` through the common Engine. It independently recomputes:

- A3 rates from each declared clock and AIC/AIV count (1.65, 1.8 and 1.85 GHz;
  20/40 and 24/48 owners), including same-owner L0A/L0B serialization versus
  different-owner overlap;
- shared-HBM read/write contention on `9381` and `9372`, with SKU HBM 1.6e12 and
  1.4e12 B/s;
- A5 per-core rates, register resources/service modes, and HBM scaling for
  `9581` (4.0e12) versus `9579` (1.6e12);
- precision ratios through ordinary matrix work: A3 FP32 is 2x slower than
  HF32, and A5 FP32 is 8x slower than HF32;
- capacity and intra-package/UB-mesh link preservation;
- the root `launch_us: 0` policy and the A3 architecture-inference/SKU-measurement
  provenance separation.

Focused result: `21 passed in 0.06s` for the new SKU tests. Together with
the six existing `9362` pipeline regression tests, `27 passed in 0.10s`. No full
suite, device workload, remote job or automation was run.

Independent acceptance replaced two prose-matching checks with 15 real
host-submission controls. For every profile, undeclared host submission serves
zero launch work; an explicit 7 us declaration adds exactly 7 us and preserves
device work. The reviewer also compared all profiles against the parent commit:
existing memory, clocks, owner counts, buffers, HBM/vector rates, links,
invocation constraints and BF16/FP16/INT8/FP8 peaks are unchanged.

The final focused set passes 126 tests in 2.80 seconds: 34 SKU controls,
51 model/quantization alias and ordinary-serving controls, the existing A3
pipeline controls, device-kernel timing and native configuration coverage.
This is local mechanism and API acceptance, not hardware calibration.
