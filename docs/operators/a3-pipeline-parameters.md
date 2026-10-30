# A3 Ascend910_9362 pipeline parameters

This note records the nominal analytical supply declared for `Ascend910_9362`
in `mindie_sim/backends/profiles/ascend.json`. The values run through the common
`Engine`; they are not measured instruction peaks and carry no accuracy claim.
They are derived from the configured 1.65 GHz clock and the pinned public
instruction documentation, not from observed model gaps.

## Declared supply

| Resource | Basis | Rate (B/s) | Owners |
| --- | --- | ---: | ---: |
| `hbm` | existing single shared HBM rate | 1.2e12 | shared |
| `mte1` | 128 B/cycle/AIC x 20 AIC x 1.65 GHz | 4.224e12 | 20 AIC |
| `mte1_l0a` | 256 B/cycle/AIC x 20 AIC x 1.65 GHz | 8.448e12 | same `mte1` owners |
| `mte1_l0b` | 128 B/cycle/AIC x 20 AIC x 1.65 GHz | 4.224e12 | same `mte1` owners |
| `mte2_aic` | 32 B/cycle/AIC x 20 AIC x 1.65 GHz | 1.056e12 | 20 AIC |
| `mte2_aiv` | 32 B/cycle/AIV x 40 AIV x 1.65 GHz | 2.112e12 | 40 AIV |
| `mte3` | 64 B/cycle/AIV x 40 AIV x 1.65 GHz | 4.224e12 | 40 AIV |
| `fixpipe` | 128 B/cycle/AIC x 20 AIC x 1.65 GHz | 4.224e12 | 20 AIC |

The generic `mte1` rate is a conservative undifferentiated L0A/L0B supply. It is
not an exact mixed-operand instruction timing. The two direction modes are
alternative instructions on that same physical pipe:

- `mte1_l0a` is L1-to-L0A at 256 B/cycle/AIC.
- `mte1_l0b` is L1-to-L0B at 128 B/cycle/AIC.
- Both declare `resource: mte1`; no `mte1_a` or `mte1_b` resource exists.

`RecipeContext.bind` preserves the mode's native byte count but converts its
service to the base pipe's units. With base rate `r_base` and L0A/L0B mode
rates `r_a`/`r_b`, A/B service on the same owner consumes

```text
A_bytes * r_base / r_a + B_bytes * r_base / r_b
```

in base-pipe units. With the profile values, two directions on one AIC owner
take `(A/256 + B/128) / 1.65e9` seconds. Different AIC owners have separate
fixed capacities and overlap normally. A single owner cannot borrow the idle
capacity of the other nineteen owners.

## Reads and shared memory

`mte2_aic` and `mte2_aiv` are local read-path supplies with their own owner
sets. They do not create independent global memory. A program that moves bytes
from GM charges the shared `hbm` resource in addition to its owning local path,
so concurrent AIC and AIV reads contend on one HBM resource. There is no generic
third `mte2` read pipe.

The AIV 32 B/cycle value is an explicit analytical per-owner mapping of the
platform memory-path `ddr_read_rate` scalar. It is not a measured physical
instruction rate and does not establish a pair-shared private AIC/AIV rate.

## Stores and output

`mte3` is the AIV-owned UB-to-GM path at 64 B/cycle/AIV. No additional AIC MTE3
pipe is declared. `fixpipe` is the 20-owner AIC output path at 128 B/cycle/AIC.
Its use for generic GM destinations is an explicit analytical assumption: the
pinned instruction table directly establishes L0C-to-L1 BF16/FP16 conversion at
128 B/cycle and INT8 output at 64 B/cycle, not universal 128 B/cycle for every
destination.

## Overrides

The existing device-input merge is unchanged. A later explicit device
declaration overrides or extends the profile map without a new selector:

```json
{
  "machine": {
    "backend": "ascend",
    "sku": "Ascend910_9362",
    "device": {
      "rates": {"mte3": 8448000000000.0},
      "resource_units": {"fixpipe": 10}
    }
  }
}
```

Unspecified rates, units, service modes and buffers remain those of the
profile. The tests exercise the override through real Engine work; they do not
only inspect the configuration.

## Provenance

Pinned `asc-devkit` revision `56500ea0dc219380f50f4a17ce73ef5264f830fc`:

- `docs/zh/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.md`,
  SHA-256 `d16edc43d897cc4eba425802e0b7d1f61b1bc76fb50768f336d458ddd4927994`:
  AIC:AIV is 1:2; the cores are independent and communicate through GM; the
  L1-to-L0A/L0B `PIPE_MTE1` rates are 256/128 B/cycle.
- `docs/zh/api/appendix/cube_instruction_theoretical_perf_summary.md`,
  SHA-256 `3a249d10fb6b794849d15f66208e43fa27ccd5edc6b635418911ebf11020f728`:
  table 3 (lines 99-109) repeats the direction rates; table 7 (lines 192-200)
  documents Fixpipe L0C-to-L1 BF16/FP16 at 128 B/cycle and INT8 output at
  64 B/cycle.

Read-only sim-runner platform declaration
`/usr/local/Ascend/cann-9.2.0/aarch64-linux/data/platform_config/Ascend910_9362.ini`,
SHA-256 `ffd0e8d061db7fa8b3877bcbaf8151edac77abcf3a50df95b42b21e2332a9ca3`:
20 AIC/40 AIV, 64 GiB, `ddr_read` 32, `ub_to_ddr` 64, `cube_freq` 1500. It has
only `AICoreMemoryRates`, with no AIV-specific rates. The retained CANN 9.1beta1
ini SHA-256 is `5135cf6bc5e2eb56811917c9010341eb3946b5a3c2b51baab7fa775ffffc315c`,
preserved in `data/verification/a3-9362-platform-metadata-20261029.json`.

Two discrepancies are retained rather than smoothed away:

- The CANN ini reports `l1_to_l0_a`/`l1_to_l0_b` at 512/256 B/cycle, while the
  public instruction table documents 256/128 B/cycle. This profile chooses the
  documented instruction rates and records the conflict.
- The ini reports `cube_freq` 1500 MHz. The profile keeps the existing
  configured 1650 MHz; the INI value is not substituted.

## Limits

- The rates are nominal analytical declarations, not measured peaks.
- Generic matrix and attention work still uses the conservative `mte1` supply.
  Explicit directional modes are wired only where known.
- No independent MTE1 A/B resources, no generic third MTE2 pipe, and no fabricated
  AIC MTE3 pipe are modeled.
- Shared HBM remains one resource; the split read paths do not add global
  bandwidth.
- The Fixpipe 128 B/cycle GM destination is an assumption beyond the direct
  L0C-to-L1 BF16/FP16 source.

## Independent acceptance (2026-10-30)

The DSH implementation was reviewed against the pinned documents and exercised
with the common Engine. Six profile controls plus the existing resource-service
and prefill frontend checks passed: 29 tests in 0.12 seconds. The controls
independently calculate same-owner A/B serialization, separate-owner overlap,
single-owner limits, 20-AIC/40-AIV read sharing, 40-owner MTE3/20-owner Fixpipe,
and contention between GM reads and both output paths on the same HBM resource.

The ordinary frontend also selected staged GQA prefill for five queries with
three preceding context tokens. Its 640 HBM bytes equal 80 query bytes, 480 K/V
bytes and 80 output bytes. It used the registered AIC read and AIV store roles,
served no work before the external input became ready, and retired all tasks,
programs and allocations. No large model or device workload was run for this
acceptance.

A structural comparison of all 15 profiles found only `Ascend910_9362` changed.
Its compute peaks, memory capacity, existing invocation parameters, host launch,
buffers and transfer channels were preserved. These checks establish declared
resource behavior; they do not calibrate A3 performance.
