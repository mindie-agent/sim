# SituMxQuant complete source call

This document describes the explicit `situ_mx_quant_arch35` program profile. It
prices one inspected native `SituMxQuant` call; it does not replace the ordinary
SiTU activation or dynamic-quantization semantic members unless that profile is
selected.

## Source identity

| Field | Value |
| --- | --- |
| Repository | `vllm-ascend` |
| Base revision | `175ef4d49c2072f9233619636f0309f4f88b4e2c` |
| Source state | `added_in_index` |
| Scope | Independent source semantics; not capture binary identity |

| Path | SHA-256 | Inspected lines |
| --- | --- | --- |
| `csrc/moe/situ_mx_quant/op_host/arch35/situ_mx_quant_tiling_arch35.cpp` | `a2c44c27ef9f4230b7bb558c31976b920bff12f278cdb4f7909d97486e2c6582` | 199-245, 255-283 |
| `csrc/moe/situ_mx_quant/op_kernel/arch35/situ_mx_quant_axis_last.h` | `2a9bd1b0058d5c5727dc313c10f1e6920ea900390c64402d5263c8a602470284` | 126-183, 186-229, 232-278 |
| `csrc/moe/situ_mx_quant/op_kernel/arch35/situ_mx_quant_common.h` | `bce5998513209f9a47579913321f6c924747829fbd5e42aad2f3a57c1284f858` | 91-123, 129-182, 188-240, 247-508 |
| `csrc/moe/situ_mx_quant/situ_mx_quant_torch_adpt.h` | `3564aba053a9d37ec4960effb538358c62d7580858d8d5fbf7c71b1f93ea95fb` | 22-68 |

The hashes identify the inspected source files. They do not establish that a
captured event executed this binary, tiling key or scalar branch.

## Supported tensor domain

```text
BF16 [..., 2H] -> FP8 E4M3 or E5M2 [..., H]
               + E8M0 [..., ceil(H/64), 2]
```

* `axis=-1` only; all leading dimensions are flattened to rows.
* Input is BF16. Output payload is `fp8_e4m3` (dst type 36) or `fp8_e5m2` (35).
* `beta > 0`; `linear_beta` is either absent or positive. Both are model
  declarations and are not inferred from a captured event.
* `activate_left` swaps the first/second input halves. It is preserved as a
  declaration; both branches have the same per-element work.
* `H` may be a non-multiple of 64 or 256; tails are handled from actual payload
  and aligned scale positions.

Selection is explicit:

```json
{
  "runtime": {
    "kernels": {
      "lowerings": {
        "native_fusion": {
          "implementation": "declared_native_fusion",
          "options": {
            "roles": ["situ_quant"],
            "program_profiles": {"situ_quant": "situ_mx_quant_arch35"}
          }
        }
      }
    }
  }
}
```

The selected group must contain exactly the SiTU activation and its
`dynamic_quantize` consumer. Its only externally visible output must be the
quantizer (payload plus E8M0 scales). An externally consumed activation
intermediate is rejected, because it would require a real BF16 HBM
materialization that this complete call does not perform. Invalid dtypes,
scale shape, axis, group membership or output boundary raise a modeling error.
The older `traffic_profile: situ_mx_quant_arch35_2a9bd1b0` remains a separate
option and keeps its existing member-recipe arithmetic; it is not this program.

## Host grid and ownership

The source computes:

```text
dimNBlockNum = ceil(H / 256)
mCore = min(T, AIV)
nCore = min(AIV // mCore, dimNBlockNum)   when mCore < AIV, else 1
used  = mCore * nCore
```

This is the source's 256-column block grid, not an aggregate "spread all work
over every AIV" assumption. For `T=1`, `H=768`, `AIV=64`: `mCore=1`,
`dimNBlockNum=3`, `nCore=3`, `used=3`. Only owners 0, 1 and 2 participate; the
other 61 AIVs are idle and never lend transfer or vector service.

Each participating owner owns one or more 256-column blocks and its source row
span. Rows are split by the source head/tail rule, so when `T` exceeds `AIV`
some owners own one extra row. With `T=130`, `AIV=64`, two owners own three
rows and 62 owners own two rows. The actual physical vector unit and copy pipe
of each owner are pinned; two owners sharing one physical copy pipe split that
pipe rather than gaining a second one.

For the T=1, H=768 example:

| Item | Per owner | Total, 3 owners |
| --- | ---: | ---: |
| Two BF16 input slices (`256` each) | 1,024 B | 3,072 B |
| FP8 payload | 256 B | 768 B |
| E8M0 scales (`8` bytes, 2 per 64 columns) | 8 B | 24 B |
| FP8 + scale write | 264 B | 792 B |

The complete GM payload is 3,072 bytes read and 792 bytes written. No BF16
SiTU intermediate leaves the owner program and no global workspace or global
sync is introduced.

## UB tiling and buffers

The source reserves 32 B and 128 B from the declared UB size, then computes
`ubTotalBasicBlock = available // 3120`. `factorDim1` is the 256-position block
count held by one UB iteration and `factorDim0` is the row factor. If the full
number of N blocks does not fit, `factorDim1=ubTotalBasicBlock` (at least one)
and `factorDim0=1`.

Per owner the fixed source partition is allocated for the call:

| Buffer | Bytes |
| --- | ---: |
| `inQuex` (two input queue slots) | `2 * factorSize * 512 * 2` |
| `outQuey` (two FP8 queue slots) | `2 * factorSize * 256` |
| `outQueScale` (two 64-byte-aligned scale slots) | `2 * align64(factorSize * 8)` |
| `situ` BF16 | `factorSize * 256 * 2` |
| `maxExp` (aligned separately) | `align32(factorSize * 16)` |
| `halfScale` (aligned separately) | `align32(factorSize * 16)` |

`factorSize = factorDim0 * factorDim1`. The buffers are acquired on the owner's
physical UB unit before its lazy tile pipeline and released at call completion;
peak live UB is bounded by participating owners. For A5 253,952 B UB, T=1,
H=768 gives `factorDim0=27`, `factorDim1=3` and 252,864 B per owner.

## Program shape

Each owner is an independent `PipelineProgram` with two source queue slots.
Tiles are generated lazily in source `mGroup x nLoop` order; only the active
window is materialized. Global work totals use closed forms over the finite
owner geometry, not a per-tile or per-element Python loop.

```text
load -> situ -> exponent32 reduction -> E8M0 scales -> quantize -> store
```

The phases are serial within a tile. `situ`, `maxExp` and `halfScale` are single
scratch buffers: the next tile's SiTU phase waits for the previous tile's
quantization to finish. The two queue slots still permit the next input load
and the previous output store to overlap that vector chain.
This is a declared simple queue approximation, not a reconstruction of the
native compiler timeline, scalar barriers or instruction overlap.

## Arithmetic ledger

The model counts source-backed vector lane operations. One lane operation is
one element-wise source expression. `Exp` and `Div` count once each; an FMA
counts as two scalar lane operations (multiply plus add); compare/select/cast/
multiply/add count once. Scalar register broadcasts and DMA/queue copies are not
lane operations. These are semantic counts, not ISA latencies or calibrated
throughputs.

| Source group | Lane operations |
| --- | ---: |
| SiTU, no `linear_beta` | 32 per active column |
| SiTU, with `linear_beta` | 54 per active column |
| BF16 exponent extract and per-32 reduction | 32 `And` + 16 pair-max + 15 reduce-max per 32-column group |
| E8M0 scale and reciprocal | 13 per 32-column scale |
| BF16-scale multiply, casts and FP8 combine | 6 per converted column |

The 32-lane SiTU group is 29 source expressions plus 3 FMAs counted as two
scalar lane operations each. It includes the two BF16→FP32 casts, the polynomial
and sigmoid `tanh` paths, the path compare/select, `sigmoid(gate)`, the beta
multiply, the up multiply and the final FP32→BF16 cast. If FMA is counted once,
subtract 3 per active column. The extra 22 lanes are the optional
`linear_beta * tanh(up / linear_beta)` poly/sigmoid/select/scale branch, also
with 3 FMAs counted twice. BF16 rounding of the SiTU result occurs before
exponent extraction, as in the source. The scale is built from the BF16 exponent mask and OCP E8M0
logic, and the reciprocal scale multiplies the BF16 SiTU value before the
FP32/FP8 casts.

The FP8 conversion loop uses a full 256-lane mask for every 256-element chunk,
including a tail chunk. Tail work therefore rounds the converted lanes up to a
256 multiple, while the GM payload remains the actual `rows * H` bytes. E8M0
destination positions are 64-column aligned, so a tail writes
`2 * ceil(columns/64)` scale bytes per row.

For 768 complete columns, exponent work is `24 * 63 = 1512` lanes.
FP8 conversion is three groups of `256 multiply + 512 cast + 768 packing-add`
lanes, or 4608 lanes. E8M0 scales add 312 lanes. The whole call is therefore
31,008 lanes without `linear_beta`, or 47,904 lanes with it. Tail accounting
combines semantic reduction work and the fixed conversion width; it does not
claim every native masked-register instruction has been reconstructed.

## Old semantic boundary versus new source boundary

| Boundary | Old ordinary semantic group | Selected complete source call |
| --- | --- | --- |
| Members | Separate SiTU activation and `dynamic_quantize` operators | One source kernel program |
| BF16 intermediate | Written and reread through HBM under `preserve_workspaces`; one traffic profile may remove it | Never materialized in HBM |
| Arithmetic | Declared scalar expression estimate (`9` or `12` per element), lower than the inspected source body | 32/54 SiTU lanes (FMA as two scalar ops) plus exponent/reduction/scale/quantize groups |
| Ownership | Aggregate resource service over the whole group | Source `mCore x nCore` 256-column owners and unequal rows |
| Scales / tails | Generic quantizer shape | Per-32 exponent reduction, OCP E8M0 scale and reciprocal, 64-aligned scale writes |
| Timing | Sum of member envelopes / resource DAG | One complete-call program per owner on the common Engine |

The source program does not borrow idle AIVs, does not add a global sync or
workspace, and does not add a kernel invocation floor or per-stage invocation
fee. It has no BF16 intermediate HBM traffic.

## The retained observed residual is unexplained

The retained observation near 65 us for this call is not modeled by this source
or resource program. The source program is substantially larger than the old
9/12-operation semantic expression because it includes two `tanh` paths,
exponent extraction, per-32 reduction, E8M0 scale arithmetic, BF16 scale
multiply and FP8 casts. Even so, the analytical resource result is much smaller
than 65 us, so the residual remains **explicitly unexplained by the inspected
source work and the declared resource controls**. No compiler reconstruction,
ISA latency, scalar issue/barrier cost, startup floor, fitted multiplier or
capture-derived admission check is added to absorb it. The 65 us value is not an
input to planning and is not used to calibrate or validate this program.

## Public isolated example

`examples/situ_mx_quant_complete_call.py` builds the T=1, H=768 source call on
the existing A5 case and runs it through `call_costs` with the shared engine. It
prints actual source work, the three owner UB allocations, released live state
and the explicit note that the retained observed residual is unexplained:

```sh
PYTHONPATH=. .venv/bin/python examples/situ_mx_quant_complete_call.py
```

## Independent review and retained controls, 2026-10-30

The DSH implementation is commit `2563d9c3`, based on `cc3e8c8a`. Independent
review corrected exponent/FP8 effective-lane overcounting and added the
cross-tile single-scratch dependency. It also preserved copy ownership when
logical service modes share a physical DMA pipe; an owner with multiple
physical units receives exactly that partition's capacity.

The ordinary frontend's current-checkpoint conditional case declares
`beta=4`, `linear_beta=25`, BF16 input and FP8/E8M0 output. Actual shared-Engine
controls retain these arguments for all three alternatives:

| Rows, width | Semantic two-member graph, us | Traffic-only fused graph, us | Source program, us |
| --- | ---: | ---: | ---: |
| 1, 768 | 0.001945636 | 0.001825284 | 0.170951273 |
| 8192, 768 | 15.938653091 | 14.952727273 | 59.645998545 |

These are predicted device-graph times, not measured accuracy. The source
program changes both ownership and arithmetic; no rate was fitted. The retained
single-row raw population contains 6,898 events, all with `BlockNum=3`,
Task Duration mean 65.134110 us and p50 64.697 us. Conditional on the selected
source and scalar branch, the new prediction remains 99.7375% below that mean.
The 8,192-row population has 368 events, all `BlockNum=64`, mean 147.124446 us
and p50 79.7395 us; source time is 59.459% below the mean and 25.199% below p50.
Whole-capture populations are not silently assigned to individual model steps.

The raw single-row population also exports mean `aiv_time=3.824011 us`,
`aiv_vec_time=0.650913 us`, `aiv_scalar_time=1.874164 us`,
`aiv_mte2_time=1.046363 us` and `aiv_mte3_time=0.413234 us`.
TaskWait is zero on all these rows. These fields are diagnostic evidence,
retained separately. The follow-up source review below establishes distinct
exporter calculation paths, while the historical capture's selected path remains
unknown. Pipe fields are not additive, they do not replace Task Duration, and
Task Duration minus AIV time is not labeled a measured dispatch latency.

All five actual graph probes finish in 0.302 seconds with 53.891 MiB peak RSS
on the local macOS process. The 8,192-row case peaks at 1,092 live tasks; every
probe returns zero live tasks, programs and allocations. Focused controls cover
1/65, 1/257 and 130/768 tails/owners, exact full-block source arithmetic,
late input readiness, and the two-tile single-scratch counterexample.
The affected suite initially returned 29 passed and seven companion-profiler
dependency skips. Re-running both dependent files with the existing profiler
development source returned ten passed, completing all 36 distinct cases.

The existing nine named A5 empty/tiny invocation classes leave this kernel
unchanged. A separate synthetic 10 us vector minimum is charged once at the
outer kernel boundary, even with multiple owners/phases. Adding a declared
7 us host submission yields 17 us. These synthetic controls verify boundary
arithmetic; they are not SituMxQuant calibration.

Reproduce the runtime report:

```sh
PYTHONPATH=. .venv/bin/python tools/verify_situ_mx.py \
  examples/cases/kimi-k3-a5-current-checkpoint-conditional.json \
  -o data/verification/situ-mx-mechanism-20261030.json
```

Evidence:

- `data/verification/situ-mx-mechanism-20261030.json`: actual graph controls,
  case SHA and package file hashes.
- `data/verification/situ-mx-native-boundary-20261029.json`: original Task
  Duration and PMU field distributions; capture SHA
  `ab1d44d3613ecb92d71af6d790ab9dfc72be5aabfe92bf4f99d584fbd0f9e648`.
- `tools/audit_situ_mx.py`: bounded streaming audit using the existing profiler
  decoder. The report was retained from the original 8.120-second scan;
  integration did not rescan the 110 MB file.

### Follow-up PMU definition evidence

The companion profiler review was committed as `ddbdbd4`. Its files are
`../profiler/examples/structure/acceptance-20261029/situ-mx-pmu-boundary.json`
(SHA-256 `f8d6e11fca93e469913a4fdfbf52906e0738764bcf215af48729df6832da0734`)
and the same basename `.md`
(`224b09abff32a8057466127e013f58ecd4d399500121ea5e94e3f73c8fd21dc5`).
This follow-up read their bodies and independently verified all eight retained
official-source file hashes against that JSON. It neither rescanned the capture
nor changed the predictions or observation denominator.

The official source is `Ascend/msprof` at immutable revision
`a11606183257bb954de14a6b5a64c44f4112ac9c`. This is the inspected source identity,
not an asserted identity for the historical collector or exporter.

| Pinned official source | Established meaning |
| --- | --- |
| `docs/en/user_guide/profile_data_file_references.md:394-395` | Task Duration includes device scheduling, accelerator execution and completion response. Task Wait is the gap after the previous task, so zero wait does not prove zero scheduling or response inside Task Duration. |
| `analysis/common_func/utils.py:175-181`, called from `analysis/mscalculate/stars/ffts_pmu_calculator.py:380-389` | The cycle path divides converted time by the task's block count, then multiplies by its wave count. A one-wave value is a block average, not a value divided by all available device cores. |
| `analysis/msconfig/data_calculator_config.py:90-95`, `analysis/csrc/application/database/db_constant.h:392`, `analysis/mscalculate/stars/chip6_pmu_calculator.py:438-472` | Chip 15 is Ascend950 and uses Chip6. Its non-MIX AIV path converts PMU `end_time-start_time` syscnt directly, without a block/core divisor. The MIX secondary-core path can use the cycle calculation. |
| `analysis/common_func/msvp_common.py:314-315` | Both calculation paths can be exported as `aiv_time`; the field name does not identify the algorithm. |
| `analysis/mscalculate/calculate_ai_core_data.py:60-82` | Individual pipe times are computed independently from counter ratios and total time; they are not an exclusive decomposition of Task Duration. |

The source can be inspected through the pinned repository tree:
`https://github.com/Ascend/msprof/tree/a11606183257bb954de14a6b5a64c44f4112ac9c`.
The companion JSON records exact immutable raw-file URLs, local source copies,
hashes and line ranges for all eight files.

The retained CSV has no exporter build identity, raw PMU syscnt endpoints,
per-block intervals or frequency inputs. Its task type and zero Mix Block Num
are compatible with the inspected non-MIX path but do not prove it was used.
Consequently, do not multiply the 3.824011 us AIV mean by 3 or 64 or derive a
startup parameter from its difference with 65.134110 us Task Duration.
One retained prefill row independently demonstrates non-additivity: the four
pipe fields sum to 60.054 us while its exported AIV time is 50.690 us.
The cause and size of scheduling, response, stalls or instrumentation effects
remain unquantified.

## Known limits

* No instruction schedule, scalar pipeline, barrier latency, queue depth beyond
  the two declared slots, cache residency or DMA transaction overhead.
* Vector lane counts are source-expression counts; they do not establish
  native instruction throughput or latency.
* The selected profile identifies inspected source semantics, not the loaded
  capture binary or its actual tiling key.
* `activate_left` changes operand selection only; its work is symmetric.
* The program covers the dense/shared `situ_quant` group shape above. Other
  SiTU contexts (grouped expert gate-up, input already quantized, externally
  visible BF16 activation) require their own source boundaries or are rejected.
