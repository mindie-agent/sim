# A5 Qwen GDN store and HCCL source controls

The four new Qwen3.6/Qwen3.8 TP2/TP4 source cases still fail the full 64-layer
Body comparison at B4, q4 and initial context 126000 after binding the retained
BF16 scalar-store primitive. The HCCL source candidates explain part of the
communication gap but do not establish a captured algorithm or a passing Body.
No new rate, fixed latency or Body-derived parameter is introduced.

The producers and evidence are under
`data/observations/a5-gdn-store-binding-20261030/`. All native populations use
the existing complete 64-layer extraction, including first-layer wait tails.
The representative geometry is the same available B4/q4 point in all four
deployments. The source case hashes read during the run are retained in
`validation.json`; the canonical Profiler templates had already acquired the
store binding during this work. The control therefore removes only that option
from its before case. The reproducible producer now spells that control as
`gating_beta_store={}` so later default inheritance cannot re-enable the queue.

## Existing beta store service

The only execution change in the first comparison is:

```json
"gating_beta_store": {
  "bytes_per_element": 2,
  "resource": "scalar_store_issue",
  "from_hardware": true
}
```

The declared 9582 profile supplies 5,657,458,563.536 stores/s over 64 owners,
eight outstanding slots, a 128-byte granule, 91.25 ns response and 7 ns row
control. These are retained primitive parameters. The new deployment's hardware
SKU and gating binary are not independently established by the delivered
artifacts. Gating/L2/norm helper shapes are N/A in the CSV; their role and
geometry are conditional on source order and adjacent shaped native calls.

| Model / TP | DMA control Body ms | Bound queue Body ms | Observed ms | Bound error |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 / 2 | 23.317 | 24.497 | 31.689 | -22.70% |
| Qwen3.6 / 4 | 21.776 | 22.442 | 28.964 | -22.52% |
| Qwen3.8 / 2 | 23.874 | 25.053 | 32.711 | -23.41% |
| Qwen3.8 / 4 | 22.024 | 22.690 | 29.240 | -22.40% |

The isolated gating duration changes from 1.312 us to 25.885 us for 24 local
heads, and to 15.176 us for 12 heads. The complete 48 GDN-layer sums remain
7.392/12.519, 5.579/10.431, 7.923/13.581 and 5.815/10.832 ms
(prediction/observation in table order).

## HCCL candidates and unresolved dispatch

Every selected complete Body has exactly 128 concrete `hcom_allReduce_`
events, two per layer. Their raw name is `AivKernel`, accelerator core is
`COMMUNICATION`, and Mix Block Num is zero. TP2 records Block Num 3; TP4
records 60. Duplicate HCOM summary rows are removed and scheduler/AICPU
controllers are excluded. The retained Body boundary is not shortened.

The semantic payload at this geometry is `16 * 5120 * 2 = 163840` bytes.
The HCCL rows have no tensor arguments, so this remains a source-derived
payload. The pinned legacy AIV selector chooses TwoShot at 128 KiB or above;
it does not establish the current cost selector or deployment overrides.
The existing c12 source control is explicitly OneShot/3 despite that threshold.

All declarations use the proper `runtime.kernels.hccl.allreduce` entry and
`axes=["tp"]`. Both candidates are retained; none is chosen because its Body
error is smaller.

| TP / candidate | Native blocks | Modeled launched owners | Per-call source us | Native steady GDN mean us, Q3.6 / Q3.8 |
| --- | ---: | ---: | ---: | ---: |
| 2 / OneShot | 3 | 3 | 14.140 | 19.176 / 20.864 |
| 2 / TwoShot | 3 | unsupported | — | 19.176 / 20.864 |
| 4 / OneShot | 60 | 5 | 24.348 | 25.643 / 24.703 |
| 4 / TwoShot | 60 | 60 | 5.959 | 25.643 / 24.703 |

TwoShot/3 raises `hccl_aiv_small_core_branch_not_selected` because Sim has not
implemented that source branch; its block count is not changed to make the
program run. The source itself has a legal small-core ReduceScatter/AllGather
path. OneShot with `block_limit=60` constructs only `R+1=5` source owners, so
its apparently close TP4 latency is not evidence of matching the recorded
60-block dispatch. The implemented source programs also leave flag/poll retries
and first-operator barriers unpriced.

| Model / TP | OneShot Body ms / error | TwoShot Body ms / error |
| --- | ---: | ---: |
| Qwen3.6 / 2 | 25.792 / -18.61% | unsupported at observed Block3 |
| Qwen3.6 / 4 | 24.096 / -16.81% | 21.742 / -24.93% |
| Qwen3.8 / 2 | 26.349 / -19.45% | unsupported at observed Block3 |
| Qwen3.8 / 4 | 24.343 / -16.75% | 21.990 / -24.80% |

All six executed candidates fail the 15% Body criterion. The run took 194.6 s
across six individually bounded probes, with peak RSS 89.9 MB and zero
remaining live tasks/programs. The beta-only and HCCL case copies are separate,
so the reports preserve their exact input identities.

## Current host-selector source audit

The subsequent read-only audit of HCCL
`c2d6790464654ccf58aca39448de4b65fdc7b184` distinguishes the current selector
from the legacy 128 KiB rule. For `MESH_1D`, `op_common.cc` sends both initial
selection and reselection to `SelectorEngine`, which generates candidate costs,
applies an optional tuner and selects the minimum. The legacy threshold is not
the active selection rule for this topology in that source revision.
[Selector entry](https://github.com/hicann/hccl/blob/c2d6790464654ccf58aca39448de4b65fdc7b184/src/ops/op_common/op_common.cc#L183),
[cost selection](https://github.com/hicann/hccl/blob/c2d6790464654ccf58aca39448de4b65fdc7b184/src/ops/op_common/selector/selector_engine.cc#L272).

With a single physical mesh level, both AIV candidates retained, source default
costs and no tuner, `HCCL_ALGO` preference or algorithm plugin override, the
ranking at `S=163840` bytes is:

| Ranks | OneShot host cost us | TwoShot host cost us | Preferred candidate / ordinary blocks |
| --- | ---: | ---: | --- |
| 2 | 6.661279 | 7.511743 | OneShot / 3 |
| 4 | 7.025368 | 4.846894 | TwoShot / 60 at core limit 60–64 |

These numbers are HCCL host-selector ranking estimates. They are not measured
kernel durations or inputs to Sim's physical rates, setup or primitive service.
The calculation uses the source equations, with seconds as the cost unit:

```text
u = 0.7135 / 0.85 * 0.65
B = 1 / 1400e9 + (R - 1) / 900e9
C_one = S * (1 / (56e9 * u) + B) + 1e-6
C_two = S * ((2 / R) / (56e9 * u) + B / R) + 2e-6
```

The executor supplies `dataRatio=1/R` and separate input/scratch buffer types;
the template coefficients, mesh utilization and fixed selector costs then give
the two expressions. The checked constants and exact line ranges are retained
in [the source audit](../../data/verification/hccl-host-selector-audit-20261030.json).
[Executor inputs](https://github.com/hicann/hccl/blob/c2d6790464654ccf58aca39448de4b65fdc7b184/src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_sole_executor.cc#L102),
[cost parameters](https://github.com/hicann/hccl/blob/c2d6790464654ccf58aca39448de4b65fdc7b184/src/ops/op_common/selector/cost_model.cc#L113),
[utilization](https://github.com/hicann/hccl/blob/c2d6790464654ccf58aca39448de4b65fdc7b184/src/ops/op_common/selector/cost_table.cc#L443).

Ordinary launch uses the host-computed block count directly. OneShot uses
`min(limit,R+1)`; TwoShot uses `floor(limit/(R+1))*(R+1)` when `limit>=R+1`,
otherwise `limit`. Thus TP2 OneShot naturally launches three blocks, while TP4
TwoShot launches 60 at limits 60–64. A legal TP2 TwoShot three-core branch also
exists at limits 3–5: owners 0/1 scatter and gather half the payload, and owner 2
reduces the local half in place. It is not the preferred candidate under the
conditions above, and no new Sim branch is added by this audit.
[Host block arithmetic](https://github.com/hicann/hccl/blob/c2d6790464654ccf58aca39448de4b65fdc7b184/src/ops/all_reduce/algorithm/template/aiv/aiv_temp_all_reduce_mesh_1D_twoshot.cc#L115),
[small-core entry](https://github.com/hicann/hccl/blob/c2d6790464654ccf58aca39448de4b65fdc7b184/src/ops/all_reduce/algorithm/template/aiv/kernel/aiv_all_reduce_mesh_1d_twoshot.h#L390).

The OneShot device program has only `R` payload producers and one consumer even
if an outer SuperKernel launches more physical cores; the remaining cores still
join synchronization. SuperKernel preserves the logical `numBlocks` argument,
so a physical launch count alone cannot identify its internal template. The
captured binary revision, selection overrides and outer launch mode remain
unknown. TP2 OneShot/3 and TP4 TwoShot/60 are therefore source-conditioned
declarations supported by the default selector and compatible native counts,
not captured dispatch proof or choices made from Body error.

## Remaining native roles

The complete first-layer native duration sum equals its device activity union
in these populations; other groups must not assume additive duration sums.

| Model / TP | Native sum / union us | Uncovered gap us | Largest five native groups, summed us per first layer |
| --- | ---: | ---: | --- |
| Qwen3.6 / 2 | 320.425 / 320.425 | 5.058 | HCOM 96.479; gating 37.872; gate-up 29.695; down 18.874; output projection 18.738 |
| Qwen3.6 / 4 | 424.886 / 424.886 | 4.897 | HCOM 257.808; gating 23.487; gate-up 18.838; down 11.575; output projection 11.268 |
| Qwen3.8 / 2 | 296.075 / 296.075 | 5.004 | HCOM 53.581; QKVZ 38.256; gating 34.286; gate-up 31.460; down 19.268 |
| Qwen3.8 / 4 | 321.899 / 321.899 | 5.005 | HCOM 142.275; gating 25.542; QKVZ 25.231; gate-up 19.297; output projection 11.656 |

`operator-residuals.json` also retains the stable 47 remaining GDN layers,
first CSV locators, complete sampled ABI fields and isolated Sim roles. For
example, Qwen3.6 TP2 predicts/observes 1.312/7.993 us for convolution,
13.356/16.369 us for recurrence, 25.885/33.032 us for gating,
1.312/3.909 us for norm/gate and 13.677/19.345 us for the output projection.
Qwen3.8 BF16 QKVZ has a larger remaining gap: 22.636/38.678 us at TP2 and
11.330/24.057 us at TP4. These are native-role diagnostics, not additive Body
corrections.

Matrix M/N/K and payload formats match the recorded selected roles. Qwen3.6
QKVZ/FFN use INT8 operands, BF16 column scales and FP32 row scales; beta/alpha
and GDN output projection remain BF16. Qwen3.8 GDN matrices remain BF16 while
FFN uses E4M3 inputs/weights and E8M0 block scales. Generic INT8 storage still
retains scale-plus-offset metadata, whereas the recorded ABI exposes scale
only; this conservative extra storage does not explain an underprediction.

The q4 convolution already reads/writes six history slots. TP2 declares and
services 696,320 read bytes and 409,600 write bytes, including the speculative
state reread/write; TP4 halves these payloads. FP32 recurrence reads the initial
state once and writes all 16 query checkpoints. TP2 HBM service is 31,853,568
bytes, of which 25,165,824 bytes are state checkpoint writes. Acceptance is not
used to reduce query work. TP4 halves the head-dependent payloads. The native
B+1 prefix vector versus the retained source's normalized-length ABI remains
an explicit interface limitation.

One native helper remains absent from the default stride1 declaration:
`Slice [256,4] -> [4,1]` before convolution. The post-recurrent
`Slice [16,64,128] -> [16,24,128]` (TP2) is consistent with materializing the
z head slice before norm/gate; a separate native `core_output_copy` event is
not identified, although the model includes one. The native FFN
`AddRmsNormDynamicQuant` is a fused boundary while the generic caller has
separate residual, norm and quantization work. These differences require
source/caller binding rather than a new per-layer delay.

```sh
python3 data/observations/a5-gdn-store-binding-20261030/produce.py
python3 data/observations/a5-gdn-store-binding-20261030/hccl_control.py
python3 data/observations/a5-gdn-store-binding-20261030/audit.py
```

The separate [frozen-response transfer](a5-hccl-frozen-transfer.md) reuses the
unchanged older TP2/TP4 native-call responses against new B1/B4/high-B captures,
preserves physical-domain misses and two bounded Body timeouts, and inventories
existing local response and norm-fusion caller bindings.
