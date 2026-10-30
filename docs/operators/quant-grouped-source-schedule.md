# Quantized grouped source geometry and schedule

This adapter exposes the owner/group/tile ordering of one pinned arch35 INT8
grouped-matmul kernel branch. It is pure geometry: it does not price work, run
the Engine, reconstruct host tiling or change default dispatch. Importing
`mindie_sim.prediction.operators.quant_grouped_schedule` is the explicit
opt-in identity `arch35_gqmm_mix_nz`.

## Source identity

Source checkout: `/Users/maoxx241/code/refs/ops-transformer`, revision
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8`.

| File | sha256 | Role |
| --- | --- | --- |
| `gmm/grouped_matmul/op_kernel/grouped_matmul_apt.cpp` | `320af4da…def0175` | lines 334-348 select `V310_GMM_QUANT_MIX`, `GMM_DEQUANT_VECTOR`, no transpose, `GQmmMixRegbaseKernel` |
| `…/quant_adaptive_sliding_window_templates/quant_block_sch.h` | `55196716…cc3f695e` | `UpdateGroupOffset`, `UpdateGroupParams`, `UpdateBasicIndex`, `UpdateTailTile`, `UpdateBlockParams`, `CalcGMOffset`, NZ size |
| `…/quant_adaptive_sliding_window_templates/gqmm_mix_online_dynamic.h` | `a2bd4d5b…72b8465` | `Init`, `SetMNK`, `CalcTailTile`, `IsLastGroupAndNeedSplit`, `IsLastGroupAndRound`, `Process`, `ProcessSingleGroup` |
| `…/quant_adaptive_sliding_window_templates/quant_utils.h` | `3d58d63f…1eeae952` | split-M/group-list constants, `CUBE_BLOCK`, NZ block constants |
| `gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp` | `68714d9b…3680ad6` | source context only; host tiling is **not** recreated |

The full hashes and byte lengths are in the module and in `.summary()["source"]`.
The historical loaded binary and its actual tiling are unknown. Producer
timings cannot identify this branch.

## Supported source domain

The adapter accepts only the following declaration:

- INT8 activation, ND, no transpose.
- INT8 weight, `FRACTAL_NZ`, no transpose.
- FP32 per-row activation scales and FP32 per-channel weight scales.
- BF16 output.
- No bias, no activation epilogue, no split-K.
- `group_type=SPLIT_M`, dense `groupListType=1` direct counts.
- Single X, W and Y; `n` is the same for every original group.
- Positive `k` multiple of 16 and positive `n` multiple of 32. Whole NZ block
  extents are required.
- `expert_rows` is an explicit nonnegative integer tuple. Every original group
  index is preserved, including empty groups; rows need not be uniform.

A caller supplies `QuantGroupedTiling(base_m, base_n, base_k, used_cores)`.
`base_m` is a positive multiple of 16, `base_n` a positive multiple of 32 and
`base_k` a positive multiple of 32. `base_k` is retained as a declared input
only; scheduling is independent of it.

## Stable API and units

```python
tile = SourceMatrixTile(
    group, ordinal, m_offset, n_offset, m, n, k,
    x_offset, weight_offset, output_offset,
    row_scale_offset, column_scale_offset,
)
```

- `group` is the original expert/group index. Empty groups are never compacted.
- `ordinal` is monotonic zero-based within one owner across the whole
  invocation, not within a group.
- `m_offset`/`n_offset` are element offsets **within the original group**.
- `m`/`n` are actual tile extents; `k` is the full K.
- The last five fields are **global byte offsets** into the respective whole
  backing tensors: INT8 X, INT8 NZ W, BF16 Y, FP32 row scale, FP32 column
  scale.

`plan_quant_grouped(expert_rows, n, k, tiling)` returns a
`QuantGroupedSchedule`. Its public `expert_rows`, `n`, `k`, `tiling` and
`total_m` are declarations. `owner_tile_counts` is a tuple with one exact count
per declared owner, including zero for idle owners. It is computed from group
rounds and the final split endpoint without traversing all output tiles.
`iter_owner_tiles(owner)` is a lazy generator in
source group then round then block order; it does not build a tile list.
`iter_weight_spans(tile, k_offset=0, k_size=None)` yields physical NZ ranges
for a whole-block rectangle or a K/16-aligned subpanel. These ranges are
storage identities only, never claimed DMA requests, cache lines or traffic.

## Scheduling transcription

For every nonempty group with `m` rows:

- `mCnt = ceil(m/base_m)`, `nCnt = ceil(n/base_n)`,
  `totalCnt = mCnt*nCnt`.
- `mCoreNum = min(4, mCnt)`, `mainRow = mCnt/mCoreNum - 1`,
  `mTailCoreNum = mCnt - mCoreNum*mainRow`.
- The group start is `0` after the previous group ended at `used_cores-1`,
  otherwise `previous_end+1`. `end = (totalCnt+start-1) % used_cores`.
- Base rounds are `ceil(totalCnt/used_cores)`. Owners outside the cyclic
  active window get one fewer round; a partial final round belongs only to the
  active window.
- `UpdateBasicIndex` maps each owner/round to a base M/N tile. Its four-row
  window is serpentine: odd `rowIdx` reverses `nIndex`.

A zero-row group updates the source offset accumulators but is skipped before
`UpdateGroupParams`. It consumes no tile and does not rotate the start owner.
Weight and column-scale offsets therefore still advance by one original bank
per group; X/Y/row-scale offsets advance by that group's row count (zero).

### Final-group tail split

`IsLastGroupAndNeedSplit` is evaluated only for `groupIdx == original_groupNum-1`
and only after that original last group has positive M. A trailing zero group
is skipped before this point, so it disables the split of an earlier nonempty
expert. This counterexample is tested directly.

`CalcTailTile` uses the pinned source `mMin=16` and `nMin=128`, the
`remainTile = used_cores/(end+1)` loop and the source reduction loop. It is
applied only when `end+1 <= used_cores/2`. `UpdateTailTile` extends the end to
`mTailTile*nTailTile*(end+1)-1`, gives the newly covered owners one extra
round, and the last round reuses the source split index arithmetic. Split M
widths are `ceil(singleCoreM/mTailTile)`; split N widths are aligned to 128.
This scheduling granule also applies to the pinned non-transposed NZ branch.
The 32-element NZ storage block is a different constraint and cannot replace
the source split granule.

An empty actual split rectangle produces no tile, exactly matching the source
`singleCoreM <= 0 || singleCoreN <= 0` early return; the owner's ordinal does
not advance.

### Remaining source differences

**Rotated-start final split.** `UpdateBasicIndex<true>` computes
   `newBlockIdx = blockIdx/(mTailTile*nTailTile)` and then subtracts
`startBlockIdx`. Direct standalone C++ evaluation of the pinned scheduling
functions with `expert_rows=[1,1]`, N=256, base M/N=16/256 and eight owners
produces the same final-group N=[128,256) rectangle on owners 1 and 3.
This is a source-geometry counterexample, not a hardware result. The adapter
conservatively rejects a real final split with a rotated start using
`unsupported_last_group_split_with_rotated_start`, rather than inventing a
different ordering under the source name. The same geometry with N=128 has
no actual split and remains supported.

Independent acceptance corrected the initial 32-element split specialization
to the source's 128-element rule. Source support is established by the pinned
branch, not by the earlier development prompt.

Everything outside the supported domain (transpose, bias, activation, split-K,
ND weight, FP8/FP4, etc.) must be selected by another implementation; this
adapter has no fallback.

## NZ storage and offsets

For no transpose, the weight of original group `g` is stored as
`[N/32, K/16, 16, 32]`; across groups the whole tensor is
`[E, N/32, K/16, 16, 32]`. With `k` multiple 16 and `n` multiple 32 there is no
padding. A tile's full-K, N-block-aligned rectangle is one contiguous physical
interval:

```text
weight_offset = g*N*K + n_offset*K          (bytes, INT8)
length        = n*K
```

Other global byte offsets follow the source linear group bases:

```text
x_offset            = prefix_m[g]*K + m_offset*K
output_offset       = (prefix_m[g]*N + m_offset*N + n_offset)*2
row_scale_offset    = (prefix_m[g] + m_offset)*4
column_scale_offset = (g*N + n_offset)*4
```

A K/16-aligned subpanel retains the holes between consecutive 32-wide N blocks:

```text
offset = g*N*K + (n_offset/32 + n_block)*K*32 + k_offset*32
length = k_size*32
```

Full-K rectangles coalesce into one contiguous range. Subpanels produce one
range per N block lazily; no cache, transfer or padding multiplier is inferred.
Sub-K/16 and non-N/32 slices are outside this storage-domain declaration.

## Summary and reproduction

`.summary()` is JSON-serializable and contains the source revision/file hashes,
implementation id, support domain, declared tiling, group/row counts, split
geometry, units and omitted assumptions. It never enumerates tiles.

The reproducible literal geometry is
`examples/calls/quantized-grouped-source-geometry.json`: 64 rows, N=64, K=16,
base M/N/K 32/32/32, eight owners. Its final M tail is split across two owners
per base tile. This is an adapter example, not a CLI call declaration or a
claimed full-call integration.

Focused command:

```sh
PYTHONPATH=. /Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python \
  -m pytest -q tests/test_quant_grouped_schedule.py
```

Tests cover literal four-row serpentine order, rotation across empty groups,
the trailing-zero split counterexample, final M/N tail split and an empty split
owner, rectangular cover/work conservation, explicit tiny NZ storage
enumeration and spans, base-K ordering independence, and lazy consumption of a
huge tile grid.

Independent acceptance on 2026-10-29 passed 54 focused schedule, complete-call
and shared-epilogue tests. A separate standalone oracle compiled the pinned
`UpdateGroupParams`, `UpdateBasicIndex`, `UpdateTailTile` and
`UpdateBlockParams` C++ bodies with a small group-loop driver and the source
tail-selection constants. Of 410 declared geometries, all 337 in the supported
domain matched every emitted owner/group/rectangle and exact owner tile count;
73 triggered the explicit rotated-start split restriction. The ten retained
historical balanced geometries were included and all were supported. These
are source-geometry checks, not timing or hardware accuracy measurements.

## Omitted assumptions

No latency, resource rate, floor, multiplier, fit, cache/L2 residency, full-bank
retention, active traffic, scalar group-loop cost, sync/startup/replay cost,
MatmulImpl internal transaction, fractal padding, host tiling inference,
historical binary identity or native dispatch is present. Bulk-bank residency
and active-expert traffic remain distinct concerns for the root integration.

## Complete-call integration and retained shape scenarios

The separate public builder `quantized_grouped_source` now consumes this
descriptor through the common matrix/refill Engine. A call explicitly declares
`arguments.tiling`; neither the historical binder nor ordinary dispatch selects
this source branch automatically. The integration adds INT32 Fixpipe publication
to two vector-local buffers, the source split-M convention, per-partition scale
loads and useful BF16 stores. Each owner's next output tile waits for its local
output completion. The inner matrix micro mapping and ideal shared HBM reuse
remain analytical declarations, not recovered native instructions.

On 2026-10-29, ten retained uniform-expert shapes were evaluated through 20 real
public `call` executions, once with `quantized_grouped` and once with the explicit
source builder. All source cases declared M/N/K tiles 16/128/128 and 32 matrix
owners. This tiling is a reproducible scenario, not inferred from event duration
or recovered host dispatch.

| Total rows | Active experts | K | N | Analytical device graph (us) | Source scenario device graph (us) | Original graph1 event median (us) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 8 | 2048 | 256 | 2.111 | 2.123 | 40.576 |
| 128 | 8 | 2048 | 256 | 2.357 | 2.388 | 40.499 |
| 32 | 32 | 2048 | 256 | 4.266 | 4.331 | 43.451 |
| 128 | 32 | 2048 | 256 | 4.340 | 4.413 | 43.814 |
| 256 | 256 | 2048 | 256 | 33.827 | 34.644 | 92.419 |
| 32 | 8 | 128 | 2048 | 0.630 | 0.831 | 39.828 |
| 128 | 8 | 128 | 2048 | 0.779 | 1.187 | 39.611 |
| 32 | 32 | 128 | 2048 | 2.240 | 3.057 | 45.317 |
| 128 | 32 | 128 | 2048 | 2.353 | 3.321 | 45.569 |
| 256 | 256 | 128 | 2048 | 17.615 | 24.449 | 103.880 |

The explicit schedule/output path changes the K=2048 family by +0.55% to +2.41%
and the wider-output K=128 family by +31.95% to +52.31%. It does not explain the
remaining event gap. Scalar group scans, address/control issue and cross-core
handshake service are still unresolved, as are native dispatch/tiling identity,
cache transactions and the device-graph/event boundary difference. The table
does not establish operator accuracy, whole-model accuracy or a service-time
improvement. No rate, calibration parameter, raw observation or default binder
was changed.

The compact evidence is
`data/verification/source-grouped-scenarios-20261029.json`. It preserves all ten
original identities, canaries and 400 `unassigned` graph1 samples;
`strict_pairs=0`. Graph8 samples were neither divided nor relabeled. Runtime
source hashes were stable across the run. Every invocation conserved FLOPs/HBM
work and released all tasks, programs and allocations. Seven physical resource
totals per source condition were also checked against independent closed-form
arithmetic over uniform expert rows and N128 rectangles.

Separate real public-call checks covered one-row, odd-row, multi-K, baseM192 and
last-group-tail cases with input readiness at 7 ms. None served a resource before
input readiness. Three independently calculated HBM/MTE1/AIC/AIV/Fixpipe/store/
vector ledgers matched; all five calls released their live state. These checks
establish the modeled dependency and resource contracts, not native timing.
