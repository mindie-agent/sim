# MoE expert weighted combination: `moe_fused_mul_sum_kernel`

Source-derived isolated operator plan for the campaign task `moe_combine`.
This module describes a device kernel; it is **not** a serving-latency fit and
does **not** use historical timing rows.

## Selected implementation

| Field | Selection |
| --- | --- |
| Source function | `moe_fused_mul_sum_kernel` launched by the `moe_fused_mul_sum` wrapper |
| Recipe | `triton_moe_fused_mul_sum` |
| Expert map | **disabled** (`has_expert_map=False`); `topk_ids`/`expert_map` are never loaded |
| Router weight placement | weights applied by this kernel (`apply_router_weight_on_input=False`) |
| Device capability | explicit `compute_capability=89` |
| Input/output dtype | declared by the case; campaign cases use `bf16` I/O |
| Weight dtype | declared by the case; campaign cases use `fp32` weights |
| Accumulator | `fp32` |
| Kernel launches in wrapper | 1 (`moe_fused_mul_sum` calls the fused kernel once) |

The selected source is the **current vLLM v0.25.1 checkout**, not the historical
capture image. The wrapper's `FakeTensor` guard (lines 177-200) skips the real
launch for fake tensors; the campaign models the concrete-tensor launch branch,
which is the only branch with device work. No `moe_fused_mul_sum` timing or
dispatch row was found in the explicit evidence paths (`data/verification`,
`data/observations`), so this report makes no capture-dispatch claim.

### Rejected nearby paths

These are different implementations and are rejected explicitly rather than
priced silently:

* `has_expert_map=True` selects the source's alternate load path (lines 42-49),
  which reads `topk_ids` and `expert_map`. It is not selected by the campaign.
* `apply_router_weight_on_input=True` matches the `TopKWeightAndReduceContiguous`
  contract (lines 107-108) that skips multiplication because weights were
  already applied. `moe_fused_mul_sum` always multiplies (line 56), so accepting
  this flag would double-apply weights.
* Compute capabilities other than 89 select different `BLOCK_M`/`BLOCK_K`
  ceilings in `_heuristic_config`. They are not the selected SM89 branch.
* `topk_weight_and_reduce.py` lines 80-121 use `mul_` followed by `ops.moe_sum`;
  that is a neighboring implementation, not the selected fused kernel.
* `TopKWeightAndReduceNoOP` (lines 44-77) is valid only when the producer has
  already applied weights and reduced; not this case.

The selected wrapper has exactly one kernel launch, so the module does not
export `kernels(op)`; the multi-launch expansion path is not needed and is not
emulated.

## Source identity

| File | SHA-256 | Bytes | Relevant lines |
| --- | --- | --- | --- |
| `/Users/maoxx241/code/vllm-v0.25.1/vllm/model_executor/layers/fused_moe/moe_fused_mul_sum.py` | `30fbb37b84b1cf6a8962d674abd9c3ab4d2be56d028fac82f55833c88d9729e8` | 5999 | kernel 10-63; program ids 25-26; masks 31-33; weight load 41; A load 50-55; FMA 56; store 58-63; heuristic 66-132; wrapper 135-202; launch 177-200 |
| `/Users/maoxx241/code/vllm-v0.25.1/vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py` | `6a269547fc2bddb5e1784676f78d63a1b3e796699a2c9bfe2c8e446e95fd3425` | 6018 | Contiguous 80-121; apply flag 107-108 |

Both entries match `tests/fixtures/operator_sources/moe_combine.json`. Source origin is
recorded in `OpSpec.metadata["source_origin"]` as
`current_checkout_not_historical_capture_image`.

## Tile heuristic (SM89)

For `element_size` of the **input** tensor:

```text
is_fp32 = element_size > 2
BLOCK_K = max(256, min(next_power_of_2(hidden), 1024 if bf16/fp16 else 256))

bf16/fp16 BLOCK_M:
    tokens <= 4   -> 1
    tokens <= 32  -> 2
    tokens <= 128 -> 4
    tokens <= 1024 -> 16
    otherwise     -> 8
fp32 BLOCK_M:
    tokens <= 4   -> 1
    tokens <= 32  -> 2
    otherwise     -> 4

total = BLOCK_M * BLOCK_K
num_warps = max(4, min(16, total // 256))     # fp32: max(8, min(16, total // 64))
num_stages = 4 if total <= 2048 else 2
grid = (cdiv(hidden, BLOCK_K), cdiv(tokens, BLOCK_M))
```

`top_k` is accepted by the source heuristic signature but does not affect tile
selection. All thresholds are exercised in
`tests/operators/test_moe_combine.py`.

Campaign case selections:

| Case | Split | tokens | hidden | topk | BLOCK_M | BLOCK_K | grid | warps | stages | padding (rows/cols) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | development | 1 | 128 | 2 | 1 | 256 | (1, 1) | 4 | 4 | 0 / 128 |
| 1 | development | 3 | 256 | 4 | 1 | 256 | (1, 3) | 4 | 4 | 0 / 0 |
| 2 | development | 16 | 512 | 8 | 2 | 512 | (1, 8) | 4 | 4 | 0 / 0 |
| 3 | heldout_geometry | 5 | 130 | 3 | 2 | 256 | (1, 3) | 4 | 4 | 1 / 126 |
| 4 | heldout_geometry | 33 | 1024 | 2 | 4 | 1024 | (1, 9) | 16 | 2 | 3 / 0 |

All five cases are valid for the selected branch (positive geometry, supported
dtypes, `has_expert_map=False`, `apply_router_weight_on_input=False`, explicit
capability 89). No case is skipped or replaced by a nearby implementation.
`hidden=130` is not a power of two, but the source's `BLOCK_K` uses
`next_power_of_2(130)=256`; that is in-envelope and is the real masked tail
tile.

## Physical work

Let `M=tokens`, `T=topk`, `K=hidden`, `BM`, `BK`, `in_b`, `w_b`, `out_b`.
The source launches one program per `(pid_k, pid_m)`. A program owns
`valid_m = min(BM, M - pid_m*BM)` rows and `valid_k = min(BK, K - pid_k*BK)`
columns. The mask is `m_mask & k_mask`.

### Tensor traffic (valid lanes only)

```text
A read          = T * M * K * in_b                 # each input element once
weight read     = cdiv(K, BK) * M * T * w_b        # re-read once per K tile
output write    = M * K * out_b                    # masked tail stores
hbm_bytes       = A read + weight read + output write
mte2_bytes      = A read + weight read
mte3_bytes      = output write
link_bytes      = 0                                # no routing or communication
```

Per program, the weight load is `T * valid_m` values; it is broadcast across
`BK` columns and **not** reloaded per lane. The same weight value is loaded
again by every K tile because each CTA owns one `(pid_k, pid_m)` tile.

### Arithmetic (full tile, including masked-off lanes)

The kernel initializes an FP32 accumulator and executes `static_range(top_k)`
iterations. Each iteration performs a multiply and an add on every tile lane;
masked lanes still execute because they were loaded as `0.0`. The ledger is:

```text
per program =
    BM*BK * 1                                   # acc = zeros
  + BM*BK * T * (2 + [in_b < 4])                # multiply, add, A upconvert
  + BM*T * [w_b < 4]                            # b_val upconvert
  + BM*BK * [out_b < 4]                         # final output downconvert
vector_ops = grid_programs * per_program
flops = 0                                       # no matrix/tensor-core work
```

The FP32 accumulation is represented by the FP32 multiply/add lane operations
and the declared `accumulation_dtype=fp32`; no cube-rate multiplier is applied.
Predicate/mask evaluation, address arithmetic and instruction issue are not
invented as fitted setup time; they are listed as unresolved primitives.

### Masked arithmetic and tail example

For `tokens=5, hidden=130, topk=3, bf16/fp32` (`BM=2`, `BK=256`, grid `(1,3)`):

```text
per-program vector work = 512 * (1 + 3*(2+1) + 1) = 5632
total vector work       = 3 * 5632 = 16896
valid-only arithmetic   = 5 * 3 * 130 * 2 = 3900
loads                   = 1584, 1584, 792 bytes
stores                  = 520, 520, 260 bytes
```

The third CTA computes all 512 lanes but stores only one valid row and 130
valid columns.

## Per-SM CTA ownership

The source grid is a set of independent CTAs. It is **not** one global
aggregate pipeline. The repair in this revision:

1. Enumerates the exact source grid `(pid_k, pid_m)` and linearizes it as
   `pid_m * k_tiles + pid_k` (Triton's first grid dimension varies fastest).
2. Creates `min(grid_programs, geometry.compute_units)` lazy CTA streams. A
   stream is one `PipelineProgram` with one resident tile (`slots=1`); a new
   CTA is issued only when the prior CTA on that owner retires.
3. Assigns global CTA index `i` to owner `i % active_owners`, so every source
   CTA runs exactly once and the final partial wave has the real tail.
4. Pins each CTA's `fma` stage to that owner's `vector` units and, when the
   backend declares split copy engines, its `load`/`store` stages to that
   owner's `mte2`/`mte3` units via `context.owned(owner, compute_units, roles)`.
   Global HBM service remains shared and unpinned.

Idle SMs therefore cannot supply a CTA the source assigned to one SM. A
`grid=(1,1)` kernel is one owner's CTA at every device width.

### Counterexample: grid(1,1) idle-SM speedup

Case 0 (`tokens=1, hidden=128, topk=2`) has `BLOCK_M=1`, `BLOCK_K=256` and one
grid program with 2048 vector lane operations. On a synthetic device with one
vector unit per SM at a fixed `1e9` vector ops/s and nonbinding memory:

| Device | CTA owners | CTA vector rate | Duration |
| --- | ---: | ---: | ---: |
| 1 SM | 1 | `1e9` ops/s | `0.002048` ms |
| 32 SM | 1 (31 idle) | `1e9` ops/s | `0.002048` ms |

The previous global unpinned pipeline gave `~0.000064` ms at 32 SM because all
32 SMs shared the single CTA's work. That is a 32x error and is rejected by
`test_singleton_grid_is_owned_by_one_sm_at_every_device_width`. A contrasting
oracle with 64 full CTAs shows 1 SM vs 32 SM scaling by exactly 32 when there
are really 32 CTAs to run. The coordinator's synthetic control hardware is a
different rate configuration and is used only for structural conservation.

## Program and buffers

Each generated CTA tile has three dependent stages:

```text
load  (hbm shared + mte2 owned: A tile + top-k weights)
  -> fma (vector owned: FP32 accumulator work)
  -> store (hbm shared + mte3 owned: valid output lanes)
```

`build()` returns a `ParallelProgram` of at most
`min(grid, geometry.compute_units)` per-owner `PipelineProgram`s. Each stream
has `slots=1`, so at most one CTA per owner is resident and only that CTA's
three stages materialize. No list of tokens, rows or scalar events is created;
a `tokens=10^9` case exposes 125,000,000 grid CTAs with 32 streams without
enumerating them.

No on-chip or global workspace is allocated by the selected kernel. The FP32
accumulator and the loaded A/weight values are Triton SSA register values; the
output is the caller's tensor (allocated by `torch.empty` only when the caller
passes `outputs=None`, outside kernel service). Therefore the plan declares no
`Allocation`, no `Geometry.buffer_units` pool and no workspace round-trip.
`Allocation.units`/per-unit pools remain available for operators that do have
scratch storage; this kernel does not. The verification report shows empty
buffer pools and no live tasks. Register-tile capacity and CTA occupancy above
one resident CTA per owner remain unresolved primitives.

## Numerical oracle

`tests/operators/test_moe_combine.py` implements an independent NumPy
weighted-sum oracle:

```text
acc = 0 (fp32)
for n in range(topk):
    acc = acc + inputs[:, n, :] * weights[:, n, None]
output = bf16_round(acc)
```

The BF16 round-to-nearest-even helper is implemented from integer bit
operations, without torch or `ml_dtypes`. Results:

* For a `(5, 4, 130)` BF16-input/FP32-weight sample, the result passes
  `rtol=2^-7, atol=2^-8` against a float64 exact weighted sum; the maximum
  relative error for entries with `|exact| > 0.1` is `0.003687 < 2^-7`.
* A BF16 accumulator that rounds after every addition loses the low-order
  contributions: for one large term plus seven `0.001` terms it returns `1.0`,
  while the source-order FP32 accumulator returns `1.0078125`. This demonstrates
  the retained FP32 accumulation.
* A CPU address oracle covers every `(token, hidden)` input address once and
  every output address once; with two K tiles the same `(token, expert)` weight
  address is visited twice, exactly as the tile loop requires.
* An ownership oracle enumerates every generated CTA and checks that each grid
  index appears exactly once, is pinned to one owner, and carries the real tail
  dimensions. A bounded-execution test starts the actual program on the shared
  engine and checks conserved service, released state and a peak live-task
  count far below the CTA count.
* The oracle is a semantic reference only; the simulator does not execute
  tensor values.

## Executed checks

```sh
PYTHONPATH=. /Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python \
  tools/run_operator_cases.py \
  mindie_sim.prediction.operators.moe_combine \
  tests/fixtures/operator_cases/moe_combine.json \
  -o data/verification/operators/moe_combine.json
# {"status": "passed", "cases": 10, "output": "data/verification/operators/moe_combine.json"}

PYTHONPATH=. /Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python \
  -m pytest tests/operators/test_moe_combine.py -q
# 72 passed
```

The case runner expands the five campaign cases across unified-memory and
split-copy synthetic controls (10 rows). It checks finite completion, work
conservation, buffer release, identity invariance and the independently
declared dispatch fields. This is a structural contract only; it is not
accuracy acceptance and the synthetic rates are not hardware calibration.

Reported per-case declared work (split-copy additionally services `mte2`/`mte3`):

| Case | vector ops | hbm bytes | mte2 bytes | mte3 bytes |
| --- | ---: | ---: | ---: | ---: |
| 0 | 2048 | 776 | 520 | 256 |
| 1 | 10752 | 7728 | 6192 | 1536 |
| 2 | 212992 | 147968 | 131584 | 16384 |
| 3 | 16896 | 5260 | 3960 | 1300 |
| 4 | 294912 | 203016 | 135432 | 67584 |

## Limitations and unresolved primitives

* Only the explicit SM89 heuristic branch is modeled. SM90+ and pre-SM80
  branches are different source paths and are rejected, not approximated.
* Only the no-expert-map and `apply_router_weight_on_input=False` paths are
  modeled. The source's other branches are rejected.
* Predicate operations, address arithmetic, instruction issue, memory response,
  L1/weight cache reuse and compiled Triton scheduling are not collapsed into a
  fitted parameter. They remain unresolved primitives in
  `OpSpec.metadata["unresolved_primitives"]`.
* CTA occupancy above one resident CTA per SM is not modeled. The plan declares
  `resident_ctas_per_owner=1`; this is an occupancy assumption, not a latency
  floor, and every source CTA still executes exactly once.
* A device whose declared vector/copy unit count cannot be partitioned equally
  across `compute_units` owners is rejected with
  `moe_combine_ownership_geometry_incompatible`. No approximate ownership
  mapping is invented.
* No register-file capacity is declared because the source exposes no register
  or scratch-pool contract. `Allocation`/`Geometry.buffer_units` are therefore
  not used for this kernel.
* `Weight`/`topk_weights` layouts are assumed contiguous, matching the launcher
  asserts at lines 164-165. The campaign cases provide no non-contiguous layout.
* The source supports `fp16`, `bf16` and `fp32` I/O with `fp16`/`bf16`/`fp32`
  weights; the campaign exercises `bf16` I/O with `fp32` weights. Additional
  dtype branches are implemented from the same kernel body and covered by
  focused work oracles, but they have no campaign case.
* No historical capture exists in the provided evidence paths for this kernel,
  so no dispatch match or latency accuracy is claimed.

## Integration

The module is not auto-discovered. The coordinator should bind:

```python
from mindie_sim.prediction.operators import moe_combine
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.program import RecipeBook

recipes = RecipeBook(recipes={
    moe_combine.RECIPE: PlannedRecipe(moe_combine.build),
})
```

`make_op(params)` is the explicit campaign geometry adapter; production
lowering should construct the equivalent `OpSpec` with the same metadata
`implementation`, `dispatch` and operand shapes, then use `build`. No shared
engine, schema, registry or case-file change is required by this task.
