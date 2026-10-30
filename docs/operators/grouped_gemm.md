# Grouped expert GEMM (`moe_grouped_matmul`, BF16 ND `TILING_KEY=10`)

This report covers the selected non-transposed BF16 ND no-quant branch of
`moe_grouped_matmul` (`GMM_CUBE_IMP(false)`). The implementation is
`mindie_sim/prediction/operators/grouped_gemm.py`; independent checks are in
`tests/operators/test_grouped_gemm.py`; the coordinator control report is
`data/verification/operators/grouped_gemm.json`.

The result is a **source-derived structural model**, not an accuracy claim
against a device capture. The coordinator owns integration and final acceptance.

## 1. Selection and dispatch

`make_op` rejects every other branch instead of selecting a nearby
implementation. The selected flags are retained in
`OpSpec.metadata["dispatch"]`:

| flag | selected value | source evidence |
|---|---|---|
| `implementation` | `ascendc_grouped_nd_bf16` | `moe_grouped_matmul.cpp:37-38` |
| `tiling_key` | `10` | `TILING_KEY_IS(10UL)`, `.cpp:37` |
| `tiling_key_is` | `TILING_KEY_IS(10UL)` | `.cpp:37` |
| `transpose_weight` | `false` | `GMM_CUBE_IMP(false)`, `.cpp:38`; `SetTilingKey`, host `271-277` |
| `transpose_template` | `false` | `transposeW = transWeight`, `.h:65`; `SetTensorB(..., transposeW)`, `.h:159` |
| `weight_layout` | `ND` | `FORMAT_WEIGHT != FORMAT_FRACTAL_NZ` default, `.cpp:14-18`; `CubeFormat::ND` at host `144` |
| `weight_format` | `ND` | same |
| `format_weight` | `CubeFormat::ND` | same |
| `template` | `KernelMoeGMMNoQuant<T,T2,CubeFormat::ND,false>` | `.cpp:28`, `.h:41-50` |
| `dtype` | `bf16` | host `185-192` accepts BF16/FP16; the selected path is BF16 |
| `quantization` | `none` | `KernelMoeGMMNoQuant`, `.h:41-42` |
| `kernel` | `moe_grouped_matmul` | `.cpp:33` |
| `group_list_mode` | `key_value_2d` | 2-D pairs, `.h:91-100`; semantic adapter in Section 4 |
| `input_adapter` | `semantic_histogram_compacted_preserve_expert_ids` | Section 4.1 |
| `raw_group_list_terminator` | `first_row_count_nonpositive_breaks_remaining_list` | `.h:95-100` |

Rejected inputs include `transpose_weight=true` (`TILING_KEY=11`), layout `NZ`,
non-BF16 activation dtype, quantized parameters, an unsupported implementation
name, a non-positive `n`/`k`, negative histogram counts, duplicate histogram
expert ids, out-of-range expert ids, a total row count of zero, and an
`x_rows` value that differs from the histogram sum. The last check is the
"No missing rows accepted" boundary.

The op recipe is `grouped_expert_gemm`; callers bind
`PlannedRecipe(module.build)` explicitly. There is no auto-discovery.

## 2. Source identity and evidence scope

All five supplied files are the **current checkout** at
`/Users/maoxx241/code/vllm-ascend`, not a historical capture image. Their
SHA256 values match both the retained `tests/fixtures/operator_sources/grouped_gemm.json`
and the wave4 campaign manifest at
`tests/fixtures/operator_sources/grouped_gemm.json`
exactly (the two manifests list the same five source files):

| file | SHA256 | bytes |
|---|---|---|
| `.../op_kernel/moe_grouped_matmul.cpp` | `7a6f14a3078351c8297915a9d452364a737fe6e41b15d562d169a22d2848214e` | 2443 |
| `.../op_kernel/moe_grouped_matmul.h` | `dfd18924241fa153f3b42c85e07a29733f575de95bb236c297a7a106ce35f628` | 7861 |
| `.../op_kernel/moe_grouped_matmul_tiling_data.h` | `4f23e438d69fa69bfdfddc82fef893f4886ef5be072bcf6f3c74199047bdd6cf` | 411 |
| `.../op_host/moe_grouped_matmul_tiling.h` | `ef061fc0afc726100a5b9ded5f4e648ea4a6d5674dd053e7941e3c19b01444c3` | 1267 |
| `.../op_host/moe_grouped_matmul_cpu.cpp` | `5953db15ff725024880f90982b531385b0187eed484c8b68a8c5d601bd792508` | 14500 |

The source identity is recorded in every `OpSpec.metadata["source_identity"]`.
No grouped-GEMM capture row was found in the supplied read-only evidence.
**No capture dispatch match or capture timing is claimed.** The only structure
compared here is the supplied source and the coordinator's synthetic control
hardware.

Review input for this repair: the `grouped_gemm` task section of
`/Users/maoxx241/code/sim-rewrite-20261027/data/verification/operators-wave2-review.json`
(finding `GMM-HISTOGRAM-BOUNDARY`). The review finding cites additional
`op_host/op_api/*.cpp` files. Those files are outside the five-file source set
and hash receipt above; this report does not claim to have read or modeled them.
The review's approved adapter behavior is implemented from the supplied kernel
loop plus the finding's counterexamples.

The wave4 repair input is finding `GMM-PHYSICAL-PARTITION-MAPPING` from
`/tmp/review-wave3-grouped-gemm-partitions-20261028.py` and its receipt
`/tmp/review-wave3-grouped-gemm-partitions-20261028.json`.  It changes only the
physical resource/buffer partition mapping described in Sections 5.1 and 6.3.
The semantic histogram compaction and the separate raw nonpositive-terminator
interpreter from Section 4 are preserved unchanged.  The receipt's old
0.016512 ms result at 16 copy units is reproduced before the repair and the
required 0.033024 ms bound is reproduced after it; Section 7 records the
independent controls.

Line map used by the model:

* `.cpp:14-18` -- ND weight format default.
* `.cpp:21-31` -- `GMM_CUBE_IMP`: AIV early return, tiling struct, kernel type.
* `.cpp:33-41` -- `TILING_KEY_IS(10UL)` selects `GMM_CUBE_IMP(false)`; 11 selects true.
* `.h:41-50` -- `xType`, `weightType`, `yType`, `biasType`, `mmT`.
* `.h:71-87` -- list-tensor initialization; first x/weight/y pointers.
* `.h:89-123` -- group-list loop, `break`, prefix offsets, weight selection, tile rotation.
* `.h:126-135` -- `AlignUp` and `Ceil`.
* `.h:137-145` -- `SetMNConfig` (`m`, `k`, `n`, `baseM/N`, `singleM/N`).
* `.h:147-161` -- `MMCompute`: M/N tails, `SetOrgShape`, `SetSingleShape`,
  `SetTensorA/B`, `IterateAll`.
* `.h:163-171` -- weight pointer and L2-cache hint when `blockDimM == 1`.
* `.h:173-185` -- ND non-transposed weight offset `tail_n`.
* `tiling_data.h:9-18` -- tiling fields (`group_num`, `core_num`, `m`, `n`, `k`,
  `single_m`, `single_n`, `TCubeTiling`).
* `tiling.h:15-23` -- identical registered host tiling fields.
* host `22-29` -- constants (`BEST_BASEN=256`, `FP32_DATATYPE_SIZE=4`,
  `MAX_BASEM=256`, L1 part A `256 KiB`, double-buffer factors 2).
* host `87-111` -- `CalcStepKaKb`.
* host `134-183` -- `GMMSetMMTiling`; `set_dbL0C(1)`; `depthA1/B1`.
* host `185-211` -- `CalMMTiling` (`baseN`, `baseK`, `baseM`).
* host `213-269` -- `Init` (shape checks, `n`, `group_num`, `single_m/n`,
  `core_num`).
* host `271-277` -- `SetTilingKey`.
* host `279-287` -- `FillTilingData`.
* host `289-310` -- `RunKernelTiling`, `SetBlockDim(core_num)`, workspace.

## 3. Host tiling model

### 3.1 Fixed fields

`single_m = 128`, `single_n = 256`, `base_n = 256`
(host `256-257`, `186`). The tiling key is 10 for non-transposed
(host `271-277`).

### 3.2 Base K and base M

With BF16/F16 element size 2 and `FP32_DATATYPE_SIZE=4`
(host `187-199`):

```text
baseK      = align_down_16((l0b / 2) / (baseN * 2))
maxBaseM   = l0c / (baseN * 4)
baseM      = min((l0a / 2) / (baseK * 2), maxBaseM)
baseM      = align_up_16(m_)     if baseM > m_ else align_down_16(baseM)
baseM      = min(baseM, 256)
align_up_16(a)   = (a + 15) & ~15
align_down_16(a) = a & ~15
```

`m_` is the **total** x row count. With the histogram adapter this equals the
sum of every histogram bin, including empty bins (which add zero), and equals
the processed row count.

`baseK == 0` would make the host divide by zero at lines 196-199; the model
raises `host_tiling_base_k_zero`. `baseM == 0` returns `GRAPH_FAILED` at lines
206-209; the all-zero row case is rejected in `make_op` (Section 3.7).

### 3.3 L1 step sizes

With `L1_PARTA = 256 KiB` (host `22-27`, `87-111`):

```text
l1_a = 256 KiB          if baseM > baseN
     = l1 - 256 KiB     otherwise
l1_b = l1 - l1_a
stepKa = (l1_a / 2) / (baseM * baseK * 2)
stepKb = (l1_b / 2) / (baseN * baseK * 2)
if stepKa > stepKb: stepKa = stepKa // stepKb * stepKb
elif stepKa < stepKb: stepKb = stepKb // stepKa * stepKa
depthA1 = 2 * stepKa
depthB1 = 2 * stepKb
```

`l1 < 256 KiB` returns `GRAPH_FAILED` at host `87-92`; the model raises
`host_tiling_l1_below_256k`. Since `baseM <= 256 = baseN` after the cap, the
first branch is not reachable for this host, but it is implemented for
completeness.

### 3.4 Core count

```text
n_tiles  = ceil(n / single_n)
core_num = min(aic_num, m_ * n_tiles)
```

(host `259-264`, then `SetBlockDim(core_num_)` at `308`). `aic_num` is supplied
by the caller's `Geometry.compute_units`; the source does not declare a device
name. The kernel's `used_core_num = GetBlockNum()` (`.h:74`) is this
`core_num`. `m_ * n_tiles` is the host's `task_num`; it is not the actual number
of MMCompute tasks, which is `sum(ceil(rows_e/128)) * n_tiles` over nonempty
experts.

### 3.5 Worked table for declared capacities

The focused test supplies `l1=524288`, `l0a=65536`, `l0b=65536`, `l0c=262144`,
`ub=196608`, `aic_num=32`, `n=128`. This table is an independent recalculation
of the host formulas; it is not a hardware claim:

| total `m_` | `baseM` | `baseK` | `stepKa` | `stepKb` | `depthA1` | `depthB1` | `core_num` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 16 | 64 | 64 | 4 | 128 | 8 | 4 |
| 12 | 16 | 64 | 64 | 4 | 128 | 8 | 12 |
| 65 | 80 | 64 | 12 | 4 | 24 | 8 | 32 |
| 84 | 96 | 64 | 8 | 4 | 16 | 8 | 32 |
| 136 | 144 | 64 | 4 | 4 | 8 | 8 | 32 |

Inside the kernel, `GMMConfig.baseM/baseN` are assigned from
`tiling_.single_m/single_n` (`.h:141-144`), so the actual M/N task tiling is the
fixed 128/256; the host `baseM/baseK` values live in `mm_tiling` for
MatmulImpl. Note that `baseM` can exceed the kernel's `single_m` (144 > 128 for
`total_m=136`). The host builds `TCubeTiling` from the total `m_` (host
`147-150`), while the kernel calls `SetOrgShape(m, n, k)` and
`SetSingleShape(cur_single_m, cur_single_n, k)` per expert (`.h:156-157`).
How MatmulImpl reconciles those two views is not defined by the supplied
sources. The plan therefore uses the kernel-requested single shape for work and
does **not** invent fractal-padded MACs from `baseM/baseK`.

### 3.6 Per-unit panel residency

When both L0 and L1 capacities are declared, `host_tiling` also returns the
`allocation_bytes` used by the planner:

```text
l0a = baseM * baseK * 2
l0b = baseN * baseK * 2
l0c = baseM * baseN * 4
l1  = stepKa * baseM * baseK * 2 + stepKb * baseN * baseK * 2
```

These are one host-derived A/B panel and one L1 step-group residency. They fit
inside the declared per-unit capacities by construction: the L0A panel is at
most half of `l0a`, the L0B panel at most half of `l0b`, the L0C panel at most
all of `l0c`, and the L1 step groups at most half of `l1`. The exact MatmulImpl
double-buffer schedule, MTE1 fill and Fixpipe transfer are not declared and are
not invented.

### 3.7 Missing platform buffers and invalid total-zero input

`Geometry.buffers` is the engine's declared pool mapping (`l0a`, `l0b`, `l0c`,
`l1`, `ub`). When the L0 buffers are absent, `host_tiling` returns
`base_m=None`, `base_k=None`, `status="unresolved_l0_buffers"`; when only L1 is
absent, `status="l0_derived"`. It never guesses platform capacities. The
coordinator control hardware declares no buffers, so the control report uses
the fixed `single_m/single_n` and the computed `core_num`, while `baseM/baseK`
remain explicit unresolved fields and no per-unit allocation is claimed.

All-zero `expert_rows` are rejected: host `m_=0` makes `baseM` align to 0 and
`CalMMTiling` returns `GRAPH_FAILED` at lines 200-208. A leading zero is valid
and is compacted away; `[0,5,7]` processes 12 rows, not zero.

Other host preconditions select the same single-tensor branch: exactly one x,
one weight and one y list tensor (host `215-223`), 2-D x and y, and BF16/FP16
activation. The model assumes one contiguous x, one `[E,K,N]` weight and one
contiguous y; multiple list tensors are outside the selected branch.

## 4. Input adapter: semantic histogram versus literal raw list

### 4.1 Histogram compaction

The campaign input is a **semantic histogram**: `expert_rows[i]` is the number
of x rows assigned to expert `group_ids[i]` (default id `i`). It is not a
cumulative offset list and not the literal kernel group list. The adapter:

```text
active = ((expert_id, count) for each histogram bin with count > 0)
```

Empty bins are skipped, expert ids are preserved, and the x/y row offsets are
the prefix sum of preceding positive counts. This is the review-approved
`GMM-HISTOGRAM-BOUNDARY` repair: compaction happens **before** the kernel list,
so a zero histogram bin can never terminate later positive experts.

`make_op` also accepts an optional `weight_experts` to declare a weight bank
larger than the histogram length. The default is `len(expert_rows)`, matching
the dense histogram/weight-bank extent; an expert id beyond that extent is
rejected unless the caller explicitly declares the larger bank. The kernel list
is padded with zero-count pairs to exactly the declared extent, so the host's
`group_num = weight_shape[0]` and the kernel's `2*group_num` loop remain
well-formed. The first padding pair is the kernel terminator; later padding is
never read.

### 4.2 Literal raw list semantics (separate oracle)

The kernel reads 2-D `(expert_id, split_value)` pairs (`.h:91-100`). Its literal
semantics are:

```text
for each pair in list order:
    if split_value <= 0: break          # no later pair is read
    expert_id selects the weight bank
    split_value is a direct row count, not a cumulative end offset
    x/y row bases advance by the preceding positive direct counts
```

`interpret_raw_group_list` exposes exactly that behavior, including the
unprocessed suffix. It is deliberately separate from `compact_histogram`:

* `compact_histogram((1, 0, 3))` returns `((0,1),(2,3))` and processes 4 rows.
* `interpret_raw_group_list(((0,1),(1,0),(2,3)))` stops at the zero pair and
  processes 1 row, with `(1,0)` and `(2,3)` unprocessed.

A literal raw list that terminates early leaves x rows unwritten; that is a
valid raw-kernel test, not a valid semantic operator. `make_op` therefore does
not accept a truncated raw list, and `x_rows`, when supplied, must equal the
histogram sum. The build path re-checks that processed rows equal input rows.

### 4.3 Campaign counterexamples

| `expert_rows` | compacted pairs | processed rows | kernel list at declared extent |
|---|---|---:|---|
| `[1,0,3]` | `(0,1),(2,3)` | **4** (old model: 1) | `(0,1),(2,3),(0,0)` |
| `[4,4,4]` | `(0,4),(1,4),(2,4)` | 12 | three pairs, `end_of_list` |
| `[64,1,0]` | `(0,64),(1,1)` | 65 | `(0,64),(1,1),(0,0)` |
| `[17,0,2,65]` | `(0,17),(2,2),(3,65)` | **84** (old model: 17) | four pairs, trailing `(0,0)` |
| `[129,0,7]` | `(0,129),(2,7)` | **136** (old model: 129) | `(0,129),(2,7),(0,0)` |

For `[1,0,3]`, `n=128`, `k=64`:

* logical FLOPs: `2*4*128*64 = 65,536`, not `2*1*128*64`;
* input bytes: `4*64*2 = 512`; weight bytes: two M panels of `64*128*2`
  (`32,768`); output bytes: `4*128*2 = 1,024`; HBM: `34,304` B;
* expert 2's weight at `w_element_offset = 2*64*128` is now read;
* y rows 1..3 are written by expert 2, not left missing.

The focused NumPy oracle computes the same result both by a direct histogram
loop and by replaying the emitted tile addresses, and shows that the literal
raw-break interpretation produces a different output with rows 2..4 zero.

### 4.4 Work ledger for the five control cases

| case | histogram | processed rows | host task_num | actual tasks | FLOPs | HBM B | MTE2 B | MTE3 B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `[1,0,3]` | 4 | 4 | 2 | 65,536 | 34,304 | 33,280 | 1,024 |
| 1 | `[4,4,4]` | 12 | 12 | 3 | 786,432 | 205,824 | 199,680 | 6,144 |
| 2 | `[64,1,0]` | 65 | 65 | 2 | 2,129,920 | 98,816 | 82,176 | 16,640 |
| 3 | `[17,0,2,65]` | 84 | 84 | 3 | 2,752,512 | 152,064 | 109,056 | 43,008 |
| 4 | `[129,0,7]` | 136 | 136 | 3 | 4,456,448 | 167,936 | 133,120 | 34,816 |

`actual tasks` is smaller than `host task_num` when several experts have fewer
than 128 rows: the host launches `min(aic, m_*n_tiles)` blocks, but only tasks
produced by nonempty entries execute MMCompute.

## 5. Expert skew, tails, static core assignment and placement

For processed entry `e`:

```text
q_e   = ceil(r_e / 128)
Nt    = ceil(n / 256)
tasks = q_e * Nt
```

Task order matches the kernel: `t = mIdx*Nt + nIdx`. The last M tile has
`rows = r_e - (q_e-1)*128`; the last N tile has `cols = n - (Nt-1)*256`. Both
tails are passed to `SetSingleShape` (`.h:147-157`), so the exact logical work is
`2*rows*cols*k` per task.

The kernel statically rotates tasks across the launched blocks. At entry to
group `g`, let `count_g = (sum of tasks of preceding groups) mod core_num`.
Core `c` executes task `t` when `(t + count_g) mod core_num == c`
(`.h:111-121`). The model materializes only per-core spans, not all tasks:
`O(experts * core_num)` metadata, with tile bodies generated lazily by
`PipelineProgram`.

`[129,0,7]`, `n=128`, `aic_num=32` is the real skew/tail case:

* `total_m = 136`, `core_num = min(32, 136*1) = 32`;
* expert 0 has two tasks, `rows=128` on core 0 and `rows=1` on core 1;
* expert 2 has one task, `rows=7` on core 2 (prefix tasks = 2);
* completion is determined by the 128-row task, not by the mean 68 rows.

With a synthetic compute-dominant control (`bf16=1e10 flops/s`, 32 units,
memory `1e18 B/s`), the independent expectation is
`32 * 1000 * 2*128*128*128 / 1e10 = 13.4217728 ms`; the focused test checks the
public path against this. This is a structural control, not a hardware timing.

### 5.1 Physical placement

Every tile stage declares `Stage.placement`:

* `compute` is pinned to the block's own cube unit;
* `load` is pinned to the block's physical `mte2` units when the hardware
  declares them;
* `store` is pinned to the block's physical `mte3` units when the hardware
  declares them;
* global HBM remains shared and is not given a private unit.

The source ids are the launched block ids, not a synthetic grid.  `Init` reads
`core_idx = GetBlockIdx()` (`.h:72`) and `used_core_num = GetBlockNum()`
(`.h:74`), and the host launches exactly `core_num` blocks with
`SetBlockDim(core_num_)` (host `308`).  Active source ids are therefore
`0..core_num-1`.  The **hardware partition**, however, is declared over the
full physical compute width `context.core_count`, not over the launched width.
Every stage maps its real source id through
`context.owned(core, context.core_count, roles)` (`ownership.unit_group`).

Let `P = context.core_count` and `U` be a role's declared physical unit count.
The mapping is:

```text
if U >= P and U % P == 0:  units(core) = [core*(U/P) .. (core+1)*(U/P) - 1]
elif P % U == 0:           units(core) = [core // (P/U)]
else:                      incompatible declared geometry -> raise
```

Copy-engine work is split equally over the units a stage lists.  With `P=32`,
`expert_rows=[1]*8`, `n=128`, `k=64`, one task moves
`a+b = 128 + 16384 = 16512` B:

| `U(mte2)` | core -> units | per-unit bytes | copy-bound at 1 GB/s/unit |
|---:|---|---:|---:|
| 16 | `0->0, 1->0, 2->1, 3->1, ...` | `2*16512 = 33024` | **0.033024 ms** |
| 32 | `core -> core` | `16512` | 0.016512 ms |
| 64 | `core -> (2*core, 2*core+1)` | `8256` | **0.008256 ms** |

At `U=16`, adjacent physical cores share one unit and the two loads serialize;
the old launched-width candidate invented eight independent units and reported
0.016512 ms.  At `U=64`, one task is served by the owner's two units, so the
copy bound halves.  With `P=1`, the single real source core owns all declared
units (e.g. `(0..31)`); with `P=32` and 32 CTAs each source id owns a distinct
unit.  An incompatible declared geometry raises instead of being silently
averaged.  Unused physical units stay idle because only launched source ids are
mapped.

Because inactive units cannot supply a pinned CTA, a launch of 12 blocks on a
32-unit device pins only those blocks' physical units, not all 32. This is why
the host's larger block count after a truncated raw list cannot change an
active core's rate; the histogram adapter avoids truncation entirely.

## 6. Physical work, stages and per-unit buffers

### 6.1 Per-entry and per-task formulas

For a processed entry with `r_e` rows, `q_e = ceil(r_e/128)`,
`Nt = ceil(n/256)`:

```text
flops_e = 2 * r_e * n * k
A_e     = r_e * k * 2 * Nt          # A rows re-read per N tile
B_e     = q_e * k * n * 2           # B columns re-read per M tile
C_e     = r_e * n * 2
HBM_e   = A_e + B_e + C_e
MTE2_e  = A_e + B_e
MTE3_e  = C_e
```

For one task `(mIdx,nIdx)`:

```text
rows = min(128, r_e - mIdx*128)
cols = min(256, n - nIdx*256)
flops = 2 * rows * cols * k
a = rows * k * 2
b = k * cols * 2
c = rows * cols * 2
```

The per-entry formulas are the exact sums of the per-task formulas; the tests
enumerate all tasks for the campaign geometries and compare both directions.
No token or scalar event is materialized up front.

The model charges B once per M tile and A once per N tile. It does **not**
assume an L2 hit for a repeated weight tile. The source's only cache hint is
`SetL2CacheHint(CACHE_MODE_DISABLE)` when `blockDimM == 1` (`.h:167-169`);
when there are multiple M tiles the hint is not set, but the supplied sources do
not declare hit behavior.

### 6.2 Stage DAG

`build` returns one `PipelineProgram(count=core_tasks, slots=1)` per active core,
wrapped in a `ParallelProgram`:

```text
load    {hbm: a+b, mte2: a+b}             acquires L1 panel, pinned MTE2
compute {matrix: {bf16: 2*rows*cols*k}}  after load; acquires L0A/L0B/L0C,
                                          frees the L1 panel, pinned cube
store   {hbm: c, mte3: c}                 after compute; frees L0C, pinned MTE3
```

Each core program serializes its own tasks; different cores overlap in the
shared engine. `slots=1` does not claim cross-task operand prefetch or
MatmulImpl internal pipelining. The outer `OperatorPlan.lower` supplies the
common setup/observation envelope; the nested program does not apply it again.

HBM and MTE2/MTE3 are two resource views of the same payload in the load/store
stages. They are serviced concurrently in one stage; the stage completes when
both demands are satisfied. On unified-memory hardware with no `mte2/mte3`
resource, `RecipeContext.bind` drops those roles and only HBM remains.

### 6.3 Per-unit L1/L0 ownership

When the caller declares platform buffers:
`Allocation(name, pool, size, units=partition(core, P, buffer_units[pool]))` is
used for each tile, with `size` taken from
`host_tiling()["allocation_bytes"]`.  The units use the same full-width physical
partition as Section 5.1, not the launched `core_num`; `size` applies to each
listed unit.  The engine charges the size to that physical unit's capacity
(`Geometry.buffer_units`, `RecipeContext.pool_units`), not to one aggregated
global pool.

In the focused `[4,4,4]`, `n=64`, `k=32` buffered control, three active blocks
each hold a `262,144` B L1 panel on their own units, so `peak_buffers["l1"]` is
`262,144`, not `786,432`; `peak_buffers_per_unit` has separate
`l1@...:unit0`, `unit1`, `unit2` keys.  For the partition counterexample, eight
active source cores on a 32-unit device with `buffer_units=8` map four adjacent
cores to each of units 0 and 1.  Four `262,144` B panels cannot fit in the
declared `524,288` B per-unit capacity, so the engine queues them and the
observed per-unit peak is `524,288` B, not four invented independent banks.
With `buffer_units=32` each active core receives its own unit and the per-unit
peak is one panel; with `buffer_units=64` each source core owns `(2c,2c+1)` and
each owned unit carries one panel.  With `buffer_units=1`, all cores share one
unit and the same panels queue.  The engine never silently sums capacity.  `ub`
is declared by the host but has no explicit kernel allocation in this path, so
it stays zero.

No private on-chip claim is made when the control hardware declares no buffers.
`workSpace` is passed to the kernel but not used by it; the host reserves
`GetLibApiWorkSpaceSize()` (`sync_workspace_size_`, host `266-267`, `302-304`).
The supplied files do not state that value, so no workspace bytes are declared.

## 7. Oracles and observed results

Focused command:

```sh
PYTHONPATH=. /Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python \
  -m pytest tests/operators/test_grouped_gemm.py -q
```

Result at the time of writing: **26 passed** (focused run; wall time varies).

The tests establish, independently of the planner's own helpers:

1. **Dispatch and split labels** -- all five campaign cases select the expected
   dispatch fields; three are `development` and two `heldout_geometry`; all
   rejected branches and invalid histograms raise.
2. **Histogram compaction** -- the review counterexamples process 4, 84 and 136
   rows; `[1,0,3]` keeps expert 2, `[17,0,2,65]` keeps experts 2 and 3, and
   `[129,0,7]` keeps expert 2. Leading and trailing zero bins also preserve all
   positive rows.
3. **Raw terminator separation** -- `interpret_raw_group_list` stops at the
   first count <= 0, reports the unprocessed suffix, treats counts as direct
   rows (2 then 3 means five rows, not three), and is shown to produce a
   different NumPy output from the histogram adapter.
4. **Host tiling** -- an independent recalculation gives the Section 3.5 table;
   missing L0/L1 buffers stay unresolved; `l1 < 256 KiB` and `baseK == 0`
   raise; per-unit panel bytes match the formulas and capacities.
5. **Per-expert work** -- for every fixed case, an independent formula
   implementation matches every per-entry FLOP/byte field and the `Work`
   totals; enumerated task sums match the same totals.
6. **Skew / no average** -- `[129,0,7]` has tile rows `[128,1,7]`, the 128-row
   task determines completion at `13.4217728 ms` on the synthetic control, and
   the mean-row prediction would be `7.1303168 ms`.
7. **Tails and task order** -- `[17,0,2,65]`, `n=300`, `k=64` gives M tails
   `17,17,2,2,65,65`, N tails `256,44`, six tasks, and the kernel's
   `mIdx*Nt+nIdx` order.
8. **Static rotation** -- for `core_num` 1,2,3,4,5,32 the model's `core`
   assignment matches an independent `(t+count)%core` interpreter on the
   compacted histogram.
9. **Addresses** -- the `[1,0,3]`, `n=5`, `k=4` oracle shows every one of the
   four input rows is consumed once, every output element is written, expert 0
   and expert 2 weights are read, and the empty expert 1 weight is never read.
10. **Small integer matrix oracle** -- direct histogram replay and emitted-tile
    replay agree; the literal raw-break interpretation differs and leaves rows
    2..4 zero. FLOPs equal `2*5*n*k`, not the raw-truncated `2*2*n*k` and not a
    per-tile mean.
11. **Per-unit buffers and placement** -- each tile's load/compute/store stages
    carry a mapping of the real source id over the full physical width in
    `placement`; every allocation lists the core's declared physical pool
    partition.  Public execution with declared buffers reports per-unit peaks
    equal to the panel bytes (or the shared capacity under contention) and no
    remaining allocation. The skewed `[129,0,7]` case repeats this with three
    different tile row counts.  A companion control with `buffer_units=1` shows
    the same panels must queue on one physical capacity, while four units hold
    them independently.
12. **Public execution** -- both the unified and split-copy control hardware
    pass work conservation, buffer release and finite completion for all five
    cases; the runner's renamed-identity control also passes.
13. **Coordinator cases** -- the unmodified `tools/run_operator_cases.py` run
    returns `passed` for 10 rows (5 cases x 2 copy models), preserving 6
    `development` and 4 `heldout_geometry` rows.
14. **Laziness** -- a 10,000,001-row geometry (2,187,528 output tiles) builds
    aggregate work and 64 lazy per-core programs without materializing tile
    stages.
15. **Source identity** -- the metadata hashes match the recorded wave3
    manifest and the wave4 manifest lists the same values; when the
    current-checkout files exist, their sizes and SHA256 values match too.
16. **Multi-task per core** -- with 2 compute units and `[4,4,4]`,
    `core_num=2`; core 0 receives two tasks and core 1 one, and the
    compute-dominant public path completes in two serialized per-core task
    times, not one.
17. **No missing rows** -- `x_rows` different from the histogram sum is
    rejected, and `build` rejects tampered metadata with `processed_m` below
    `input_rows`.
18. **Copy-unit physical partition** -- with 32 compute units and
    `expert_rows=[1]*8`, an independent full-width partition oracle matches the
    actual `mte2`/`mte3` placements for 16, 32 and 64 copy units: adjacent core
    pairs share a unit at 16, each core owns one at 32, and each core owns two
    at 64.  The independent per-unit byte bound gives exactly
    `0.033024`/`0.016512`/`0.008256` ms at 1 GB/s per unit, and the public
    execution agrees.  The old launched-width 0.016512 ms at 16 units is
    explicitly rejected.
19. **1 and 32 compute units / 32-CTA scaling** -- one real source core owns all
    32 declared copy units and 32 tasks serialize; a real 32-CTA launch has 32
    lazy per-core programs and 32 distinct copy units.  At the same 1 GB/s
    per-unit physical rate, one CTA/one unit takes 32x the copy time of 32
    CTAs/32 units (`0.528384` vs `0.016512` ms), with total copy work conserved.
20. **Buffer partition and shared-bank contention** -- `buffer_units` 8, 16, 32
    and 64 use the same full-width partition as the copy engines.  With 8 pool
    units and 8 active cores, four cores share each bank; four 262,144 B L1
    panels are capped at the declared 524,288 B per-unit capacity (observed
    per-unit peak `524,288`), not four independent banks.  A non-divisible
    `24`-unit copy or pool geometry raises instead of being averaged or
    silently collapsed to one invented unit per block.

The planner never reads the case `split`; `development` and `heldout_geometry`
exercise the same selected source path. No case timing is read by `make_op`,
`build` or any parameter; the report merely retains the split label. The
coordinator's runner independently checks `expected_dispatch` on every row.

`data/verification/operators/grouped_gemm.json` is the coordinator runner's
output. Its `scope` is `structural_contract_only_no_accuracy_claim`; its
durations are synthetic control-hardware results, not device measurements.

## 8. Unresolved parameters and proposed core interfaces

These are explicit gaps, not multipliers or disabled checks.

1. **MatmulImpl micro-tiling.** `baseM/baseK` govern L0 fragments, K panels and
   double buffering; `IterateAll` may pad tails to fractal/instruction shapes.
   The supplied files do not define that padding, and the host `baseM` can
   exceed the kernel's `single_m`. The plan counts the exact logical
   `2*rows*cols*k` requested by `SetSingleShape` and uses one host-derived
   panel per unit. Proposed interface: an explicit MatmulImpl tile recipe with
   fractal padding, K-panel counts and double-buffer lifetimes, supplied by a
   later source capture.
2. **MTE1/Fixpipe roles.** L1-to-L0 fill and L0C-to-GM output are real transfer
   stages. The control hardware declares no separate `mte1`/`fixpipe` rates, so
   the model charges L0 residency but only MTE2/MTE3/HBM service. Proposed
   interface: declared `mte1` and `fixpipe` roles on the matrix recipe, without
   charging both the fallback and the specific role.
3. **Scalar/control work.** `group_list_gm_.GetValue(loop)` performs scalar GM
   loads of `2*group_num` values on every launched block (`.h:95-100`). The
   engine has no scalar-load/control role. Counterexample: `E=256`, 32 blocks
   reads 16,384 scalar pairs/values before any MTE2 work. Proposed interface:
   an explicit `scalar_load`/`control` resource with declared operations and a
   rate, plus the group-list element width (`T2` int32 vs int64), which is
   itself missing from the supplied files.
4. **Host tiling CPU time.** `CalMMTiling`, `CalcStepKaKb` and `FillTilingData`
   are host integer work. The source supplies no host instruction count or rate.
   Proposed interface: a declared `host_tiling` work class with instruction
   counts and a separately calibrated rate; do not fit a millisecond constant.
5. **Workspace.** `GetLibApiWorkSpaceSize()` is platform API and is not stated
   in the supplied files. Proposed interface: a `Geometry`/caller-provided
   workspace declaration or a named platform buffer pool; no guessed bytes.
6. **L2/cache behavior.** The model charges repeated A/B reads; it does not
   claim an L2 hit from the conditional `SetL2CacheHint`. Proposed interface: an
   explicit cache policy in `OpSpec.metadata` with declared residency; a
   measured hit rate is a primitive, not an efficiency multiplier.
7. **Per-unit rates.** Pinning uses the declared aggregate rate divided across
   the engine's physical units. The supplied kernel/host sources do not declare
   per-block rates. A future measured per-core rate belongs in primitive
   calibration, separate from this structural plan.
8. **Wrapper/group-list construction.** The wrapper that turns routing into the
   histogram and the weight bank extent is not supplied. The model accepts a
   histogram with explicit ids and an optional `weight_experts`; it does not
   claim the caller compacts or pads the same way. A wrapper that passes a
   literal truncated raw list would violate the "no missing rows" boundary and
   must be rejected or fixed at the boundary.
9. **Historical capture.** No grouped-GEMM capture row was present in the
   supplied read-only evidence. No capture-dispatch match or timing comparison
   is possible from this task alone.

## 9. Integration instructions

```python
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.operators import grouped_gemm

op = grouped_gemm.make_op(params)                 # histogram -> compacted raw list
recipes = RecipeBook(recipes={op.recipe: PlannedRecipe(grouped_gemm.build)})
```

`Execution.prepare` supplies `Geometry` from the active device (compute units,
resource units, buffers and buffer units). `build` declares the physical `Work`,
per-unit allocations and a lazy per-core program.  The program maps each real
launched source id over the full physical width
(`RecipeContext.core_count`), so the coordinator's device geometry is the only
source of unit counts; no shared-core change is required for this mapping.
The coordinator owns registry wiring and final acceptance; this module is not
auto-discovered.

For the campaign, the exact runner command is:

```sh
PYTHONPATH=. /Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python \
  tools/run_operator_cases.py \
  mindie_sim.prediction.operators.grouped_gemm \
  tests/fixtures/operator_cases/grouped_gemm.json \
  -o data/verification/operators/grouped_gemm.json
```

Observed output: `{"status": "passed", "cases": 10, ...}`. This is structural
verification only; independent review and acceptance remain with the
coordinator.
