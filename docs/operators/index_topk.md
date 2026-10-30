# index_topk: sparse-attention block scoring and bitonic top-k

## Scope and status

This planner is the source-derived extension for the common MiniMax-M3 Triton
indexer in `vllm/models/minimax_m3/common/ops/index_topk.py`. It represents the
two prefill launch boundaries:

1. `_index_block_score_kernel` -- paged, causal, per-128-token block max scores;
2. `_topk_index_kernel` -- a streaming bitonic top-k over one explicitly chosen
   tuning tile, after a barrier on the fp32 score workspace.

The selected implementation id is `triton_block_index_topk`. The fixed campaign
geometry is a single request with `query` new tokens after a `context`-token
prefix (`request.context` semantics in this repository: `kv_len = context +
q_len`, `mindie_sim/schema/workload.py:43`), one shared index head, bf16 index
keys and 128-wide pages. `context` maps to the source's `prefix_lens`, not to
`seq_lens`: the score kernel's causal window is `prefix_len + query position`
(source lines 134, 151-152) and the top-k kernel's `valid_blocks` is
`(prefix_len + pid_q + 128) // 128` (line 229). Consequently `index_topk-4`
has `seq = 32769`, 257 visible blocks and two 256-wide sort iterations; reading
`context` as total sequence length would erase that tail. The small `query`
values are chunked-prefill queries after a long prefix, not decode: the decode
split-K path is a different implementation selection and is not substituted.

This is a structural source model, not a hardware measurement. The case-runner
rates are synthetic controls. No historical capture observation for this
operator was found in the read-only evidence paths, so nothing here claims a
capture dispatch match or latency accuracy. The source at the manifest path is
the **current checkout** file, not a supplied historical capture image. The
current `mindie_sim/model_configs/minimax-m3.json` records a 64-wide index block
and eight index heads, while the supplied source fixes 128-wide pages and the
campaign case supplies one shared head; the planner follows the supplied source
and case and reports that integration mismatch in the limitations below.

## Source identity

| Role | Path | SHA-256 |
| --- | --- | --- |
| Selected source | `/Users/maoxx241/code/vllm-v0.25.1/vllm/models/minimax_m3/common/ops/index_topk.py` | `1ca8557aed4260d2f4a742b9eda81d6f774e49d4b016ccc555dde40976c5a1a0` |
| Dispatch / supported sizes | `/Users/maoxx241/code/vllm-v0.25.1/vllm/models/minimax_m3/common/indexer.py` | `43787aeb1c8a7fd0f990aa62267e6c47f802b511b8850a00f99de0fad2fb468a` |
| Independent dense reference test | `/Users/maoxx241/code/vllm-v0.25.1/tests/kernels/attention/test_minimax_m3.py` | `0e88afe33ad85542da6ca02ecbaa144af12bbad504546097b062f643a8742bd2` |
| AMD variant, not selected | `/Users/maoxx241/code/vllm-v0.25.1/vllm/models/minimax_m3/amd/ops/index_topk.py` | `6cc2ee849a0d015bca64f4f826be0478c43dd3cd36ae601712b2aa9971d52999` |

The selected file is 37,281 bytes. Selected line map (current file):

| Lines | Content |
| --- | --- |
| 24 | `SPARSE_BLOCK_SIZE = 128` |
| 31-52 | `_compare_and_swap` bitonic lane operation |
| 55-70 | `_bitonic_merge` stage builder |
| 81-163 | `_index_block_score_kernel` |
| 110-113 | request/sequence scalars |
| 117-125 | 64-row q block load with boundary padding |
| 134 | `hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)` |
| 135-148 | per-page K load and dot |
| 149-154 | `tl.dot`, causal mask, `tl.max(qk, axis=1)` |
| 155-163 | fp32 score store with row mask |
| 173-184 | top-k `BLOCK_SIZE_T` heuristic and six `BLOCK_SIZE_K` autotune configs |
| 185-284 | `_topk_index_kernel` |
| 226-246 | sentinel fill, loop, causal/init/local masks, index build |
| 247-268 | build merges and first/later streaming top-k merges |
| 269-284 | first-`BLOCK_SIZE_T` extraction and int32 store with `-1` padding |
| 647-703 | `minimax_m3_index_score` wrapper: `BLOCK_SIZE_Q = 64`, score stride rounded up to 16 |
| 706-756 | `minimax_m3_index_topk` wrapper: `sample_interval = 1`, grid `(max_query_len, batch, heads)` |
| 753-754 | literal wrapper arguments `MASK_INIT=False, MASK_LOCAL=False` |
| 287-641 | decode split-K score/partial/merge path; **not selected** |

`indexer.py` lines 89-94 fix supported head size 128 and kernel block size 128;
lines 511-515 reject non-bf16 for the Triton indexer. The prefill caller passes
`init_blocks` and `local_blocks` through as runtime scalars, but the wrapper
hard-codes both constexpr mask flags to false.

## Selected dispatch

`OpSpec.metadata["dispatch"]` records every selection axis:

```text
implementation          triton_block_index_topk
mask_init               false (fixed cases; wrapper-selected)
mask_local              false (fixed cases; wrapper-selected)
mask_branch_selection   wrapper_fixed_false_false | raw_constexpr_branch
wrapper_mask_init       false
wrapper_mask_local      false
score_kernel            _index_block_score_kernel
topk_kernel             _topk_index_kernel
kernel_phases           [score, topk]
block_size_q            1
score_block             128
score_q_tile            64
topk_block              case value (fixed cases: 256)
block_size_t            next_pow2(topk) (2/4/8/16 for the fixed cases)
dtype                   bf16
launch_count            2
topk_block_selection    explicit_case_tile
autotune_configs        [64, 128, 256, 512, 1024, 2048]
init_mask_semantics     force_1e30 | mask_minus_1e29
local_mask_semantics    force_1e29 | mask_minus_1e28
decode_split_k_path     false
```

The top-k tile is **the case-declared tile**, not an autotune winner. The source
autotuner would benchmark the six `BLOCK_SIZE_K` configs on first use; tuning
cost and winner selection are not modeled. The plan prices only the declared
tile, and the padded bitonic network is charged for that tile.

`OpSpec.metadata["launch_boundaries"]` also records the score per-block stages
and the top-k per-chunk stages with their local dependencies, and
`OpSpec.metadata["source_cadence"]` marks the load/dot stage separation and the
carried per-chunk merge. These are provenance for the program, not additional
work: the aggregate ledger above is unchanged.

### Flag semantics and the wrapper-selected branch

The kernel's constexpr names are inverse to their effect:

- `mask_init=false` executes `score = where(causal & init, 1e30, score)`;
  `mask_init=true` executes `score - 1e29` (lines 237-240).
- `mask_local=false` executes `score = where(causal & local, 1e29, score)`;
  `mask_local=true` executes `score - 1e28` (lines 241-244).

The actual in-tree prefill wrapper calls the kernel with
`MASK_INIT=False, MASK_LOCAL=False` (source lines 753-754). The fixed campaign
cases therefore select the wrapper path, and the planner records
`mask_branch_selection="wrapper_fixed_false_false"`.

The raw constexpr branches remain available as explicit source paths for
separate branch tests. When either flag is true, dispatch records
`mask_branch_selection="raw_constexpr_branch"`; it never rewrites the true
constexpr into the wrapper's forced branch. The fixed cases and production
wiring use the wrapper-selected false/false values. A concrete counterexample
from the review is checked in the tests: scores `[5,4,3,2,1]`,
`local_blocks=1`, `topk=2`. The wrapper's `MASK_LOCAL=False` forces the local
block to `1e29`, selecting blocks `{4,0}`. The raw `MASK_LOCAL=True` branch
subtracts `1e28`, so the local block ranks last and the selection is `{0,1}`.
Both branches have identical declared work; only the selection semantics
differ.

## Geometry and work ledger

Inputs: `query = Q` new tokens, `context = C` prefix tokens, `batch = B`,
`heads = H`, `topk = K`, `topk_block = T`, sparse block `S = 128`, score query
tile `QT = 64`, head width `D = 128`, bf16 width 2 bytes, int32/fp32 width 4.
`seq = C + Q`, `NB = ceil(seq / S)`, `stride = align_up(NB, 16)`,
`BT = next_pow2(K)`.

### Launch 1: block score

Grid: `(ceil(Q/QT), B*H)`. The linear CTA index advances the query-tile axis
fastest. For each query tile `b = 0 .. ceil(Q/QT)-1` of each request/head:

- `rows_b = min(QT, Q - b*QT)`
- `q_start_b = C + b*QT`
- `hi_b = min(seq, q_start_b + QT)`
- `blocks_b = ceil(hi_b / S)`
- `masked_blocks_b = blocks_b - floor(q_start_b / S)` (the blocks whose
  128-position window satisfies `q_start_b < i*S + S`; at most two because
  `hi_b <= q_start_b + QT`).

The launch declares:

```text
qk_flops       = B*H * sum_b 2 * QT * S * D * blocks_b
q_bytes        = B*H * sum_b rows_b * D * 2
k_bytes        = B*H * sum_b blocks_b * S * D * 2
blocktable     = B*H * sum_b blocks_b * 4
score_store    = B*H * sum_b rows_b * blocks_b * 4
score_mask     = B*H * sum_b QT * S * masked_blocks_b
score_max      = B*H * sum_b QT * (S-1) * blocks_b
score_addr     = B*H * sum_b (S*blocks_b + QT)
score_vector   = 2*score_mask + score_max + score_addr
```

The 64-row `tl.dot` is charged in full even when only one query row is valid
(the source wrapper hard-codes `BLOCK_SIZE_Q=64`). The full 128-position page is
loaded for every visited block because the source intentionally does not mask the
K load; positions beyond `seq` are masked later in `qk`. q traffic counts only
valid rows (the block-pointer load is boundary-checked). Score stores count
`rows_b * blocks_b`: for a row in an early tile, the kernel also writes `-inf`
scores for blocks beyond that token's causal window, so score-store traffic can
exceed the later top-k read traffic. The fp32 workspace is
`B*Q*H*stride*4` bytes; stride padding is allocated but not touched.

The CTA is a serial source loop (lines 135-163): load the block-table entry,
load that page's K tile, consume it in `tl.dot`, mask/reduce and store the block
score. `build` therefore emits `q_load` once, a `q_addr` stage for the 64 row
offsets, and a nested lazy per-page loop:

```text
page_load -> k_load -> qk_dot -> mask_max -> score_store
```

for every visited block. `page_load`/`k_load` declare only memory traffic and
`qk_dot` declares only bf16 FLOPs; `mask_max` is vector-only and `score_store`
is the fp32 store. Per block the declared work is 4 block-table bytes,
`S*D*2` K bytes, `2*QT*S*D` dot FLOPs, `2*QT*S` mask lanes when the block
intersects the causal window, `QT*(S-1)` max lanes, `S` address lanes and
`rows_b*4` store bytes. The source loop declares no `num_stages` hint or double-buffered
operand contract, so this candidate models the repeated load/dot cadence
serially; it does not infer compiler software pipelining or fold all pages into
one read-plus-matrix envelope.

The independent single-block control is `Q=1, C=0, topk=1`: its score read is
`33028` bytes and its dot is `2097152` FLOPs. At `330280000 B/s` and
`20971520000 FLOPs/s` each operation costs `0.1 ms`. The earlier single
`load_mma` `Work` let the engine overlap them and charged
`0.100012110936 ms`; the repaired source DAG charges `0.200012110936 ms`.
Those rates are a structural counterexample, not hardware calibration.

### Launch 2: bitonic top-k

Grid: `(Q, B, H)`. The linear CTA index advances the query-token axis fastest.
For query token `j = 0 .. Q-1` of each request/head:

- `v_j = floor((C + j)/S) + 1`
- `iters_j = ceil(v_j / T)`
- first iteration CAS stages `n(n+1)/2`, later iterations
  `n(n-1)/2 + 2n`, where `n = log2(T)`;
- total CAS stages `first + (iters_j-1)*later`;
- padded lanes `T * iters_j`;
- combine lanes `T * (iters_j-1)`.

The launch declares:

```text
score_read     = B*H * sum_j v_j * 4                 (masked load: valid lanes)
topk_causal    = B*H * sum_j T*iters_j               (causal mask / sentinel)
topk_init      = B*H * sum_j T*iters_j               (init branch evaluated)
topk_local     = B*H * sum_j T*iters_j               (local branch evaluated)
padded_lanes   = B*H * sum_j T*iters_j
sort_stages    = B*H * sum_j (first + (iters_j-1)*later)
cas_lanes      = T * sort_stages
merge_lanes    = T * B*H * sum_j (iters_j-1)
prepare_ops    = 15 * padded_lanes
sort_ops       = 6 * cas_lanes
merge_ops      = 6 * merge_lanes
init_fill_ops  = 2 * T * B*H*Q
extract_ops    = (3*T + BT) * B*H*Q
topk_vector    = prepare_ops + sort_ops + merge_ops + init_fill_ops + extract_ops
selected_bytes = B*H*Q*K*4                           (all slots, including -1)
valid_bytes    = B*H*sum_j min(K, v_j)*4              (subset of selected bytes)
```

`prepare_ops` is 15 logical lane operations per padded iteration layer: causal
mask/load (3), NaN sanitize (2), init compare/select (3), local compare/select
(3), index build/select (2) and previous-top-k copies (2). `sort_ops` charges 6
logical operations per compare-exchange lane; `merge_ops` charges 6 for the
score/index two-multiply-plus-add combine. `extract_ops` is `T` multiply, `T-BT`
reduction adds, `2*BT` store-mask/select and `T` top-k mask lanes. These are
**declared elementary vector counts**, not calibrated instruction latencies.

The top-k loop carries `topk_score` and `topk_idx` from one `T`-lane chunk to
the next. `build` emits one `init` fill, then a nested lazy per-chunk loop, then
`extract` and `output_store`:

```text
first chunk:  load -> masks -> bitonic_merge_scan -> bitonic_merge_asc
later chunks: load -> masks -> bitonic_merge_scan
              -> bitonic_merge_desc -> bitonic_merge_combine
              -> bitonic_merge_asc
```

`bitonic_merge_scan` is the `n(n-1)/2` in-loop compare-exchange stages
(`n = log2(T)`); `bitonic_merge_desc` and `bitonic_merge_asc` are each `n`
stages. For a later chunk the source first descends the current tile, combines
its top half with the previous iteration's carried top half (6 lanes per
lane), and only then ascends. The later chunk's `load` is after the preceding
chunk's `bitonic_merge_asc`, so the carried state is a real dependency, not one
aggregate `sort_ops+merge_ops` envelope. After the last chunk, `extract`
reduces/masks the first `BT` sorted slots and `output_store` writes the int32
result; the store is a dependent stage after extraction. The sum over chunks is
exactly the aggregate ledger above; only the dependency history changed.

### Total ledger and resource mapping

```text
hbm_read  = q_bytes + k_bytes + blocktable + score_read
hbm_write = score_store + selected_bytes
matrix    = {"bf16": qk_flops}
mte2      = hbm_read   (split-copy geometry only)
mte3      = hbm_write  (split-copy geometry only)
vector    = score_vector + topk_vector
```

On unified-memory geometry the engine does not declare `mte2`/`mte3`; the core
skips those roles, as it does for the tiled-matrix recipe. They remain declared
physical bytes on the op fields for split-copy hardware.

Fixed-case ledger (per one head, one request):

| Case | split | Q | C | seq | NB | stride | bf16 FLOPs | HBM read B | HBM write B | vector ops | iters | padded lanes | sort stages | CAS lanes | selected B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| index_topk-0 | development | 1 | 512 | 513 | 5 | 16 | 10,485,760 | 164,136 | 28 | 118,146 | 1 | 256 | 36 | 9,216 | 8 |
| index_topk-1 | development | 4 | 2048 | 2052 | 17 | 32 | 35,651,584 | 558,420 | 336 | 398,480 | 4 | 1,024 | 144 | 36,864 | 64 |
| index_topk-2 | development | 2 | 16384 | 16386 | 129 | 144 | 270,532,608 | 4,229,132 | 1,096 | 1,202,320 | 2 | 512 | 72 | 18,432 | 64 |
| index_topk-3 | heldout_geometry | 3 | 640 | 643 | 6 | 16 | 12,582,912 | 197,472 | 96 | 247,238 | 3 | 768 | 108 | 27,648 | 24 |
| index_topk-4 | heldout_geometry | 1 | 32768 | 32769 | 257 | 272 | 538,968,064 | 8,423,688 | 1,092 | 2,271,632 | 2 | 512 | 80 | 20,480 | 64 |

The `index_topk-4` geometry is the padded-sort boundary case: 257 visible blocks
with a 256-element tile forces a second iteration, so the sort charges
`36 + 44 = 80` stages over `512` padded lanes although only 257 scores are read
and 16 indices are kept. In the table, `iters` is the total number of padded
sort iterations across all query tokens (`B*H*sum_j iters_j`), not a per-token
count.

`init_forced_candidates` and `local_forced_candidates` report
`sum_j min(init_blocks, v_j)` and `sum_j min(local_blocks, v_j)`;
`visible_candidates` is `sum_j v_j`. The top-k mask lanes report every padded
lane on which the kernel evaluates the causal, init or local branch, including
lanes whose score is ultimately ignored.

## CTA ownership, laziness and launch boundaries

The source launches two grids of CTAs. A grid CTA is the physical owner of its
compute/copy slices; splitting the launch into an aggregate device-wide stage
would let one CTA borrow every idle unit, which is a mechanism error rather than
a missing timing parameter.

`build` lowers each launch as follows:

- `units = min(total_CTAs, context.core_count)` (the engine's compute-unit
  count supplied by `Geometry.compute_units` in the public path);
- one `PipelineProgram` per participating unit, with
  `count_u = ceil((total_CTAs - u) / units)` CTAs;
- CTA global index `g = index*units + u` follows the source grid flattening
  (query-tile/token axis fastest), so the per-unit streams cover every grid CTA
  exactly once;
- each CTA's stages use `context.owned(u, context.core_count, roles)`
  (implicitly through the recipe context) for `cube`, `vector`, `mte2` and
  `mte3`. A vector partition is widened by the declared `resource_units`
  ratio (for example two vector lanes per compute unit when
  `resource_units={"vector": 2*compute_units}`);
- a score CTA lowers to `q_load`, `q_addr` and a nested lazy per-page
  `PipelineProgram` (`slots=1`) holding `page_load`, `k_load`, `qk_dot`,
  `mask_max` and `score_store`;
- a top-k CTA lowers to `init`, a nested lazy per-chunk `PipelineProgram`
  (`slots=1`) holding `load`, `masks`, `bitonic_merge_scan`,
  `bitonic_merge_desc`, `bitonic_merge_combine` and `bitonic_merge_asc`,
  followed by `extract` and `output_store`;
- the stream issues its next CTA only after the previous CTA completes
  (`slots=1`), so a unit never runs two CTAs concurrently in this structural
  model.

Both nested loops are lazy `PipelineProgram`s: each stores a finite count and a
`tile()` callback, so planning never materializes one event per page or per
chunk. The large-geometry test builds a 200,000-token top-k grid as 32 streams
and verifies that no per-token statistics function is called during planning.

The singleton counterexample is explicit: with one score CTA and one top-k CTA,
the program contains one stream per launch, each with count 1, and its cube/
vector placements are a single partition. With fixed per-SM rates, the isolated
duration at 1 SM and 32 SMs is identical; a device-wide aggregate stage would
have shortened the 32-SM run by about 32x. The test scales total device rates
with the declared unit count, so each per-SM partition has the same rate at 1
and 32 SMs. `tests/operators/test_index_topk.py` checks both the placement and
the equal durations.

| Run | score host submits | top-k host submits | total host service at 50 us |
| --- | ---: | ---: | ---: |
| eager (`replay_scope="none"`) | 1 | 1 | 0.10 ms |
| fully captured (`replay_scope="full"`) | captured group | same group | 0.05 ms |

`kernels(op)` exports the two launches as child operators `score` and `topk`
with local ids and `topk.deps == ("score",)`. Both children keep the same recipe
id; `build` reads `OpSpec.metadata["kernel_phase"]` to select the phase. The
shared `expand_call` prefixes the child ids, preserves the caller's dependencies
and adds a terminal join. In the public `Execution` path the eager run submits
each real kernel (two host submissions); a full capture submits the captured
group once. The test checks `Engine.serviced["host@..."]` and the retained trace,
not only the metadata `launch_count`. Binding no `kernels` callable leaves one
host submission even though `launch_count` says 2, which is the independent
counterexample that metadata alone changes nothing.

The CTA-to-unit assignment is a declared static round-robin in source grid
order. Real hardware may dynamically backfill a freed unit with a different CTA;
this model does not reconstruct occupancy, register pressure, warp scheduling or
backfill. No fitted load-balancing or efficiency multiplier is applied.

## Buffers and lifetimes

- Score workspace: fp32 `[H, B*Q, stride]`; launch 1 stores, the launch barrier
  separates it, launch 2 loads. Allocated bytes are reported as
  `score_workspace_bytes`; untouched stride padding is not counted as traffic.
- Selected-index output: int32 `[H, B*Q, K]`; the final store writes every `K`
  slot, with `-1` in slots at or beyond `v_j`. `selected_index_bytes` is the full
  output; `selected_valid_bytes` only separates real block ids.
- On-chip residency is reported, not capacity-checked: score tile
  `64*D*2 + S*D*2 + 64*S*4 + 64*4` bytes and logical sort state
  `(7*T + BT)*4` bytes per program (`onchip_score_tile_bytes`,
  `onchip_sort_bytes_per_program`). The synthetic control geometry declares no
  on-chip pools, so no `Allocation` is created for these scratch regions.

The score workspace is a real cross-launch lifetime, but the current public
extension interface prices each exported child through its own isolated
`Execution`/engine. `KernelProgram` scopes an `Allocation` name with the
program prefix, so a score-child allocation cannot be freed by the separately
built top-k child; a cross-child allocation handle is not available. The plan
therefore charges the real store/load traffic and reports allocated bytes, but
does not pretend to reserve a pool. If a later interface supplies a stable
cross-expansion allocation id plus acquire/free points, the score buffer can be
added without changing any work formula. This is a reporting limitation, not a
disabled capacity check: the control hardware has no declared pool at all.

## Independent oracles

`tests/operators/test_index_topk.py` contains independent NumPy/CPU checks:

- per-tile/per-token enumeration of every ledger field, compared to the planner's
  closed forms for all fixed cases and for non-multiple contexts/tails
  (`Q=128,C=100`, `Q=7,C=129`, `Q=1,C=0`, `Q=65,C=8192`, batch/heads/forced
  blocks);
- dense block scores from random q/pages: causal mask, per-block max,
  init/local forced branches, and `np.argsort` top-k;
- an exact NumPy mirror of the source `_bitonic_merge`/`_compare_and_swap`
  network and the first/later streaming merge, including `-1` padding, checked
  against the dense sorted indices (single and two-iteration cases);
- explicit raw constexpr tests for all four `MASK_INIT`/`MASK_LOCAL`
  combinations, plus the wrapper `false/false` and raw `true` counterexample;
- score-workspace, K-page and output address maps: touched addresses, stride
  padding, page rereads per query tile, selected-index bytes;
- actual public-path execution work through `verify_operator(...,
  kernels=kernels)` on unified and split-copy synthetic hardware: two distinct
  child rows, work conservation, finite completion, buffer release, identity
  invariance, and declared versus serviced totals;
- source-cadence counterexamples with independent rates: the one-page
  `Q=1,C=0` score must cost more than `0.2 ms` when its 33,028-byte read and
  2,097,152-FLOP dot each cost `0.1 ms`, with distinct `page_load`/`k_load`/
  `qk_dot` events and no task carrying both matrix and memory demand; a
  five-page score must emit five load/dot pairs and take more than the
  aggregate one-read/one-dot envelope; and a 257-block top-k must emit
  per-chunk `bitonic_merge_scan`/`bitonic_merge_desc`/
  `bitonic_merge_combine`/`bitonic_merge_asc` stages whose carry order matches
  the source loop;
- per-CTA placement checks (one cube unit, two vector lanes per compute unit at
  `resource_units={"vector": 64}`, one copy unit), total grid coverage by the
  lazy streams, per-CTA stage-work conservation against the aggregate plan, and
  the same ownership check on nested page/chunk leaf stages;
- singleton 1-SM versus 32-SM fixed-per-SM-rate invariance, plus an actual
  `Execution` run showing two eager host submissions, one fully captured host
  submission, equal device work in both modes, and one host submission when the
  `kernels` callable is not bound;
- explicit-tile checks (`T=64/256/1024` changes padded work independently of the
  top-k count) and rejection of nearby unsupported paths (vendor implementation,
  fp16, 64-wide heads/blocks, `block_size_q != 1`, non-autotune tiles, too-large
  `BT`, negative/bool/unknown parameters);
- a 200,000-token geometry that builds with a constant number of stages and a
  fixed-size metadata record, with zero token-statistics calls during planning.

These are structural/semantic verification only. They do not establish hardware
accuracy, occupancy, autotune behavior or capture agreement.

## Limitations and unresolved parameters

1. **No historical capture observation** for this operator was supplied or found
   in `data/verification` / `data/observations`; no timing number enters the
   planner. The manifest source is a current checkout file; its relation to an
   older capture image is not established.
2. **Autotune is not emulated.** The case's `topk_block=256` is priced as a
   declared tile, not as the winner among `[64,128,256,512,1024,2048]`. Tuning
   cost, compile time and winner selection are open. The source config's
   `num_warps`/`num_stages` (8/2 for the 256 tile) and resulting occupancy are
   not modeled; no guessed efficiency multiplier is applied.
3. **Raw true mask branches are not production.** The wrapper always passes
   `MASK_INIT=False, MASK_LOCAL=False`. The planner still supports the raw true
   constexpr branches and records `mask_branch_selection="raw_constexpr_branch"`;
   the tests exercise those branches separately and the fixed cases stay on the
   wrapper path.
4. **Other platform/dispatch variants are out of scope.** Decode split-K
   (`_decode_index_score_kernel`, `_topk_index_partial_kernel`,
   `_topk_index_merge_kernel`; three launches, partial score/index buffers,
   fixed-count chunks), the AMD source variant and the Blackwell
   fmha_sm100/MSA score path are different implementation selections and are
   not modeled or replaced by this prefill Triton pair.
5. **Head count defaults to 1.** The source docstring describes a single shared
   index head, and the fixed cases do not provide `heads`. The current model
   config (`minimax-m3.json`) records `index_n_heads=8`; the real adapter must
   pass `heads=8` structurally if that is the intended geometry. No model-name
   branch exists in this module.
6. **Only `block_size_q=1`, `head_dim=128`, `score_block=128`, bf16.** These are
   source-supported invariants; other values are rejected rather than mapped to
   a nearby path. fp16 is rejected because the Triton indexer implementation is
   bf16-only (`indexer.py` 511-515).
7. **Element-op constants are not calibrated.** Score mask 2, max 1, top-k
   prepare 15, compare-exchange 6 and merge-combine 6 are explicit logical
   counts. Instruction issue, reduction trees, bitcast/broadcast overhead,
   register spills and synchronization are unresolved primitive parameters.
   The two source launches are preserved as separate dependency stages, but
   launch issue/setup latency is not separately modeled; no fitted launch floor
   was added.
8. **Scalar metadata path is not bulk traffic.** `cu_seqlens_q`, `seq_lens`,
   `prefix_lens`, `cu_seqblocks_q` and block-table page scalars are O(program)
   loads (block-table bytes are counted; the 4-5 int32 per-program scalar loads
   are not). Their issue/response cost needs an explicit scalar program, not a
   fitted setup time.
9. **On-chip pools are declared, not owned.** The control geometry declares no
   pools; per-program score/sort footprints are metadata only. Grid-level
   concurrency, register occupancy and pipeline overlap are not reconstructed.
   Cross-child pool lifetime has no stable allocation handle in the current
   extension API, as described above.
10. **Ties and forced-score ordering.** The source bitonic network is not a
    stable sort. The dense oracle uses distinct random scores and forced
    constants with at most one init and one local candidate; tie order is not
    claimed.
11. **Current model config disagrees with the supplied source.** The checkout's
    `mindie_sim/model_configs/minimax-m3.json` records
    `sparse_attention_config.index_block_size=64`, `index_n_heads=8`,
    `index_topk_blocks=64` and `index_local_blocks=16`. The supplied source
    fixes the sparse block and kernel page to 128, and `indexer.py` reports
    `get_supported_kernel_block_sizes() == [128]`; the fixed campaign case also
    uses `score_block=128` and no head count. This planner follows the supplied
    source/case (128, one shared head) and rejects `score_block != 128` or
    unsupported head widths explicitly. The model-config/source mismatch must be
    resolved by the coordinator before real graph wiring; no model-name branch
    is present here.
12. **CTA scheduling is a declared static assignment.** The per-unit streams
    preserve grid order and owned partitions, but do not model hardware CTA
    backfill, occupancy or warp scheduling. A dynamic scheduler could finish a
    mixed-work grid at a slightly different time. No calibrated correction or
    fitted per-CTA rate is applied.
13. **Per-request query lengths are uniform.** The case schema supplies one
    `query` for every request. The source grid uses `max_query_len` and each
    kernel early-returns CTAs whose local extent is shorter; that per-request
    tail is not represented here. Batch/heads scaling assumes every request has
    the declared query length.
14. **Per-page/per-chunk cadence is serial; no inferred prefetch.** The score
    loop is modeled as `page_load -> k_load -> qk_dot -> mask_max ->
    score_store` per page and the top-k loop as a serial carried merge per
    chunk. The source does not annotate these loops with an explicit
    `num_stages` hint or double-buffered operand contract; the planner
    therefore does not invent cross-iteration overlap. If a compiled schedule prefetches the next K page
    or score chunk, that overlap must be added as an explicit bounded prefetch
    dependency (producer and prefetch depth), not by recombining the aggregate
    read and matrix totals into one envelope. The current shared
    `PipelineProgram` already expresses the per-iteration stream order and
    `slots=1` serializes one iteration; a double-buffered variant would need a
    declared buffer allocation and an explicit release point.

The existing `KernelProgram`, `ParallelProgram`, `PipelineProgram`, `Stage`
placement, `RecipeContext.owned` and `PlannedRecipe(build, kernels)` are
sufficient for this planner. The only missing shared feature is the cross-child
workspace-allocation lifetime described above; the work and traffic ledger does
not depend on it.

## Integration

`make_op(params)` accepts the campaign params plus optional `batch`, `heads` and
the source's raw `mask_init`/`mask_local` true branches:

```python
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.operators.index_topk import build, kernels, make_op
from mindie_sim.prediction.program import RecipeBook

op = make_op({
    "query": 1, "context": 32768, "head_dim": 128, "topk": 16,
    "block_size_q": 1, "score_block": 128, "topk_block": 256,
    "init_blocks": 0, "local_blocks": 1, "dtype": "bf16",
    "implementation": "triton_block_index_topk",
    "mask_init": False, "mask_local": False,
})
book = RecipeBook(recipes={op.recipe: PlannedRecipe(build, kernels)})
# The caller owns graph placement, request ids and final acceptance.
```

The module rejects unknown parameter keys, non-boolean mask flags, non-integer
geometry and every unsupported source path with a named `ValueError`. The
coordinator should keep
`op.recipe == "triton_block_index_topk"` and let the shared `Execution.prepare`
apply `plan.work` before pricing. `make_op` uses a placeholder
`requests=("batch",)` and `op_id="index_topk"`; the graph adapter must substitute
the real request ids and a unique op id for wiring. Physical cost is
identity-independent. The returned program has exactly two source launch
boundaries and a score-store -> top-k-load dependency; do not collapse it into
one envelope or replace the bitonic stage with a vendor top-k recipe.

The current checkout's `minimax-m3.json` would pass `score_block=64` and
`heads=8` if used verbatim. `score_block=64` is rejected by this module because
the supplied source's kernel block/page size is 128; the config mismatch must be
resolved before real graph wiring. Passing `heads=8` is structurally supported
and scales q/K/score/output work by tenancy, but it does not change the
128-block source invariant.

## Reproduction

```sh
PYTHONPATH=. /Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python \
  tools/run_operator_cases.py \
  mindie_sim.prediction.operators.index_topk \
  tests/fixtures/operator_cases/index_topk.json \
  -o data/verification/operators/index_topk.json
```

```sh
PYTHONPATH=. /Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python \
  -m pytest -q tests/operators/test_index_topk.py
```

The generated `data/verification/operators/index_topk.json` is the unmodified
campaign runner's structural contract report for all development and
heldout_geometry rows on both synthetic copy geometries. A green result there is
not accuracy acceptance, and this session does not claim integrated acceptance.
