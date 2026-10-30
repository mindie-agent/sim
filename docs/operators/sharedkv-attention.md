# SparseAttnSharedkv SCFA source phases

`prediction.sharedkv_attention` provides an explicitly selected SCFA source
planner on the ordinary Engine. It does not cover SWA or CFA, does not infer
cache residency, and does not assert the loaded binary or metadata of a
historical capture. The public frontend selects `vllm_ascend_sharedkv_scfa`
through `runtime.kernels.lowerings.sharedkv_attention`. The declaration's
options must explicitly provide `owner_schedule`, `selected_rows`, `sinks_dtype`,
`metadata_s2_split` and `return_softmax_lse` as shown below; `scratch_policy`
defaults to `offchip`. No custom recipe construction is needed.

The binding applies only to a single contiguous prefill view with an actual
compressed-index producer. SWA/CFA and decode keep their separate programs.
Even a short query with zero valid compressed rows keeps the SCFA branch when
the compressed-index tensor exists. An unsupported selected source geometry
is an error. The retained V4 A3 profile does not yet declare all split copy
roles required by this planner; synthetic resource tests are not its accuracy
validation.

The five dispatch, host and SCFA source files in local `vllm-ascend` were
independently hashed against git revision
`175ef4d49c2072f9233619636f0309f4f88b4e2c`. Hashes and source line references
are retained in
`data/verification/sharedkv-attention-coarse-20261029.json`. Unrelated source
worktree changes are not evidence for this plan.

## Exact selection contract

`recipe(scratch_policy="offchip", slots=2)` returns a `PlannedRecipe`;
`build(op, geometry, scratch_policy="offchip", slots=2)` returns its physical
work and executable program. The operation must be `prefill` and have the
following `sharedkv_attention_geometry` metadata:

```json
{
  "query": 4096, "context": 0, "window": 128, "cmp_ratio": 4,
  "topk": 512, "heads": 8, "width": 512,
  "branch": "SCFA", "query_layout": "TND", "cache_layout": "PA_ND",
  "sequences": 1, "views": 1, "kv_heads": 1, "cp": 1, "pcp": 1,
  "sparse_block_size": 1, "query_dtype": "bf16", "cache_dtype": "bf16",
  "output_dtype": "bf16", "sinks_dtype": "fp32",
  "return_softmax_lse": false, "metadata_s2_split": false,
  "selected_rows": "valid_prefix", "owner_schedule": "one_query_per_core"
}
```

`query` is positive; `context` nonnegative; window is 1..512; ratio is4 or128;
selected capacity is512 or1024; local query heads are4,8,12 or16. These are
source-domain restrictions, not model-name switches. A whole-query cyclic
owner policy is an explicit scenario: it does not reconstruct metadata's
`bN2/gS1/S2` partition. Multiple views/sequences, cross-core S2 splits, decode,
LSE output, other dtypes and other branches are rejected. The dense causal
window and independent compressed selections remain different regions.

Host `GetSASTemplateMode` selects SCFA only when both original/compressed KV
and compressed sparse indices exist. SWA and CFA dispatch to the other C++
class. A nonzero compression ratio alone cannot establish SCFA. Source
`GetSinks` requires FP32 sinks; both enabled AIV subblocks read them once,
including the odd AIV when it has no softmax rows.

`selected_rows="valid_prefix"` declares valid indices followed by sentinels.
Only the valid-slot index and page-table bytes are counted. Arbitrarily
interspersed invalid entries and scalar response/issue cost are not inferred.

## Physical paths and ownership

- Original KV has no AIV gather. `CalcParams` sets `v0S2DealSize=0`; QK and PV
  each use AIC `DataCopyPA` from original KV. Both copies are external reads in
  this no-hit scenario. Original page-table control work remains unmodeled.
- Compressed KV uses AIV gather, GM publication, then AIC QK/PV scratch reads.
  The first256 valid rows belong to AIV0, the remaining rows to AIV1. Each
  transfer keeps its private owner; there is no aggregate two-AIV borrowing.
- Matrix shapes pad M/N to16. Score output uses padded matrix rows; vector
  score reads use actual heads and the source32-element score stride. PV
  Fixpipe writes padded FP32 rows; vector accumulation reads actual heads.
- Softmax and accumulation use AIV0 for heads<=16. One combined S2 chain has
  one first tile and one final tile. Non-final tiles write FP32 accumulation;
  the final tile writes BF16 external output through MTE3 **and HBM**.
- Source has no per-tile1024-byte valid-size publication/read in this branch.
  The allocated valid-size workspace is preserved in capacity; fictitious
  transfers are removed.

Default `offchip` sends scratch reads/writes through HBM and plain physical
copy roles. Explicit `resident` instead requires `mte2_aic_cached` and
`mte2_aiv_cached` service modes sharing their ordinary physical pipes, plus
`l2_read/l2_write` modes on one shared cache resource. It reserves the complete
per-owner host workspace: four gathered-KV buffers, two score/probability/PV/
accumulator buffers and the declared valid-size allocation. Cache residency
never turns the final output or input KV into a resident temporary.

Each owner uses an existing `PipelineProgram` with two active tiles. Phase
DAGs preserve gather/publication before AIC consumption, softmax before PV,
PV before accumulator read, vector math before output store, the previous tile's
accumulator store before its successor read, and global input readiness. Fresh lazy iterators are
created for each replay. Counters are bounded by source tile shapes; neither
planning nor execution retains a list for every query. This coarse expansion
omits instruction latency and inner-matrix overlap; local UB/L1/L0 capacity is
not yet reconstructed. It is not a cycle-accurate compiler schedule.

## Bounds and execution cost

For query4096, local heads8, width512, window128, topk512, unchanged illustrative
1200GB/s HBM and270TF/s BF16 bound inputs:

| Ratio | Scratch scenario | HBM bytes | HBM lower bound | Matrix work |
| --- | --- | ---: | ---: | ---: |
| 128 | resident | 1,222,273,280 | 1.018561ms | 20.033GF |
| 128 | offchip | 2,023,243,008 | 1.686036ms | 20.033GF |
| 4 | resident | 2,780,403,712 | 2.317003ms | 68.971GF |
| 4 | offchip | 8,417,287,168 | 7.014406ms | 68.971GF |

Each case includes33,554,432 final-output HBM bytes and2,048 sink-read bytes.
These bounds are not completion times or acceptance. SWA's retained0.489ms
observation and the44-event mixed V4 Flash attention average are not matched
SCFA validation.

Two full-query expanded Engine controls used synthetic test resources, solely
for CPU cost and conservation: ratio4 took11.32s and ratio128 took10.07s;
peak process RSS was48.9MB. They completed137,484 and129,238 tasks respectively,
with at most1,116/1,058 live tasks, all work conserved and no remaining tasks,
programs or allocations. This expansion must remain explicitly selected:
repeating it for dozens of layers would exceed the full-model time budget.
The compact generic attention path remains the screening option.

Independent acceptance passed41 focused tests. New counterexamples check the
odd/even MTE3 distinction under real contention, output HBM under resident
scratch, zero valid compressed rows, topk1024's single final output, delayed
input readiness, repeated replay, bounded scratch/live tasks, and a lazy
10^12-query stream that yields three tiles below64KiB temporary allocation.
No observations or elapsed device times enter the planner.
