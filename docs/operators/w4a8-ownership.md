# A8W4 MSD per-owner distribution

The explicit `ascend_w4a8_msd` and `ascend_w4a8_msd_grouped_down` recipes keep
their public selection and one-kernel boundary. Their physical work now retains
private panel reads, owner-local assist copies and the down metadata prelude.
This note records how the source kernels distribute that work over declared
AIC/AIV owners. It is not a native dispatch or instruction reconstruction.

## Source ownership

The gate host sets `A8W4_BASEM/N/K = 128/256/256` for MSD
(`vllm-ascend` revision `8fd0bc4885cb52e58791aaa8efacc571b9bd115d`,
`op_host/grouped_matmul_swiglu_quant_v2_tiling.h` 147-149,
`base_tiling.cpp` 354-392). `DynamicTilingSingleN` affects A4W4, not MSD. The
mid kernel scans every group entry, turns each count into `2*rows` INT4 rows,
and walks `ceil(2*rows/baseM) * ceil(N/baseN)` tiles. The rolling `preCount`
cursor continues across experts, so global tile `t` is owned by AIC
`t % coreNum`. Zero-row groups keep their original `group*N*K` bank offsets but
issue no tile or weight read. Each independently owned N tile reads its own
activation slice, and each M tile reads its own weight panel. These transfers
use the declared off-chip path; an L2 hit or multicast is not inferred.

The gate pre and post bodies partition contiguous input rows by
`ceil(M/aivCoreNum)` with a remainder correction. With M=8 only eight of the
declared AIVs are used, regardless of the full-width count. The post body keeps
one weight-assist row per owner/group segment. If one expert crosses several
AIV chunks, each chunk loads that assist row; this duplicate private read is
retained rather than hidden by a global unique-expert allocation.

The down host's non-whitelist API-dequant branch chooses baseM 64 when
`M/expert_count < 32`, else 128, and baseN 512 or 256
(`ops-transformer` revision `a3925fd49cd256ad478d2faf983bf2961b8ae2c8`,
`op_host/op_tiling/grouped_matmul_tiling.cpp` 2242-2274). The kernel uses the
same rolling tile cursor. The host selects CV11 when `floor(M/E)<=128`,
`E<=4`, `K<=8192` and `N<=8192` in this non-whitelist MSD domain, otherwise
CV12. That choice is independent of the physical AIV inventory. CV11 uses the
first AIV in each physical pair; CV12 uses both. Each participating AIV copies
the tile's assist row before the row-chunk guard.

The vector counter starts at zero, increments before the modulo test, and
persists across each AIC's tiles. Its unit of progress is a UB row chunk of at
most `8192/baseN` original rows, not a complete matrix tile. For 33 rows in one
expert, baseN 512 and one AIC, the 16/16/1-row chunks run on AIVs 1/0/1. A
whole-tile alternation would incorrectly distribute 32/1 rows.

The down pre-kernel explicitly copies all E INT64 counts to every participating
AIV, casts E values, reduces them with E-1 adds and casts the result. This runs
before the input-row loop, including on AIVs with no row. It adds `8*E` read
bytes and `2*E` vector operations per launched AIV. Scalar GetValue scans and
cross-core flags remain visible but unpriced; no scalar rate is inferred.

`prediction/w4a8_schedule.py` implements these loops as bounded arithmetic. It
retains one record per expert and per active owner, not one record per tile.
Each record keeps the real M tail, N tail, weight-bank offset, active-tile count
and assist-load count. HBM remains one shared physical resource; cube, vector,
their declared read/write pipes and a declared Fixpipe are pinned to owner
units.

## Program boundary

Owner phases remain the existing coarse pre -> matrix -> post chain. All matrix
owner stages wait for the complete pre phase, and post waits for the complete
matrix phase. This preserves input/output readiness but deliberately does not
claim the native pre(n+1)/matrix(n)/post(n-1) workspace overlap. Workspace chunk
boundaries are still not reconstructed, and the declared column-scale vector
work stays a vector-equivalent interpretation rather than a Fixpipe rate claim.

The plan's physical work is the sum of the submitted per-owner stages, including
duplicate assist reads. Zero experts contribute neither weight nor scale bytes,
and E256 with eight one-row experts reads eight banks, never 256. Expert scan
counts and zero bank offsets remain visible. The source metadata ledger labels
its unresolved one-AIC/two-AIV reference inventory; execution replaces its work
with the actual device-bound owner ledger before pricing.

## Limits

- Native chunk overlap, workspace slot timing and per-tile issue order are not
  modeled.
- These source recipes require two physical AIVs per AIC. The synthetic examples
  declare that inventory explicitly. Incompatible geometry fails visibly;
  generic analytic matrix recipes remain separate.
- Cached reuses, scalar instruction/flag latency, native UB issue details and
  large-workspace chunk overlap are not priced by this refinement.
- Capture-binary identity and exact CANN dispatch remain unestablished.
- No measured latency, fitted rate or model-name branch enters the schedule.

`tests/test_w4a8_ownership.py` checks the rolling owner mapping with an
independent preCount/curBlock loop, E256/M8 versus one active expert, middle-zero
and nonuniform histograms, N and double-row M tails, 8-of-40 AIV gate chunks,
the real two-AIV down split, physical work conservation, shared-HBM contention,
late input readiness and zero live tasks on the public graph path.

## Independent review and retained descriptor comparison

The DSH implementation is commit `27e687b3c62cb75695b336434acb999a6c6e6385`.
Independent review corrected its whole-tile AIV alternation, missing metadata
DMA/reduction, omitted activation reads on nonzero N tiles and unsupported
owner-ratio adaptation. The maintained suite is run once by the integrating
root; focused review checks cover the actual corrected files.

Using the unchanged GLM case's explicit HBM/vector rates and M8/E256 declared
balanced histogram, public isolated prediction changes as follows:

| Complete selected kernel | Before (us) | Reviewed (us) | Retained observed mean (us) |
| --- | ---: | ---: | ---: |
| Gate/up + SwiGLU + INT8 quant | 10.755987 | 11.471689 | 88.038726 |
| Grouped down -> BF16 | 6.152036 | 6.400320 | 52.907483 |

Both populations contain 5,550 captured events. The 75 repeated model candidates
are not individual occurrence assignments. The captured group-list values and
loaded binary dispatch are unknown, so this remains a declared-histogram
comparison, not exact capture replay or heldout accuracy. The remaining
underprediction is preserved. No timing ratio, event minimum or inferred
all-256-active weight traffic was added.
