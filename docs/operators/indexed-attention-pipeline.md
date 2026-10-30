# Indexed attention copy and output pipeline

`programs.indexed_attention.kind = "indexed_attention_pipeline"` is an
explicit physical plan for the reviewed arch22 sparse MLA vector-gather
path. It uses the ordinary Engine and requires `scratch_policy` to be
`resident` or `offchip`. Neither is inferred from an observed duration.

The reviewed source is vllm-ascend
`8fd0bc4885cb52e58791aaa8efacc571b9bd115d`; file identities are retained in
`data/verification/sparse-attention-mechanism-20261029.json`. The implemented
domain is BF16, one latent KV head, latent/rope widths 512/64, at most 16
local query heads, CP1, no compressed cache or dense sliding window. This
does not identify V4 `SparseAttnSharedkv` as the same implementation.

The ordinary frontend also accepts the explicit
`lowerings.indexed_attention.implementation =
"ascend_sparse_flash_attention_arch22"` binding. Its `scratch_policy` option
defaults to `offchip`; `resident` still requires independently declared cache
capacity and rates. The binding applies to both prefill and decode, preserves
the real top-k/query/cache producers, and keeps one device kernel boundary.
`examples/source-families/indexed-mla-a3-control.json` exercises this public
path. An explicit `programs.indexed_attention` declaration, including the
supported decode body default, selects an analytic pipeline without claiming
captured native dispatch. With neither declaration the generic envelope remains.

The source binding requires the two operand service modes `mte1_l0a` and
`mte1_l0b` to share the physical `mte1` pipe. Their byte counts stay separate,
so a slower B transfer cannot borrow A's instruction rate. Existing explicit
`indexed_attention_pipeline` programs retain the aggregate-operand mode unless
`operand_modes=true` is declared. The hardware profile's aggregate MTE1 rate
therefore remains a distinct approximation, rather than a reconstructed
mixed-direction schedule.

## Physical distinctions

- Each query owns its selected KV rows. Gather reads the selection and page
  indices and stages KV through the two AIVs associated with its AIC. Separate
  1024-byte latent and 128-byte rope rows already meet 32-byte alignment;
  an extra cacheline inflation factor would be unjustified.
- QK and PV retain 16-row matrix padding. Each 128-column matrix subdivision
  rereads its private Q or probability operand through MTE1. Shared GM/L1
  input bytes do not acquire this factor.
- For at most 16 heads, the source gives all softmax and output rows to the
  first AIV. The second AIV still participates in KV gather. Score Fixpipe
  writes padded rows, while the vector read consumes actual head rows.
- The source publishes an integer bias, atomically updates that bias on later
  key tiles, and performs atomic Fixpipe accumulation. It reads the output
  accumulator into the vector core only on the last key tile. The plan
  preserves that path instead of inventing a full vector accumulator update
  on every key tile. Destination read/modify/write consumes backing-store
  read/write service; it is not an additional MTE2 input load.

The fixed tile size is 512 selected keys with three in-flight pipeline slots.
Four KV scratch rings and three score/probability/output rings retain finite
capacity. `resident` reserves those bytes in a shared `l2` pool and still
charges copy-pipe and shared cache service. `offchip` charges scratch reads
and writes to HBM. Capacity sufficiency does not prove cache hits.

## Execution abstraction

Compact execution retains phase ordering, prior partial-output dependencies,
finite slots, per-owner/per-unit service lower bounds and ordinary Engine
contention. It caches at most four geometry plans and four nominal rate
signatures per plan. These caches retain arithmetic summaries, never Engine
tasks, readiness events, observations or layer histories.

The compact body is a fluid approximation. It does not reproduce exact native
arbitration or claim equivalence to the expanded `compact=false` control.
That control uses the same work in lazy finite `PipelineProgram` stages.
The phase model conservatively orders PV computation before the bias-update
and Fixpipe publication; the source may overlap part of that work.
Scalar index response, cross-core flag latency, atomic response latency and
external cache eviction remain unmodeled. Vector demand is an arithmetic
ledger, not a reconstructed instruction issue schedule.

Independent acceptance checks actual Engine service, single-AIV ownership,
delayed inputs, rate changes across cached shapes, atomic backing traffic,
finite allocations, ragged tiles and contention. The source review corrected
private A-operand reloads, active AIV ownership and output accumulation before
the final replay.

## Conditional supply scenarios

`data/verification/a3-9362-platform-metadata-20261029.json` records the exact
current-host platform file, not the historical capture's toolkit. It lists
L2 read/write values 110/86 in `AICoreMemoryRates`, but does not independently
establish whether those values should be multiplied by the AIC count to
obtain chip bandwidth. The platform nominal clock and memory capacity also
differ from the retained capture declaration.

The historical
`data/verification/model-scoreboard-20261029-glm53-indexed-pipeline-case.json`
explicitly labels multiplication by 20
and use of the existing 1.65 GHz clock as an **unverified supply assumption**.
Its resident/offchip results are conditional sensitivity scenarios; they do
not calibrate or replace the default GLM hardware profile. Neither the policy
nor a cache hit fraction is selected to approach the measured 13.2 ms event.
The default scoreboard row keeps its independently declared supply and
query-specific gather envelope, with these alternatives shown separately.
The newer `examples/source-families/indexed-mla-a3-control.json` instead uses
the SKU's published nominal copy declarations and the explicit offchip policy;
it does not inherit those historical L2 supply assumptions.

## Finite query ownership and the decode boundary

The arch22 kernel assigns query head-group tasks to AIC owners; it does not
give one query row the aggregate bandwidth of all 20 AICs. `SplitBalanced`
sets `usedCoreNum_ = aicNum_`, but the work distributor places each
`gS1Idx` task on one owner (`gS1Idx = gS1LoopIdx * mBaseSize`,
`mBaseSize = gSize`). A single-token TND decode therefore owns one AIC and the
two AIV subblocks paired with it. The plan now takes this owner count from
`sparse_attention.selected_query_owners`; 1, 2 and 20 query rows own 1, 2 and
20 AIC tasks respectively. `InitCalcParamsEach` and `GetAxisStartIdx` assign
contiguous intervals of `ceil(query/AIC)` rows. The planner now preserves those
intervals instead of round-robin query assignment: with 21 causal queries and
20 AICs, the heaviest owner reads 39 selected rows, not the round-robin bound of
22. Owners past the final interval have no work. This matters for causal
prefill where later queries carry more keys, even when total bytes are unchanged.

For the reviewed GLM geometry (query1, top-k2048, 512+64 BF16 rows) the main
generic envelope's 7.25 GB prefill ledger is not the decode body. Its
body-only full-chip bound is 2.36 MB / 1.2 TB/s = 1.99 us. The indexed plan
predicts 24.73 us with the explicitly conditional per-core copy/cache supply
already used by the resident alternative: one owner reserves 2.6 MB scratch,
gathers 2.37 MB to GM, publishes 2.54 MB through MTE3 including the
scratch rings, writes 0.26 MB through Fixpipe, and rereads 4.49 MB cached on
the AIC path. The coordinator-supplied decode observation is 90.7 us, so 65.97 us remains
outside this reviewed mechanism; it is not independently reproduced here. Scalar index
response, page-table dependency, cross-core flags and atomic response are not
fitted or assigned a residual-derived rate.

## Top-k entries are selected rows in this domain

`sparse_indices` has one INT32 entry per selected row in the reviewed GLM
capture (`sparse_block_size=1`; arch22 tiling selects the vector template for
block sizes up to4, while DAV_3510 requires block size1). `GetRealS2Idx`
multiplies an entry by `sparse_block_size` exactly once to obtain the cache
row; the attention driver already passes row-count top-k. The recipe
therefore rejects a geometry that supplies `sparse_block_size != 1` rather
than multiplying an already-row-counted top-k again. The V4 `SparseAttnSharedkv` wrapper's declared
`cmp_topk=index_topk` reading is a separate source family and is not bound to
the retained arch22 binary: its
`candidate_topk_blocks=2048` and `candidate_block_size=8` are a candidate
filter expansion, not a factor on the selected 512 `index_topk` entries.

## Batched analytic caller

The earlier frontend emitted one `latent_attention{i}` operator per request.
Each default-stream operator restarted its owner grid at AIC0, so B requests
were serialized even when their query rows fit on otherwise idle owners.

The retained kernel source independently confirms the shared scheduling axis:
`sparse_flash_attention_kernel_mla.h` lines 525–580 first sums
`actualSeqLengthsQ[b]`, then distributes the `(batch, query-position)` tasks
over the common AIC grid. Lines 324–365 retain each batch's KV length, and
lines 661–675 retain the independent Q/KV prefix offsets. Its SHA256 remains
`cec94843ad2144f80e5a504b33f92a682ba8908037435de65da26f5682ce7cdd`, matching
the already retained `8fd0bc4885cb52e58791aaa8efacc571b9bd115d` source.

The analytic frontend now emits one batch operator when the selected program
is `indexed_attention_pipeline`. Its geometry stores every `(query, context)`
pair. Owner assignment uses the sum of query rows, then maps each task back
to its sequence-local position before computing `min(context+position+1,topk)`.
No KV contexts are concatenated. Scratch ownership, each query's accumulator
reset, physical work and shared HBM contention remain explicit.

Explicit per-request recipes, attention-family recipes and native source
lowerings retain their own boundaries. The exact native selector still requires
one contiguous sequence; this change does not silently enlarge that ABI domain.
CP, compressed/window cache and INT8 cache are not admitted into this BF16
pipeline. The batch path is an analytic scheduling adaptation.

`data/verification/indexed-attention-batch-20261030.json` records 32 real local
Engine executions, with B1…B8 at q1 and q6, context4096, TP16 and the current A3
9362 resource profile. All are one-layer attention component controls. At q6,
the B1/B2/B4/B8 spans change from 85.06/170.12/340.25/680.49 μs to
85.06/99.18/198.37/396.74 μs. B8 reaches the shared HBM limit; the program does
not grant independent full-chip bandwidth to each owner. At q1, eight requests
use eight actual AIC owners. In the compact fluid approximation they complete
in the single-query pipeline span; finite-phase timing differs as described below.
Every pair retains identical physical FLOPs, vector work and HBM bytes, and
all tasks/programs drain. Delayed-input checks show no early service.

The ragged test `(q,c)=[(1,0),(2,510),(1,1024)]` preserves 2,049 logical selected
rows; it is not treated as a four-token causal sequence. Expanded and compact
batch controls retain identical physical Engine service. These are schedule
verification results, not a claim about whole-body or new INT8-KV capture error.

Independent acceptance checked 1,000 owner geometries, 20 actual Engine pairs
and three explicit selection controls. The retained
`indexed-attention-batch-independent-review-20261030.json` separates service
conservation from timing approximation: ragged sequences
`[(4,1023),(7,1023),(4,1023),(4,15),(2,1023)]` take 85.925 us compact versus
110.901 us expanded (-22.52%). Even uniform B8/q1/context4096 takes
85.062 versus 102.500 us (-17.01%); B8/q6 takes 396.739 versus 409.271 us
(-3.06%). The fluid compact program omits finite-phase arbitration barriers.
Both paths retain the same service, and neither row/context mapping nor
owner assignment differed. Compact timing is therefore not a universal
15% approximation to expanded timing. No multiplier is applied to conceal
this limitation.
