# Explicit GDN decode caller

`runtime.kernels.lowerings.gdn_decode = "vllm_ascend_gdn_decode"` selects the
non-interleaved QKVZ/BA caller in the ordinary frontend. `target_graphs`,
`operator_costs`, `graph_costs`, body prediction and serving use the same
lowering and physical programs. No capture or model name selects this binding.

## Implemented boundary

The selected pure-decode branch has one update convolution, one gating kernel,
two separate Q/K L2-normalization invocations, one recurrence invocation and
one gated output norm, followed by the ordinary output projection/reduction.
With multiple projection rows and value heads, the caller has nine source materializations: two
BA slices, one contiguous convolution input, three flat Q/K/V slices, one
concatenation, one copy into the caller output and one z reshape before norm.
An explicitly strided cache-index column adds its own materialization.
Its zero fill is separate.
View-contiguous singleton shapes may remove source `.contiguous()` copies.
Source-known copy and fill operators explicitly declare vector invocation
class, so the hardware's existing invocation declaration applies once.
They are not modeled as host gaps or additional launch latency parameters.

BA slices are made contiguous over all padded rows before gating consumes the
actual-row prefix. The convolution input is sliced to actual rows before its
contiguous copy. Core recurrence uses separate request lengths, never the graph
bucket or accepted-token count. The caller output zero retains padded rows.
Before native gated norm, the caller reshapes z to `[tokens * heads, head_dim]`.
For multiple rows and heads, that reshape crosses the QKVZ projection row gap
and materializes the selected z payload. Singleton/view-compatible cases stay
free. The source assignment into the caller output remains separate; absence
of a small TensorMove in kernel-only CSVs does not prove that source copy is
free. See [the copy-boundary audit](gdn-caller-copy-boundaries.md).

Supported caller domain: BF16 activations, FP32 recurrent state, PCP1,
non-interleaved QKVZ/BA, and decode-only requests. All one-token requests select
non-speculative convolution. All requests with more than one query token select
the speculative update. Mixed speculative/non-speculative batches have a
separate source packing/splitting/merging branch and are explicitly outside this
binding. The ordinary analytical frontend remains available for them.

Q/K norms and recurrence retain their explicit analytical defaults. Selecting
`options.recurrent_implementation = "arch35_fp32"` enables the resident-QKV/state
program described in [gdn-recurrent-source.md](gdn-recurrent-source.md).
Its 247,808-byte UB layout is an arch35 source contract: it must not replace
the default on an A3 profile whose UB is 196,608 bytes. The selected source
domain, scalar control limitations and conservative tile handoffs stay visible.

## Update convolution

The preserved source's exact-divisor-first tile selection considers channel
widths 4096, 2048, 1024, 512, 384 and 192. Tasks are sequence/channel-tile pairs;
they are assigned round-robin to the actual launched AIV owners. Independent
owners run concurrently, with bounded token prefetch/output windows.

Each task reads weights and the previous `width-1` state rows once, performs the
source casts, convolution and SiLU, and writes every token output. In the
width-4 speculative branch, `WriteBackStateSpec` then rereads two retained
history positions plus all current input tokens, and writes `2+query_length`
state rows. Acceptance changes their addresses, not the amount of query work.
The non-speculative branch writes the final `width-1` resident history rows.
The fixed source ring/output/calculation allocation is 188,416 bytes per
executing AIV. Scratch is released when its per-owner program finishes.

All movement uses the declared AIV read pipe and MTE3 output pipe, together
with shared HBM service. Scalar metadata issue/cache latency is not inferred
from profiler counters. This finite resource abstraction does not reconstruct
all scalar instruction costs or compiler scheduling.

The six new A5 captures under `/Volumes/data/mindie-sim-data/a5/` corroborate the
source's launch geometry. They are conditional source comparisons, not proof
that retained source and the capture binary have identical revisions.

| Capture | Query rows / sequences | Channels | Observed Block Num | Source selection |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 c4 | 16 / 4 | 1,280 | 28 | 28 |
| Qwen3.6 c40 | 160 / 40 | 1,280 | 64 | 64 |
| Qwen3.6 c128 | 512 / 128 | 1,280 | 64 | 64 |
| Qwen3.8 c4 | 16 / 4 | 2,560 | 20 | 20 |
| Qwen3.8 c32 | 128 / 32 | 2,560 | 64 | 64 |
| Qwen3.8 c128 | 512 / 128 | 2,560 | 64 | 64 |

CSV inspection is streaming. The counts are read from `kernel_details.csv`;
`op_statistic.csv` independently reports equal convolution/gating/recurrence
counts and twice as many L2-normalization invocations. The shared metadata
slice visible before convolution is not charged repeatedly as tensor data;
its control boundary needs separate metadata ownership evidence.

## Gating programs and stores

The retained Triton kernel uses 64-row by 8-head tiles. Masks suppress tensor
loads/stores, but the vector expressions and per-program parameter loads still
execute. `g` is FP32 and `beta` is BF16. G uses the declared DMA output resource.
The default beta path also uses DMA as an analytical memory-service model;
a scalar-store path requires an explicit independent queue declaration.

The caller accepts `gating_beta_store` with `resource` and `queue` fields, or
reuses the explicit beta queue in an existing `gdn_gating` tiled-pointwise
family declaration. The historical A5 queue is **not** imported automatically.
Its BF16 issue/response parameters must not be applied to FP32 g or treated as
universal launch overhead. Queue evidence and transfer-domain suitability are
independent of the caller's source geometry.

The logical grid defaults to the declared physical vector-core count, matching
`get_vectorcore_num()` in the retained wrapper. An explicit `gating_programs`
option can express a separately observed logical grid. Programs reuse physical
owners sequentially (`program_id % AIV_count`); a grid of 64 cannot borrow
64 physical cores on a 40-AIV device. Scalar issue resources also retain their
actual owner mapping. The shared `QueuedStore.issue_placement` parameter is
optional; old callers retain their previous behavior.

## Sources and verification

- `data/observations/sources/gdn-image-c3d00a196905/gdn.py.txt`, SHA-256
  `e9ed210b4ca4431ee40916f06f0272e5e781eb04b61b084bacea216799fbca98`:
  BA materialization and caller zero at 298–360; gating, normalization and
  recurrence at 532–575 / 705–725; final output assignment at 725–737.
- `data/observations/sources/gdn-image-c3d00a196905/fused_gdn_gating.py.txt`,
  SHA-256 `5587bf3440e64eda125899f5de2d1ab609061e2b5336f847af101ada51d81058`:
  masked kernel at 14–56 and wrapper at 59–101.
- `data/observations/sources/native-causal-conv-a4a7958/`, revision
  `a4a795857868729fbdbb0118c1e06c8655a7b31b`: tile selection in
  `op_host/causal_conv1d_tiling_planner.h:25–73`; weight/history loads,
  token loop, speculative state restoration and update dispatch in
  `op_kernel/causal_conv1d.h:310–405,463–548,1008–1097,1216–1288`.

Focused validation: 55 tests passed, including 13 new source-caller controls.
They run actual Execution/Engine programs: late input, physical-byte and vector
work conservation, AIV bandwidth response, logical-to-physical core reuse,
source scratch release, padded BA work, ordinary graph completion, one
invocation constraint per copy and shared scalar-pipe contention. No whole-model
accuracy claim follows from these controls. No hardware parameters, captured
measurements or original implementation files were removed.
