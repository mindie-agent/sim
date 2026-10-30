# Causal convolution prefill with state and tails

## Selected source and boundary

| Item | Value |
| --- | --- |
| Source | `/Users/maoxx241/code/vllm-v0.25.1/vllm/model_executor/layers/mamba/ops/causal_conv1d.py` |
| SHA-256 | `5f3d16d730fd6dc427cf6981a3de58b1e1e625e624ae8fa0fe77891f58b8bd5c` |
| Bytes | 49533 |
| Kernel | `_causal_conv1d_fwd_kernel`, lines 16-465 |
| Wrapper | `causal_conv1d_fn`, lines 468-745 |
| Selected implementation | `triton_causal_conv1d_fwd` |
| Selected wrapper paths | explicit `null_block_id=None` (`HAS_NULL_BLOCK=False`) and the literal wrapper default `null_block_id=0` (`HAS_NULL_BLOCK=True`) |
| Auxiliary constant source | `/Users/maoxx241/code/vllm-v0.25.1/vllm/v1/attention/backends/utils.py` lines 45-46, `PAD_SLOT_ID = -1` and `NULL_BLOCK_ID = 0`; SHA-256 `192e6b5df24ff52ae57a6a23d6cf58d1b4dd34f14c9283ce268e5f88b1123b96` |
| Layout | channel-last `x:(channels, sum(lengths))`; `out` same shape |
| Element dtype | `bf16` for every campaign case; fp16/fp32 are representable by the same work model |

This is the current vLLM 0.25.1 Triton CUDA prefill source, not the historical
capture image. The historical A5 artifact
`examples/artifacts/a5-linear-attn-k3-shape.json` describes a different
`aclnnCausalConv1d` FLA NPU path, so no captured-dispatch match or timing
comparison is claimed here. The source path and hash above are the only
implementation authority for this plan. The campaign manifest entry for this
task lists the same path, SHA-256 and byte count; the file was present and
matched.

APC remains unselected: a request for `is_apc_enabled=True` is rejected rather
than silently mapped to a nearby branch. The update kernel
`_causal_conv1d_update_kernel` is never a substitute; a request for it is also
rejected.

## Selected dispatch

`OpSpec.metadata["dispatch"]` retains every source dispatch axis used by the
plan:

```text
implementation                 triton_causal_conv1d_fwd
kernel_function                _causal_conv1d_fwd_kernel
wrapper                        causal_conv1d_fn
is_apc_enabled                 False
has_null_block                 per case
null_block_id                  None when has_null_block is false, else the
                               source value (default 0)
pad_slot_id                    -1 (source default; no padded grid programs)
has_bias                       per case
kernel_width                   per case (fwd token loop supports 2..4)
silu_activation                per case
state_len                      kernel_width - 1
np2_statelen                   next_power_of_2(state_len)
block_m                        8
block_n                        256
num_stages                     2 (Triton launch argument)
cta_window                     2 (declared per-owner dependency window)
layout                         channel_last
cache_indices                  explicit per-sequence list, or identity
cache_index_stride             per case
num_cache_lines                per case
initial_state_mode             per_sequence_flags
dtype                          bf16 (canonical)
```

The source derives the dispatch flag as
`HAS_NULL_BLOCK = null_block_id is not None` (line 738), with the wrapper
default `null_block_id: int = NULL_BLOCK_ID` (line 478) and
`NULL_BLOCK_ID = 0` (`vllm/v1/attention/backends/utils.py:46`). The planner
mirrors that derivation:

* omitted `null_block_id` and omitted `has_null_block` -> default 0 -> true;
* `null_block_id=None` -> false;
* explicit `has_null_block=False` without an explicit non-None
  `null_block_id` -> the None specialization;
* explicit `has_null_block=False` together with `null_block_id=0` is rejected
  as a contradictory caller flag, because the source would compile the check
  in.

The campaign case file explicitly sets `has_null_block=false` on all five
cases, so those cases select the explicit no-null path. The literal wrapper
default is independently tested.

When `cache_indices` is omitted, the planner's declared default is the identity
one-line-per-sequence list. With `has_null_block=true` and `null_block_id=0`,
that means sequence 0's cache line is the null block and its CTA early-returns.
Callers that need sequence 0 processed must pass explicit non-null
`cache_indices` (for example `[1, 2, ...]`); the planner does not silently
renumber the caller's block table.

## Source map

| Source lines | Mechanism | Work/model consequence |
| --- | --- | --- |
| 67-69 | `state_len = KERNEL_WIDTH - 1` | state columns `S=width-1` |
| 75-87 | `idx_seq`, `chunk_offset`, `query_start_loc` | one program per `(sequence chunk, channel block)`; flat chunk order is sequence-major |
| 91-121 | APC disabled | `n_block_to_fill=0`, `current_last_index=0`, `conv_state_init_index=0` |
| 123-129 | token offset, `segment_len`, `x_base` | segment length is real; token tails are not padded into the loop |
| 132-139 | cache-index scalar and `HAS_NULL_BLOCK` early return | a null sequence's CTA stops after 20 scalar bytes; no state, weight, conv or output traffic |
| 140-146 | `conv_states_base`, `w_base` | address arithmetic only; no additional payload |
| 151-235 | chunk-0 state branch `S <= seqlen` | read last `S` x tokens; store `S` state values; one `tl.debug_barrier`; one output-coordinate load |
| 237-280 | chunk-0 state branch `S > seqlen`, prior state present | load shifted old state; load first `L` x values; `tl.debug_barrier`; `tl.where`; store `S` state values |
| 281-307 | chunk-0 state branch `S > seqlen`, no prior state | load first `L` x values; store `S` state values; no barrier and no `tl.where` |
| 309-338 | `chunk_offset > 0` prior tokens | read `S=width-1` prior x token vectors; no state access |
| 340-384 | APC intermediate state stores | not executed (`n_block_to_fill=0`); the line-383 barrier is also not executed |
| 386-409 | bias preload and weight preload | bias `.to(float32)` over `BLOCK_N` lanes and all `width` weight columns per program |
| 411-465 | token loop | one x load, `width` multiply-adds, rolling-window moves, SiLU, one output store per real token |
| 543-545 | wrapper dtype/out allocation | `x`/`out` are the selected bf16 dtype, so no conversion kernel |
| 552-562, 630-662 | wrapper chunk table and device metadata arrays | outside the device-kernel boundary; see limitations |
| 563-600 | channel-last strides | `stride_x_dim=1`, token stride is the channel extent |
| 689-693 | grid | `(sum(ceil(length/8)), ceil(channels/256))` |
| 699-744 | launch | `BLOCK_M=8`, `BLOCK_N=256`, `num_stages=2`, `HAS_NULL_BLOCK=null_block_id is not None` |

## Physical CTA ownership and source grid

The source launches a 2-D grid with `program_id(0)` selecting a sequence-major
flat chunk and `program_id(1)` selecting a `BLOCK_N` channel block. CUDA
enumerates the x dimension fastest, so the linear CTA identity used by the
planner is

```text
linear CTA id = flat_chunk + total_chunks * feature_block
```

Each such CTA is the physical unit of work. The plan does not aggregate the grid
into a single instruction stream and does not let one CTA borrow another SM's
vector rate.

```text
owners       = geometry.compute_units (one physical SM owner per compute unit)
streams      = min(owners, total_programs)
assignment   = local index i of owner o handles linear CTA id o + i*owners
window       = OWNER_CTA_WINDOW = 2 in-flight CTAs per owner
```

The scheduling is lazy: `build` stores only `O(sequences)` prefix sums and
`O(owners)` stream descriptors. A stream evaluates `tile(i)` on demand. The
per-owner window is a declared dependency window; it is deliberately not
Triton's `num_stages=2` and not a measured CTA residency/occupancy value. Empty
owners are not given a stream, so a four-CTA grid on an eight-SM device creates
four streams rather than eight empty ones.

Every vector stage of a CTA is pinned with the shared ownership API:

```python
context.owned(owner, owners, ("vector",))
```

A device with two vector subblocks per SM maps owner `o` to vector units
`(2o, 2o+1)`; a device with one vector unit per SM maps owner `o` to `(o,)`. The
engine's `unit_rates` gives each pinned unit its share of the declared total
rate, so one CTA receives exactly one SM's vector service. Idle owners cannot
supply a pinned CTA. Global memory (`hbm`) remains a shared resource, as stated
by the shared ownership contract; it is not pinned to a single SM. The selected
CUDA kernel has no separate DMA/copy-engine stage, so no `mte2`/`mte3` work is
fabricated. If a future source declares a copy-engine stage, the same
`context.owned` mapping must be applied to that stage.

### Ownership counterexample

The original failure was an unpinned single program: a grid of one CTA received
the whole device vector rate. With a fixed per-SM rate `R` and 32 SMs, the
unpinned CTA was 32x faster than the same CTA at one SM. The repaired plan pins
the CTA to one owner:

| Geometry | Total vector rate | One CTA's owned units | Structural time for 4352 vector ops |
| --- | ---: | --- | ---: |
| 1 SM, 2 vector units | `1R` | `(0,1)` | `4352/R` |
| 32 SMs, 64 vector units | `32R` | `(0,1)` | `4352/R` |
| 32 SMs, 64 vector units, total rate fixed to `1R` | `1R` | `(0,1)` | `32*4352/R` |

The first two rows are the required fixed-per-SM counterexample: the single CTA
does not speed up at all when SM count grows from 1 to 32. The third row is the
complementary counterexample: when the total device rate is fixed instead, the
single CTA receives one 1/32 share and is not 32x faster. These are synthetic
structural controls, not hardware accuracy claims.

## Tiling and physical work

Definitions:

```text
C = channels                 W = kernel width (2..4)       S = W - 1
N = next_power_of_2(S)       e = element bytes (2 for bf16)
c = 1 for bf16/fp16 output  (fp32 accumulator store conversion)
c = 0 for fp32 output
BLOCK_M = 8                  BLOCK_N = 256
B = ceil(C / 256)            P = 256 * B  (padded lanes per chunk)
L = sequence length          n = ceil(L / 8)
A = min(256, C - 256*f)      feature-block active lanes
active = L > 0 and not (HAS_NULL_BLOCK and cache_index == null_block_id)
```

### Per-program work

For a processed program at feature block `f`, chunk offset `c`, with
`first = (c == 0)`, `seg = min(8, L - 8c)`, and `init` the sequence's
`has_initial_state` flag:

```text
prep_bytes  = A*W*e + (bias ? A*e : 0) + (first & init ? S*A*e : 0)
              + 20 + (first ? 1 : 0)
conv_load   = A*seg*e + (first ? 0 : (W-1)*A*e)
state_read  = first ? (S <= L ? S*A*e + 4
                              : (init ? (S-L)*A*e : 0) + L*A*e) : 0
state_store = first ? S*A*e : 0
output      = A*seg*e
conv_vector = 256*seg*(3W + a + c) + (bias ? 256 : 0)
state_vect  = first & (S > L) & init ? 256*N : 0
barrier     = first & (S <= L or init) ? 1 : 0
```

`a = 4` for SiLU (`negate`, `exp`, `add`, `divide`) and `0` when activation is
none. The per-token elementary count is `1` accumulator copy + `2W` for the
multiply-adds + `(W-1)` rolling-window moves + `a` + `c`, where `c` is the
optional fp32-accumulator-to-output-dtype store conversion (`1` for bf16/fp16,
`0` for fp32). The `20` scalar bytes are `batch_ptr`,
`token_chunk_offset_ptr`, two `query_start_loc` entries and the cache index;
the chunk-0 program additionally reads the one-byte `has_initial_states` flag.
The extra 4-byte chunk-0 scalar is the state output cache-index load at line
219, executed only in the `S <= L` branch. For `S > L`, the store reuses the
already-loaded state base and does not perform that extra load.

Every program reloads its weight columns and bias. Weight reuse across programs
is deliberately not assumed. The active channel tail uses `A` for payload
bytes and `BLOCK_N` for arithmetic and register work; inactive lanes are masked
memory accesses but still execute vector arithmetic. There is no vector work
for unused token slots in the last chunk because the loop bound is the real
`segment_len`.

### Null-block early return

If `HAS_NULL_BLOCK` is true and the sequence's cache index equals
`null_block_id`, the source returns at lines 136-139 before `conv_states_base`,
`w_base`, the bias/weight preload, the state branch and the token loop. The
planner therefore declares only the 20 scalar bytes loaded at lines 75-87 and
132-134:

```text
early_return_program = 20 bytes, no vector work, no barrier,
                       no state read, no state store, no output
```

The source grid still launches the CTA; only its payload work disappears. A
sequence with `length=0` has no chunk and therefore no CTA, regardless of its
cache index. A null-block sequence's final state and output are left untouched,
which is why the independent oracle fills them with a sentinel and checks that
they stay unchanged.

### Aggregate work

Summing the per-program terms over feature blocks and chunks gives closed-form
per-sequence totals. For a processed sequence:

```text
hbm = n*W*C*e
    + (bias ? n*C*e : 0)
    + (init ? S*C*e : 0)
    + 20*n*B + B
    + C*L*e + (n-1)*C*(W-1)*e
    + (S <= L ? S*C*e + 4B
             : (init ? (S-L)*C*e : 0) + L*C*e)
    + S*C*e
    + C*L*e

vector = P*(L*(3W + a + c) + (bias ? n : 0))
       + (S > L and init ? P*N : 0)
barriers = (S <= L or init) ? B : 0
```

For a skipped (null-block) sequence with `n` chunks, `hbm = 20*n*B` and
`vector = 0`. `P*L` is the padded-lane count for all feature blocks over the
whole sequence; `P*n` is the padded-lane count for per-program bias preloads.
For a zero-length sequence there are no chunks, no programs, no state read and
no state write. The only state writes are the chunk-0 programs: with no APC,
chunk offsets greater than zero never enter the intermediate-state store path.

Final state values are exactly the source branch:

```text
L >= S:  new_state = x[L-S : L]                       (S values)
L <  S:  new_state = [old_state[L:S] if init else 0]  (S-L values)
                    + x[0:L]                          (L values)
```

The model writes that state for every active, processed sequence and feature
block once, even when `initial_state=false`; a false flag selects a zero prior,
not a suppressed final store. A null-block sequence keeps its old state.

### Buffers and stage dependencies

The kernel uses registers for `acc_preload`, `acc`, `w_col*` and `col*`; it
declares no shared-memory or global workspace and the source has no workspace
roundtrip. External tensors are:

| Tensor | Shape | Traffic |
| --- | --- | --- |
| `x` | `(C, sum(lengths))` | one token vector per token plus prior-token loads for chunks > 0 |
| `weight` | `(C, W)` | all `W` columns per program |
| `bias` | `(C,)` when present | one vector per program |
| `conv_states` | `(num_cache_lines, C, S)` | initial/shift reads and final-state writes |
| `out` | `(C, sum(lengths))` | one token vector per real token |
| launch scalars | small int32/bool arrays | 20 bytes per program plus chunk-0 flag/coordinate |

No `Allocation` records are emitted. The synthetic devices in the campaign
declare no pool for these external tensors, the state is global cache storage
and the kernel allocates no on-chip buffer beyond registers. `Allocation.units`,
`Geometry.buffer_units` and `RecipeContext.pool_units` are therefore not used by
this selected path; there is no resident on-chip state to pin.

Reads and writes both consume the unified `hbm` resource. This is a CUDA
global-memory kernel, so the plan declares no artificial `mte2`/`mte3`
copy-engine bottleneck. The split-copy synthetic controls therefore exercise
the same unified-memory path; they are compatibility checks, not a claim that
the CUDA kernel uses two copy engines.

The executable program is a `ParallelProgram` of per-owner `PipelineProgram`
streams. Within one CTA:

```text
chunk offset 0, S <= L:
  prep -> state_read -> barrier -> state_store
       -> conv_load -> conv_compute -> store

chunk offset 0, S > L, init:
  prep -> state_read -> barrier -> state_rewrite -> state_store
       -> conv_load -> conv_compute -> store

chunk offset 0, S > L, no init:
  prep -> state_read -> state_store -> conv_load -> conv_compute -> store

chunk offset > 0:
  prep -> conv_load -> conv_compute -> store

HAS_NULL_BLOCK early return:
  scalar (20 bytes only)
```

`barrier` is a zero-work dependency marker for `tl.debug_barrier()`; it is not
priced without a declared barrier primitive. The barrier appears only where the
source executes it: the `S <= L` store path and the `S > L` shifting path with
a prior state. The no-prior-state `S > L` branch has no barrier. Stage names are
shared across tiles, so each name denotes a serial instruction stream while
distinct names may overlap within the per-owner window. Vector stages carry the
owner placement; global-memory stages remain shared.

## Independent numerical and work oracles

`tests/operators/test_causal_conv_prefill.py` contains independent paths:

1. `rolling_reference`: a direct per-token causal convolution with history
   shifting, optional initial state, optional bias, optional SiLU, variable
   lengths, null-block skipping and final-state extraction.
2. `source_tiled_convolution`: a separate NumPy implementation of the kernel's
   `(chunk, BLOCK_N)` programs, including the `S <= L` and `S > L` state
   branches, chunk-0-only state writes, channel tails, token tails and
   null-block skips.

The two paths were compared on:

| Lengths | Channels | Width | Initial flags | Bias | Activation | Max output diff | Max state diff |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| `[3, 7, 8, 9, 17]` | 130 | 4 | `[T,F,T,F,T]` | yes | SiLU | 0 | 0 |
| `[1, 2, 64]` | 256 | 4 | `[F,T,T]` | no | none | 0 | 0 |
| `[9, 0]` | 257 | 4 | `[T,F]` | yes | SiLU | 0 | 0 |
| `[9, 5]` | 7 | 3 | `[T,T]` | yes | SiLU | 0 | 0 |
| `[9]` | 16 | 2 | `[T]` | yes | SiLU | 0 | 0 |
| `[8, 9, 16]` | 257 | 4 | `[T,F,T]` | yes | SiLU | 0 | 0 |
| `[3, 4]` with null seq 0 | 130 | 4 | `[T,T]` | yes | SiLU | 0 on processed columns | 0 |

"0" means bit-identical float32 results in these runs; it is an oracle check,
not a hardware accuracy claim. The null-block case fills the output with NaN
and verifies that the skipped sequence's columns remain NaN while its final
state is untouched.

An independent final/initial-state test also establishes:

* for `L >= S`, the final state is the last `S` real x tokens and does not
  depend on the initial state;
* for `L < S` with `initial_state=false`, the final state is zero-prefixed x;
* for `L < S` with `initial_state=true`, the final state is the shifted old
  state followed by x;
* tokens at chunk offsets greater than zero depend only on x, not on the initial
  state, while tokens in the first chunk may depend on it.

A separate explicit program walk applies the source-line formulas above to
every `(sequence, chunk, feature block)` program and sums them, including
conditional barriers and null-block scalar traffic. The tests compare that walk
against `build(...).work` for variable lengths, initial-state flags, bias
on/off, activation on/off, channel tails (`130`, `257`, `7`), token tails,
`S <= L` and `S > L`, and null-block sequences. The public `verify_operator`
path additionally checks that engine service equals the declared work exactly,
that buffers are released, and that renaming the op/layer/requests does not
change completion.

## Campaign cases and results

The case file is unchanged. All five cases select `kernel_width=4`, non-APC,
explicit `has_null_block=false` fwd execution, so every supplied case is valid
for the selected source under the declared cache-index assumption; none was
skipped or mapped to another implementation. The cases do not carry a
`cache_indices` table, so the plan uses the identity one-line-per-sequence
assumption described above; the disabled null-block branch makes every identity
line a real cache line. Both development and `heldout_geometry` splits are
preserved in the generated report.

| Case | Split | Lengths | C | Init | Bias | Programs | Barriers | HBM bytes | Vector ops |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `causal_conv_prefill-0` | development | `[1]` | 128 | no | no | 1 | 0 | 2581 | 4352 |
| `causal_conv_prefill-1` | development | `[65]` | 256 | yes | yes | 9 | 1 | 106681 | 285184 |
| `causal_conv_prefill-2` | development | `[64,1]` | 128 | yes | no | 9 | 2 | 52666 | 283904 |
| `causal_conv_prefill-3` | heldout_geometry | `[3,127]` | 130 | no | yes | 17 | 2 | 104870 | 570112 |
| `causal_conv_prefill-4` | heldout_geometry | `[129,2]` | 256 | yes | yes | 18 | 2 | 214382 | 575744 |

`Barriers` counts the conditional chunk-0 `tl.debug_barrier()` instructions,
one per chunk-0 program where the source executes it. `Programs` counts every
source-grid CTA, including null-block CTAs in cases where one is selected.

Command and result:

```sh
PYTHONPATH=. /Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python \
  tools/run_operator_cases.py \
  mindie_sim.prediction.operators.causal_conv_prefill \
  tests/fixtures/operator_cases/causal_conv_prefill.json \
  -o data/verification/operators/causal_conv_prefill.json
# {"status": "passed", "cases": 10, "output": "data/verification/operators/causal_conv_prefill.json"}
```

The report has 10 rows because every case is run on unified-memory and
split-copy synthetic controls. Every row has `dispatch_mismatches == {}`,
finite completion, exact work service, released buffers, and identity
invariance. The synthetic rates are structural controls only; they are not
hardware accuracy evidence and none of their durations is used as a parameter.

Focused tests:

```sh
PYTHONPATH=. /Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python \
  -m pytest tests/operators/test_causal_conv_prefill.py -q
# 43 passed
```

The tests include independent ownership counterexamples:

* one CTA, per-SM vector rate fixed, 1 vs 32 SMs: structural completion equal
  (`4352` vector ops at `1e12` ops/s per SM = `4.352e-6` ms in both);
* one CTA, total device vector rate fixed: 32 SMs gives one 1/32 share, not a
  32x speedup (`1.39264e-4` ms for `4352` ops at `1e12` total ops/s);
* four CTAs on one, two, four and eight SMs: completion falls until four
  owners are used, then stays flat because no fifth CTA exists;
* two owners with four CTAs expose two two-CTA windows, never one global
  two-slot window;
* a two-channel-block grid follows the CUDA x-fastest order (feature block 0
  before feature block 1 for one flat chunk);
* a later sequence can be the null block while an earlier sequence is processed;
* a non-divisible 32-owner/48-vector-unit geometry is rejected with a named
  error rather than silently borrowing units.

## Unresolved implementation and primitive parameters

These are visible limitations, not timing multipliers or floors:

1. **Barrier service.** `tl.debug_barrier()` appears at source lines 234 and
   269; the line-383 APC barrier is not executed because APC is disabled. The
   plan records one barrier per applicable chunk-0 program in a zero-work stage,
   but the shared engine has no barrier primitive or declared synchronization
   rate. Proposed interface: a typed `sync` resource or a sourced barrier
   primitive keyed by the exact instruction, rather than a kernel minimum time.
2. **Global-memory transactions and sector granularity.** HBM work counts active
   bytes. A 130-of-256 channel tail and scalar loads may form partial sectors or
   separate transactions; no sector/coalescing model exists. Source
   counterexample: `mask_w = idx_feats < dim` (lines 157, 398) with `C=130`.
   Because `hbm` remains a shared resource, one CTA can consume the full
   declared HBM rate when no other CTA is running; this is why the ownership
   counterexample fixes memory as nonbinding. Proposed interface: explicit
   transaction granules, issue counts and a per-SM memory-request limit on the
   memory resource.
3. **Instruction and transcendental rates.** `vector` work counts elementary
   multiply/add/move/negate/exp/add/divide operations. The engine applies one
   declared vector rate to all of them; it has no separate SFU/transcendental
   or instruction-issue resource. FMA is counted as multiply plus add. The
   explicit bias `.to(tl.float32)` and the fp32-to-out-dtype store conversion
   are counted, but a compiler-chosen promotion of bf16 x/weight/state operands
   is not separately declared. This is a declared analytical work count, not a
   compiled instruction schedule.
4. **Wrapper host metadata.** The default wrapper builds `batch_ptr` and
   `token_chunk_offset_ptr` with `np.repeat`/`range` and copies device arrays
   (lines 552-562, 630-662), and reads `query_start_loc` to the CPU. This is
   outside the isolated `device_kernel` boundary. It is not emulated with a
   setup multiplier. Proposed interface: a host-metadata/H2D-transfer recipe with
   a declared launch or link resource.
5. **Launch overhead and cache replacement.** The plan starts at the Triton
   program, not the host driver launch. Weight and bias are charged as full HBM
   reads for every program even though real caches may retain them; no L2/L1
   replacement model is asserted. This is conservative and is not an efficiency
   factor.
6. **Mixed dtype conversion.** The wrapper performs `x.to(conv_states.dtype)`
   and returns `out.to(original_x_dtype)` (lines 543-545). All selected cases use
   one bf16 dtype, so no conversion kernel is modeled. Weight, bias, x and state
   are assumed to share that element dtype; the explicit bias `.to(tl.float32)`
   is counted once per program lane. A future case with an fp32 bias or mixed
   operand dtypes would need a separate dtype axis and an explicit conversion
   operator.
7. **Cache-index block table.** The selected path uses one scalar cache index per
   sequence (`conv_state_init_index=0`, `current_last_index=0`). The planner
   models the null-block compare and early return but not a multi-block table:
   `cache_index_stride` is retained as dispatch metadata, while the state traffic
   is counted per sequence rather than per table address. APC is rejected; the
   single-line null-block early return is supported. A multi-block table path
   would need its own source branch and scalar-address traffic.
8. **Kernel-width coverage.** The fwd token loop implements explicit branches
   only for `KERNEL_WIDTH` 2, 3 and 4 (lines 411-452). Width 5 has state-load
   branches but no width-5 token-loop branch; width 6 belongs to
   `_causal_conv1d_update_kernel`. Widths outside 2..4 are rejected rather than
   borrowing update semantics.
9. **Pointer/control instruction issue.** Loop control, pointer arithmetic,
   predicate setup, masked load issue and register renames are not counted as
   vector operations. They remain part of the unresolved instruction-issue
   primitive described in item 3.
10. **Occupancy and concurrent program count.** Each owner's
    `PipelineProgram` uses a declared two-CTA dependency window. It is not derived
    from `compute_units`, registers, shared memory, `num_stages` or a measured
    occupancy. Work conservation and tail retention are independent of that
    window.
11. **Non-divisible ownership geometry.** `context.owned` uses the shared
    `unit_group` integer partition. A geometry with 32 SM owners and 48 vector
    units cannot be partitioned exactly; the planner raises
    `causal_conv_prefill_owner_geometry_incompatible:vector:32:48`. Proposed
    interface: a weighted or fractional per-owner unit partition for vector
    subblocks that are not an integer multiple of the SM count.
12. **Copy-engine ownership is not exercised.** The selected CUDA Triton kernel
    uses global memory, so the plan declares no `mte2`/`mte3` copy work and
    there is no copy stage to pin. If a future source revision or a different
    selected implementation declares a copy-engine stage, that stage must use
    the same explicit `context.owned` mapping; otherwise the same ownership bug
    would reappear on the copy resource.

## Integration notes

The module exports `make_op(params)` and `build(op, geometry)` only; it is not
auto-discovered. To integrate, bind
`PlannedRecipe(mindie_sim.prediction.operators.causal_conv_prefill.build)` under
`recipe="triton_causal_conv1d_fwd"` in the coordinator's recipe book. The
selected source has one kernel launch, so no `kernels(op)` expansion is
exported or required.

No shared engine or registry change is required for the selected path. Optional
shared-core extensions are the barrier/synchronization primitive in item 1 and
the non-divisible ownership partition in item 11. Until they exist, those
limitations remain visible rather than hidden in a timing floor or a silent
fallback.

No timing number from a heldout capture enters this plan. The historical A5
observation is a different implementation and remains a provenance distinction.
The coordinator owns Git metadata, integration and final acceptance; this
worktree is returned for independent review, not claimed as accepted.
