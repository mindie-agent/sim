# KDA prefill FWD_H state pipeline

`mindie_sim.prediction.operators.kda_state` binds the retained vLLM-Ascend
`chunk_kda_fwd` **FWD_H** stage. It is selected explicitly through the ordinary
model frontend:

```python
from mindie_sim.schema.case import Software

runtime = Software.from_mapping({"kernels": {"lowerings": {
    "kda": "vllm_ascend_kda_chunk_state",
}}})
```

The existing `vllm_ascend_kda_in_kernel_gates` selection and an unselected KDA
model keep their prior generic `recurrent_scan` work. The new binding changes
only the prefill scan member; projections, QK normalization, gate placement,
the wide KDA semantic group, output normalization and the output projection
remain caller-owned. Decode scans are never source-bound; a mixed batch keeps
ordinary decode work while all prefill sequences are fused into one actual
FWD_H launch.

## Source branch

Pinned repository/revision:
`/Users/maoxx241/code/vllm-ascend-v41-private`,
`ae923905948f360f4b7ab06df52d0789a0c9316e`. The selected branch is the arch35,
varlen (``cu_seqlens``), key-gated KDA call:

- `chunk_kda_fwd_common.h` `RunFwdH` constructs
  `GDNFwdHKernel<T,float,float,float,GDNFwdHTileShapes128,true,false,true>`
  (K/V/W/U BF16, FP32 gates/state/workspace, ``kGated=true``,
  ``scalarGated=false``, ``useExp2=true``). Lines 225-241.
- `chunk_kda_fwd.cpp` dispatches stage 2 (`KDA_STAGE_FWD_H`) to that kernel for
  ``vHeadDim <= 128``. Lines 55-64.
- `gdn_fwd_h_kernel.hpp` owns the C1/V1/C2/V2 chunk exchange and the per-stream
  state snapshots. Lines 508-1067.
- `block_scheduler_gdn_fwd_h.hpp` assigns each core up to two live
  ``(sequence, value_head)`` streams and advances them by the source stride.
  Lines 214-225 and 300-353.
- `chunk_kda_fwd_tiling.cpp` allocates the GM V, V-update, K-decay and H
  workspaces with two pipeline slots per core. Lines 274-293.
- `kimi_kda.py` `_run_prefill` supplies raw FP32 gates, an FP32 state and varlen
  CU metadata; external QK L2 normalization and the output norm/projection are
  outside the selected phase. Lines 468-519.

SHA-256 identities are embedded in `family_kda_state.SOURCE`:

| Source | SHA-256 |
| --- | --- |
| `csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_common.h` | `e98c17e5...e74f8` |
| `csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp` | `0ea78773...a74d0` |
| `csrc/attention/chunk_kda_fwd/op_host/chunk_kda_fwd_tiling.cpp` | `72148ca1...f813c` |
| `arch35/gemm/kernel/gdn_fwd_h_kernel.hpp` | `aa9695e2...4cf13` |
| `arch35/gemm/block/block_scheduler_gdn_fwd_h.hpp` | `3e7730a0...8ba741` |
| `arch35/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp` | `8ec011c4...a953dd02` |
| `arch35/epilogue/block/block_epilogue_gdn_fwdh_update.hpp` | `67dbb475...6d88fedb` |
| `arch35/epilogue/block/block_epilogue_gdn_fwdh_regbase.hpp` | `c2221dbb...daac4cc` |
| `vllm_ascend/ops/kimi_kda.py` | `35365b3b...d54f11f3` |

These are source identities, not capture-binary identities or accuracy
evidence.

## State branch

The source FWD_H kernel internally uses ``[K, V]`` FP32 state. The public KDA
wrapper keeps a ``[H, V, K]`` cache and the L2 API transposes before and after
FWD_H. That transpose, the cache gather/scatter and the final-state export are
outside this selected phase. The planner records both layouts so callers do
not confuse the internal phase representation with the public cache layout.

The FP32 state is not resident across chunks in this branch. Every chunk reads
the current FP32 state, writes the updated FP32 state back to the same
tiling-assigned final-state location, and additionally publishes a BF16 H
snapshot (except on the last chunk) for the next C1. The initial H0 conversion
and the first V2 update each read the initial FP32 state, so the first chunk has
two initial-state reads and all later chunks have one state read. This differs
from the arch35 GDN resident-state branch in `gdn_state.py`.

## Ownership and phases

The source scheduler starts a core at tasks
``core * slots + slot`` and advances by ``core_count * slots``, where
``slots = 2`` when there are more head tasks than cores and ``1`` otherwise.
The planner keeps that assignment. Each slot advances by one chunk per source
iteration and admits its next stream when that slot finishes. A short stream
does not wait for the other slot's complete sequence. A two-iteration window
keeps expansion bounded while retaining the source AIC order (both C1 calls
before both C2 calls) and AIV order (both V1 calls before both V2 calls). The next
C1 also depends on its own slot's preceding state publication. Idle cores and
the second AIV of a different AIC are not borrowed for vector or copy service.
Within a stream, chunks are strictly serial.

Vector phases use the owning AIV pair's aggregate supply. For an odd V1 tail,
the source divides rows as `ceil(valid/2)` and `floor(valid/2)` while this phase
model averages work over the pair. At a one-row tail, only one source lane does
the V1 row work; its modeled V1 service can therefore be optimistic. The
8192-token scenario has only even 64-row chunks and does not exercise this
approximation. This limitation concerns V1 phase timing, not GM byte totals.

Each ``(sequence, value_head, chunk)`` uses four source phases:

1. **C1** (AIC): ``V_work = W @ H``. W is BF16 rows, H is the BF16 snapshot;
   output V_work is FP32 GM scratch.
2. **V1** (AIV pair): ``V_new = U - V_work``; K is copied to the BF16 K-decay
   workspace and BF16 V-update is published for C2.
3. **C2** (AIC): ``H_work = K_decay^T @ V_update`` in FP32 GM scratch.
4. **V2** (AIV pair): ``H = exp2(g_last) * H + H_work``. The key-wise FP32 gate
   is read for the K rows, the FP32 state is read and written, and a non-final
   BF16 H snapshot is published.

An **initialization** phase reads the FP32 initial state, casts it to BF16 and
writes H0 for every assigned stream. It is aggregated per owner and completes
across all cores before any C1, preserving the source initialization `SyncAll`
dependency without inventing a synchronization duration. C1 reads that
snapshot; C1 of every later chunk reads the previous non-final snapshot. The
first-chunk V2 update reads the initial state again. Each matrix phase separates
GM input, MTE1 operand transfer, matrix compute and Fixpipe publication; operand
transfer cannot overlap its own dependent matrix compute.

## Work accounting

For a real chunk with ``valid = min(64, length - chunk * 64)`` rows:

| Phase | Matrix | HBM read / write | Inner service |
| --- | --- | --- | --- |
| init | - | `K*V*4` read, `K*V*2` write | `K*V` cast |
| C1 | `2*valid*K*V` | `valid*K*2 + K*V*2` read, `valid*V*4` publish | `2*(valid*K + K*V)` MTE1 |
| V1 | - | `valid*V*2 + valid*V*4 + valid*K*2` read, `2*valid*V*2 + valid*K*2` write | `4*valid*V` vector |
| C2 | `2*K*valid*V` | `valid*K*2 + valid*V*2` read, `K*V*4` publish | `2*(valid*K + valid*V)` MTE1 |
| V2 | - | `K*V*4 + K*4 + K*V*4` read, `K*V*4 + (K*V*2 if non-final)` write | `2*K + K*V*(2 + non-final cast)` vector |

`fixpipe` is charged for the C1 FP32 V_work and C2 FP32 H_work publications.
When the hardware declares separate read pipes, aggregate MTE2 is partitioned
between the AIC (C1/C2) and AIV (V1/V2/init) owners. The engine keeps the
native byte/FLOP units; no rate constant, measured duration, model multiplier,
startup floor or residual is read or fitted by this planner.

The plan's total matrix work is the exact real-row arithmetic over every
sequence/head/chunk interval. Ragged tails use their actual rows. The selected
varlen branch does clear two full 128-by-128 BF16 L1 operands for each tail C1
and C2, or 131072 local bytes per tail/head. Metadata records this local fill
payload separately as an unpriced service because no L1-fill resource rate is
declared. It is neither an HBM transfer nor evidence that MMAD executes fully
padded matrix dimensions. The 8192-token, chunk-64 scenario has no tail fill.
GM scratch activity is explicit: V_work and V-update, K-decay, H_work and the
H snapshot/FP32 state are separate reads/writes. Per-core tiling allocation
sizes are recorded in metadata, including the source's two-slot V, V-update,
K-decay and H workspaces.

## Supported domain

- BF16 K/V/W/U inputs; FP32 key-wise gates; FP32 state and workspace.
- ``dk == dv == 128``, ``chunk == 64``, equal key and value heads.
- One or more non-empty varlen sequences with meaningful ragged tails.
- Initial state present, final state stored, internal ``[K,V]`` representation,
  ``use_exp2`` and saved V_new.
- Prefill ``PCP == 1``; no speculative prefill tokens.
- Mixed prefill/decode is allowed: only the prefill scans are fused and bound;
  decode remains the generic `recurrent_scan` mechanism. Unsupported decode
  variants are not silently source-bound.

Other geometry, flags and branches raise a ``ModelingError``. The source call's
Prepare/Post-WU/Finalize stages, external QK L2 normalization, gate placement,
output norm/projection, public state transpose, state gather/scatter and host
launch are not modeled here.

## Bounded execution and tests

The recipe is a two-window `PipelineProgram` per core over lazy slot queues.
Only the active pipeline window and active streams are expanded; finished
tasks, programs and allocations are released.
The focused tests independently enumerate the source intervals and establish:

- plan/engine work conservation for tails, multiple sequences and split/unified
  copy pipes;
- source task ownership and the absence of idle-core supply for one-head work;
- two distinct state streams and two initialization/final-state writes;
- vector/copy bottleneck response with unchanged declared work;
- delayed input readiness, C1 -> V1 -> C2 -> V2 and next-chunk ordering;
- global H0 initialization, separate MTE1-to-compute dependency and independent
  slot refill for unequal sequence lengths;
- zero live tasks/programs/allocations and bounded live state for long chunks;
- ordinary `target_operators`/`graph_costs` selection, model-label invariance,
  unchanged generic/old explicit KDA paths, mixed prefill/decode behavior and
  visible rejection of unsupported geometry/flags.

Focused command:

```sh
PYTHONPATH=. /Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python \
  -m pytest tests/test_kda_state_pipeline.py -q
```

Passing these resource-contract checks does not establish source completeness,
instruction timing accuracy or agreement with device measurements. Not modeled
include cross-core/AIV synchronization and flag latency, `SyncAll` cost,
Fixpipe-to-vector response, MTE2/MTE3 queue latency, L1/UB residency and cache
hits, L1-tail-fill service duration, the L2 state transpose, scalar/control work,
and any measured stage duration. Each source phase is a resource envelope; its
internal row-wise copy/vector overlap is not reconstructed. The selected
program is a bounded source phase model, not a source-compiler reconstruction.

## 2026-10-29 retained case execution

The ordinary frontend selects this phase for the existing
`examples/cases/kimi-k3-a5-current-checkpoint-conditional.json` case. No physical
rate or model geometry is changed; the verification driver isolates its first
KDA layer and explicitly selects the source implementation.

```sh
PYTHONPATH=. .venv/bin/python tools/verify_kda_state.py \
  examples/cases/kimi-k3-a5-current-checkpoint-conditional.json \
  --rss-byte-factor 1 -o data/verification/kda-state-source-20261029.json
```

`--rss-byte-factor` describes the host's `getrusage` units (macOS bytes, Linux
KiB); it does not choose execution placement. The retained result runs both
the graph and isolated public APIs with input ready at 3 ms.

| Tokens / local heads | Source duration ms | Completed tasks | Peak live tasks | CPU wall s |
| --- | ---: | ---: | ---: | ---: |
| 512 / 12 | 0.037715487 | 2331 | 569 | 0.066 |
| 8192 / 12 | 0.596000520 | 35451 | 569 | 0.965 |

Both executions finish with zero live tasks/programs and no allocated buffers.
The two-invocation call protocol then completes in 1.192001040 simulated ms
with zero live state. Final peak process RSS is 31.16 MiB, including this call
check. CPU wall time is host-dependent and is not a device performance measure.
This model records GM workspace capacities but
does not simulate local L1/UB allocation lifetimes; zero local allocations is
not a capacity validation claim.

At 8192 tokens the independent arithmetic is 6.442450944 GFLOPs, 806879232 HBM
bytes, 429391872 MTE2 bytes, 125829120 MTE1 bytes, 226492416 MTE3 bytes and
150994944 Fixpipe bytes. The live supply is 12 AIC and their 24 AIV, with HBM
remaining a shared chip resource. Work serviced by the engine matches these
totals.

Against the independently recovered 276 closed call blocks, the prior generic
FWD_H projection was 0.106556866 ms/call and 29.409695 ms cumulative. This source
phase predicts 0.596000520 ms/call and 164.496143 ms cumulative, versus the
conditionally identified observed state stage's 298.471487 ms. Signed residual
changes from -269.061792 to -133.975344 ms (-44.89%). The stage-order condition,
uncaptured binary identity and valid token interval 8129--8192 remain; 8192 is
the explicitly selected scenario endpoint. No measurement supplied a rate.

This does not close KDA or Kimi accuracy. The unchanged prepare projection
still differs by -508.303740 ms across these blocks. That remains the next
dominant mechanism gap; wrapper copies and the complete model critical path
are outside this stage comparison.

The public small-model sample is
`examples/source-families/kda-state-control.json`. It uses synthetic rates and
two heads, so it is a frontend/packaging control rather than a hardware claim:

```sh
python -m mindie_sim graph examples/source-families/kda-state-control.json \
  --phase prefill --tokens 65 --sequences 2
```

The retained historical case can be priced through the Kimi scoreboard driver
without losing the explicit source choice:

```sh
PYTHONPATH=. python tools/model_scoreboard_kimi.py predict \
  --observed data/verification/kda-state-source-observed-20261029.json \
  --case examples/cases/kimi-k3-a5-current-checkpoint-conditional.json \
  --output-dir data/verification/kda-state-source-prediction-20261029 \
  --kda-implementation vllm_ascend_kda_chunk_state --timeout 10 --total-timeout 15
PYTHONPATH=. python tools/kimi_kda_alignment.py \
  --blocks data/verification/model-scoreboard-20261029-kimi-kda-call-blocks.json \
  --prediction data/verification/kda-state-source-prediction-20261029/prediction-kda.json \
  -o data/verification/kda-state-source-alignment-20261029.json
```

Without `--kda-implementation`, the tool inherits the case selection. If the
case declares none, it retains the original in-kernel-gates semantic control.
Only an explicit CLI value replaces a declared selection. The resulting
source phase is included exactly once in the four-stage partition.

The wheel verifier now defines the two-sequence ordinary graph and a real
two-invocation `call_costs` check under the installed package. The local source
checks have passed; the installed-wheel checks await the integrated wheel
run. These test work, readiness and finite active state, not measurement
accuracy.
