# KDA GATE_PREPARE source stage

`mindie_sim.prediction.operators.kda_prepare` binds stage 0 of the retained
vLLM-Ascend `chunk_kda_fwd` call. The same explicit implementation also reuses
the existing source FWD_H plan. A selected prefill call has one source
GATE_PREPARE, the generic Post-WU phase, one source FWD_H and generic Finalize:

```python
from mindie_sim.schema.case import Software

runtime = Software.from_mapping({"kernels": {"lowerings": {
    "kda": "vllm_ascend_kda_chunk_prepare_state",
}}})
```

`vllm_ascend_kda_in_kernel_gates` and `vllm_ascend_kda_chunk_state` remain
selectable. The new implementation is prefill-only; it does not certify a
decode gate/recurrent branch. A mixed prefill/decode selection is rejected with
`kda_gate_prepare_source_requires_prefill_only_views` rather than charging a
whole generic gate and a source gate. Decode-only KDA keeps the generic
`recurrent_scan` with no source-boundary claim.

The selected source recipe uses compact owner runs by default. The diagnostic
control is available through `kda_prepare.make_op({...,"compact": False})`;
both use the same source work and owner mapping. This setting changes schedule
representation, not source selection or native precision.

## Source branch

Pinned `/Users/maoxx241/code/vllm-ascend-v41-private` at
`ae923905948f360f4b7ab06df52d0789a0c9316e`.

- `chunk_kda_fwd.cpp` lines 15-19 and 35-46 dispatch stage 0
  (`KDA_STAGE_GATE_PREPARE`) to `RunGateCumsum` followed by
  `KdaPrepare::RunChunkKdaPrepare`.
- `chunk_kda_fwd_common.h` lines 160-199 call the factory-selected
  `KdaGateCumsum::DispatchKdaGateCumsum<bfloat16_t>` when `computeGateInPrepare`
  is false, then execute the source `SyncAll` before prepare.
- `csrc/attention/kda_gate_cumsum/op_kernel/kda_gate_cumsum_kernel.h` owns the varlen task loop
  `seqNum * valueHeads` (line 188), BF16 row path and double row slots. For one
  sequence and 12 heads only 12 of the 64 AIVs are active.
- `arch35/chunk_kda_fwd_prepare.h` SHA-256
  `b808a1a1c6e21c0a3098433764675a1f507b04bc8f77d22eaffedd2c4de866c7` owns
  the paired-head score (`4731-4810`), paired AIC solve (`4636-4710`), the
  generic `ComputeAkkMergeCubeWorkspace` call (`4009-4018`), the six FP32
  GEMMs (`3505-3545`) and diagonal `ForwardSubDiag16StridedRegbase` (`609`,
  called at `3018`).
- `op_host/arch35/chunk_kda_fwd_tiling_impl.h` and
  `op_host/chunk_kda_fwd_tiling.cpp` lines 219-293 and 324-354 fix the
  prepare scratch and `prepareUsedCoreNum = blockDim`.
- `vllm_ascend/ops/kimi_kda.py` retains the caller beta sigmoid, QK L2 norm and
  output boundary.

## Stage structure

The source launch has two ownership regions separated by a global barrier:

1. **GATE cumsum, AIV.** Task count is `sequences * heads`. A task owns one
   head and walks its whole sequence (chunks serially, rows serially). Each row
   reads 128 BF16 values and applies the safe-gate row chain: cast, bias add,
   `expA` scale, `-1` scale, `exp`, `+1`, duplicate-one, divide, lower-bound
   scale, `1/ln 2`, accumulator add and output copy, then writes 128 FP32
   values. The sum is reset once per 64-row chunk. The model charges ten
   ordinary 128-lane operations and two special-function lane operations per
   row plus the reset. A-log, bias and `cu_seqlens` require 532 bytes per task.
   The declared `reg_special_lanes` resource supplies Exp/Div independently
   of ordinary vector arithmetic; no latency is inferred from observations.
   The model
   keeps a per-owner two-slot row pipeline: row `i+1` may load while row `i`
   accumulates, but the accumulator stage itself is serial. No per-row latency
   constant is invented. With one sequence and 12 heads, only 12 AIVs are
   placed; idle AIVs are not added as supply.
2. **Barrier.** A dependency-only stage reproduces `SyncAll` after cumsum. It
   carries no fabricated duration.
3. **Prepare, paired AIC/AIV.** The source task is
   `chunk * (heads / 2) + pair`, advanced by `usedCoreNum` (the AIC count). A
   task computes two heads; each AIV subblock owns one head. The model keeps a
   finite per-core task queue. This phase abstraction serializes successive
   paired tasks and the two score blocks. It does not reproduce the source's
   deferred-solve overlap.

Per full head-chunk:

| Phase | Owner | Source work |
| --- | --- | --- |
| Score factors | AIV per head | 108,800 B read and 147,456 B written, including score scratch, qg, qgScaled, wSeed, uSeed and finalKg; 368,704 ordinary and 36,864 special-function lane operations |
| Score | AIC | two BF16 32-row blocks with key width 32 then 64; `2*2*32*(32+64)*128 = 1,572,864` FLOPs; 57,344 B GM/MTE1 input per head; 24,576 B Fixpipe output direct to UB, not HBM |
| Diagonal | AIV per head | four serial 16-row blocks, each with 14 dependent row phases; `119` Mul + `119` dependent Add + `14` row Add + 16 GetValue/SetValue = 4,048 vector lane operations per block |
| Seed/export | AIV per head | diagonal completion precedes 16,384 B FP32 X seed and 8,192 B BF16 Aqk publication; after both heads' merges, final Akk reads 16,384 B FP32 and writes 8,192 B BF16 |
| Merge | AIC | six serial IEEE-FP32 GEMMs: four `(16,16,16)` and two `(32,32,32)` = 163,840 FLOPs; each has GM load, MTE1 operand, Cube and Fixpipe publish; 24,576 B read and 12,288 B written per head |

The model exposes `bf16` and `fp32` matrix work separately. The solve has an
explicit per-GEMM GM input / L1-L0 operand / cube / Fixpipe completion chain,
and the diagonal retains its per-row dependency. Gate and score-factor reads
are AIV copy work; score and merge reads are AIC work. Direct-UB score output
is Fixpipe service and is not charged as HBM. If the device declares split
AIC/AIV copy pipes they retain separate ownership, otherwise one `mte2` pipe
serves both. Output and operand bytes are source interval totals.

Retained geometry `T=8192, H=12, K=V=128, chunk=64` gives 1,536 head-chunks,
768 paired tasks, 24 tasks on each of 32 AICs, `2,415,919,104` BF16 FLOPs,
`251,658,240` FP32 FLOPs and 9,216 small FP32 GEMMs. The gate cumsum uses 12
active AIVs for one sequence, not 64 and not the 32 AIC count.

## Frontend placement

The binder replaces exactly the generic prefill `delta_gates`,
`chunk_products` and `triangular_solve` members with one `GATE_PREPARE` op
whose dependencies are the external decay, the external beta sigmoid and QK
normalization. The generic `wy_transform` remains a separate consumer, as do
the projections, output norm/gate and output projection. The existing FWD_H
binder then replaces only `recurrent_scan`; its dependencies are rewritten to
the Post-WU member and all sequence outputs move behind the one FWD_H launch.
The generic prefill gate is removed, so it is never charged alongside the
source cumsum.

## Scratch and GM assumption

Metadata records the tiling workspaces: two FP32 Aqk/Akk matrices, solve and
score scratch for the used AIC count, and the separate Post-WU scratch. These
are footprint declarations, not timed allocations or capacity validation.
The selected recipe explicitly charges declared GM reads and writes to HBM.
That is an off-chip service assumption, not proof that the source misses L2.
Direct Fixpipe-to-UB score output is excluded from HBM.

## Approximations and remaining unknowns

- Score-factor work follows `PrepareScoreFactorsBulk`, including the first
  block's 8,192 B K-negative padding and all five BF16 exports. `storeQG=true`
  is part of this selected domain. The factor ledger is retained separately
  from its later AIC consumption.
- Arithmetic counts describe semantic lane operations. They do not claim
  compiler instruction counts, scalar address issue, register-load timing or
  a complete register pipeline model.
- The diagonal phase keeps the source row dependency but not each masked
  instruction's issue latency or barrier duration.
- Paired tasks and score blocks are serialized; cross-task deferred solve and
  score/diagonal overlap are omitted. This is an explicit timing approximation,
  not a claim of the exact native schedule or a universal latency upper bound.
- The GM/off-chip split, L1 cache residency and actual MTE1/L0 packet counts
  are not measured.
- Non-64 tails are rejected. The comparison explicitly selects 8192 valid
  tokens; the captured buffer/chunk geometry alone permits 8129--8192. Generic
  Post-WU remains outside this source stage; no dense Prepare/Post-WU fusion
  is assumed.

## Retained-call execution

The source-selected expanded control ran on committed runtime `cda4df21`.
The worker took 11.2684 seconds and peaked at 49.5 MiB. Its runtime package
hash was identical before and after execution. The Prepare projection
completed 482,497 tasks, peaked at 7,845 live tasks, and finished with zero
live tasks/programs. These are local simulator costs, not device timing.

The comparison uses the independently anchored 276 four-launch blocks in
`data/verification/model-scoreboard-20261029-kimi-kda-call-blocks.json`.
The selected scenario is 8192 valid tokens, one sequence and 12 heads.
The source-compatible stage order is conditional: the captured stage
scalar and loaded binary identity are not recovered.

| Stage | Observed cumulative ms | Predicted cumulative ms | Signed gap ms |
| --- | ---: | ---: | ---: |
| GATE_PREPARE | 517.895295 | 103.968770 | -413.926525 |
| POST_WU | 20.208037 | 7.813988 | -12.394049 |
| FWD_H | 298.471487 | 164.496143 | -133.975344 |
| FINALIZE | 45.423019 | 13.891535 | -31.531484 |
| Complete core | 881.997838 | 290.170437 | -591.827401 |

The previous generic Prepare prediction was 9.591555 ms over the same
276 calls. Source ownership, work and dependencies therefore recover
94.377215 ms of its former -508.303740 ms gap. The remaining Prepare gap
is still 413.926525 ms; this change does not establish accuracy convergence.
No observation duration was used to choose rates or add timing work.

Actual prediction, runtime identities and execution receipt are retained in
`data/verification/kda-prepare-source-prediction-20261029/`; the compact
comparison is `data/verification/kda-prepare-closed-call-alignment-20261029.json`.
The 68 focused controls include multi-sequence ordinary frontend placement,
independent work arithmetic, owner coverage, delayed inputs and zero final
live state. The expanded implementation remains a diagnostic schedule
control; its task count is unsuitable as a default whole-model expansion.

## Compact owner execution

`kda_prepare_compact.py` retains one coupled resource task per active gate AIV
and one per active Prepare AIC. Every gate owner completes before any Prepare
owner begins. Owners with no source task contribute no work or resource supply.
Each task preserves its physical resource demand and the exact per-unit shares
of the expanded phase template.

The nominal gate time is computed once by the same Engine from that owner's
complete task-length sequence, including the two live row slots, per-task
auxiliary reads and chunk resets. It does not restart the pipeline between
heads. Prepare uses one actual paired-task execution, multiplied by that AIC's
integer task count because the selected source abstraction serializes paired
tasks. A bounded 64-entry cache keys these scalar times by geometry, rates,
precision peaks, service modes and physical lane aliasing. No observation
duration enters this calculation.

Under isolated per-owner supply, these nominal times retain the expanded
finite schedule, including shared load/store service inside one gate owner.
During main execution, concurrent owners and other kernels use coupled fluid
resource progress. This is an explicit contention approximation: it does not
preserve every discrete interleaving or claim a universal upper/lower bound.
The diagnostic expanded control remains available for counterexamples.

The cold nominal calculation still scales with gate rows, but its active
state remains limited to two rows; repeated layers reuse the scalar result.
The main Engine never receives one task per gate row or Prepare instruction.

Committed compact runtime `618237c2` (verifier correction `7fb0876c`) produced
the following device-only controls; all inputs were released at 3 ms and that
release delay is excluded from reported duration:

| Tokens / H12 | Expanded ms | Compact ms | Cold / warm wall seconds | Compact tasks / peak live |
| --- | ---: | ---: | ---: | ---: |
| 64 | 0.012084391 | 0.012084391 | 0.0227 / 0.0173 | 50 / 47 |
| 128 | 0.012861361 | 0.012861361 | 0.0299 / 0.0225 | 62 / 59 |
| 512 | 0.029074610 | 0.028818641 | 0.0674 / 0.0394 | 102 / 99 |
| 8192 | 0.376698442 | 0.370555186 | 0.4371 / 0.0389 | 102 / 99 |

The verifier peaked at 47.27 MiB and ended every run with zero live tasks,
programs and allocations. The largest per-resource service difference is
`9.33e-14` relative. The 8192-token expanded control created 482,494 tasks;
the earlier isolated-cost projection counted three additional boundary tasks.
The 1.63% timing difference at 8192 tokens is the declared owner-contention
approximation. It is not a claim that the compressed schedule is more accurate.

The complete Kimi core worker now takes 1.1169 seconds and 36.73 MiB, with an
unchanged package fingerprint throughout the run. Its prediction is
1.045199 ms/call, or 288.474898 ms over 276 closed calls, versus the observed
881.997838 ms. Prepare alone is 102.273231 ms, leaving **415.622064 ms** of
negative residual. The other three stage predictions are unchanged.

The report and reproducible command are:

```sh
python tools/verify_kda_prepare_compact.py \
  data/verification/kda-prepare-source-prediction-20261029/input-kda.json \
  -o data/verification/kda-prepare-compact-control-20261029.json
```

`data/verification/kda-prepare-compact-prediction-20261029/` retains the actual
core worker inputs/output. `kda-prepare-compact-alignment-20261029.json` preserves
the prior expanded values and the current conditional stage gaps. The 80
focused tests cover finite nominal matching, physical service conservation,
input readiness, rate-sensitive caching and bounded state. An independent
per-unit review additionally checks split copy, shared copy and special-function
service aliased onto the vector pipe; its report is
`data/verification/kda-prepare-compact-review-20261029.json`.
