# Source-selected KDA Decode

`vllm_ascend_recurrent_kda_arch35` selects one **RecurrentKda kernel** for
BF16 Q/K/V, K=128, V=128 or 256, FP32/BF16 V-first initial state and 1–8
query rows per sequence. Gate/beta may be FP16/BF16/FP32. Value heads may
repeat fewer key heads, with integral HV/HK. The registered selector is
`runtime.kernels.lowerings.kda`; the frontend supplies raw dependencies and
replaces the normalization, gate and scan members covered by this kernel.
Caller projections, short convolution, gate/beta contiguous copies and
output reshapes remain outside its boundary.

The implementation follows committed local vLLM-Ascend revision
`8c30e4f491f73cba737a1dacc522894e437cdf3d`:

- `csrc/attention/recurrent_kda/op_kernel/arch35/recurrent_kda.h`,
  SHA256 `f35dd2c5fb604d6d9711c11139c755414d49fb27fc6aea0f3a4bec3d3adb0199`:
  queue allocations 151–198, ownership 235–270, QKV/gates 375–559,
  state input 560–596, compute/output queues 736–920.
- `csrc/attention/recurrent_kda/op_host/recurrent_kda_tiling_processor.h`,
  SHA256 `c9eac28bd528eb03bdeb456c4fdbed4951f716937570ee9d810e0fb5b3ef3fbd`:
  shape domain and AIV count 592–621; UB profile selection 648–756.
- `vllm_ascend/ops/kda.py`, SHA256
  `ca2f56280746f45a347578e7191decda716557c93aca69b5c3baf7b22a578803`, calls
  `torch.ops._C_ascend.recurrent_kda`. The torch adapter requests V-first
  inplace state updates and no duplicate final-state output tensor.

This explicitly selected source is distinct from the retained Triton KDA
BV=8 program and the cannbot DSL implementation. A profiler kernel name
`FusedRecurrentKda` alone does not establish compiled binary equivalence.

## Executed physical mechanism

The host launches `min(B*HV, physical AIVs)` blocks. Task
`sequence*HV + value_head` goes to `task_id % launched_blocks`, exactly as
the kernel's block-stride loop. Each block has its own declared UB allocation
and the corresponding physical MTE2_AIV, vector and MTE3 supply. A partial
grid cannot borrow idle AIVs or unrelated copy pipes.

For every value head, the program loads and converts the **whole q-row
Q/K/V and gate tensors**. Fewer key heads do not create shared Q/K residency
between different value-head tasks. `CopyInBeta` reads `q*HV` beta elements
for **each** value-head task; it does not load only q scalar elements. This
repeated payload is retained. Q/K may have independent token/head strides;
only selected payload is counted, without invented contiguous copies or
cacheline amplification.

The selected flags control Q/K L2 normalization, Q scaling, A_log
load/exp, dt_bias addition, raw-gate softplus or bounded sigmoid, gate exp
and optional beta sigmoid. A preprocessed beta producer stays external
when `use_beta_sigmoid=False`. Scalar reciprocal and flag issue costs are
not fitted into vector work.

Initial state is read once per head across its value tiles, converted to
resident FP32, and updated across all q rows. **Every query still emits a
state checkpoint and BF16 attention output.** A speculative acceptance
ratio never reduces the number of query rows executed. The first adapter
release originally supported no accepted-token input. The explicit
[accepted-state extension](kda-accepted-state.md) now carries either exact
addresses or a declared valid-range conditional domain. A bare nonempty
accepted-token flag remains unbound without that state-slot/index contract.

The host derives value-tile size and state/attention output queue depths
from real per-owner UB capacity. It evaluates `(1,1)`, `(1,2)`, `(2,2)`,
minimizes value-tile repetitions, then prefers deeper output queues. The
program reserves the kernel's actual buffers; the host's 2048-byte guard
and beta alignment slack are not fictitious live allocations.

One input state tile may be prefetched while the previous tile computes.
State remains resident across queries. Depth-two outputs are submitted
only after the following query's `Compute`; finite output-slot reuse waits
for the corresponding DMA. All attention/state writes follow one MTE3
FIFO. Cross-value-tile prefetch/output overlap is retained. Head-to-head
handoff conservatively waits for retirement; this does not assert a source
head-level DMA drain.

## Compaction, scope and remaining uncertainty

Complete launch waves with the same q share a resource cycle; their
leading/trailing partial waves keep individual heads. This preserves
unequal owner counts without rephasing them into unequal scaled phases.
Every phase's work is multiplied by the **number of real heads**, including
distinct state/QKV/gate/beta payloads. Buffers and physical owner count do
not grow with that multiplicity. `build(..., compact=False)` retains the
expanded head oracle. Under other resource consumers, compaction changes
fine-grained arbitration; `program_execution.kda_recurrent_schedule`
reports `serial_cycle_fluid_rephasing` explicitly.

Scalar prefix/state-index `GetValue` access counts are exported as
diagnostics. Their cache hits and issue latency, queue/flag instruction
issue, vector instruction issue and transcendental-specific throughput
are not independently characterized here. Scalar bytes are not silently
treated as guaranteed HBM misses, and no PMU span or residual is added as
a fixed delay. The existing explicit hardware invocation policy applies
once around the kernel. This preserves an analytic prediction while
showing what remains outside the physical service model.

## Local executable evidence

`data/verification/kda-recurrent-source-20261030.producer.py` generates the
retained JSON using the unchanged A5 hardware declaration from
`examples/cases/qwen38-a5.json`; the model label contributes no timing.
The controls contain q1/q4, B=1/4/32/128, a ragged q1/2/3/4/5/8 sequence
and four explicit competing-resource workloads. All programs release their
buffers/tasks, and no service precedes input readiness.

For B128, q4, HV=HK=12, K=V=128, expanded and compact isolated programs both
take 165.027003 us under that declaration. Engine tasks fall from 89,606 to
4,230. The four contention controls differ by at most 0.0572%; all resource
work is conserved within floating arithmetic. This is an implementation
comparison, **not measured accuracy or an end-to-end body result**.

The focused tests additionally execute FP32/BF16 state, all gate/beta
precisions, actual tail value tiles, shared copy-pipe ownership, all four
bandwidth axes, finite queue reuse, source MTE3 order and one outer launch.
