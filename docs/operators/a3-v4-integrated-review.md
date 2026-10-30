# Final V4 A3 caller integration review

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

Reviewed mainline `7b3d7c8b` (independent `mla.o_a` precision) and `e0c44aee`
(retained caller inputs) on runtime `287f932b`. The integrated runtime and input
files match the previously reviewed source commits. This is a fresh
post-integration check by the patch author, not a claim of separate authorship.
No runtime or existing input was edited during the review.

No blocking finding:

- The c4/c8 parent SHA, machine and runtime parameters are preserved. Only the
  explicit TP caller, `mla.o_a` override, input identity and evidence metadata
  change. Geometry is q2/B4 or q2/B8, DP1/TP8/EP8, scope `body`, conditional
  context4096; no MTP auxiliary, sampler or whole serving step is substituted.
- Both actual 43-layer graphs have exactly 43 finalize AllGathers, versus zero
  in the parent. Each depends on that layer's `route_unpack` and reconstructs
  full M8/M16 from local M1/M2. Router remains full-row; only top-k is sliced.
  All 86 expert matrix operators are identical after removing the newly added
  caller metadata. There is no second expert or generic/native transport charge.
- Every grouped output-A is BF16 and has one matching resident `wo_a` term of
  8,388,608 bytes per layer/rank; all 43 terms total 360,710,144 bytes. Its
  activation quantizer is absent, while `q_b` and output-B remain INT8. Existing
  parent-MLA inheritance, child precedence and prepared-MX consistency tests
  pass. Two real isolated output-A Engine calls wait until 11 ms and release all
  tasks/buffers. Five focused integration tests pass.
- Generic AG remains the baseline. The AICPU Mesh arm expressly states its
  algorithm identity is conditional. The evidence preserves its lower predicted
  span even though c4 becomes less accurate; no algorithm is chosen to recover
  a passing result. V2 protocol and unknown group-list values remain distinct
  limitations. Prior conditional body results are not relabeled as a new full
  mainline replay.

The compact review receipt and producer
retain hashes and checks. No 66-source or whole-body sweep was repeated.

Ready for two targeted profiler replays:

- `examples/cases/high-concurrency-20261030/a3-v4-caller-conditional/v4flash-a3-tp8-c4-caller-bf16.json`, q2/B4/context4096.
- `examples/cases/high-concurrency-20261030/a3-v4-caller-conditional/v4flash-a3-tp8-c8-caller-bf16.json`, q2/B8/context4096.

These use generic AG. The separately retained AICPU arms are optional algorithm
counterfactuals, not replacements for the observed caller evidence.
