# A3 native FA/RoPE capability pilot

Status: predeclared capability manifest. The worker accepts only manifest-declared
`split` in `{train, heldout}` and preserves it exactly; it performs no splitting
heuristics and never forces results to train. This example manifest declares all
four capability cases as train. No device work, lease, network or heldout timing
was used to create these artifacts. `sim_audit`
owns the actual default-queue pilot.

## Cases (4; all `target_match=false`, example-declared `split=train`)

| Case | Native primitive | Geometry | Scope |
| --- | --- | --- | --- |
| `fa-prefill-b1-hq3-hkv1-q32-kv32-d256-causal` | `torch_npu.npu_fusion_attention` | B1, 3 Q / 1 KV heads, Q32/KV32, Dqk=Dv=256, bottom-right causal, scale 1/16 | Standalone prefill FA representative of Qwen27B TP8 GQA 24Q/4KV, 1 replicated KV head. Not full model/cache/framework/MLA/sparse/FP8. |
| `fa-decode-b1-hq3-hkv1-q1-kv128-d256-fullvisible` | same | B1, 3 Q / 1 KV heads, Q1/KV128, D=256, fully visible mask, scale 1/16 | Standalone decode-shaped primitive with explicit full-visible mask; not paged/cache/framework coverage. |
| `rope-half-b1-s8-h3-d64` | `torch_npu.npu_rotary_mul` | B1, S8, H3, D64, BSND, half | Standalone rotated plane; Qwen partial_rotary_factor 0.25 gives 64 of 256. Not mRoPE/concat/QK composite. |
| `rope-half-b1-s128-h3-d64` | same | B1, S128, H3, D64, BSND, half | Same primitive capability scope, longer sequence. |

Protocol: 10 warmup calls before **each** of 2 trials, 30 samples/trial, 240 raw
blocks total. One explicit stream; events precreated/reused; inputs preallocated;
native default outputs allocate (no `.out`); only the latest output reference is
retained. Budgets: 24 s active case, 60 s setup, 240 s global work, 60 s cleanup;
external supervisor >=370 s. Cleanup checks the cooperative deadline again after
completion synchronization, so a synchronized-but-over-budget cleanup is failed
and nonzero.

FA uses the installed spelling `pre_tockens`/`next_tockens`, BNSD, explicit
contiguous bool `atten_mask`, `keep_prob=1.0`, `sparse_mode=0`, `head_num=3`,
`scale=0.0625`. The initial correctness dispatch is executed on the same explicit
measurement stream as the pilot and synchronized there before any CPU copy or
reference. The mixin supports a caller without a stream by using the current
stream and synchronizing the device. It validates all 7 returned items' actual types/shapes/dtypes but
interprets only output0. A dispatch error is `pending`; malformed, non-finite or
incorrect output0 is `failed`, never pass/pending. The GQA CPU reference repeats
K/V heads by `qheads/kvheads` and computes full qk/softmax/pv from the actual
BF16-rounded CPU inputs.

RoPE passes fully materialized contiguous BF16 cos/sin of exactly x's shape,
with nonconstant positions 0..S-1 and base 10000. The CPU half-rotation reference
is `cat(-x[...,D/2:],x[...,:D/2])`, using the rounded cos/sin values. No
broadcast, alias, mRoPE or full-model claim is made.

## Provenance is optional attribution

`source_contract` is not a measurement-validity gate. It may be absent or contain
any supplied attribution object; the worker preserves it verbatim in the plan and
results, or records `{}` when absent. It never invents a hash. Shape, dtype, mask,
native API and protocol validity remain required and independent of provenance.

## Commands

```bash
python -m mindie_sim.a3_operator_worker_attention \
  --manifest examples/operator-attention/capability.manifest.json \
  --results /tmp/a3-attention/capability.json \
  --run-id a3-attention-capability-<id> --device npu:0 --dry-run

python -m mindie_sim.a3_operator_worker_attention \
  --manifest examples/operator-attention/capability.manifest.json \
  --results /tmp/a3-attention/capability.json \
  --run-id a3-attention-capability-<id> --device npu:0
```

`cpu_submit_ms` stops immediately after the native API call. Output0 extraction
and release of the returned auxiliary items happen after that clock and before
end-event submission; the returned tuple stays alive until the call returns and
is never included in the submit interval. The wall clock still includes that
host-side extraction/release. The capabilities are independent physical conditions, not a split-origin claim.
For heldout-declared manifests, `split=heldout` is retained in the plan, dry-run
output, per-case results, and summary counts; the worker never relabels it.
Dry-run imports no torch and writes nothing. A real run writes partial results
after each case and exits nonzero for partial/failed/pending cases or
cleanup/persistence errors. `analytic_ms` is always null; no calibrated or
validated performance label is emitted. No silent retry is performed.

## Integration

`mindie_sim.a3_operator_attention_adapter` connects campaign requests to this
standalone worker through `attention_manifest_from_campaign`,
`dry_run_campaign_attention` and `run_campaign_attention`. Requests must provide
the exact FA/RoPE geometry and semantics; missing fields remain explicit pending
conditions. No manual patch to the generic worker is required.

`attention_results_to_campaign` preserves per-trial samples, all three clocks,
native output allocation and the declared split. Use
`campaign_requests_from_attention` and `attention_query_from_request` to build
pre-use curve queries from the original manifest. `target_match=false` remains
a standalone primitive candidate and does not establish full-model equivalence.

`manifest.schema.json` and `results.schema.json` describe the current contracts.
The manifest uses `kind=a3_attention_rope_manifest`; results use
`kind=a3_attention_rope_capability`. Provenance stays optional as described above.
Native dispatch and output validation run against the selected CANN/torch-npu
stack. Cooperative budgets cannot preempt a hung driver call; the external
supervisor supplies the hard bound.
