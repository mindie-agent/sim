# Staged GQA prefill

`StagedGQAPrefill(query_tile=64, key_tile=128, slots=2)` is an explicit
analytical phase recipe for generic causal GQA prefill. It is registered as
recipe kind `staged_gqa_prefill`; the ordinary frontend selects it only when the
runtime declares:

```json
{"kernels": {"recipes": {"flash_attention_prefill": {
  "kind": "staged_gqa_prefill", "query_tile": 64,
  "key_tile": 128, "slots": 2
}}}}
```

The unconfigured default remains `flash_attention`. Decode and draft selectors
are unchanged. A prefill op carries `metadata['gqa_prefill']` with per-request
`(query_len, context)` pairs, query/KV head counts, head dimension, dtypes,
causal/window and CP declarations. The recipe accepts `prefill` and
`draft_prefill`; the batched contract retains every request separately.

## Declared mechanism

One task is `(request, query_head, query_token_tile)`. Its KV head is
`query_head // (query_heads / kv_heads)`. A task reads Q once, keeps Q and the
online max/sum/numerator resident, and reloads K/V for every key tile. K/V is
therefore charged once per query head, not once per KV head: the recipe does not
reuse the flattened decode GQA mapping across heads.

Tasks are cycled over the declared physical owners by global task index,
`owner = index % core_count`. Owners process their own stride lazily; no owner
borrows another owner's idle supply. Key tiles are generated through a bounded
`slots` pipeline window, so live state is proportional to owners and slots, not
to all query/key pairs or all output tasks.

For a task with `R` actual query rows and actual horizon
`H = context + last_query + 1`, each key tile of `N` actual key rows executes:

```text
QK cube      = 2 * R * N * D FLOPs
PV cube      = 2 * R * N * D FLOPs
vector       = 7*R*N + R (score scale/mask and softmax) + R*(3D+9) (online)
             = 7*R*N + R*(3D+10)
K/V HBM      = 2 * N * D * dtype_bytes
```

The final output is written once per task, with `R*(D+1)` declared vector work
and `R*D` output bytes on HBM and the declared store pipe. Q is read once per
task. On-chip score/probability/accumulator copies are not HBM traffic. Cube
service uses the full executed rectangle `R*N`, including the causal-masked
diagonal. The logical valid-pair count is retained separately by
`StagedGQAPrefill.report` as `logical_pairs`; `executed_rectangle_pairs` and
`causal_masked_pairs` make the physical/logical distinction explicit. The report
also records actual tasks, key-tile loads, Q/KV/output bytes and policy values.

Stage order is `q_load -> {k_load -> qk -> softmax -> (v_load,
probability_operand) -> pv -> online_update} -> final_store`. `online_update`
has an explicit `previous` dependency, so the prior accumulator is complete
before the next key tile updates it. Probability data is ready before PV. Every
stage depends on the task-ready event, including K/V prefetches. Declared L1,
L0A, L0B, L0C and UB buffers are acquired per physical owner/slot and released
by the owning task; supplied capacities are enforced, not ignored.
Score scaling and masking begin after QK finishes; the PV accumulator remains
allocated until the online update consumes it. The default 64/128 tiles were
chosen to fit the retained A3 L0A/B 64 KiB and UB 192 KiB declaration at D256,
independently of observation durations. A declared larger tile can require a
different micro-tile implementation: this model rejects incompatible resident
working sets instead of shrinking allocations or entering a stalled schedule.

## Fidelity and source boundary

This is a practical analytical phase model, not a source planner, captured
dispatch or accuracy claim. The pinned ops-transformer source at
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8` was inspected for mechanism evidence:

- `attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp`
  lines 748-864 iterate query heads and outer query blocks as tiling axes;
  lines 1095-1178 construct BMM1 over `sOuterFactor x sInnerFactor x headSize`.
- `attention/fused_infer_attention_score/op_host/arch22/
  fused_infer_attention_score_tiling_v3.cpp` lines 1403-1408 show `RouteToFia`
  rejecting TND without rope for head dimensions other than 64/128/192 (thus
  D256 is not routed there).
- `attention/fused_infer_attention_score/op_host/flash_attention_infer_tiling.h`
  lines 220-253 and 373-399 independently produce one query head per task for
  query length 8192, with source query tile 128 and KV stack 512.

These lines support a declared head-local tiling policy. After the first route
rejects D256, `IsUsingFAI` can select the separate FAInfer path before IFA or PFA;
it depends on sparse mode, precision flags and PA descriptors not retained by
the tensor cohort. These sources do not prove what a historical capture dispatched,
what private copy rates applied, or the native instruction schedule. The model
does not claim source equivalence. Scalar, address-generation, control and cache
miss costs are unmodeled and no latency floor is invented. The recipe's
`report(op, resource_roles=...)` explicitly lists missing input, operand and
output transfer roles. Existing A3 profiles without private copy rates are not
calibrated here.

The bounded acceptance control is a single H6/KV2/D256 request with 8192 query
tokens uses synthetic service rates but the actual A3 local-buffer capacities,
20 matrix owners and 40 vector units. It completes with the default tile policy,
within 30 seconds and 512 MiB. The retained public A3 reference-case runs in
`data/verification/paged-fia-prefill-20261029.json` also record wall time,
executed service, physical work and missing private-copy roles. These are single
operator controls, not a claim that repeated full-model detailed execution meets
a 60-second budget.

| Public A3 query-8192 probe | Envelope ms | Selected ms | Conditional observed mean ms | Selected difference |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5, four query heads | 0.5085 | 1.8608 | 2.4765 | -24.86% |
| Qwen3.6, six query heads | 0.7627 | 2.7851 | 3.6784 | -24.28% |

Both use unchanged reference hardware and caller-declared paged context zero.
They finish in 5.52 and 8.48 seconds, respectively, with peak process RSS
40,288,256 bytes across the run. No observation has been bound to a specific
layer occurrence, and all roles remain `unassigned`. The remaining discrepancy
is not treated as startup or a fitted rate.

## Test boundary

`tests/test_attention_prefill.py` derives traffic, FLOPs and vector work from
independent closed-form geometry, including per-query-head KV reload, causal
rectangle vs logical triangle, ragged batches, query tails, a partial last key
tile, nonzero readiness, inactive-owner supply and resource/buffer release. It
also checks the actual Engine service against those ledgers.
`tests/test_attention_prefill_frontend.py` exercises ordinary frontend recipe
selection, batched per-request metadata, draft-prefill selection, unchanged
default/decode selectors, source-required binding precedence and real graph
execution.

## Separate FAInfer workspace candidate

`data/verification/paged-fia-workspace-evidence-20261029.json` retains exact
source hashes and a candidate physical ledger. The source FAInfer path binds
FP32 scores, BF16 probabilities, FP32 PV outputs and FP32 running-output spills
to GM workspace. QK publishes scores; softmax reads them and publishes
probabilities; PV reads probabilities and publishes output; rescale consumes
that output. At query tile 128 and D256, each vector half has 64 rows but its
output loop holds 32, so nonfinal running outputs also spill and reload.

For query 8192, context zero, D256, one KV head and the source 128/512 tiling,
the declared causal geometry gives:

| Query heads | Score write/read | Probability write/read | PV write/read | Running output write/read | Total GM bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1,090,519,040 | 545,259,520 | 570,425,344 | 503,316,480 | 2,709,520,384 |
| 6 | 1,635,778,560 | 817,889,280 | 855,638,016 | 754,974,720 | 4,064,280,576 |

These are address-space transactions, not established HBM traffic. A completely
offchip scenario would require 2.258/3.387 ms of service at the case's existing
1.2 TB/s HBM rate; that division is neither extra measured time nor a parameter
fit. Source workspace allocation is 55,050,240 bytes for 20 owners and three
slots, below the retained 176,160,768-byte L2 capacity. Capacity alone does not
establish residency, cache hit rate, active-owner transfer supply or contention.
An explicit source scratch policy needs these distinctions before replacing the
analytic resident-state recipe. No part of this candidate ledger is silently
added to its default work.
