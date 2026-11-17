> **Historical Archive (Non-Gate Snapshot)** — Preserved from `sim` baseline commit `6800d9d7`.
> Original path: `docs/kimi-k31-vllm-ascend.md` · Point-in-time wave/audit/review record excluded from active gates.
> Retained for historical modeling provenance; consult active `docs/` guides for current interfaces.
> Current living specification: [`docs/architecture.md`](../architecture.md).

# Kimi K3.1 decode alignment with vllm-ascend

This adapter models the K3.1 D-worker's cache ownership and operator graph on
the common simulator engine. It does not execute PyTorch, load checkpoint
weights, initialize an NPU, or claim measured kernel timing accuracy.

The source baseline is vllm-ascend
[`c3d1d1fc1dbaee5c38c0601d23be99f14129fa46`](https://github.com/vllm-project/vllm-ascend/tree/c3d1d1fc1dbaee5c38c0601d23be99f14129fa46)
and vLLM
[`d07b7db9785564549d163937b5d5d9c04c76d626`](https://github.com/vllm-project/vllm/tree/d07b7db9785564549d163937b5d5d9c04c76d626).
The corresponding investigation is [issue #37](https://github.com/mindie-agent/sim/issues/37).

## Native input and explicit deployment choices

Pass the native target configuration to `compose_case`. The Kimi adapter reads
`text_config.dsa_config`: `dsa_layers` uses native **one-based** layer numbers,
like `linear_attn_config.full_attn_layers`; `index_num_heads`, `index_head_dim`,
`index_topk` and `index_rope_dim` describe each sparse indexer. Invalid layer
ownership and unimplemented TopK-reuse schedules fail explicitly.

The adapter emits structural `attention_execution` and `cache_allocator`
contracts. Execution and capacity consume those mechanisms, not a model-name
branch or a model-specific performance multiplier. Existing K3 configurations
without `dsa_config` keep their previous paths.

Cache precision is a deployment input. For the FP8 scenario, declare
`quant.kv_dtype="fp8_e4m3"`, `quant.index_dtype="fp8_e4m3"` and
`quant.index_scale_dtype="fp16"`. Do not infer runtime cache precision from
expert weight quantization or from the checkpoint's indexer training dtype.
Explicit `cache_representations` declarations retain precedence.

An external five-layer MLA DSpark is a separate structure: select
`dspark_attention_kind="mla"`, `dspark_block_size=5` and the actual
`dspark_target_layer_ids` in the native configuration adapter. It keeps BF16
weights/cache and does not inherit target DSA, sparse indices or output gates.

## Cache capacity and ownership

| Region | Physical row charge in the reference FP8 deployment |
| --- | --- |
| Target latent KV | 512 FP8 values + 64 BF16 RoPE values + 4 FP32 scales = **656 B** |
| Indexer | 128 FP8 values + one FP16 scale = **130 B** |
| External MLA draft | (512 + 64) BF16 values = **1152 B** |

These are byte charges; the indexer's key/scale planes remain page-local in the
reference implementation. They do not assert a globally contiguous scale plane.
See `attention/utils.py`, `worker/v2/attn_utils.py` and
`models/kimi_k3/paired_sfa_cache.py` in the pinned Ascend source.

The ordinary `CachePages` ledger owns a paired group per pipeline stage. For
each pool ID its price includes the main slot region, every indexer page and
the selected draft pages. KDA groups alias main backing with distinct live IDs;
they do not overwrite live attention pages. Balanced recurrent groups are
derived from that stage's main-layer count.

For 24 target layers, 24 indexers, five draft layers and block1536:

```
pool_ID_bytes = 1536 * (24*656 + 24*130 + 5*1152) = 37,822,464
history_IDs = ceil(sequence_tokens / (1536 * DCP))
state_groups = ceil(69 / 24) = 3
align_state_IDs_per_group = 2 + speculative_blocks + prefill_checkpoint_blocks
request_bytes = (history_IDs + state_groups * align_state_IDs_per_group)
                * pool_ID_bytes
```

Three proposals and no extra prefill checkpoints need five resident IDs/group,
not one resident checkpoint per historical token block. `none` mode uses
`1 + speculative_blocks`. The block table's position-indexed length is not its
simultaneously allocated state count. Runtime metadata can explicitly declare
`mamba_cache_mode`, `num_speculative_state_blocks` and
`num_prefill_checkpoint_blocks`.

The pool participates in ordinary admission, resize, release, rank budgets and
allocation reports. Inactive paired regions are reserved bytes, not free HBM.
Consequently concurrency must be solved again from this ledger. A user-provided
`memory.cache_page_groups` cannot simultaneously price the same paired backing.
A main page too small for a recurrent checkpoint is rejected explicitly.

## Decode graph

- TP projections and key absorption retain their local query heads. The DCP
  branch packs and gathers distinct head shards before attention. TP8/DCP8
  supplies **96 target heads**, while KDA remains **12 heads/rank**.
- Index history is token-sharded. Local score/TopK work precedes candidate
  packing and candidate-only exchange (`int32 index + FP32 score`, 8 B each),
  followed by global TopK merge. The full historical index cache is not gathered.
- Sparse attention is batched across requests into one selected FP8 QSFA
  boundary per layer. Output uses head AllToAll and weighted LSE merge;
  BF16 wire payload includes the reference packed LSE lanes. It does not issue
  one output collective per request or use generic output/LSE AllGather.
- The existing INT8 CP1/heads<=64 indexed-attention implementation is unchanged.
  FP8 QSFA binds its own explicit **analytical resource-envelope recipe**, with
  its source identity, cache traffic and gathered-head arithmetic preserved.
- External DSpark separates context feature projection/norm and all five
  context-KV precomputations/stores from proposal embeddings and query-body work.
  For three proposals this is **four context rows and three query rows/request**.

## Fidelity and supported domain

The new gathered-head DCP schedule covers **decode with PCP1**. A partitioned
prefill schedule is not substituted silently. Matrix and memory work still run
on the existing shared resource engine; tests validate structure, conservation,
admission boundaries and executable planning, not real NPU timing.
The alternate RecoverSSM page layout is explicitly outside this binding.
Conflicting explicit attention implementations are rejected instead of being
silently replaced by the FP8 envelope.

The FP8 source call's internal tiling, quantization instruction schedule,
pipeline overlap and host runtime calibration remain separate refinement work.
The external context path explicitly models the materialized QKV-A/KV-copy
path; alternative fused prolog selection is not inferred from a device label.
The BF16 draft attention body remains an analytical history/current envelope.
These fidelity limits are visible in operator metadata and must accompany
reported theoretical latency/throughput. No whole-model correction factor or
INT8 timing substitution is applied.
