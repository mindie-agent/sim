# Online pointer attention

This extension models the selected one-sequence, 2D pointer branch of vLLM
unified attention. It separates source tensor operations, logical memory
operands, matrix work, physical service rates and dependency ordering.

The implementation is `vllm_triton_unified_attention_2d_pointer`, selected by
recipe `triton_unified_attention_2d`. Inputs are BF16 or FP16 Q/K/V, with FP32
running maximum, sum and accumulator. KV quantization, tensor descriptors,
segmented 3D attention, sinks, ALiBi, QQ bias, softcap, sliding/chunked windows,
multimodal prefixes and R-SWA are outside this selected branch. Unsupported
branch parameters are not silently interpreted as this branch.

## Source boundary

The source checkout is `752a3a504485790a2e8491cacbb35c137339ad34`:

| Source | SHA256 |
| --- | --- |
| `vllm/v1/attention/ops/triton_unified_attention.py` | `a653bd1e1c200be4189804956f680bb462ca80e582b418d279b5afc6ff855311` |
| `vllm/v1/attention/ops/triton_attention_helpers.py` | `8c730611e7b3c5fb7579ec7846d56a2ab7e348ce06b39136da22072ecc363c95` |

These are the supplied local source files under
`/Users/maoxx241/code/vllm-v0.25.1`. They do not identify a historical capture's
installed Triton compiler or generated binary.

`query` is the new query length; `context` is the preceding KV prefix. The
source sequence length is their sum. `heads/key_heads` is the GQA ratio. The
declared capability defaults to `(8,9)`; no device query occurs.

The wrapper at lines 932-955 chooses:

| Source condition | BM | BQ | Tile | Launch parameters |
| --- | --- | --- | --- | --- |
| Ordinary branch, ratio at most 16 | 16 | BM//ratio | 32 | Triton defaults unspecified |
| Ordinary branch, ratio above 16 | next power of two of ratio | BM//ratio | 32 | Triton defaults unspecified |
| Capability family 10.x, D=256, query>1, ratio<=16 | 32 | BM//ratio | 128 | 8 warps, 2 stages |

Outside the tuned branch, an explicit `block` can request a manual kernel tile;
metadata distinguishes that from the wrapper tile. In the tuned branch the
wrapper's 128 selection remains authoritative. Manual tiny tiles are arithmetic
source controls; this extension does not establish compiler shape admission.

The launch grid is `(floor(query/BQ)+1, key_heads)`, including possible
metadata-only CTAs. This extension selects 2D explicitly; it does not infer 3D
segment-buffer availability from a query length of one.

## Geometry, memory and state

The packed row expression is:

```
query_position = query_block * BQ + row // ratio
query_head = kv_head * ratio + row % ratio
```

For a non-power-of-two ratio, adjacent CTAs can address overlapping query rows.
The model retains those source accesses. Head width is padded to the next power
of two for both dot products; Q/K/V/output payload includes only valid lanes.
The causal horizon is
`min(context + query_block*BQ + (BM-1)//ratio + 1, sequence_length)`.
Noncausal attention reads the entire sequence. Every executed QK or PV tile
retains its rectangular `2*BM*tile*padded_head_dim` matrix work.

Sequence resolution at helpers lines 45-106 makes five logical int32 loads over
three distinct addresses for every grid CTA. The early-return test is later,
at kernel lines 319-320. Page-table loads at kernel lines 420-426 are unmasked:
every tile has `tile` logical int32 operands, even when its final K/V payload is
partial. Repeated page indices are counted as source operands; no claim is made
that each operand causes a separate HBM transaction.

Only an explicitly declared `register` hardware pool constrains the lower bound
`BM*padded_head_dim*(2+4) + 8*BM` for Q, FP32 accumulator, M and L. Without that
pool, local state is **unconstrained**. UB, L1 and shared-memory pools are never
substituted: the source does not establish that these state tensors occupy them.
Metadata states this conditional scope without guessing the eventual hardware.
When declared, the register allocation belongs to the CTA's owner and lives
from Q load through output. It excludes K/V staging, score/P temporaries,
register allocation granularity, spills and shared-memory layout, so it does not
establish complete compiled occupancy or feasibility.

## Typed source inventory

`Work.resources` holds operation-specific source quantities. `vector_ops` is
zero. `SERVICE_ROLES` only tells the synthetic runner which physical resource
to declare for each role; it supplies no default rate to real hardware.
`Device.service_modes` or direct resource declarations must provide the rates.
Missing `fp32_exp`, for example, raises `physical_resource_rate_missing` instead
of substituting the generic vector rate.

Let R=BM, C=tile, Dp=padded head width. Counts below are per executed tile.

| Source expression | Role | Operand count |
| --- | --- | --- |
| Token-position comparison | `int32_compare` | R*C causal; C noncausal, then broadcast |
| Query-mask and sequence-mask combination | `predicate_and` | R + R*C |
| Score scale | `fp32_mul` | R*C |
| Score mask selection | `fp32_select` | R*C |
| `tl.max(S,axis=1)` | `source_tl_reduce_max_elements` | R*C inputs in R groups |
| `maximum(M,row_max)` | `fp32_max` | R |
| `m_j > -inf` | `fp32_compare` | R |
| `where(...,m_j,0)` | `fp32_select` | R |
| `M-m_j`, `exp(...)` | `fp32_sub`, `fp32_exp` | R each |
| `S-m_j`, `exp(...)` | `fp32_sub`, `fp32_exp` | R*C each |
| `tl.sum(P,axis=1)` | `source_tl_reduce_sum_elements` | R*C inputs in R groups |
| `L*alpha`, addition of row sum | `fp32_mul`, `fp32_add` | R each |
| Accumulator rescale | `fp32_mul` | R*Dp |
| `P.to(V.dtype)` | `cvt_fp32_bf16` or `cvt_fp32_fp16` | R*C |

After the final tile, division and output conversion have separate stages, each
with R*Dp tensor operands. Masked output stores write only valid rows and head
lanes. These conversion operand counts do not identify compiled predication,
warp issue or dead-lane elimination.

The reduction roles preserve input cardinality. They are neither scalar
max/add instruction counts nor a guessed shuffle tree. A hardware lowering can
replace them when its compiler/layout evidence is available. Rate tables for
actual instructions cannot be attached to these roles without that translation.

The current inventory does not price address/index arithmetic, preloop mask
construction, register initialization, compiler synchronization, cache
transactions, instruction issue, occupancy or software pipelining. Metadata
retains these omissions independently of measurement availability.

## Dependency graph and ownership

The page-table result precedes the K/V payload addresses. QK depends on K;
PV depends on V. Score scale and masking precede the row maximum. Running
maximum, comparison and clamp selection have separate dependencies.

After the clamped maximum:

- Alpha subtraction/exp feeds accumulator rescale and `L*alpha`.
- P subtraction/exp feeds its low-precision cast and the row sum.
- PV waits for P cast, rescaled accumulator and V.
- The new L waits for `L*alpha` and the row sum.

The next tile waits for all carried M/L/acc state. Output division waits for the
completed tile loop; output cast waits for division; the store waits for the
cast. A slow sum reduction therefore does not falsely block PV, while it still
blocks the next iteration and final normalization.

Each CTA is assigned one owner. Matrix and typed arithmetic service is pinned
to that owner's physical resources; HBM is unpinned and remains shared.
Independent owners can execute concurrently. The schedule declares one CTA per
owner, one tile in flight and static CTA assignment. These are explicit virtual
schedule choices, not measurements of CUDA residency, scheduling or Triton
software pipelining. The two tuned launch parameters do not become an invented
occupancy factor.

## Independent controls

The retained original independent review is
`/tmp/review-wave6-attention-online-independent-20261028.json`. It found four
issues despite the original 61 tests passing: generic vector pricing, missing
running-max/clamp operations, noncausal mask overcount, and numerical tests
omitting P conversion. The original module is preserved beside the report.

The repaired tests replay the pinned softmax expression with an independent
shape/count interpreter and enumerate query/KV addresses. They cover source
capability families, GQA ratios 1/3/7/16/17, query tails, rectangular matrix
padding, typed rate sensitivity, delayed inputs and per-owner resource service.
They also slow the sum reduction independently to distinguish the source
dependency branches, and compare two long loops for an unchanged peak active
task count. Storage controls distinguish an explicitly declared register pool
from tiny UB/L1/shared pools, which cannot constrain this source state.

The precision control uses exactly representable scores `[0,-2]` and values
`[0,1]`. BF16 P conversion changes the stored output from `0.119140625` to
`0.11962890625`; FP16 changes it from `0.11920166015625` to `0.1192626953125`.
This control establishes the conversion's semantic necessity. It does not prove
NumPy's exp/reduction rounding matches a compiled Triton binary.

`data/verification/operators/attention_online.json` contains the synthetic
public-contract runs. Results are source-structure and engine checks only.
They do not establish kernel timing accuracy, full-model costs, TTFT or TPOT.
