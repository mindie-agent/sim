# Query-specific sparse attention traffic

Indexed attention cannot inherit dense attention's assumption that all queries
in a query tile reuse one KV sweep. Each query supplies its own selected indices.
Even two queries with disjoint selections of two rows need four logical row
reads; putting them in a tile of 128 queries does not reduce that to two.

`prediction.sparse_attention.sparse_attention_traffic` records the causal
selection demand as
`sum(min(floor((context + i) / compression_ratio), topk), i=1..query)`.
It uses constant-space integer arithmetic. Index reads cover those declared
valid slots only; extra sentinel scans are not inferred. Local query heads
share the latent KV rows, so this demand is not multiplied by head count.

Compressed indexed attention can also have an uncompressed dense window.
The helper keeps the window's query-tile sweeps separate from indexed gathers.
It returns an unsharded workload ledger; the caller retains placement and
context-parallel ownership. Dense indexed attention passes `window=0`.

The helper establishes logical read demand. It does not predict cache hits or
measure DRAM transactions. The existing envelope assigns this declared demand
to its memory service. Metadata explicitly leaves source scratch materialization,
physical copy ownership/transactions, scalar index/page dependencies and physical
matrix padding/pipelining unresolved. No multiplier or measured duration enters
the calculation.

## GLM retained descriptor and source evidence

The original single-request `pf4k_b1.csv.gz` has 78 `SparseFlashAttention`
events with identical descriptors:

- Query `[4096, 8, 512]`, query rope `[4096, 8, 64]`, BF16.
- Shared key/value paged cache `[703, 128, 1, 512]`, rope cache
  `[703, 128, 1, 64]`, BF16.
- Per-query sparse indices `[4096, 1, 2048]`, INT32; output `[4096, 8, 512]`.
- 20 AIC blocks and 40 mixed vector blocks.

The allocated 703 pages are capacity, not a 89,984-token active context.
Index selection is a separate `LightningIndexer` boundary: 21 events totaling
31.341753 ms. The 78 attention events total 1029.573832 ms, with median
13.208966 ms. Their BF16 cache descriptors exclude a KV quantization explanation
for this particular difference.

The capture declares vllm-ascend revision
`8fd0bc4885cb52e58791aaa8efacc571b9bd115d`. Official source at that revision
provides a concrete compatible mechanism:

- `vllm_ascend/device/device_op.py:560-578` passes the same KV tensor as key and
  value, separate rope, TND/PA_BSND layouts and sparse block size 1.
- `csrc/attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp:282-284`
  selects the vector-gather template for small sparse blocks; lines 337-343
  set the outer matrix row block to the query-head group, one query here.
- `op_kernel/arch22/sparse_flash_attention_service_vector_mla.h:824-845`
  reads query-specific top-k and page indices. Lines 882-989 gather KV/rope
  through UB and publish merged GM workspace. Lines 997-1088 retain independent
  query offsets, finite buffers and transfer dependencies.
- `op_kernel/arch22/sparse_flash_attention_service_cube_mla.h:815-858` and
  `1105-1120` reread that workspace for QK and PV. The kernel pipeline in
  `sparse_flash_attention_kernel_mla.h:866-902` orders gather, QK,
  softmax, PV and output stages.

The installed binary is not independently proven equal to that source.
The mechanism evidence is stronger than a name association, but the source
schedule is not silently selected as the historical implementation.

Counters support prioritizing the omitted transfer/control path: per-event
medians include AIV scalar 9.310818 ms, AIV MTE2 7.967597 ms, AIC MTE2
7.309014 ms and AIC MAC 1.608153 ms. These counters overlap and cannot be
added into a wall-time decomposition or converted into new resource rates.

## Independent envelope control

For the declared 4096-token causal prefill, top-k 2048 and widths 512+64:

| Quantity | Dense query-tile reuse | Independent indexed queries |
| --- | ---: | ---: |
| Logical KV rows read | 50,176 | 6,292,480 |
| KV payload bytes | 57,802,752 | 7,248,936,960 |
| Valid index bytes | omitted | 25,169,920 |
| Full envelope memory demand bytes | 129,105,920 | 7,345,410,048 |
| Isolated envelope time per call ms | 0.405198 | 6.121175 |
| Projection over 78 calls ms | 31.605411 | 477.451653 |

Both controls execute the public operator-cost path with the same matrix work,
vector work and declared hardware rates. Only logical read demand changes.
Both leave zero live tasks. This is a mechanism comparison, not a complete-model
graph result or strict observed-kernel validation. The remaining difference
against the native activity sum is 552.122179 ms.

Adding every source GM transfer to off-chip HBM would be another mistake:
gather, workspace publication and QK/PV rereads alone total 28.1903104 GB
under this geometry, exceeding 23 ms at the declared 1.2 TB/s memory rate.
That result exceeds the observed event time and cannot establish cache behavior.
Source GM access, on-chip copy work and off-chip traffic need separate treatment.

The compact record is
`data/verification/sparse-attention-mechanism-20261029.json`. It retains raw/source
hashes, descriptors, counter coverage, current source hashes and both controls.
Four focused tests cover 540 independently enumerated geometries, disjoint
query selections, large exact integers and the causal-ramp/saturation identity.

## Owner count versus traffic

Rows remain an independent, chip-wide logical ledger. They are not divided by
AIC count, and a single decode row still selects its full top-k. Physical
service is different: `selected_query_owners(query, compute_units)` returns
the finite number of source TND AIC tasks. One row activates one AIC and its
two AIVs, so the envelope's full-chip rate is an optimistic lower bound for
that call. The indexed pipeline applies this ownership; the generic envelope
does not. `tests/test_sparse_attention_traffic.py` checks 1, 2, 20 and zero
owners independently.
