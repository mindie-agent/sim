# Native GLM KeyPool caller contract

`vllm_ascend_cann_keypool_caller` selects the retained CANN caller explicitly.
The default `causal_pool_indexer_analytic` remains the older Triton fallback
contract. The native caller uses simple resource envelopes for `KeyPool` and
`PoolKeyIndexer`; it does not claim their private tiling or installed binary.

```json
{
  "runtime": {
    "kernels": {
      "lowerings": {
        "pool_key_indexer": {
          "implementation": "vllm_ascend_cann_keypool_caller",
          "options": {
            "tail_capacity": 9,
            "block_table_columns": 8195,
            "score_mode": "linear_weighted_query"
          }
        }
      }
    }
  }
}
```

`tail_capacity` declares the real persistent
rollback ring, not a value inferred from q6. It defaults to the pool size;
a larger capacity enables the source's retained-hidden reprojection. The
retained 131K capture's `_write_keypool_caches_kernel` output
`[7646,9,256]` establishes capacity 9 for this capture.
`block_table_columns` is the allocated table width, which can exceed the
active context. It defaults to `ceil(runtime.max_model_len / 128)`; the retained
B4/B8 tables have 8195 columns. All current requests must be NoPE BF16 decode
with BF16 pooled keys and independent FP32 head weights. Context partitioning
is not reconstructed by this caller declaration.

The caller is pinned to vllm-ascend revision
`c9aebf229e999f54bedf5b0f13f8e3ffd9989e61`:

| File | SHA256 |
| --- | --- |
| `attention/indexer_kpool.py` | `7b185038d89cd46d245b4e2a8d768bea51e3d8c82bbe999305d97621a105d498` |
| `ops/cann_keypool.py` | `57348e3889486d75015a7cb67d3dde3cdd50f003156749887dd86d800d362ba6` |
| `ops/triton/glm5_next_cann_keypool.py` | `39ab2a610de3db8259b44a1196ae91c3c9eb83103254de8eafab89366d394595` |
| `ops/triton/prepare_nonempty_keypool_inputs.py` | `1864a2affc64dee1f033aeae7ce92d9159948ee3b6bb80b6fa81c739b9745184` |

## Preserved boundaries

- `KeyPool` takes BF16 hidden, WK and gate matrices. Its APE, normalization
  parameters and state are FP32. Head weights are a separate
  `F.linear(hidden.float(), head_weight_f32)`, then BF16 cast and scale. Native
  residency retains only the head-weight FP32 copy; it does not retain the
  fallback's FP32 WK/gate copies.
- State gathering copies B pools into pages `[1:B+1]` of a temporary
  `[1+2B, pool, 2D]` state tensor. With D128/pool4, B4/B8 produce the captured
  `[9,4,256]` / `[17,4,256]` shapes.
- Pooled output capacity is `min(T, T//pool+B)`. B4 q6 allocates 10 rows and B8
  q6 allocates 20. Actual completed rows remain the sum of per-request causal
  pool boundaries; unused output capacity has invalid slot mappings. Shape
  alone does not establish actual context or valid pool count.
- A larger rollback ring gathers `min(capacity,max_query_len)` hidden rows
  per request, reprojects separate BF16 K/Gate, converts them to FP32,
  normalizes K and updates the retained ring. Ordinary mode copies the tail
  from CANN state. Masked rows count toward launch capacity but not valid cache
  writes.
- The nonempty adapter allocates query/weights at T+1 and metadata/table at
  B+1, sets the appended sequence length to one, and clamps real key lengths
  to at least one. The new query end is padded T+1. Therefore graph padding
  also belongs to the appended request: 18 valid rows padded to 24 produce
  seven dummy-request query rows, not one real query plus six extra requests.
- Native `PoolKeyIndexer` returns token indices at capacity `topk+pool-1`.
  The packing adapter copies valid historical indices and appends the causal
  tail. It does not repeat the fallback's pool-ID expansion. Invalid padded
  query rows remain `-1`; they never enlarge sparse-attention candidates.

## Resource and service model

The two native kernels have recipes `cann_key_pool` and
`cann_pool_key_indexer`. Their envelopes charge logical arithmetic and
external tensor traffic on the common Engine. `score_mode` declares the
reference algebra (`linear_weighted_query` or `per_head_relu`); the retained
Python caller does not prove CANN's internal algebra. The indexer envelope
assumes no cross-query pooled-cache reuse. KeyPool state I/O uses a full pool
per request as an explicit reference bound. Internal spills, reuse and native
owner scheduling remain unresolved. Adapters and rollback work stay separate.

Existing `runtime.kernels.service_regimes` can match either recipe or native
kernel name. An inclusive response may be a completion floor. An explicit
`execution.kind: joint_native` response can replace one native envelope on a
declared single `busy_ns` queue; analytical bytes/FLOPs remain reference work.
It cannot replace the whole caller or swallow adapter/head-weight work. Scope
response parameters with physical metadata such as `native_query_rows`,
`scanned_pools`, dtype, revision and execution identity. Use independent
calibration; no retained B4/B8 validation durations are bundled here.

Tests cover captured B4/B8 shapes, padded dummy ownership, valid pool/token
conservation, rollback work and residency, and isolated synthetic native
service replacement. These establish structure and Engine behavior. They are
not evidence that Body accuracy or native kernel accuracy passed.

## Explicit operator calibration and remaining errors

`data/observations/a3-native-keypool-20261030/producer.py` creates the explicit
`examples/cases/longctx-decode-20261030/glm53flash-a3-native-keypool.json` case.
It streams the retained CSVs, assigns two native responses, and runs all 45
layers for B1 through B8. It does not change the default caller or fit Body
latency. The case declares q6 graph rows from the native shapes; B5/B6 are
prediction-only scenarios.

- KeyPool uses independent short-context static-on q6 calls with the same BF16
  WK/gate and FP32 state ABI. Equal-geometry affine assignment gives
  `28.96730454 + 0.1060826424 * query_rows` microseconds.
- PoolKeyIndexer uses only long-context B1/B2 means. Nonnegative affine
  assignment gives `41.14708824` microseconds with a zero row slope. Its
  extension to larger batches is an explicit extrapolation. Short-context
  calls include approximately 55/489 microsecond regimes and are not blended.
- Long-context B3/B4/B7/B8 were examined earlier; they are validation after
  parameter assignment, not pristine holdouts. Native KeyPool passes all six
  recorded geometries within 15%. PoolKeyIndexer fails B7/B8 at -27.9%/-32.2%.
- The full Body passes five of six observed populations within 15%; B8 remains
  +24.6%. B1/B2/B3/B4/B7/B8 predictions are
  22.621/25.407/28.159/30.978/37.364/39.344 ms. Per-layer comparisons and native
  operator checks remain in `validation.json`; a Body pass does not imply
  that every native operator or layer passed.

Installed CANN binary identity, internal score-cache reuse, actual sequence
length values, sparse-attention valid indices and native scheduling remain
unbound. These are observed service responses in a declared execution regime,
not recovered hardware throughput or a proof of the private kernel algorithm.

`residuals.py` reuses the retained complete-Body segmentation and streams the
same capture to locate the remaining B7/B8 error. B7 has one Forward; B8 has
12. Layer 0 alone decreases from 3.540 ms to 1.193 ms between those populations,
so their total means are not a matched steady-state workload pair.

The dominant B8 operator differences are routed FFN gate-up and down: predicted
duration sums are 9.379 and 4.733 ms, versus observed 2.899 and 1.524 ms across
42 layers. Native B7/B8 shapes are identical: 384-row activation capacity and
18 expert weight banks. Actual group-list values are not retained. The default
probability-moment model assumes 24 local rows and 13.344 active experts at B8,
charging 224.31/112.37 MB of gate-up/down weight reads per layer. The resulting
223.31/112.68 us costs differ substantially from native means 69.02/36.28 us.
Expert grouping and weight reuse therefore dominate the next source-model
investigation. Capacity 384 is not evidence of 384 valid rows, and observed
kernel duration sums are diagnostic rather than additive Body wall time.

## Independent short-context GMM service check

`gmm_service.py` keeps the probability-moment reference and every routing
parameter unchanged. It selects complete native gate-up/down calls from the
independent 4K static-on capture only when their recorded ABI matches the 131K
population: q6, TP16/EP16, 384-row capacity, 18 banks, identical shapes, dtypes,
formats and block count. Short-context B6/B7 each contain one complete Forward
and 42 calls of each kernel. The actual count-list values remain unavailable.
The recorded up weights are INT8 `[18,128,256,16,32]` in NZ, which contains the
full 8-bit `[18,4096,4096]` payload; down weights are INT8
`[18,2048,4096]`. Up weight/input scales are FP32; down weight scale/output are
BF16 and its per-token scale is FP32. This is not an implicit INT4 path.

Short-context affine responses in expected local rows `r` are:

```
gate-up us = 54.29521429 + 1.501968254 * r
down us    = 40.67923810 + 0.153317460 * r
```

The explicit `glm53flash-a3-native-keypool-gmm-service.json` case replaces only
those native boundaries with `joint_native` service for `r` in 18–24. Its
original logical reference bytes, FLOPs, expert probabilities and active-bank
moments remain available but are not executed at the replaced boundary. No
whole-step or layer timing entered either coefficient assignment.

| 131K validation | Gate-up prediction / native us | Down prediction / native us | Full Body prediction / native ms |
| --- | ---: | ---: | ---: |
| B7, one Forward | 85.837 / 79.909 | 43.899 / 41.409 | 29.843 / 35.748 |
| B8, 12 Forwards | 90.342 / 69.023 | 44.359 / 36.276 | 30.890 / 31.580 |

B7 native primitive errors are +7.4%/+6.0%, but its Body remains -16.5%.
B8 Body is -2.2%, while its two primitives **fail** at +30.9%/+22.3%. The
short-context conditional response therefore does not demonstrate independent
routing generalization. B8 mean AIC MTE2 times are 44.10/19.74 us versus
short-context B7 59.44/27.50 us, supporting a changed observed memory-service
population without identifying its exact transferred bytes or expert counts.

The current expected grouped-matrix implementation also passes a byte-accounting
check: B8 down services 112.620 MB of HBM, exactly its declared payload, while
113.510 MB of MTE2 and 119.276 MB of MTE1 are separate private-buffer transfers.
The bank set is not reread once per aggregate M tile. The model's 24 local rows
equal `48 query rows * topk8 / EP16`; it does not apply a second TP division.
The model has router groups 1/1, and the delivered artifacts do not establish
force-EPLB. Neither an unobserved routing correlation nor capacity 384 is used
to lower the active-expert count.
