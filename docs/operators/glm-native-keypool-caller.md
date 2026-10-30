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

This is a conditional example. `tail_capacity` declares the real persistent
rollback ring, not a value inferred from q6. It defaults to the pool size;
a larger capacity enables the source's retained-hidden reprojection. The
example capacity 9 is a scenario, not an observed cache dimension.
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
