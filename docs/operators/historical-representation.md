# Historical representation and cache mechanisms

This review uses preserved source revision
`d141b68616015aff8bbac59e0073417b65b5ebd5`. It reads source; it does not execute
the retired predictor or import old fitted rates. The scope is representation,
cache ownership, allocator capacity and valid transfer payload.

## Source identities and current consumers

Hashes are SHA-256 of the complete Git blob at the preserved revision.

| Preserved path | SHA-256 | Reviewed scope and current consumer |
| --- | --- | --- |
| `mindie_sim/a3_kv_pages.py` | `30590c6bf754a3fe12706a85c9bba0d751f7a39fe80ccfe65ea465f01ac1f06c` | Independent group-page geometry, fixed state ring, window plus declared in-flight span and recycling page. Re-expressed in `catalog/cache_pages.py`, consumed by `prediction/capacity.py`. |
| `mindie_sim/a3_catalog_cache.py` | `aff53c7c8f0f4bdb236973683ccee8c6d22dec9531c3a06e85034ce03d56f8b4` | Valid compressed rows, bounded windows/state, producer ownership, allocation versus transfer rows. Current `catalog/cache.py` serves live capacity and P/D transfer through `Memory.cache_bytes`. |
| `mindie_sim/a3_catalog_types.py` | `ed2d74505495ddd0e45004fb2ced831dafa632514a6bd5718788046135b2c88f` | Row-local FP4 packing, FP8 block scales, per-channel/group INT4 scales and explicit table representation. Current `catalog/types.py` feeds weight residency and GEMM traffic. |
| `mindie_sim/a3_catalog_weights.py` | `276230775b94a6e07a46807848d1fedeaff5032fa7eea9bc02d12934fde6e8a8` | Shard matrix geometry before packing/scales; separate resident payload and metadata. Current `catalog/weight_terms.py`, `catalog/weights.py` and `prediction/capacity.py`. |

These are scoped mappings. They do not establish that every old model adapter,
field or layout has been reconciled.

## Material counterexamples and repairs

The former new-version `Memory.cache_bytes` rounded committed tokens to the
global KV block before applying each explicit page group. With a global block
of 8, four valid tokens, a 3-token page priced at 192 bytes and a 5-token page
priced at 320 bytes, this charged 1,216 bytes. Independent allocators actually
need `2 * 192 + 1 * 320 = 704` bytes. The repair makes the 704-byte capacity
admit that request through ordinary `predict`; changing the unrelated global
block to 16 does not change explicit group allocations.

The same premature rounding turned a one-token sequence into completed C4
rows. Window and compressed rows now derive from committed tokens first;
their allocations then use the respective component page size. For an 8-token global block,
a five-token window and C4 compression, one valid token owns one window page
and zero compressed pages. Nine tokens own one window page and one two-row
compressed page. Fixed compressor state is charged separately.

An allocator page can contain several layer components. The previous layer-wide
replacement could not express a shared window group plus a fixed compressor
state group while retaining the long-cache owner's storage. Component coverage
now removes only the covered analytic capacity components. A page priced for a
whole group is charged once; a page priced per layer is charged once per layer.
Overlapping ownership is rejected. A physical group cannot straddle PP stages;
declare a separate group for each stage. All prices are per physical rank.

`padded=False` now always returns model-derived valid cache/state payload,
independent of allocator declarations. A real P/D test with 3/5-token source
pages and 7-token destination pages transfers exactly 384 bytes:
three tokens, two layers and 64 bytes per layer row. Both memory resources and
the connector consume 384 bytes. It does not transmit unused page capacity or
recycling slots.

The old and new `matrix_payload_bytes` both mislabeled half of an Engram table's
BF16 device payload as quantization metadata. The repaired decomposition uses
the declared runtime representation. A 3-by-65 table has 390 BF16 payload bytes
and zero metadata, while its packed source has 195 payload bytes and nine E8M0
scale bytes. The INT8/FP32 runtime representation has 195 payload bytes and 36
scale bytes. Every decomposition still sums to actual stage resident weights;
no total-capacity correction or timing coefficient was added.

## Explicit page declaration

`runtime.memory.cache_page_groups` retains the existing `layers`,
`tokens_per_page`, `bytes_per_layer_page` and optional `resident_tokens`
declaration. New fields express allocator mechanisms:

| Field | Meaning |
| --- | --- |
| `components` | Names from `layer_cache_components`; omission covers all components of the declared layers. Separate groups may cover distinct components of the same layer. |
| `members` | Explicit `{"layer": 0, "component": "index_k"}` entries for a group containing heterogeneous components. Replaces `layers` and `components`, allowing one physical page to hold several long-cache producers and only their actual index producers. |
| `bytes_per_group_page` | One price for the complete group, mutually exclusive with `bytes_per_layer_page`. |
| `fixed_pages` | Fixed pages per live request, suitable for a declared state ring. Zero uses token-based allocation. |
| `allocation` | `live` uses committed tokens; `max_model_len` reserves the configured maximum span when a request owns cache. |
| `max_in_flight_tokens` | Explicit additional resident span reserved while preceding work still reads old rows. It is not inferred from the async flag. |
| `recycling_pages` | Additional pages reserved for recycling after the token span has been rounded. |

For non-fixed groups, the span is the live or maximum sequence span, capped by
`resident_tokens` when declared, plus `max_in_flight_tokens`, capped by
`max_model_len`. Page count is the ceiling of this span divided by
`tokens_per_page`, plus `recycling_pages`.

For example, the historical window rule is expressible without a model-name
branch:

```json
{
  "layers": [0],
  "components": ["window"],
  "tokens_per_page": 128,
  "bytes_per_group_page": 462848,
  "resident_tokens": 128,
  "max_in_flight_tokens": 8192,
  "recycling_pages": 1,
  "allocation": "max_model_len"
}
```

At `max_model_len=16384`, this reserves 66 pages. Increasing only the explicitly
declared in-flight span to 16384 reserves 129 pages. These reproduce the old
declared arithmetic, not the old model-specific group selector, a measured
total-memory replacement, or a claim that all current engines use these pages.
The layer list and per-group byte price must describe the actual deployment.

## Independent behavior checks

`tests/test_cache_pages.py` verifies independent page geometries, real serving
admission and release, P/D link and memory payload conservation, component
ownership including heterogeneous sparse-group members, fixed state, window retention, compressed completion and the
66/129-page in-flight counterexample.

`tests/test_representation_geometry.py` verifies odd-width packing and scale
tails after ETP sharding. A 3-by-65 MXFP4 matrix occupies
`3 * ceil(65/2) + 3 * ceil(65/32) = 108` bytes. Two rank-local 3-by-33 shards
occupy 57 bytes each; simply dividing the unsharded total is incorrect.
It also checks grouped and per-channel INT4 scales, a two-dimensional FP8 scale
grid in actual GEMM traffic, and table payload/metadata conservation through
the stage and `Memory` consumers.

Existing `tests/test_structure.py` checks long-cache ownership, bounded
recurrent state and whole CP blocks versus valid transfer rows.
`tests/test_operator_mechanisms.py` checks separate activation scale geometry
and actual activation producers; weight scale grouping does not replace these
activation contracts.

Focused run on 2026-10-29:

```text
.venv/bin/python -m pytest -q tests/test_cache_pages.py \
  tests/test_representation_geometry.py tests/test_structure.py \
  tests/test_prediction_deployment.py tests/test_prediction_system.py \
  tests/test_parallel_mechanisms.py tests/test_operator_mechanisms.py
56 passed
```

These checks establish arithmetic, ownership and consumer behavior. They are
not a hardware timing validation.

## Remaining scope

- The old fixed V4.1 group selector, hardcoded group membership and measured KV
  pool substitution are not revived. An actual engine dump still needs an
  explicit deployment declaration; this change does not claim that an old
  recorded group layout is the current backend layout.
- In-flight retention is an explicit capacity budget. The current scheduler
  does not infer that budget from asynchronous overlap or simulate per-page
  retirement epochs, fragmentation, prefix sharing or allocator free lists.
  Default analytic windows use a compact retained-row footprint; an engine
  that keeps extra absolute-position pages during recycling must declare them.
- P/D currently moves valid modeled tensors. Protocols that transfer whole
  pages, require repacking between different cache formats or compress the
  wire need an explicit connector/representation mechanism.
- Fixed pages cover the declared component set. They do not infer omitted
  hidden state or manufacture producer ownership.
- INT8 scale/offset schemas, vendor-specific blocked weight layouts, layout
  conversion workspace and additional checkpoint adapters remain separate
  audits. This review verifies the supported FP4/FP8/INT4 formulas and explicit
  table modes, not every old quantization preset.
- Raw measurements and the preserved checkout remain intact. No data role,
  calibrated rate or independent validation outcome is changed by these repairs.
