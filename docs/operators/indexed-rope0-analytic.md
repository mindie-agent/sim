# Explicit rope-free indexed MLA

`programs.indexed_attention` accepts
`{"kind":"indexed_attention_pipeline","scratch_policy":"offchip","rope0_analytic":true}`.
The option extends the existing BF16, latent-width 512 analytic pipeline to a
zero-width rotary suffix. It keeps the same query owners, query-local sparse
indices, finite slots, QK/softmax/PV dependencies and shared Engine. K and V alias
the latent cache, so each selected row contributes one 1,024-byte BF16 payload
plus two INT32 indices to the gather. The rotary suffix contributes no payload
or matrix work.

This is an explicit geometry extension, not a native kernel identification.
The retained arch22 source implements latent 512 plus rotary 64. Its native
`ascend_sparse_flash_attention_arch22` selector still requires that geometry.
The extension does not install a new body default, change resource rates, or
claim that a captured private rope-free binary uses the retained source's
schedule. `program_execution` reports `coverage=analytic_geometry_extension`,
the source reference and unresolved dispatch, scalar index response, flag
latency and external cache eviction. An input with INT8 KV remains outside this
BF16 adapter; the separately named packed INT8 adapter has its own contract.

As with the existing generic batch path, an explicitly selected program may
share a query-owner grid across independent sequences while retaining their
separate contexts and causal selected-row counts. Compact execution is the
existing fluid approximation; `compact=false` remains the detailed control.
`operand_modes=true` requires explicit `mte1_l0a` and `mte1_l0b` service modes.
Do not enable it on a profile that declares only the shared MTE1 rate.

`tests/test_indexed_attention_rope0.py` executes one-key, tile-tail and batch
controls through `target_graphs` and the real Engine. It checks byte/service
conservation against detailed execution, late producers, finite ownership,
resource release, explicit source-domain rejection and HBM rate response.
These are mechanism checks, not an observed-accuracy result.
