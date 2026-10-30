# Paged MLA source structure export

The selected `ascend_paged_mla_decode_arch35` kernel now exports its active native
tensor contract and the physical work from its existing source planner. This
repairs an export-only gap: the frontend kernel deliberately starts with zero
work, and execution previously prepared it while structure export prepared only
selected native fusion nodes. Timing programs, rates and source cases are unchanged.

`export_structure` supplies each replica/stage's declared `Geometry` to
`export_op_structure`. Source calls still expand into kernel children before
planning, preserving parent/child lineage. `RecipeBook.prepare` supplies the
kernel's resource ledger without starting the timing Engine. The optional
`resource_geometries={(replica, stage): Geometry(...)}` argument also enables this
for direct graph exports. Without geometry, source declarations still export
their tensors, but `physical_work_basis` explicitly says
`source_declaration_without_resource_geometry`.

The `attention_paged_mla_qk_rope` contract retains BF16 queries in
`[B, physical_heads, 1, 512]`, rotary queries in `[B, physical_heads, 1, 64]`,
shared latent cache tails `[1, page_size, 512]`, rotary cache tails
`[1, page_size, 64]`, and native output `[physical_heads, B, 1, 512]`.
The subsequent logical-head view remains outside this kernel. Query-head count
never multiplies the shared cache head count. This active interface matches the
existing profiler compact7/full31 contract without fabricating a cache capacity,
block table value, sequence-length array, binary identity or event occurrence.

Context and explicit AIC/FD dispatch remain separate physical inputs. Changing
them can retain the same tensor contract while changing its physical signature
and resource work. `physical_work.resource_work` now preserves directional
`mte2_aic`/`mte2_aiv`, `mte1`, `fixpipe` and other extended services. Subgraph totals
sum these independently; they are not folded into HBM or counted again as generic
MTE2 traffic.

## Verification

`tests/test_structure_paged_mla.py` checks the real full 93-layer frontend,
the existing planner, actual Engine work with a 7 ms input dependency, context
and page changes, subgraph accumulation, wrong axis/precision counterexamples,
and the retained A5 full31 ABI. The focused structure suite passed **34 tests**,
including source-call expansion and native-fusion exports. Engine execution is
explicitly disabled during the export tests.

The reproducible compact receipt is
`data/verification/paged-mla-structure-20261030.json`; its neighboring producer
runs two original conditional source-host cases without changing input geometry:

| Full checkpoint probe | Exported nodes | MLA kernels | Export wall | Per-kernel BF16 FLOPs | Per-kernel HBM bytes |
|---|---:|---:|---:|---:|---:|
| B1, context4096, q1 | 5,251 | 24 | 1.006 s | 143,163,392 | 7,431,428 |
| B32, context16384, q1 | 5,088 | 24 | 1.122 s | 18,271,436,800 | 605,147,264 |

Each export matches the actual prepared Engine operator's complete work ledger;
both kernel executions finish with no premature input service, live tasks or
buffer allocations. The combined producer used 2.992 s wall, 2.990 s CPU and
226.0 MiB peak RSS with an unchanged source hash. This verifies export and
execution agreement, not measured performance accuracy. The cases retain their
original conditional context/host-dispatch evidence; newer captured-context
variants require their own input declaration.
