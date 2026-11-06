# Explicit HCCL AICPU source templates

This is the ordinary public recipe for the new source-backed AR/RS/AG/RS
contracts. They return existing `HcclPlan` DAGs and therefore use
`hccl_graph.lower`, the common `Engine` and `DirectPeerGroup`. No second
lowering, engine, capture registry or per-shape schedule table is added.

## Templates

| collective | template | required declarations |
|---|---|---|
| AllReduce | `mesh_chunk_twoshot` | one-level `MESH_1D`, `buffer_mode=separate` |
| ReduceScatter | `rs_mesh_chunk` | explicit ordered `segments` summing to `payload_bytes/ranks` |
| AllGather | `z_axis_detour_allgather` | explicit ordered `segments` and `level1_weights` |
| ReduceScatter | `z_axis_detour_reduce_scatter` | one explicit segment and `level1_weights` |

`level1_weights=(6,2)` is an explicit native byte-share declaration used by the
captured A5 structure. It is **not** a hardware bandwidth, port inventory or
global default. If the selector is unknown, the caller must declare the
template and all required physical parameters; missing declarations fail
visibly. Existing `mesh_oneshot`, `mesh_twoshot`, `mesh`, `nhr` and
`hccl_alltoallv` remain valid and are not replaced by a device-label default.
`nhr` remains the correct source declaration for its own cyclic task graph;
no SKU or model label silently switches a call to a mesh template.

## API

```python
from mindie_sim.prediction.hccl_contracts import select
from mindie_sim.prediction.hccl_aicpu import contract

selection = select(
    {
        "expansion": "AICPU_TS",
        "template": "mesh_chunk_twoshot",
        "dtype_bytes": 2,
        "buffer_mode": "separate",
    },
    collective="allreduce",
    ranks=4,
    payload_bytes=2006,
)
plan = contract(selection, rank=0)
```

For a segmented source:

```python
selection = select(
    {
        "expansion": "AICPU_TS",
        "template": "z_axis_detour_allgather",
        "dtype_bytes": 2,
        "buffer_mode": "separate",
        "segments": (52428800, 14680064),
        "level1_weights": (6, 2),
    },
    collective="allgather",
    ranks=8,
    payload_bytes=67108864,
)
```

The resulting `HcclPlan` is consumed by the ordinary lower:

```python
from mindie_sim.prediction.hccl_graph import lower

ops = lower(ops, runtime, source_dtype)   # runtime.kernels['hccl'][family] holds the declaration
report = graph_costs(raw_case, ops)
```

Each phase exposes `native_direction` (`send`, `read` or `local`) as provenance.
The `exchange` matrix always remains the physical send matrix; AG channel
phases are marked `read`, AR RS steps and RS phases are `send`.

## CLI

```bash
python3 tools/hccl_source_contract.py \
  --collective allgather --ranks 8 --payload-bytes 67108864 \
  --template z_axis_detour_allgather \
  --segments 52428800,14680064 --level1-weights 6,2
```

The CLI calls the same `select`/`contract` path and prints phases, work,
matrices, directions and residual boundaries.

## Source provenance

Pinned `hicann/hccl` revision
`c2d6790464654ccf58aca39448de4b65fdc7b184`:

- AR mesh-chunk `ins_temp_all_reduce_mesh_1D_two_shot_mesh_chunk.cc`
  (`6401880c…`): 182-198 element-exact chunks, 207-245 peer map, 268-316
  write-reduce/sync, 330-438 AG/copies.
- RS mesh-chunk `ins_temp_reduce_scatter_mesh_1D_meshchunk.cc`
  (`3f9fe719…`): 181-207 4 KiB floor+tail; 226-304 frontRank write-reduce;
  139-334 PreCopy/PostCopy.
- Z detour `template_utils.cc` (`532c3aee…`): 146-220 split; AG/RS detour
  templates for channel/thread layout.
- AG plain mesh `ins_temp_all_gather_mesh_1D.cc` (`74acb7b9…`): peer/channel
  loops and double local copy.
- `alg_param.h` (`c6258678…`): `AICPU_ALIGN_SIZE=4096`,
  `MESH_CHANNELS_NUM=1`.

The same hashes and native structure comparison are retained in the analysis
v2 evidence. The product recipe is a **source-compatible declaration**, not a
claim that the deployed binary equals the source or that a measured whole-call
duration is reproduced.

## Residual boundaries

- Local work and network exchange share the public phase dependency. Work
  inside a phase is an aggregate envelope; per-message reduce/channel progress
  is not modeled.
- AR chunks are element-exact. RS mesh-chunk uses the source 4 KiB floor and
  exact tail; AG/RS detour chunks use the source split helper.
- Explicit segment order and native direction are preserved; segment values are
  not an allocator predicate.
- AllToAllV remains the existing product contract. Its observed peer
  start-order is a concurrent-slot micro-order residual, not an exact claim.
- No measured primitive rate, local startup, inclusive remote-reduce rate,
  coarse control rate or whole-collective alpha is imported. Unsupported rate
  fields fail visibly. The existing `observations/a5_comm_binding.py` A5 capture
  service seam remains separate; this structural recipe adds no capture registry.
- The source-compatible declaration is not deployed-binary identity. Phase
  local work and network exchange overlap as the public lowering's phase
  envelope; per-message reduce timing and independent channel supply are not
  established.
