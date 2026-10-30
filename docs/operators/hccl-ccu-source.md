# Explicit narrow CCU Mesh1D programs

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

`hccl_ccu.contract(selection, rank)` and `hccl_ccu.build(op, geometry)` add two
source-compatible programs. The central ordinary caller is a separate
integration; this module does not change `hccl_graph`, `program`, hardware
profiles, or default dispatch.

| Explicit declaration | Executable domain |
| --- | --- |
| `CCU_MS`, AllReduce, `mesh_oneshot_v1` / `mesh_oneshot_v2` | 2–8 ranks; one fragment of at most 4096 bytes; non-expanding same-dtype SUM with 2- or 4-byte elements |
| `CCU_SCHED`, AllGather, `mesh_mem2mem_v1` / `mesh_mem2mem_v2` | One equal-contribution executor repeat; separate input/output; eight finite 32 KiB local-copy slots |

Both require a single-level `MESH_1D`. The version suffix selects the retained
CCU V1/V2 event protocol, not a hardware SKU. The two expansion engines are not
interchanged. `TWO_DIE_REGULAR` is a communication topology for different
executors; it is not inferred from an A3 card having two compute dies.
The separate [two endpoint-die CCU_MS AllGather](hccl-ccu-two-die-source.md)
now supports an explicitly mapped, homogeneous single-fragment domain through
the public caller, with two physical CCU owners and routed remote notifications.

Source revision is `c2d6790464654ccf58aca39448de4b65fdc7b184` of
[hicann/hccl](https://github.com/hicann/hccl/tree/c2d6790464654ccf58aca39448de4b65fdc7b184).
The receipt records full source paths and hashes. Executor registrations in
`ins_v2_all_reduce_sole_executor.cc:555–565` and
`ins_v2_all_gather_sole_executor.cc:428–434` bind these distinct engines/templates.
This is not a claim about capture binary identity or successful runtime dispatch.

## Memory and execution boundaries

The one-shot `GroupReduce` loads each peer directly from its HBM into the
receiver's CCU buffer. Local input is copied into another CCU buffer. After all
loads finish, `LocalReduce` produces the reduced buffer and a local copy writes
output to HBM. Incoming peer bytes are **not written to HBM scratch**. For R ranks
and N input bytes per rank, local work is:

- HBM `(R+1)*N`: `(R-1)*N` reads serving peers, N local input read, N output write.
- CCU LocalCopy `2*N`: local input and output movements.
- Logical CCU SRAM accesses `(2*R+2)*N`: R input writes, R reduction reads plus
  one reduction write, then one output read.
- Reduction `(R-1)*N/dtype_bytes` binary SUM element operations.
- Network `R*(R-1)*N` useful bytes across the complete cohort, counted once.

The source template requests `R*N` HBM scratch capacity even though this narrow
kernel does not reference that scratch. The capacity request remains metadata;
it does not create scratch traffic. `AllocGoResource` declares 128×8 CCU buffer
handles of 4096 bytes; only R handles carry this single fragment. Handle counts
are not silently converted into NPU UB or HBM allocations. CCU resource-handle
availability is currently an explicit coverage gap.

AllGather issues nonblocking writes to remote output while `GroupCopy` copies
the local contribution into local output. `GroupCopy` loads through CCU SRAM and
then writes out, with read→wait→write→wait dependencies. Full groups use eight
32 KiB slots; the final group retains its remainder-first source order. The
finite program does not materialize all local-copy tiles in memory. Local work
is `2*R*N` HBM bytes, `2*N` CCU LocalCopy bytes and `2*N` CCU SRAM access bytes.
There is no local arithmetic reduction or HBM scratch.

The phase contract exposes source pre/post notification and wait counts.
Transport remains caller-owned through `HcclPhase.exchange`, with whole-phase
readiness. A central caller must count the outer invocation once and must not
add a separate per-phase kernel launch floor.

## Service units and missing information

Optional declared hardware services have explicit units:

| Role | Unit |
| --- | --- |
| `ccu_copy` | bytes moved by the CCU LocalCopy primitive |
| `ccu_buffer_bytes` | logical CCU SRAM bytes read plus written |
| `ccu_reduce_elements` | binary SUM element operations |
| `ccu_control` | explicit notification/event-wait API calls |

Absent roles remain in `program_execution.unpriced_service_work`. Physical HBM
traffic still executes. This gives an explicitly incomplete lower bound, not
evidence that control/reduction takes zero time. None maps automatically to
AIV, Cube, vector work, or a native CPU submission resource.

`cost_model.cc:121–122` initializes `ccuLocalCopyBw_ = 200e9` and
`ccuLocalReduceBw_ = 35e9` for source algorithm selection. These are retained as
**selector heuristic evidence only**. They are not hardware measurements, not
runtime rates, and not imported into any profile. No timing residual or launch
floor is added.

Larger CCU_MS AllReduce payloads need the source's parallel `LoopGroup` and
subfragment transport readiness; this implementation rejects that explicitly
selected unsupported domain. The existing generic path remains available when
this source program is not selected. Multi-level, two-die, in-place, mixed output
dtype, source reduce operations other than SUM, and executor repeat>1 are not
implemented here.

## Verification

`tests/test_hccl_ccu_source.py` runs actual Engine programs for both versions,
2/4/8-rank byte equations, finite 32 KiB tails and >8-slot local-copy groups,
separate copy/reduce/SRAM/control/network rate sweeps, late input, and a single
outer startup/inclusive minimum. All 15 new tests and the combined 37 affected HCCL tests passed.
The retained receipt
records source hashes and four actual Engine controls. These establish mechanism behavior,
not real-device CCU latency or model accuracy.
