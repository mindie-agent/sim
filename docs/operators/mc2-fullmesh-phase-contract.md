# MC2 arch35 full-mesh phase contract

`prediction/operators/mc2_fullmesh.py` is a pure source work builder. It returns
`RowLayout`, `ProtocolPhase` and `ProtocolPlan` records for the ordinary caller
to bind to shared copy, vector and transport programs. It is not an additional
simulator and supplies no fitted timing or invocation floor.

The source revision and dispatch/combine SHA256 values match the retained
references in `direct-peer-moe-transport.md`. Quantization row layout comes
from `moe_distribute_dispatch_v2_quant.h`, SHA256
`46fec97abaec206c9b41bd1b3ab690d7c0ff7fe7a09961599b8c56f304543ed5`,
lines 68–110. No exact historical compiled binary or dispatch full-mesh tiling
key has been established.

## Caller interface

```python
from mindie_sim.prediction.operators.mc2_fullmesh import (
    dispatch_layout, dispatch_plan, combine_layout, combine_plan,
)

row = dispatch_layout(
    width=1536, input_dtype="bf16", payload_dtype="fp8_e4m3",
    quant_mode="mx", has_expert_scales=True,
)
plan = dispatch_plan(
    row, source_tokens=1, top_k=8,
    source_assignments=8, received_assignments=8,
    experts=384, ep=32, aiv_count=64,
)
```

These numbers illustrate the contract, not a claimed Kimi physical case.
`source_tokens` is the actual source row count seen by this dispatch kernel,
not model batch before TP/SP partitioning. `source_assignments` includes local
destinations. `received_assignments` is this rank's expert work. The caller
retains the peer matrix and removes its diagonal only from network traffic;
local-window copy/quant work still includes local assignments.

Each phase exposes `name`, `after`, `owner_domain`, `ownership`, `work`,
`details`, `owner_basis` and optional source-ordered `steps`.
`dict(phase.work)` uses native units: HBM/MTE bytes and vector
element operations. Network phases expose bytes per assignment and count
record sizes separately. Their `mte3` work belongs to this rank's outgoing
messages, while their `hbm` work belongs to this rank's incoming window
writes. They must not be split into new host submissions or per-phase kernel
floors.

`owner_domain` is an eligible physical AIV set, not permission to spread a
single active row across all cores. Payload assignment index modulo the
payload-core count, contiguous expert/count ranges and contiguous return rows
are source rules. Exact destination/source-expert cells are needed to identify
active receiving owners. Fractional rows are explicitly expected work and
cannot establish captured owner occupancy.

## Supported branches and physical work

Dispatch supports identical BF16/FP16/FP8 input/output (with optional supplied
scale bytes), per-token INT8/FP8 dynamic quantization, FP8 MX group32, and FP8
group128 scale layout. These branches exclude smooth quantization matrices,
shared-expert ranks, elastic remapping and special zero/copy/constant experts.
The first return contract supports unquantized BF16/FP16. Unsupported
representations raise explicitly rather than choosing a different branch.

`TokenToExpert` / `TokenToExpertInQuant` frame **one token assignment** at a
time. The dispatch payload/scale area is aligned, followed by a reserved 32B
tail containing the 12B route triple and optional 4B expert weight. Every
480B data region occupies 512B on the remote window. Calculate a complete
row's frame first, then multiply by assignment count; applying `ceil` to
fractional expected peer rows is incorrect. The BF16 return contains no triple
inside its packet, so its framing is separately calculated.

Dispatch reserves
`min(max(experts // 16, 1), AIV // 2, 8, experts)` count cores, with the
remaining cores available for payload. Every source emits one 32B count/status
record per expert, including zero-count experts. Count computation/exchange
and payload read/quantize/frame/exchange are parallel. Receiving data waits
for both payload and counts, then checks flags, copies the 480B regions,
materializes payload/scales/12B triples and clears flag blocks. Optional
expert weights have a separate 4B receive DMA even though the packed data
copy already brought their containing region into UB.

At least one successful flag/status read is represented. Extra spin polls,
scalar GetValue/SetValue costs, control cache behavior, some prefix metadata
scalar work and actual native UB/window selection remain visible in
`unmodeled_features`. MX/group128 quantization layout is implemented, but its
full vector instruction work is not silently declared zero: the caller must
retain an explicit quantization program until that work is replaced. The
per-token dynamic branch already counts the source cast/abs/reduction/multiply
and output-cast sequence, with scalar reciprocal/store service still explicit
as unmodeled.

## Replacement boundaries

The profile replaces corresponding generic `route_pack` payload copying and
dispatch-fused expert-input quantization only where the physical calls match.
It does not delete router IDs/weights or upstream projections. With explicit
`source_tokens` and `top_k`, the return contract includes token-owned FP32
weighted reduction and final publication; the ordinary caller replaces all
equivalent `route_unpack` work while preserving weight producers. The older
standalone layout-only return contract leaves these downstream services to
its caller. Source framing/local receive
work must not be added on top of an unchanged equivalent copy program.

The builder intentionally creates no end-to-end accuracy claim. Source-layout
tests cover tail widths, optional scales, count/payload partitioning, parallel
dependencies, expected fractional assignments, flag stripping and the distinct
return layout. Runtime and ordinary-caller integration have separate evidence.
