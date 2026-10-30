# Native MC2 fullmesh caller

The ordinary MoE frontend accepts an explicit software selection:

```json
{"native_moe_protocol": {"kind": "arch35_fullmesh", "max_inflight_per_rank": 8}}
```

This selects the retained arch35 DispatchV3 fullmesh / CombineV3 MTE source
protocol with unquantized BF16/FP16 rows. It replaces generic payload packing,
expert dispatch, expert return and unpermutation through final publication.
Router weights, grouped matrices, explicit expert-input quantization, weighted
reduction and the final token output store remain in the graph. The first
supported caller has EP > 1 and one physical source partition (including the
explicit sequence-parallel caller). Other representations require their own
selection; this branch is not identified with a captured binary. Reduction is
an owned source phase: each output token reads its top-k slots on one AIV,
casts/accumulates in FP32, then casts/stores output. The local count-prefix
workspace and successful ready checks are described in the
[BF16 phase audit](kimi-bf16-native-mc2.md).

Each dispatch has parallel count/status and framed payload paths. Both must
arrive before receive materialization. Counts include zero-token experts. A
payload row reserves source metadata, frames 480 useful bytes into 512 bytes,
and only then scales by assignment count. Return frames use their own width;
the complete cohort converts widths before transposing routed peer traffic.
Physical MTE3 send work includes local destinations. Incoming window HBM writes
are charged to receiving work. A separate network phase charges only remote
wire, once. No `(EP - 1)` serial setup is added to the direct-peer program.

All resource phases share one `device_kernel_group`: one eager submission and
one device startup/inclusive completion minimum. The readiness boundary waits
for every external input and preceding branch stream. Independent branches keep
independent stream order. An incoming-only rank remains a real kernel invocation.
No whole-kernel primitive is rebound to an internal resource phase.

The planner gets actual AIV ownership from hardware Geometry. Count work uses
the source count-core domain; payload work uses a declared row-grid ownership
approximation, capped by rows and eligible AIVs. Expected fractional counts
scale whole-row frames. Native token queue occupancy, scalar control/polling,
metadata cache service, destination-specific owner assignment and fused MX
quantization remain unresolved. Whole-phase copy/wire overlap is a fluid
approximation, not a reconstruction of every token-level queue. The peer window
is an explicit concurrency parameter, not inferred from a timing residual.

Verification covers the actual public graph/body paths, whole-row wire totals,
count/payload concurrency, asymmetric DP and zero senders, width-changing return
traffic, delayed inputs, one native invocation per direction, preservation of
router-weight arithmetic and output stores, bandwidth response and task release.
The independent caller review also checked dispatch 1024-byte frames against
512-byte return frames. These are mechanism controls; transport-only timing is
not a comparison to complete native Dispatch/Combine event duration.
