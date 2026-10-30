# Local MoE initialization with MXFP8 routing output

The explicit selector is
`runtime.kernels.lowerings.moe_init_routing = "ascend_moe_init_routing_v3_mxfp8"`.
It binds an ordinary EP1 route pack and its gate/up activation producer to one
source-compatible `MoeInitRoutingV3` boundary. Router logits/top-k, shared
experts, matrix weights, expert down quantization and output combination remain
separate. The original gate/up quantizer is removed because the routing kernel
now publishes both E4M3 data and E8M0 scales. A contiguous scale reshape retains
the same storage; no conversion or copy is invented for it.

The retained CANN source is pinned at
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8` under
`moe/moe_init_routing_v3/`; individual source hashes are retained in the program
and verification report. `op_host/moe_init_routing_v3_tiling_arch35.cpp:1893–1900`
selects `useGatherCopy` when `active_experts * topk > experts * 3`. The current
domain has all experts local and valid, top-k > 3, BF16 input, E4M3/E8M0 output,
one-core sorting capacity and a full column tile divisible by 256. It explicitly
rejects larger sorting/index tiling domains and transport quantization.

The source sequence is one-core sort → parallel expert counts → owner-zero
INT64 count publication → inverse row indices → parallel gather/MX quantization.
The first three stages retain actual workspace traffic and ordering. Count's
SIMT direct global memory accesses are charged as HBM, not mislabeled as AIV
DMA. Each launched AIV retains the maximum source UB arena, including the
host's 64-KiB SIMT cache reservation. Actual source Sort/Concat/Extract issue,
SIMT/scalar instruction service, atomic cache RMW, SyncAll flag latency and
UB-register ports have no declared service model and are explicitly unpriced.
Known elementary vector expressions retain the existing lane-work envelope.

`moe_v3_gather_mxfp8_quant.h:543–600` reuses a quantized token across its top-k
output positions. Assignment ranges are split contiguously across AIV owners.
A token crossing an owner boundary is loaded and quantized by both owners;
the implementation does not pretend an inter-owner shared cache exists. The
finite execution model retains two whole-row payload slots. This conservatively
couples separate source input/output queue lifetimes and is not a binary
dispatch reconstruction. The program writes each expanded activation and
scale once and keeps the source expert count and inverse-index outputs.

For the complete GLM Flash q4 inputs B16/B32/B64 there are 42 initialization
calls/body. Predictions are 1.312 / 1.756 / 3.275 us/call. At B64 the observed
family mean is 28.768 us; the remaining gap is **not resolved**. The complete
four-output ABI is exported for future independent cohort matching:

| Tensor | Shape | Type |
|---|---|---|
| Input | `[Bq,4096]` | BF16 |
| Expert IDs | `[Bq,8]` | INT32 |
| Expanded input | `[8Bq,4096]` | FP8 E4M3 |
| Inverse row IDs | `[8Bq]` | INT32 |
| Expert counts | `[288]` | INT64 |
| Expanded scales | `[8Bq,128]` | FP8 E8M0 |

This tensor match cannot establish top-k values, exact sort order, flags,
tiling or binary identity. No prediction reads measured durations. Source
counts and physical service controls must not be presented as 15% acceptance.

Validation: 13 new tests cover complete frontend producer replacement,
nonuniform owner intervals, repeated-token boundaries, exact payload/scales,
the ordinary fused gate/up consumer, late readiness and buffer cleanup. All
four isolated resource axes (HBM, AIV read, AIV write and vector) respond
monotonically over 0.5/1/2. The combined affected suite passed 54 tests, including the existing grouped
INT32 caller. Reproduce compact evidence with
`data/verification/glm-moe-init-mxfp8-20261030.producer.py`.
