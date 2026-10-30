# Explicit MX32 dispatch producer

`runtime.kernels.native_moe_protocol` now accepts the explicit option
`dispatch_quant: "mx"` with `kind: "arch35_fullmesh"` and a declared
`max_inflight_per_rank`. The existing `none` default is unchanged. This first
ordinary branch supports BF16 input, E4M3 output, group-32 E8M0 scales, widths
divisible by 256, no smooth scales, no dedicated shared-expert ranks, and no
source TP partition. It selects the non-Clip MX source program; the observation's
FP8 dtype alone does not distinguish MX from MX_CLIP or identify its tiling key.

The pinned `a3925fd4` source `TokenToExpertInQuant` reads and quantizes each
token-to-expert assignment. `QuantDynamicMx` calls `ComputeMaxExp`, `ComputeScale`
and `ComputeFp8Data`, then appends routing metadata and frames each row into
480-byte data regions plus 32-byte flags. The new shared ledger prices those
three dependent vector phases. It includes the source exponent masks/reductions,
scale predicates and construction, rescaling, lane rearrangements and casts.
It does not invent issue latency for register loads/stores or scalar control.
The exact helper hash and line references are exported with the phase.

For width 5120 each assignment reads 10,240 BF16 bytes and performs 38,784
vector element operations. Its output contains 5,120 E4M3 bytes, 160 E8M0 bytes,
routing metadata and framing: 6,144 transmitted bytes per assignment. The
BF16 expert return remains 11,264 bytes per assignment. Count/status traffic
still includes zero-count experts. Receive materialization waits for both
payload and count/prefix completion; flags are cleared after copying.

The receiver's `expert_up` may omit its ordinary activation quantizer only when
it directly consumes this dispatch and its input format, group size, E8M0 scale
shape and byte count match the transport output. The existing explicit transport
producer contract carries both payload and scales. The expert-down quantizer,
router weights, local shared expert and weighted top-k combine remain intact.
An incompatible representation fails visibly; it is not silently requantized
or accepted as the same producer. No default model or hardware profile changes.

## Conditional V4.1 controls

Two new inputs under `examples/cases/high-concurrency-20261030/v41-mc2-mx32-conditional/`
retain their forty-layer parent configurations and exact parent SHA. The evidence
producer limits execution to one layer. It uses rank-zero observed B1/B4 and
q1; symmetric eight-DP sequences and context 2048 remain declared conditions.
The source capture does not establish other ranks' valid assignments, active
mask values, native fullmesh window, clip flag or compiled dispatch identity.
These evidence limitations are separate from model structural coverage.

| Rank-zero geometry | Legacy wire, layer ms | Direct-peer wire, layer ms | Fullmesh BF16, layer ms | Fullmesh MX32, layer ms |
|---|---:|---:|---:|---:|
| B1, q1 | 0.181508 | 0.159908 | 0.160275 | 0.159011 |
| B4, q1 | 0.304496 | 0.282896 | 0.283448 | 0.282051 |

The modeled native-group spans remain far below the retained native call
durations. This source-compatible composition fixes representation and dependency
coverage; it does **not** establish protocol timing accuracy:

| Geometry | Predicted dispatch µs | Observed dispatch mean / P50 µs | Predicted combine µs | Observed combine mean / P50 µs |
|---|---:|---:|---:|---:|
| B1 | 2.617 | 45.332 / 33.641 | 4.297 | 11.198 / 11.141 |
| B4 | 2.865 | 18.548 / 17.867 | 4.923 | 12.381 / 12.230 |

Successful flag checks, count-prefix bytes, row framing and local copies are
modeled. Additional polling, scalar metadata/control service, the native UB token
window and actual peer arrivals remain unresolved. They cannot be replaced by
assigning the entire observed kernel duration to Vector or network latency.

`data/verification/mc2-mx32-dispatch-20261030.json` retains all eight one-layer
runs, phase timelines, source hashes, original input hashes, source ABI and
package fingerprint. Each finished with no live tasks/programs/buffers and
preserved the 7 ms input dependency. The dedicated tests also execute unequal
and idle DP cohorts, conserve both framed network directions, and independently
scan Vector, HBM, AIV read, MTE3 and network rates at 0.5×/1×/2× with unchanged
work. Those controls validate mechanism response, not measured accuracy.
