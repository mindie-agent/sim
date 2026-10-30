# W4A8 source matrix pipeline

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The selected gate/up and grouped-down MSD programs now execute their matrix
phase through the shared steady matrix pipeline. This fixes the zero-MTE1
omission identified in the [GLM Decode audit](glm-decode-grouped-audit.md).
It does not resolve the large scalar/control discrepancy in that capture.

`w4a8_matrix.py` streams the source's rolling owner rectangles. Each active
expert retains its own doubled-M grid and packed INT4 bank; an empty expert
retains its count entry and issues no matrix tile. A matrix-only descriptor
uses one bank's width with cache reuse disabled, so each rectangle loads its
own A/B operands independently. It is not a dense matrix made by concatenating
all experts or an assumption that equal shapes share cached weights.

The source's fixed output tile is preserved (gate 128×256; short-row down
64×512; tall-row down 128×256). The inner matrix mapping is explicitly declared
as divisors of 128 with 256-wide K refill panels. It is a shared physical model,
not a recovered native MatmulImpl dispatch. L1, L0A, L0B and L0C reservations
retain their existing device capacities and owners.

MTE1 directional mode selection is explicit: these source programs opt into
`mte1_l0a`/`mte1_l0b` when both are declared. The two modes contend on the same
physical MTE1 resource. Ordinary `HierarchicalMatmul` and `SteadyMatmul` calls
retain their previous generic MTE1 semantics unless opted in. A profile without
MTE1 still has an unresolved operand-transfer cost; no rate is invented.

The former matrix envelope is replaced, not added. Existing fused work remains:

- Preprocessing and its INT4 scratch, including the down kernel's count-vector
  bulk copy/reduction on all participating AIVs.
- One 8-byte column scale per output column per source tile, and the existing
  vector-equivalent scale operation.
- FP16 workspace publication via HBM and declared Fixpipe, once per output tile.
- Existing dequantization, assist, SwiGLU, output quantization and MTE3 work.

The shared `SteadyMatmul.lower` accepts an optional fixed hierarchy, lazy owner
tile streams and source publication stages. Regular K loops and repeated
owner-local shape classes are compressed while retaining multiplicity of all
loads and nominal serial output-tile duration. The detailed hierarchy remains
the small-control oracle. No per-K Engine task explosion is introduced in the
default source path.

The pinned down source also establishes 256 `GetValue` group reads per launched
AIC/AIV in the matrix/output loop. For the retained 20-AIC CV12 case this is
15,360 logical scalar loads. It separately establishes 40 × 2,048 = 81,920 bytes
of pre-phase count-vector DMA and 20,480 vector operations, already charged.
Mapping diagnostics expose both quantities. Logical scalar reads are not
silently converted into uncached HBM transactions, vector work, or a calibrated
latency: scalar issue, cache response, cross-core flag and polling timing remain
unpriced. The 54/25 µs AIC scalar PMU intervals cannot be added as a floor.

Independent checks cover source owner/tail geometry, detailed-vs-compact
duration and service, no service before external readiness, released buffers,
MTE1 bandwidth sensitivity, one-time scale/Fixpipe publication, and constant
Engine task count for a million K panels. Synthetic scalar rates or historical
timings do not enter those checks.

The first retained GLM shape replay changes gate from 15.932 to 16.110 µs and
down from 7.264 to 8.954 µs against observed 88.039/52.907 µs. MTE1 service is
now nonzero and other physical work is conserved; the remaining errors are
still approximately −82%/−83%. This is a physical mechanism correction, not a
claim of 15% accuracy or a full-body Forward validation.

## Retained execution evidence

The native-contract report
retains the original 5,550-call observations and the new physical-service ledger.
The complete geometry report
and its producer
exercise the committed runtime `036cf85e`. Package content SHA256 stayed
`0dec1c03303ebba1784a5d55c5153b76eef33919f3161a5240da0e82092772b7` throughout.

| Declared geometry | Gate µs | Down µs | Target body Forward ms |
|---|---:|---:|---:|
| Decode q=1, sequences=1, context=4096 | 16.110 | 8.954 | 23.962 |
| Decode q=1, sequences=64, context=4096 | 410.859 | 286.516 | 107.351 |
| Decode q=4, sequences=32, context=4096 | 428.966 | 311.656 | 115.193 |
| Prefill q=4096, sequences=1 | 1934.778 | 2296.641 | not run |

Source kernel task counts remain 257–297, with at most 293 live tasks, even
for the Prefill source matrix with 32,768 routed rows. All source/body runs
released every task and program. The local four-geometry verification took
57.69 seconds and used 266,797,056 bytes (254.44 MiB) peak RSS. This wall time
includes complete frontend construction and three complete body simulations.
The source pair's one-token producer alone took 0.577 seconds.

The high-row cases are workload generalization checks. They have no matched
body elapsed-time observation and do not establish production TPOT or body
accuracy. The representative source kernels start after an explicit 7 ms input
dependency; their reported durations subtract that readiness boundary. The body
results exclude host submission, sampling and cross-step waits.
