# Scalar transfer mechanism

The missing dominant GDN gating service is a scalar UB-to-GM output loop.
It cannot be represented by bulk DMA bytes divided by HBM bandwidth.
The new queue program runs on the same `Engine` as other resources. It contains
no model names, hardware-name dispatch, width table or latency multiplier.

## Compiled path and reproduction boundary

The source and compiler products come from the original capture image:
`sha256:c3d00a196905159dbf0b97588d34f085086c2708712fa0d1c8fedd807f0e48a4`.
Artifacts are under `data/observations/gating-probe-20261028/`.

The original GDN binary is byte-identical for 64, 256, 4096 and 16384 tokens:
SHA-256 `fbbc7bcc3d2a29f079ebcee08d0a65276c076061af110d5e690a2db9ad85227b`.
Its `store_ubuf_to_gm_2d_bfloat16_t` has a positive-stride loop at addresses
`0xb14` through `0xb30`: address arithmetic, `LD_IO.b16`, `ST_DEV.b16` and
loop control. The UB row advances by 16 bytes. MTE3/scalar synchronization
surrounds the loop. The tile is 64 rows by 8 heads; 16384 rows on 64 units
require four row iterations and six column iterations.

A separate primitive loads a 64-by-8 BF16 seed, converts/multiplies/converts it,
then writes masked tiles. Its inspected binary retains the same eight-instruction
inner store loop and UB stride. A copy-only candidate produced a different
32-byte padded UB/DMA path and was rejected for this calibration.

The stride-isolation variant adds one runtime destination leading dimension to
retained compiler IR. Logical masks and loops still use the column count.
The original and modified IR, exact transformation, disassembly and binary
are preserved. This variant is labeled as an experiment, not an original kernel.

## Discriminating observations

The native ACL runner uses no Torch/model imports, allocates at most 16 MiB,
and checks every output element and padding cell. It warms up 512 launches,
then records five event intervals of 20 launches each. Stream synchronization
has a 10-second bound and the process a 25-second outer bound plus cleanup.
Measured runner RSS is approximately 171 MiB.

The initial Python compiler/runtime import used about 1.96 GiB. It is retained
as an over-budget observation, not a compliant routine measurement path.
Subsequent compilation uses retained IR without importing that runtime.

1. Reversing execution order with extensive warmup preserves the width effect.
   At 64 units and 4096 rows, 32 columns take 126.494 us; 64 columns take
   166.928 us. A constant per-element transfer rate cannot explain both.
2. The width effect also exists with one unit. Multi-unit contention alone
   cannot explain it.
3. With one unit, 64 rows and 32 logical columns, changing only destination
   leading dimension from 32 to 64 elements changes the median from 103.830
   to 71.951 us. Both execute 2048 scalar writes with the same binary.
4. The address sweep varies stores per row, leading dimension and base offset.
   Repeated requests to the same 128-byte region explain the observed trend
   through a finite queue. The region size and queue parameters are inferred
   from these experiments; they are not published hardware specifications.

Native replay of the unchanged GDN binary gives medians 152.812, 172.806 and
647.665 us at 64, 4096 and 16384 tokens. Every output is checked against
`g=-log(2)` and `beta=1/2` for zero inputs. These replay values are independent
observations, not queue fitting inputs. The historical 16384-token kernel median
is 673.704 us; image identity does not guarantee identical runtime conditions.

## Queue semantics and calibration

`QueuedStore` explicitly models:

- Finite outstanding requests; a slot releases on completion of the response.
- Serial issue service per participating unit.
- Address-granule dependencies between writes.
- Separate payload memory service, response latency and row-control work.
- Backpressure when memory service is contended.

Live address records are removed when their final pending request completes.
Storage scales with active requests. `TiledPointwise` groups only units with
identical row ownership and address phase. Different groups retain independent
instruction streams; there is no artificial barrier between their tiles.
Nested programs preserve buffer lifetimes until their child completes.

Only the first 24 rows of `native-address-sweep/stdout.jsonl` train the candidate:
256 rows, one unit, column counts 1/2/4/8, and six leading dimensions.
The finite capacity search checks 4 through 16 outstanding requests. It identifies
8 slots, 91.25 ns response latency, 11.3125 ns per scalar-loop issue, and
7 ns row control. The fitted common setup is zero. These are primitive service
parameters, not FLOP/byte inflation or a whole-operator correction.

The native max-plus oracle is retained for reproducible parameter identification.
Production predictions use the Python resource engine, including actual memory
contention. The frozen protocol was written before measuring the ten validation
cases; validation does not change parameters.

| Heldout geometry `(rows, columns, units, leading dimension)` | Engine us | Measured us | Error |
| --- | ---: | ---: | ---: |
| `(192,3,1,12)` | 31.889 | 31.041 | +2.73% |
| `(512,5,1,20)` | 138.320 | 141.276 | -2.09% |
| `(257,7,1,47)`, offset 5 | 80.613 | 81.915 | -1.59% |
| `(128,15,1,96)` | 68.296 | 68.666 | -0.54% |
| `(1024,24,1,80)`, offset 9 | 898.578 | 909.559 | -1.21% |
| `(384,13,1,64)` | 169.331 | 170.402 | -0.63% |
| `(64,5,1,64)` | 9.396 | 9.380 | +0.18% |
| `(4096,13,64,64)` | 28.222 | 32.510 | -13.19% |
| `(8192,24,64,80)`, offset 9 | 112.322 | 118.329 | -5.08% |
| `(4097,7,64,47)`, offset 5 | 39.981 | 43.924 | -8.98% |

The seven single-unit cases are within 3%; the three 64-unit cases still
underpredict by 5-13%. That remaining shared-service effect is not corrected
with a unit-count multiplier. The production engine validation retains at most
50 live tasks and retires every task/program on these cases.

## Integration and limits

The example's declared compiler implementation binds `gdn_gating` to
`TiledPointwise`: four input bytes and eight per-column parameter bytes,
four-byte DMA decay output, and two-byte queued scalar beta output. Its
`scalar_store_issue` resource rate scales the independently declared unit
count; the queue parameters stay fixed. Parameter provenance is carried in
the output. Other hardware uses its own declared implementation/resources.

For `(16384,48)`, the model changes from 1.966 us to 582.938 us, against
673.704 us in the historical capture, leaving -13.47% error. The 1.966 us value
is the intermediate resource envelope after correcting the per-head geometry,
not a shape-matched row from baseline revision `81b376b`. The machine-readable
baseline comparison therefore leaves this row's `before_us` unset. This repairs the
dominant missing transfer mechanism but does not establish full operator
acceptance. Vector transcendental expansion, synchronization, kernel entry
and shared transfer response remain incompletely modeled.

The separate short-row sweep exposes a roughly 4.2-4.5 us event interval for
tiny work. That interval may include host submission and entry/exit work.
It is not installed as a minimum kernel time. The queue comparison excludes
the primitive's seed DMA/conversion and entry/exit stages; small-work timing
requires a boundary-matched device trace.

Failed compiler/ABI experiments remain beside successful captures. The initial
loader error was resolved by matching the image's `aclrtBinaryLoadFromData`
path. A hidden FFTS pointer initially shifted launch arguments; the retained
compiler flags use `--disable-ffts`, as the original loader does. Failed launches
are never calibration samples. The original checkout remains untouched.

## Reproduction

```sh
.venv/bin/python tools/measure/fit_store_queue.py \
  data/observations/gating-probe-20261028/native-address-sweep/stdout.jsonl \
  -o /tmp/store-training.json

.venv/bin/python tools/measure/verify_store_queue.py \
  data/observations/gating-probe-20261028/native-queue-holdout/protocol.json \
  data/observations/gating-probe-20261028/native-queue-holdout/stdout.jsonl \
  -o /tmp/store-heldout.json
```

The fitting command requires NumPy and a C++17 compiler; it never opens a device.
`tools/measure/compile_native_probe.sh`, `prepare_strided_ir.py` and
`native_kernel_probe.cpp` preserve the actual build/launch input contracts.
