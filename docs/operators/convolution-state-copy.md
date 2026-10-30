# Noncontiguous convolution-state caller copies

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

`lowerings.conv_state_copy = {"implementation":
"triton_noncontiguous_conv_state_copy", "options": {"state_slots": 6}}`
explicitly selects the retained caller that packs a dim-first persistent state
view before convolution and writes it back afterwards. It is currently a decode
KDA caller extension; it does not automatically select a native convolution or
claim its speculative implementation.

The retained source is `vllm_ascend/models/glm5next/ops/causal_conv1d.py` in local
revision `8c30e4f491f73cba737a1dacc522894e437cdf3d`, SHA256
`106311478c25ef1981b95b33b9c4e7fb94fc4856cef53bca6a484ef2cd0061b8`.
Lines 13–47 describe a 256-element Triton program; lines 60–105 describe output
zeroing, optional state packing, the convolution call and state writeback.
Source identity is separate from the captured binary's compiler configuration.

The physical payload is `B × state_slots × local_qkv_channels × dtype_bytes` in
each direction for each copy. The grid is `(B, ceil(state_slots*channels/256))`.
Every program reads an index and two query-prefix entries. Gather also writes
one INT32 packed index per request. An inactive request writes a zero packed
state but does not read the persistent state; scatter reads the packed state
and masks its persistent write. The public caller currently declares distinct,
valid slots for its active request views. It does not derive cache indices or
state capacity from an acceptance probability.

The executable physical lower bound assigns source programs round-robin with
the request coordinate varying fastest. Each owner has one 256-element UB
payload slot and executes read then write; owners retain the existing physical
AIV/MTE2/MTE3 mapping. This is an explicit execution approximation. Triton
compiler grid assignment, transaction amplification of the strided persistent
view, integer address issue, scalar metadata latency and register spills are
not determined by the retained Python source. They remain listed as unpriced
mechanisms. Metadata memory bytes are counted without pretending they are AIV
DMA or charging an inferred scalar latency.

Ordinary GLM Flash q4 cases at B16/B32/B64 create 68 copy calls around 34
convolutions, preserving the convolution's existing work. The capture's first
CausalConv1d has state `[64,6,3072]`, INT32 prefix and index metadata; its two
neighboring `_copy_conv_state_kernel` events have no tensor descriptors.
The source-compatible packing contract therefore remains conditional on the
retained caller, not an exact binary/layout proof.

Current payload-only predictions for all six representative copy calls are
1.312 us, limited by the existing inclusive A5 vector minimum. The c64 captured
copy-family mean is 21.777 us/call. This implementation establishes the missing
call and state boundary but **does not resolve that residual**. No measured
duration, PMU sum or residual has been installed as a service rate or floor.

Validation: 57 affected tests passed. Controls cover full-layer call counts,
unchanged inner convolution, all-active/inactive masks, 256-element tails,
one owner that cannot borrow idle AIVs, late input, finite UB release, and actual
gather → convolution → scatter execution. Compact evidence:
`data/verification/glm-convolution-state-copy-20261030.json`.

## Directional service audit, October 30

The streamed directional audit
selects the same-stream copy → convolution → copy chain, allowing the captured
input/output materialization calls inside the convolution wrapper. The
convolution's `[B,6,3072]` state identifies B16/B32/B64 populations. It does not
assign a tensor ABI to a copy event that lacks one. There are respectively
1,020 / 1,054 / 1,054 pairs; every copy uses 64 AIV blocks.

| B | Gather mean µs | Scatter mean µs | Gather MTE2 busy µs | Scatter MTE2 busy µs |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 9.384 | 7.293 | 5.780 | 3.056 |
| 32 | 15.044 | 10.748 | 10.963 | 5.685 |
| 64 | 26.799 | 16.755 | 21.862 | 10.295 |

Vector busy time is approximately zero. Scalar and MTE2/MTE3 busy time overlap,
so their sum is not another latency to add to the existing program. The source
grid gives 18 / 36 / 72 programs per declared owner, each with three scalar
metadata loads and a dependent payload load/store. Gather MTE2 busy time per
program is approximately 0.30–0.32 µs; scatter is 0.14–0.17 µs. These are
diagnostics from a mixed kernel, **not isolated DMA/metadata calibration**.

For a declared 32-byte transaction granule, each request's persistent state
has 36,864 unique bytes, while summing footprints independently across its
72 source programs gives 221,184 bytes. All six history coordinates revisit
the same cache lines. Multiplying compulsory HBM traffic by six would assume
away cross-program cache reuse; the source and PMU rows do not establish that
policy. A compiler transaction/issue contract and independent small-transfer
service parameters remain necessary before changing this planner's rates.
The BF16 scalar **store** primitive used by GDN is not a measurement of these
scalar index **loads** or the MTE2 strided payload. No such substitution or
whole-copy residual floor is made here.

## Conditional source-program busy capacity

An explicit `joint_program_service` lowering option now adds overlapping
owner-local busy bounds to the same source program. The retained parameter
packet is [a5-copy-joint-service-20261030/parameters.json](../../data/observations/a5-copy-joint-service-20261030/parameters.json).
Its `joint_program_service` object goes in the copy lowering's `options`; its
`compute_arch` object goes in `machine.device`. No default hardware rate changes.

Only the B16/B32/B64 PMU busy populations supply these affine parameters:

| Direction | Pipe | Owner intercept ns | ns/source program |
| --- | --- | ---: | ---: |
| Gather | Scalar | 1112.636 | 181.161 |
| Gather | MTE2 | 330.776 | 298.518 |
| Gather | MTE3 | 571.969 | 139.955 |
| Scatter | Scalar | 1922.270 | 136.334 |
| Scatter | MTE2 | 750.865 | 133.207 |
| Scatter | MTE3 | 642.036 | 134.413 |

One owner consumes `intercept + programs * slope` on each pipe's conditional
busy resource. Those three demands overlap with each other and with the native
payload program; completion joins all four paths. Busy values are not summed
into wall time, and the intercept is not labeled as launch, barrier or poll.
An enclosing whole-call response stays an inclusive completion constraint.
The 64 owner domains remain separate, so idle owners cannot accelerate a busy
one. Supply perturbations change busy service without changing source programs,
metadata loads, HBM payload or MTE bytes.

These resources describe joint source-schedule capacity, not universal Scalar,
MTE2 or MTE3 opcode prices or occupancy against unrelated kernels. The observed
features have rank one: each program adds three metadata loads and 512 payload
bytes. Compiler descriptor counts, physical grid mapping and exporter owner
aggregation remain unconfirmed. The declared applicability contract therefore
keeps 64 AIV owners, all-active BF16 state `[B,6,3072]`, 256 elements per program
and 12 metadata bytes. Other layouts, tails or masks retain the existing
analytical program and report `analytic_outside_joint_program_contract`.
Nothing in this packet prices unknown polling or generalizes a descriptor
price into `CausalConv1dUpdate`.

B1/B4/B8 captures remain independent checks and are absent from fitting.
Small-batch scatter busy predictions overestimate individual counters by up
to 258%; that failure is preserved. Busy lower bounds also leave substantial
whole-call residuals:

| B | Gather lower bound / observed µs | Scatter lower bound / observed µs |
| ---: | ---: | ---: |
| 1 | 1.475 / 3.810 | 2.195 / 3.559 |
| 4 | 2.018 / 4.896 | 2.604 / 4.153 |
| 8 | 3.017 / 6.403 | 3.149 / 5.242 |
| 16 | 5.704 / 9.385 | 4.376 / 7.293 |
| 32 | 11.077 / 15.044 | 6.830 / 10.748 |
| 64 | 21.824 / 26.799 | 11.738 / 16.755 |

For uneven B1/B4 owner counts, execution uses the slowest integer-count owner;
the evidence packet separately compares the fitted mean-owner coordinate with
the original PMU aggregation. Neither interpretation proves the compiler map.
The full GLM 45-layer B16 q4 Body runs through `predict(scope="body")` with
existing whole-call responses retained. Its measured local wall/RSS, exact
result, unchanged-work controls and all residuals are in the
verification packet.
This is execution validation, not a new Body accuracy claim.

Reproduce from the retained profiler summary (no raw scan or NPU required):

```sh
python3 tools/fit_copy_joint_service.py /Users/maoxx241/code/profiler/data/observations/sources/arch3510-physical-services/scalar-dma-physical-constraints.json data/observations/a5-copy-joint-service-20261030/parameters.json
python3 tools/verify_copy_joint_service.py --rss-byte-factor 1 --output data/verification/copy-joint-service-20261030.json
```

The RSS conversion is 1 for the recorded macOS run and 1024 for Linux. Each
verification child has a 60-second outer budget and checks a 512 MiB peak limit.
Focused validation also covers inclusive call timing, late readiness, same-owner
contention, owner overlap, finite buffer release and unchanged native-byte work.
The 4D public prediction controls and switch-fabric controls passed alongside
these checks. Review of `5d234037` confirms each routed hop frames once and
couples Tx/Rx/optional plane demand; `wire_bytes` counts physical transmissions
once while `capacity_byte_service` reports all simultaneous port service.
