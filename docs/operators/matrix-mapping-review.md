# Generic matrix mapping review

This review records independent physical constraints and source comparisons on
2026-10-29. It is a design review of analytical mappings, not identification of
the tile selected by a deployed vendor kernel. The observed matrix residuals
motivate checking the mapping; none of the calculations below uses an observed
duration as a rate, multiplier or selection objective.

## Missing level in the fixed mapping

The reviewed `TiledMatmul` maps one M/N output tile and its K panels to one core.
Its defaults are `128 x 128 x 128`. It correctly distinguishes ideal shared
HBM reuse from private operand copies: an HBM hit does not eliminate the copy
into another owner's private memory.

That fixed mapping also makes the output compute tile the L1 refill tile. It
reloads both operands for every output tile, even when one owner could retain
a larger pair of L1 panels and compute several output subtiles from them. It
does not represent a separate L1-to-L0 copy stage or L0A/L0B residency.

The reusable extension is an explicit hierarchy:

| Boundary | Work removed by reuse | Work still required |
| --- | --- | --- |
| Shared HBM/cache | Declared ideal shared-cache hits can avoid HBM reads | Every owner's private copy and downstream work |
| Private L1 macro panel | A panel can serve multiple N subtiles; B can serve multiple M subtiles on that owner | Refill on another owner; L1 residency while consumers remain |
| Private L0 compute fragments | Explicit local reuse may avoid another fragment load | The declared L1-to-L0 transfers and buffer capacity; arithmetic and accumulator residency |

For macro shape `(BM, BN, BK)` and micro shape `(tm, tn, BK)`, an interior
macro panel refills `BM*BK*wa + BN*BK*wb` operand payload bytes into L1.
A simple legal micro mapping reloads both fragments for each micro product;
its L1-to-L0 work is
`(BM/tm)*(BN/tn)*(tm*BK*wa + tn*BK*wb)`.
These are different byte counts. Scales and layout transformations require
their own declared representation, rather than being removed by either
formula.

Across the complete dense GEMM, the corresponding private L1 payload reads
are `M*K*wa*ceil(N/BN) + N*K*wb*ceil(M/BM)`. This formula conserves tails
when each logical macro slice is assigned once. It does not grant private
reuse between cores. Global ideal HBM reads can remain `A+B`, independently.

## Rates and capacities in the actual A5 case

Reading `examples/cases/qwen38-a5.json` through `parse_case` gives:

| Role | Aggregate bytes/s | Owners | Per-owner bytes/s |
| --- | ---: | ---: | ---: |
| `mte2` | 4,000,000,000,000 | 32 | 125,000,000,000 |
| `mte2_aic` | 4,224,000,000,000 | 32 | 132,000,000,000 |
| `mte2_aiv` | 4,224,000,000,000 | 64 | 66,000,000,000 |
| `mte1` | 13,516,800,000,000 | 32 | 422,400,000,000 |
| `fixpipe` | 6,758,400,000,000 | 32 | 211,200,000,000 |

The case explicitly supplies legacy `mte_bw_bytes_per_s=4e12`. Hardware
normalization preserves that as ordinary `mte2`; it coexists with the profile's
split AIC/AIV roles. The reviewed fixed matrix recipe uses ordinary `mte2`.
Using `mte2_aic` in a new recipe is a separately reportable physical-role
choice, not a hidden rate correction.

The split nominal supply follows the retained platform's 80 B/cycle/AIC and
40 B/cycle/AIV declarations at the profile's 1.65 GHz. These are platform
memory-path supply declarations, not measured MTE2 instruction-response
curves. Shared HBM supply remains a separate constraint. The case's scalar
store issue fit is unrelated to matrix rates.

The retained architecture documentation declares L1=512 KiB, L0A=64 KiB,
L0B=64 KiB and L0C=256 KiB. Both L1-to-L0A and L1-to-L0B use `PIPE_MTE1`
at 256 B/cycle. Charging A and B to two independent 256 B/cycle resources
would invent parallel supply. The existing profile exposes one shared MTE1
resource with 32 owners.

This per-owner interpretation is consistent with retained profiler source:
`CalMemoryUnitBw` multiplies rate by frequency and core count, and
`InsertPipeMaxBw` scales MTE1/MTE2/FIXP with cube-core count while vector
transfer paths use vector-core count. It is not a reason to multiply an
already aggregated device rate by the core count a second time.

## Independent arithmetic controls

Use BF16 operands, FP32 accumulators, 128-wide K panels, 32 cores and the
case's BF16 aggregate peak of 432,537,600,000,000 FLOP/s. Per core this is
13,516,800,000,000 FLOP/s. These arithmetic controls isolate operand mapping;
they exclude setup, shared-HBM contention and output stores.

| L1 macro M x N | Private refill bytes | Refill us at ordinary MTE2 | Compute us | Micro MTE1 us | Double L1 bytes | L0C bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 x 128 | 65,536 | 0.524288 | 0.310303 | 0.155152 | 131,072 | 65,536 |
| 128 x 256 | 98,304 | 0.786432 | 0.620606 | 0.310303 | 196,608 | 131,072 |
| 256 x 128 | 98,304 | 0.786432 | 0.620606 | 0.310303 | 196,608 | 131,072 |
| 256 x 256 | 131,072 | 1.048576 | 1.241212 | 0.620606 | 262,144 | 262,144 |

At the reviewed fixed 128-cubed mapping, private refill service is exactly
1.6896 times compute service. Even perfect load/compute overlap cannot remove
that resource bottleneck. With the profile's split AIC supply it is 1.6 times;
the problem is still present. A 256-square macro halves L1 refill per useful
FLOP and can move the bottleneck to compute without changing rates. This
explains a plausible mechanism, not the accuracy of the resulting kernel
prediction.

Four counterexamples prevent interpreting larger macros as free reuse:

1. **Same-owner versus cross-owner copies.** For a 256-square output and
   K=128, four independent 128-square owners refill 262,144 bytes in total.
   A single 256-square owner refills 131,072 bytes. Both retain the same
   16,777,216 FLOPs and, with the simple micro mapping, the same 262,144
   MTE1 bytes. Ideal HBM reads are 131,072 bytes in both cases.
2. **Occupancy.** The four-owner mapping can execute its independent work in
   parallel. With non-limiting HBM and no stores, its single-panel
   load/fragment/compute path is 0.989743 us. The one-owner 256-square
   mapping takes 2.444940 us with a two-slot micro pipeline. Less total
   private traffic does not imply a shorter finite makespan.
3. **Capacity.** A 256-square FP32 accumulator consumes all 256 KiB L0C.
   A second such output cannot accumulate on that owner until the first
   releases storage. A 512-square macro would require 1 MiB L0C and is not
   legal under this resident-accumulator schedule. Splitting or spilling
   changes work and dependencies; capacity screening cannot waive them.
4. **Two buffer levels.** Two BF16 128-by-128 A fragments consume all
   64 KiB L0A; B does the same in L0B. Double-buffered L1 does not imply
   unlimited L0 fragments. FP32 fragments or larger inner K may require
   smaller tiles or single buffering.

The timing arithmetic can be reproduced independently of the simulator:

```python
cube_per_core = 432537600000000 / 32
refill_per_core = 4000000000000 / 32
mte1_per_core = 13516800000000 / 32
for m, n in ((128, 128), (128, 256), (256, 128), (256, 256)):
    k = 128
    copies = (m + n) * k * 2
    micros = (m // 128) * (n // 128)
    print(m, n, {
        "refill_us": copies / refill_per_core * 1e6,
        "compute_us": 2 * m * n * k / cube_per_core * 1e6,
        "mte1_us": micros * 65536 / mte1_per_core * 1e6,
        "double_l1_bytes": 2 * copies,
        "l0c_bytes": m * n * 4,
    })
```

## Finite pipeline contract

A two-level phase model is sufficient for this question. It need not recreate
MMAD instructions, address generation or compiler scheduling. It must retain
the following causal facts:

- A macro panel is ready only after its declared input service; every inner
  fragment waits for that readiness.
- A panel remains allocated until all its inner consumers complete. A
  fragment remains allocated through its compute consumer. Accumulators
  remain live across K panels through the final store.
- A/B fragment service shares MTE1; arithmetic shares the matrix resource.
  Separate stage names permit overlap only where dependencies and storage
  allow it.
- A finite number of slots includes prologue, drain and ragged work. Mapping
  must preserve owner placement and inactive-core limitations.

Starting a separate inner pipeline after every K-panel barrier is a valid
but more restricted mapping. It can introduce an avoidable fragment-prefetch
bubble: the next K panel's first fragment cannot load during the current
panel's last compute stage. In this control, one fragment takes 0.155152 us
to load and 0.310303 us to compute. With four micros per panel, repeated
first-fragment fill adds 12.5% of compute time per panel in the long-stream
limit; with one micro it adds 50%. A continuous micro stream across K can
prefetch into the other L0 slot when the next L1 panel is ready. An analytical
mapping should either express this dependency pattern or explicitly identify
the barrier policy in its result.

The L1 producer window must also be independent of the micro consumer window.
For one owner, `M=N=4, K=4`, macro `4x4x2`, micro `2x2x2`, and BF16 operands,
each panel contains four 16-FLOP micro products and refills 32 bytes. Set cube
to 16 FLOP/ms, MTE2 to `32/3` bytes/ms, other rates effectively unlimited,
L1=64 B, L0A/B=16 B each, and L0C=64 B. A legal two-panel schedule loads
panel 0 at `[0,3]`, loads panel 1 at `[3,6]`, and computes the two panels at
`[3,7]` and `[7,11]`: 11 ms plus negligible transfers. A two-micro issue
window that constructs the next L1 load only after the previous panel's
penultimate micro retires instead finishes at 13 ms. Both are bounded
schedules, but the latter does not implement early double-buffered L1
prefetch. Separate bounded producer/consumer windows preserve this mechanism
without materializing all K panels.

Mapping selection needs the same physical ownership for every resource.
For `M=N=3, K=2`, two owners, 2-square micro tiles, candidates 2-square and
4-square macros, and MTE1=MTE2=8 B/ms, private refill bounds are 7 and 6 ms.
The 4-square candidate sends all 48 MTE1 bytes to one owner: its MTE1 bound
is 12 ms, not the 6 ms obtained by dividing by the entire device rate.
The 2-square candidate distributes 28 and 20 MTE1 bytes, giving a 7 ms
bound. Ignoring MTE1 ownership selects the wrong candidate even before
considering pipeline fill/drain. With all other rates at 1e9 and unrestricted
buffers, the reviewed causal programs take approximately 11 and 18 ms.

## Representation and selection limits

Transferred packed bytes, L1 resident bytes and L0 compute-format bytes are
separate quantities. Current matrix metadata identifies activation, weight
and compute formats, packed weight bytes, and activation scales. It does not
identify every vendor's quantized L1/L0 layout or scale path.

Direct dense formats can use a declared payload layout. FP4-to-BF16 and
W4A8 fused conversion cannot use packed weight size as the unpacked L0
footprint. Scales must remain in HBM/private refill accounting and any buffer
in which the mapping retains them. Assuming that every scale follows both
ordinary MMAD operands through MTE1 is also a distinct approximation, not an
established instruction contract. A first hierarchy should expose its
supported formats and assumptions rather than silently invent such paths.

An explicit `HierarchicalMatmul` and a small finite set of legal mappings
provide a useful next experiment. Compare equal shapes, rates, HBM policy and
submission boundaries. Record the selected mapping and rejected capacity
domains. A predicted best mapping is a design-space result, not evidence that
the deployed library selects it. Small and ragged geometries are necessary
controls alongside the large prefill hotspot. Do not silently replace a
source-selected implementation or use measured residuals to choose the tile.

## Inspected source evidence

These sources justify mechanisms, not hardware accuracy or vendor dispatch:

- LLMCompass,
  `2e015fd2ee750e6cad8d7152df52551e5b41ef20`,
  [software_model/matmul.py](https://github.com/PrincetonUniversity/LLMCompass/blob/2e015fd2ee750e6cad8d7152df52551e5b41ef20/software_model/matmul.py#L217):
  `Mapping` and capacity screening at lines 217-395; nested tile I/O, reuse
  and double buffering at 740-1050; operand-read sets and core-affinity
  commentary at 1160-1370. Reuse and capacity are useful; its whole-grid
  arrays and idealized read-set behavior should not replace our bounded
  per-owner lifetime model.
- NVIDIA CUTLASS v3.5.1, resolved to commit
  `f7b19de32c5d1f3cedfc735c2849f12b537522ee`,
  [media/docs/efficient_gemm.md](https://github.com/NVIDIA/cutlass/blob/f7b19de32c5d1f3cedfc735c2849f12b537522ee/media/docs/efficient_gemm.md#L10):
  threadblock/warp/instruction tiling, the locality-versus-occupancy tradeoff,
  and separate shared-memory/register double buffers. This is a GPU example
  of the reusable hierarchy, not an Ascend layout or rate source.
- Retained Ascend architecture documentation,
  `hicann/asc-devkit@56500ea0dc219380f50f4a17ce73ef5264f830fc`,
  [npu_arch_3510.md](../../data/observations/sources/arch3510-physical-services/platform-units/npu_arch_3510.md):
  storage table at lines 82-91 and transfer-pipe table at 124-132.
  SHA256 `3a69176f4d4eab9e3883755e92598fb9defb3d51dc117de0bd94bb1acaa03caa`.
- Retained platform declaration
  [Ascend950DT_9582.ini](../../data/observations/sources/arch3510-physical-services/Ascend950DT_9582.ini),
  AIC memory rates at lines 73-85 and vector rates at 234-242.
  SHA256 `b8e7d61780056889e795639504530ecacb2b759ad1602e929b863d837fdf0099`.
- Ascend/msopprof,
  `a52c8694979016be703006ec242901519bef31d8`,
  [retained roofline.cpp](../../data/observations/sources/arch3510-physical-services/platform-units/profiler_roofline.cpp):
  `CalMemoryUnitBw` at 237-244 and `InsertPipeMaxBw` at 277-290.
  SHA256 `0dbe5a22cf5187252115084a4dae587f3a5d97bdfd19840bd0ce1a547df75009`.
- The exact-image TBE schedule excerpt in
  [exact-image-rate-consumers.json](../../data/observations/sources/arch3510-physical-services/platform-units/exact-image-rate-consumers.json)
  divides L0 transfer geometry by separate platform L1-to-L0A/B rates.
  This verifies the existence of the boundary in the retained CANN image;
  it does not identify the matrix kernel used in the historical capture.
