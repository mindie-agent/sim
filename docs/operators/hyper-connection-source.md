# Hyper-connection source resource plans

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The optional HC profiles select inspected native implementations on the ordinary
engine. They neither identify a historical binary nor use measured durations as
inputs. `prediction.family_hc` supplies `OperatorPlan` work and finite programs;
the semantic six-node HcPre group remains available for attribution.

| Profile | Independent source | Resource program |
| --- | --- | --- |
| `hc_pre_m_k_split_175ef4d4` | vllm-ascend `175ef4d49c2072f9233619636f0309f4f88b4e2c`, non-arch35 M/K split | Cast; shared-X projection and Gram; partial reduction and coefficients; input mixing and output |
| `hc_post_d_split_175ef4d4` | Same revision, `hc_post_d_split.h` and its host tiler | Per-owner 2048-column load/cast/multiply/reduce/store loops with two queue slots |

Select them through the existing native fusion declaration:

```json
{
  "native_fusion": {
    "implementation": "declared_native_fusion",
    "options": {
      "roles": ["hc_pre", "hc_post"],
      "traffic_profile": {
        "hc_pre": "hc_pre_m_k_split_175ef4d4",
        "hc_post": "hc_post_d_split_175ef4d4"
      }
    }
  }
}
```

This object belongs under `runtime.kernels.lowerings`. The default unselected
semantic graph and `preserve_workspaces` reference retain their existing
meaning. Structure export prepares selected source work without running timing
or importing observations; its physical signature includes the source profile
and hardware ownership geometry.

## HcPre: shared operands and real partials

`hc_pre_cube_compute.h:267-341` loads an FP32 X panel once into L1.
`MmadA2` computes its 16-row Gram block, then `MmadAB` uses the same operand for
the projection. `MmadA2:418-436` computes the off-diagonal products as well;
stage 2 gathers the diagonal. A separate vector reduction reading all of X
would describe a different implementation.

For T tokens, hidden width H, four residual streams, and K=4H:

- The cast reads `2TK` BF16 bytes and writes `4TK` FP32 bytes.
- Projection plus Gram performs `2 * padded_T * K * (24 + 16)` matrix
  operations. `MmadAB` explicitly enables HF32 for the 24-column projection;
  `MmadA2` leaves HF32 off for the 16-column Gram. The plan therefore charges
  the projection to an explicit `hf32` compute peak and the Gram to `fp32`.
  There is no implicit HF32 fallback or rate scaling: a device without a
  declared `hf32` peak rejects the selected plan.
- The one X read is `4TK`. Projection weights are read for each source M tile.
  Split-K partitions produce `4T * split_K * (24 + 16)` bytes of projection
  and Gram partials, which stage 2 reads.
- Inverse RMS operates on those partials. Sinkhorn operates only on the
  `4 x 4` combination coefficients; pre/post gates do not enter its iterations.
- Projection and Gram outputs use the source `Fixp`/`CopyOut` path (lines
  234-250), not AIV MTE3. The plan preserves the same HBM bytes and charges
  `fixpipe`; a device without a fixpipe service rejects the selection.
- Casting and input mixing read through the AIV private read pipe; the shared
  projection reads through the AIC pipe. A generic-only device keeps using its
  single `mte2` service.
- Mixing rereads the original BF16 X and writes y, post and combination
  coefficients. Input/output casts remain vector work.

The host tiler uses M tiles up to 256 rows and K partitions aligned to 256.
It derives partitions from the declared compute units. The 20-core cases have
16 M partitions and one K partition at T=4096, and four M partitions and five
K partitions at T=1024. Neither partition is a model-name rule.

For the Part2 vector phase, the host sets `rowFormer = ceil(T / AIV)` and
`used = ceil(T / rowFormer)`, not `min(T, AIV)`. Each participating owner reads
its own base/coefficient payload, so at T=41 and 40 AIVs the plan has 21 owners
and 21 * 24 * 4 = 2016 parameter bytes rather than 40 owners. Owners with equal
row counts share a bounded pinned group; the shorter tail owner has its own
group. Coefficients and mixing are pinned to those participating AIVs and their
private read/write units, so idle units cannot supply the work. Equal owners
are grouped only when their physical resource partitions are disjoint; a
shared read/write pipe preserves each owner's row count separately.

Stage1 uses `hc_pre_m_k_split_core.h:82-99,135-140`: AIC owner
`m * Kparts + k` owns its M/K rectangle, and AIV owners `2 * AIC` and
`2 * AIC + 1` divide each M panel's rows by ceil/floor. They do not divide K
again. At T1/H7168 there are 19 active AICs and only the even AIVs 0 through 36
carry cast payload; the final K slice is 1024 rather than 1536 elements. At
T1024/H7168, all 20 AICs and 40 AIVs carry work, with the final K slice 5120
rather than 5888 elements. The finite owner phases preserve these unequal
payloads for vector, private copies, matrix and Fixpipe service. They keep
the existing global cast-to-projection barrier and do not claim to recover
the native cross-core panel pipeline.

The FP32 cast workspace is a double-buffered ring:
`cores * 2 * min(256, per_core_rows) * 1024 * 4` bytes. At 20 cores and full
256-row tiles that is 40 MiB. The plan retains every read/write through GM as
HBM service. It does not infer a cache hit from the ring fitting in L2.
Searching this source's HC trees finds no explicit L2 cache hint or bypass;
the cache-line constants describe strides, not measured cache residency.

The source profile currently supports BF16 X, FP32 parameters, four residual
streams and K divisible by 32. It describes the three-output source ABI
(y/post/comb). A capture exposing an additional optional pre output can share
the compared core tensor contract, but this profile does not include that
extra output write and does not claim binary identity.
If a declared external consumer requires an internal projection, affine or
inverse-RMS tensor, this source selection is rejected: those tensors are not
outputs of the selected native ABI.

This is a compact four-phase program. It preserves mandatory stage ordering
and the declared HF32/FP32 precision split, but does not reproduce stage-1
vector/Cube panel overlap, vector padding, instruction issue or a compiler
schedule. Source paths, hashes and line ranges live alongside the planner.

## HcPost: dependencies and conversions

The D-split source reads x and residual once for each token/column chunk,
and post/comb coefficients once per token. It casts BF16/FP16 inputs to FP32,
initializes the output with `x * post`, accumulates the residual combinations,
casts the result back, and stores it. Its external tensor byte count already
matched the former envelope; the missing work was conversion and sequencing.

The plan conserves those bytes, adds the two input and one output conversion
domains, preserves load-to-consumer and conversion-to-store dependencies, and
assigns token rows to the declared vector owners. Independent owners execute
concurrently; a pair of source queue slots bounds in-flight chunks. Tail
columns and uneven owner row counts retain their actual work.

FP32 coefficients are required by this source profile. The source's
`InitBuffer` calls reserve, per vector owner, two input queue slots, two output
queue slots, two FP32 cast buffers, one temporary sum buffer, coefficient
queues and broadcast scratch. At D=2048, hc=4 and BF16 x with FP32
coefficients that is 164640 requested bytes. The plan reserves that sum on the
owner's UB partition, holds it through the owner's pipeline and releases it at
completion. The configured 2048-column tile is retained even when the actual
tail width is smaller. Allocator alignment is not inferred, and the reservation
does not establish throughput. A5/arch35 uses a different source branch.

Vector instruction issue, broadcast/barrier delays and different vector
instruction rates remain unmodeled; an aggregate vector rate is not proof that
those costs are zero. No unexplained residual has been converted into a startup
constant.

The selected D-split domain requires D divisible by 64 and at most eight
residual streams. Its `DoMal`/`DoAdd` loops use complete 64-element repeats
without a remainder path, and coefficient padding assumes `hc <= 8`.
An unsupported tail such as D2051 is not silently treated as a correct native
three-column loop; the unselected semantic operator remains available.

An independent shared-pipe control uses 41 rows on 40 AIVs with 20 MTE3 units.
Owner 0 writes two rows and owner 1 one row through the same pipe, giving three
row-service times on the busiest pipe. Collapsing the 39 one-row owners to a
uniform union incorrectly gave 3.95; retaining physical owner multiplicity
gives 3. This control changes no supply rate to match an observation.

## Historical comparison and remaining error

The October 29, 2026 comparison reuses complete historical captures and their
already computed HcPre/HcPost cohort statistics. This is a new local prediction,
not a new device capture or per-layer binding. The source selection remains a
caller declaration. The table predates the ownership, HF32, fixpipe and UB
reservation corrections above; it is retained as history, not as a current
accuracy claim, and none of its durations enters the planner.

| Case and operator | Observed mean ms | Previous prediction ms | Source-plan prediction ms | Relative error |
| --- | ---: | ---: | ---: | ---: |
| V4 Flash, T4096, HcPre | 0.431534 | 0.925671 | 0.721545 | +67.20% |
| V4 Pro, T1024, HcPre | 0.191803 | 0.406399 | 0.316457 | +64.99% |
| V4 Flash, T4096, HcPost | 0.418375 | 0.251931 | 0.297336 | -28.93% |
| V4 Pro, T1024, HcPost | 0.202098 | 0.110169 | 0.129751 | -35.80% |

The historical differences above do not establish a 15% tolerance. HcPre still
conservatively charges its ring traffic to HBM. HcPost still lacks
independently established instruction and synchronization supply. These are
explicit limitations, not grounds to fit a multiplier.

The current host's 9362 platform metadata separately declares a 176160768-byte
L2 and `l2_read_rate=110`, `l2_write_rate=86` under `AICoreMemoryRates`.
It comes from a different toolkit than the capture and conflicts with other
case fields. It does not establish the captured clock, aggregation of those
rates, cache policy or actual hits. It must not silently replace the hardware
profile or remove GM bytes.

`tests/test_hc_source_resources.py` verifies independent case geometry,
shared-X Gram arithmetic, split-K partial bytes, manifest/execution work
agreement, tail chunks, conversion counts, bounded execution and absence of
service before inputs become ready. `tests/test_hc_pipeline_ownership.py`
checks the selected split read roles, fixed HF32/FP32 rates, fixpipe service,
per-owner Part2 reads and pinning, D=2048 UB reservation and release, and
engine work equality for tail owners. Historical durations are not assertions
or planner inputs.

## October 30 owner and pipe correction

The independent control
retains the baseline, corrected original input, explicit arch2201 precision
input and equal-precision control. The source producer reads no measurements.

| V4 Pro T1024/H7168 candidate | Before us | After us | Observed mean us | New conditional difference |
| --- | ---: | ---: | ---: | ---: |
| HcPre, explicit arch2201 FP32/HF32 input | 316.456747 | 333.628916 | 191.803266 | +73.94% |
| HcPost, unchanged rates | 129.750575 | 130.816269 | 202.098462 | -35.27% |

The precision input is derived from the retained AscendC table at revision
`56500ea0dc219380f50f4a17ce73ef5264f830fc`, file
`cube_instruction_theoretical_perf_summary.md`, SHA-256
`3a249d10fb6b794849d15f66208e43fa27ccd5edc6b635418911ebf11020f728`.
Its arch2201 table gives cube K parallelism 16/4/8 for BF16/FP32/HF32.
With the case's existing BF16 peak 270.336 TFLOP/s this declares
67.584/135.168 TFLOP/s for FP32/HF32. This lane does not edit hardware profiles.
The unchanged original input has no HF32 peak, so corrected HcPre reports
unsupported. An equal-peak control retaining the old FP32 supply gives the
same 333.628916 us: matrix throughput is not the limiting service here.

Work is conserved: HcPre still services 379,748,096 HBM bytes and
2,348,810,240 matrix operations. Its private reads are now 128,450,560 AIC
bytes and 118,275,840 AIV bytes; the 819,200-byte cube output is assigned
to Fixpipe instead of MTE3. HcPost retains 132,202,496 HBM bytes,
73,482,240 AIV read bytes and 58,720,256 output bytes. The corrected resource
mapping does not close the observed residual, and the larger HcPre difference
is retained.

The Pre all-GM-as-HBM lower bound is already 316.456747 us at 1.2 TB/s,
above the population mean. Establishing workspace/cache placement is therefore
more useful than changing compute peaks or fitting bandwidth. Source panel
overlap remains absent. Post still lacks instruction-specific vector/flag
supply. Its observed AIV vector mean 165.028 us and total AIV mean 195.359 us
are independent PMU fields, not additive phases or planner inputs.

The observations are 496 calls per role. Pre is MIX_AIC with BlockNum20 and
MixBlockNum40; Post is AI_VECTOR_CORE with BlockNum40. The captured Pre ABI
also exposes an optional fourth `pre` output, while this selected source
candidate emits three outputs. The table is a conditional source-candidate
comparison, not proof of identical binary, complete ABI or per-layer occurrence
binding.

Reproduce the corrected explicit-rate control:

```sh
PYTHONPATH=. python tools/verify_hc_source.py examples/cases/v4pro-a3-w4a8-tp16.json \
  --matrix-peaks '{"fp32":67584000000000,"hf32":135168000000000}' \
  -o /tmp/hc-source-control.json
```

The three HC/native-fusion test files pass 37 focused checks in 2.18 seconds.
Four descriptor controls each finish in less than 0.27 seconds. All modeled
calls preserve delayed-input readiness and release their finite resources.
