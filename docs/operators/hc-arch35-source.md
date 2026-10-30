# Explicit arch35 HC programs

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The `hc_pre_arch35_mk_175ef4d4` and `hc_post_arch35_bf16_175ef4d4`
traffic profiles select separate source programs through ordinary
`native_fusion`. The old HC profiles and default selection remain unchanged.
A runnable declaration is
[`v41-hc-arch35-source-control.json`](../../examples/source-families/v41-hc-arch35-source-control.json).

This is a conditional source model, not proof that the captured r31 binary
executes that source. Retained V4.1 `HcPre` has an extra `[B,4]` input and
output absent from the source's three-output interface. Those tensors are
not silently assigned an invented operation or ignored in an accuracy claim.
Neither measured residuals nor measured durations enter these programs.

## Source and support domain

All source files are from `vllm-ascend` commit
`175ef4d49c2072f9233619636f0309f4f88b4e2c`; complete paths, line ranges and SHA256
values are embedded in each prepared operator's `program_execution.source`.
The local files were checked byte-for-byte against that Git object.

- `hc_pre.cpp:49–74`: C310 mixed entry dispatches arch35 key1000/key1001.
- `hc_pre_tiling_arch35.h:295–429`: host tile and stage2 UB sizing.
- `hc_pre_m_k_split_core_arch35.h:36–231,276–389`: source buffers, AIV/Cube
  ownership, direct UB→L1 publication, RMS partials and stage transition.
- `hc_pre_cube_compute_arch35.h:65–190,216–233`: HF32, L0 ping/pong,
  MTE1 operand copies, and GM projection output.
- `hc_pre_base_arch35.h:134–317,431–553,591–703,879–1075`: cast/RMS,
  group reduction, gate/Sinkhorn and residual mixing.
- `hc_post.cpp:39–59`, `hc_post_tiling_arch35.h:184–203`, and
  `hc_post_bfloat16.h:96–362`: BF16 register path, D4096 and active row owners.

Pre covers BF16 X, FP32 projection/scales/base, HC4/mixed24, M1–256,
H divisible by32 and H≤32768, and two AIV per AIC. This keeps one source M
partition and host key1000; actual K partitions may still be one for a small H.
Post covers BF16 X/residual, FP32 coefficients, HC4, M1–256 and H divisible
by32 with H≤32768. Geometry and buffer capacity are explicit; unsupported cases error.
These are physical capability declarations, not SKU-name timing branches.

## Physical execution

Pre stage1 reads BF16 X through each active AIV's MTE2 pipe, casts and computes
partial RMS, transposes to NZ and publishes FP32 X directly from UB to L1 via
MTE3. It reads FP32 projection weights via AIC MTE2. There is no intermediate
cast-ring GM write/read and no Cube Gram operation in this arch35 branch.

The finite input and transpose buffers, two L1 slots and two L0 operand slots
have separate readiness dependencies. MTE1 completion releases an L1 slot;
Cube may continue using its L0 operands. Source microK is
`(256 // max(aligned_M, 32)) * 32`; M65 therefore uses K96, not K102.
Logical M×24×K FLOPs are recorded separately from issued M16/N32-rounded HF32
work. M1 assigns its only row to the odd AIV; the idle partner does no payload
or vector work.

Pre allocates the source's fixed512KiB L1,64KiB L0A,64KiB L0B,256KiB L0C
and computed UB per launch owner, including launch owners that have no tile.
Source initialization precedes the active-owner guard. After stage1 completes,
its buffers are released before stage2 allocation. Stage2 reads projection and
RMS partials, performs group reduction/gates/Sinkhorn and writes only public
outputs. The three scalar `GetValue` payloads occur at their source stages;
no scalar latency is inferred. Post coefficient output may overlap Sinkhorn.

Post assigns at most one active owner per row until rows exceed AIV count.
It reserves213152B UB per active owner. Each D tile reads X/residual, computes
four outputs and writes BF16 once. Its source casts all4×4096 scratch elements
even on a short last tile; this fixed tail work is retained. Coefficients are
read once per row, not once per D tile. Post has no MTE1 or Cube work.

Both programs use the common Engine and one outer physical invocation.
Nested stages acquire no separate launch floor. Pre lets existing hardware
policy choose the maximum Cube/Vector completion minimum once from real work.
The physical source program replaces all selected semantic member work;
it does not add onto the old HC envelope.

## Limits

Vector work is a disclosed elementary arithmetic/cast/transpose envelope.
Register permutations, instruction issue throughput, scalar control latency
and extra polling are not calibrated from residuals. The reduction formula
retains source masked arithmetic, rather than replacing it with ideal n−1
reductions. MTE1 uses declared base-pipe bytes; directional operand instruction
modes are not asserted by this profile. Control flags impose readiness, with
no invented fixed latency. Pre stage2 conservatively completes one row batch
before the next; this is not an instruction-exact overlap claim.

## Local verification

Producer and
receipt contain28 real
ordinary-frontend/Engine controls for B1/B8,H5120, delayed input7ms and
MTE1/AIV/HBM rates0.5×/1×/2×. Same work, no early service, zero live tasks and
all buffers released. Nominal isolated source spans:

| B | HC Pre | HC Post |
|---|---:|---:|
| 1 | 2.297µs | 3.841µs |
| 8 | 2.316µs | 3.841µs |

Pre responds monotonically to all three axes. Post responds to AIV rates;
MTE1 has no Post work and HBM is below its private-pipe bottleneck, so those
scans plateau (roundoff≤1e−9µs). Plateaus are not converted into fabricated
strict sensitivity. These are source operator predictions, not observed
accuracy, full-body latency, or a linear sum to apply to a model residual.

The focused32 tests additionally cover M65 integer microK, a D4128 tail,
finite capacity, late inputs, one startup, graph padding, external consumers
and the unchanged old HC/native-fusion paths.

## Complete-body template residency

The source owner DAG is immutable and shared through a four-geometry cache.
The cache contains unbound work and dependencies; each invocation still binds
its own Engine, rates, physical owners, buffer allocation and input readiness.
Device capacity validation runs before each lookup. No measured timing is cached.

The memory receipt
separates reduced-layer allocation diagnostics from complete 40-layer acceptance.
At ten layers, retained HCPre preparation allocation fell from63.97MB to0.935MB.
The production c1 body retained exactly the same7.430227510823243ms and all
serviced work, while peak RSS fell from682.75MiB to557.625MiB with HC caching alone.
After also using the shared read-only Engine rates view, the complete DP8 c32
body fell from844.234MiB to436.563MiB. Its12.36084511861599ms prediction and
all resource service totals were unchanged, with zero live tasks/programs.
Its wall time was55.36s versus57.11s:2.4million Engine tasks still execute, so
this removes retained compilation state without claiming second-scale execution.
