# Kimi MXFP4 grouped SiTU: explicit weight supply

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The retained independent delivery overlay contains a VCV implementation of
`GroupedMatmulSituQuant`. Its narrow Decode path reads FP4 weights on AIV,
widens them to FP8, publishes them into L1 through MTE3, and then uses the
scale-aware FP8 Cube path. Treating this as a direct FP4 matrix omits actual
AIV and copy-pipe service. The current ordinary fused-member composition did
not already contain this program: it declared the native MXFP4 path unmodeled.

`prediction/mxfp4_supply.py` now executes the **weight-producer segment** using
the existing Engine. It is intentionally not registered as a complete native
grouped kernel. `config_from_group` accepts the ordinary frontend's explicit
FP8E4M3 × MXFP4 grouped SiTU tensor contract and integer `matrix_group_rows`.
Allocation capacity, expected routing moments, and captured `M=512` are never
substituted for those valid row counts.

## Source and supported domain

The source is `gmmsituquant-latest-0916/gmmsituquant-latest-overlay.zip`, SHA256
`fd5ded62862cabd7b686feceadcd8f39b692bdbb1989eb93c72eea29f4fc11da`, against base
revision `75f52c1d7df8df8548d854a01b64935727a70e88`. It is a retained independent
delivery overlay, **not an established identity for the historical capture
binary**. The report verifies the archive and individual source members.

The executable segment covers integer group counts below 34 rows per expert,
K and half-output widths divisible by 64, the interleaved Decode branch, and
one AIC with two physical AIV owners. It requires declared AIV MTE2, MTE3,
231,424 bytes UB per AIV and 131,072 bytes weight L1 per AIC. The larger-row
dual-role branch, its different AIV ownership, and M splitting remain outside
this program's domain. BF16 E16 and one-bank FP8 × FP8 calls are separate domains.

Source evidence:

- Host tiling lines 113–146: base M128, half-N128 where divisible, otherwise64;
  full AIC grid and device `group_list` declaration.
- `gmsq_vcv_controller.h` lines 128–171 and 214–333: scan actual counts twice,
  skip empty banks, assign output tiles with a rolling owner, select K256/512.
- `gmmsq_weight_quant_vec_compute.h` lines 245–399: two disjoint gate/up NZ
  reads, two AIV K halves, FP4 UB expansion, and MTE3 publication into L1.
- `gmmsq_quant_mx_vf.h` lines 340–383: five 256-lane bit operations plus a
  two-output interleave for every 512 FP8 values. The program counts 3.5
  elementary vector lanes per expanded value. Register loads/stores are not
  extra HBM traffic or invented arithmetic. This is a vector-work envelope,
  not an instruction issue/throughput model.

## Execution and conservation

An owner lazily traverses its source tiles and K panels. Each of the two AIVs
per AIC executes HBM/MTE2 read → vector expansion → MTE3 L1 publication.
At most two panels are active per owner. Source UB and weight L1 reservations
are fixed and released; empty experts create no weight read. Both source
group-list scans are retained as scalar-access diagnostics, with no invented
uncached HBM traffic or per-entry latency.

For K3584, output width3072, valid group rows `(1,1,0,…,0)` over28 experts:

| Resource work | Amount |
|---|---:|
| Packed FP4 HBM / AIV MTE2 read | 22,020,096 bytes each |
| Expanded FP8 MTE3 → L1 write | 44,040,192 bytes |
| Elementary vector lanes | 154,140,672 |

The declared A5 resource profile predicts 31.196 µs for this **supply-only**
control. The actual current-checkpoint ordinary frontend with explicit balanced
routing produces this integer workload. Changing allocation capacity from512
to2 leaves both service and duration unchanged; putting the two valid rows in
one expert halves weight traffic. Neither workload is inferred to be the
historical capture's unknown expert histogram.

The retained report
records the real frontend declaration, source hashes, late-input readiness,
zero live tasks/buffers, tails, empty experts and four isolated resource
sensitivity controls. Tests use the actual Engine and ordinary full catalog
frontend; no measured duration enters the program.

## What remains before full native composition

The supply-only segment retires its L1 slot after publication. It does not
yet hold that slot until Cube consumption. It excludes activation/scales
GM→L1, scale-aware FP8 MTE1/Cube, Fixpipe FP32→AIV UB, correction/rounding,
SiTU/MXQuant output, and cross-basic-block epilogue overlap. Reserving the source
buffers establishes a bounded segment; it does not prove full-kernel overlap.

The next composition must share L1 publication/consumption dependencies with
the existing matrix engine and replace the corresponding generic FP4 reads.
It must also replace generic intermediate HBM traffic with the source Fixpipe
relay, then use the SiTU output program. **Adding 31.196 µs to an existing
whole-kernel estimate would duplicate reads and ignore overlap.** The internal
segment deliberately receives no kernel invocation minimum; one complete
native call may receive one outer invocation model only after composition.

The historical task's ~74.25 µs and high AIC/AIV PMU activity justify focusing
on this missing physical path. They do not establish its expert counts or
authorize fitting an additional floor. Its measurement status remains
`reference_only` until source binary, quantization and real group counts align.
