# MLA Prolog arch22 full-int8, BF16-KV source phases

This is an explicit opt-in source profile, not a default MLA Prolog prediction:

```yaml
runtime:
  kernels:
    lowerings:
      mla_prolog: vllm_ascend_mla_prolog_v3
      native_fusion:
        implementation: declared_native_fusion
        options:
          roles: [mla_prolog]
          traffic_profile: mla_prolog_arch22_fullquant_a3925fd4
```

`mla_prolog` remains the semantic frontend. The selected native-fusion role
replaces only the complete `mla_prolog` group with the source phase plan in
`mindie_sim/prediction/mla_prolog_arch22.py`. The public operator, graph and
cohort paths bind the same `PlannedRecipe`; the profile is not an export-only
annotation. Other roles and all unselected/default cases keep their existing
recipes and traffic behaviour.

## Selected source domain

Inspection source: `ops-transformer` revision
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8`.

The profile requires the complete V3 dtype/attribute branch:

| Axis | Required value |
| --- | --- |
| token / activation | `int8` |
| `weight_dq`, `weight_uq_qr`, `weight_dkv_kr` | `int8` |
| `weight_uk` | `bf16` |
| RMS gamma / rope sin-cos | `bf16` |
| KV and KR cache | `bf16` dense `PA_BSND` |
| query / query rope | `bf16` |
| query norm | `int8`; scale `fp32` |
| `weight_quant_mode` | `FULL_QUANT` |
| `kv_cache_quant_mode` | `NO_QUANT` |
| `splitM` | disabled |
| `enableDequantOpt` | true, selected by int8 UQ/QR and `N >= 8` |

The host checks constrain `Hcq ∈ {1536, 2048}`, `Hckv = 512`, `Dr = 64`,
`D ∈ {128, 192}`, `He` to the registered hidden sizes, `N ∈ [1, 128]`, and
AIV:AIC is 1:2 for this quantized branch. The host explicitly rejects every
quant mode in CV1:1 even though device templates contain CV1:1 code. This
profile requires AIV:AIC 1:2 and `N >= 8`; a
selected profile with any different dtype, cache, quant, split-M, group-compute
or dequant-option branch is an error. The profile does not assert that a
retained binary or historical tiling used this branch.

Evidence:

- `attention/mla_prolog/op_kernel/arch22/kernel_mla_prolog_split_n_arch22.h`
  sha256 `ea32ad7...2998`, lines `628-675` (workspaces), `716-775`
  (step/tail loop), `788-815` (AIC order), `828-890` (AIV barriers and
  consumers), `1073-1090` (QcQr), `1305-1400` (Cq dequant/RMS/quant),
  `1666-1775` (Ckv/Kr), `1983-2143` (split-N Qc dequant and Qr rope).
- `attention/mla_prolog/op_host/mla_prolog_tiling.cpp` sha256 `08a5a78...1bd2`,
  lines `393-495` (per-stage N/owner distribution), `496-527`
  (`stepBatchSize = min(128, T)`, dequant option), `655-702` (tiling key).
- `attention/mla_prolog_v3/op_host/mla_prolog_v3_def.cpp` sha256
  `66305b0...0011`, lines `19-164` (dtype columns including index 4) and
  `171-174` (attributes).
- `attention/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp` sha256
  `da68a7a...3684`, lines `103-112`
  (`FULL_QUANT_KV_NO_QUANT` → `MLAPType<int8_t, int8_t, bfloat16_t, float>`).
- Service files and hashes are retained in the plan metadata.

All hashes, line spans and the selected domain are copied into
`fusion_boundary.source_program` on the fused `OpSpec`. The metadata also keeps
`fusion_semantic_ids`, `fusion_semantic_recipes`, external outputs, source work
per token and the complete source work ledger. Physical owner-dependent work is bound
when `RecipeBook.prepare` receives the device geometry; the bound ledger and
prepared operator fields agree. The earlier declaration explicitly labels its
unbound payload rather than claiming a device-bound total.

## Source phase DAG

One outer invocation runs up to 128-token chunks (`stepBatchSize = min(128, T)`).
For a chunk the plan keeps the inspected cube/vector order:

```text
cq (AIC) ──> ckvkr (AIC, same cube stream)
  │              │
  ├─> rms_cq (AIV all) ──┬─> qcqr (AIC, after CkvKr + all RMS Cq)
  │                      └─> rms_ckv (AIV) -> rope_kr (AIV)
  qcqr + rope_kr ──> dequant_qc ──┬─> qn (AIC)
                                └─> rope_qr (AIV, overlaps qn)
```

- `cq`, `ckvkr`, `qcqr` are INT8 matrix products accumulating in INT32 GM.
- `qn` is a BF16 matrix product consuming the BF16 Qc dequant workspace and
  BF16 `weight_uk`; it writes BF16 query directly.
- `rms_cq` dequantizes Cq once, applies RMS and dynamic quant, writes int8
  query norm (external when the layer exports it) and its scale workspace.
  When the layer exports query norm, the external `fp32` scale is written too.
- `rms_ckv` and `rope_kr` are separate consumers of the Ckv/Kr INT32 scratch
  and write BF16 cache rows.
- `dequant_qc` and `rope_qr` are separate consumers of their Qc/Qr INT32
  slices; read their per-channel scales; their row scales are already local after
  the all-AIV aggregation. The two Qr AIVs each load their channel scales.
- Chunks are serialized through one pipeline slot. Within-chunk cube/vector
  overlap is preserved; inter-chunk overlap is not claimed.

The plan is a bounded lazy `PipelineProgram`: only the current chunk's stages
are issued. Metadata uses at most three counted chunk templates (initial,
regular, tail), rather than one record per chunk. Distinct source owners retain
unequal N/head/token tails instead of averaging their work over an owner set. `remaining live tasks/programs = 0` after execution.

## Work and traffic accounting

- Matrix work is exact `2*M*N*K` by compute precision: int8 for Cq, CkvKr and
  QcQr; bf16 for Qn. Logical FLOPs are the same sum; no native padded M/N/K
  work is introduced by this model.
- INT32 scratch is written once and read once by its actual dequant consumer.
  For a chunk of `n` tokens: Cq `4*n*Hcq`, CkvKr `4*n*(Hckv+Dr)`,
  QcQr `4*n*N*(D+Dr)`.
- BF16 Qc dequant scratch is `2*n*N*D` written and read by Qn.
- The Cq row-scale workspace is padded by the source to `32*n` bytes,
  written once and gathered in full by every AIV. Qc/Qr consume local scales
  afterward. The separately exported query-norm scale stays `4*n` bytes.
- The inspected caller supplies no smooth-scales input. Gamma and Cq/Ckv/Kr
  channel-scale vectors are copied once per invocation by every AIV. They are
  not reloaded once per chunk or charged to INT8 matrix products. Query RoPE
  sin/cos are gathered per active AIC pair; KV RoPE uses a separate token gather.
- Cq, CkvKr and QcQr output shapes are the source INT32 workspaces, not the
  semantic BF16 GEMM epilogue. The semantic `matrix_epilogue` work is therefore
  not counted again: dequant/normalization arithmetic appears once in
  `rms_cq`, `rms_ckv`/`rope_kr`, `dequant_qc` and `rope_qr`.
- External query norm, query norm scale, query rope, BF16 query, BF16 KV cache
  and BF16 KR cache writes remain real payload traffic. `weight_uk`, query and
  cache are never silently converted to int8.
- All workspace traffic is charged at the declared default GM/HBM rate. No L2
  hit, residency, or cache-miss rate is inferred. Logical operand payload is
  charged once per product; internal L1 tile reloads and fixpipe issue latency
  are not instruction-reconstructed.
- UB/L1/L0 buffers are not reserved because the loaded per-phase footprint is
  not established here. The metadata states this limitation instead of
  inventing whole-chip occupancy.

## Ownership

Owners are derived from the host tiling, not from a shrunk core count:

- Cq uses `max(align32(ceil(Hcq/(32*AIC))*32), 64)` per active AIC.
- CkvKr uses fixed `N=64` when `AIC >= 9`; otherwise the aligned host fallback.
- QcQr aligns whole `(D+Dr)` slabs across AIC (`enableDequantOpt`).
- Qn assigns `ceil(N/AIC)` heads per active AIC.
- All AIV token phases use `min(128, AIV)` front/back token split.
- Qc dequant uses both AIVs in each active pair to split columns. Qr splits
  tokens; a one-token tail gives work only to odd AIVs.

For the normal GLM-shaped group (`T=4096`, `He=6144`, `Hcq=2048`, `Hckv=512`,
`N=8`, `D=192`, `Dr=64`, AIC=20/AIV=40): Cq uses 16 AIC, CkvKr 9 AIC, QcQr
8 AIC, Qn 8 AIC, all 40 AIV for token phases, and 16 AIV for Qc/Qn consumers.
`ceil(4096/128) = 32` chunks. These are source-owner facts for the declared
geometry; they are not a fit to any retained duration.

## Fidelity limits

This is a phase/resource model, not an instruction simulator. It does not
reconstruct scalar synchronization, L1/L0 tile residency, fixpipe latency,
instruction issue, cache replacement, or compiler scheduling. Per-head QcQr/dequant streaming is conservatively
coarsened to chunk phase barriers; it is not claimed to be a complete native
pipeline reconstruction. It does not
replace measured totals or use timing residuals to pick a branch. It does not
model split-M, group-compute, quantized KV, MXFP8/FP8/HIF8, or other V3
dtype/cache branches; those require their own source plans.

## Independent acceptance corrections

The reviewer corrected CV1:1 rejection, constant-sized chunk metadata, real
32-byte row-scale workspace and its all-AIV gathering, one-time parameter
loads, missing Ckv/Kr-to-Qc vector dependencies, and unequal physical owner
loads. A true Engine test checks dependency times and one-token odd-AIV Qr
ownership. A ten-million-chunk arithmetic check keeps metadata below 30 KiB
without executing or materializing the chunk history.

## Retained capture comparison

The replay in `data/verification/mla-prolog-arch22-wave4-20261029/` uses the
real public structure export, profiler streaming mapper and `compare_cohorts`.
The selected case changes only the explicit MLA source traffic profile. Both
native contracts retain identical capture hashes, descriptors and observations.
The semantic reference predictions keep their earlier report provenance.

| Native output contract | Calls | Observed us/call | Prior semantic us/call | Source-phase us/call |
| --- | ---: | ---: | ---: | ---: |
| Exported INT8 query norm | 21 | 1342.506476 | 491.634484 | 836.800853 |
| Query norm not exported | 57 | 1248.166263 | 491.634484 | 836.787200 |

The 78-call observed activity totals 99.338113 ms; the new predicted activity
is 65.269688 ms, a -34.295% residual. This improves the semantic baseline but
still fails the relative 15% criterion. It does not establish full-model
accuracy or identify the captured binary's tiling. Remaining source pipeline,
copy/residency and scalar costs stay explicit limitations; no residual floor
or multiplier is inserted.

Runtime revision `4d9cb391` and the exact package hash are retained before and
after the 2.012-second replay. The explicit devices in this case do not carry
A3 invocation minima; the source profile does not silently add them. Replay:

```sh
python -m mindie_sim.cohort_comparison \
  data/verification/mla-prolog-arch22-wave4-20261029/a3_glm53_prefill-arch22.case.json \
  data/verification/mla-prolog-arch22-wave4-20261029/a3_glm53_prefill-arch22.mapping.json \
  -o /tmp/glm53-mla-arch22-comparison.json
```
