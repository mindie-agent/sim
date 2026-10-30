# Packed projections and native precision

Projection geometry is a layer contract. `packed_input_projection` declares
one shared-input GEMM with strided output slices. `gate_rank > 0` declares a
replicated hidden-to-rank projection and a rank-to-local-value-width projection;
zero preserves the direct gate. `projection_dtype` supplies the layer's native
precision default. Explicit parent and child quantization roles take precedence.
Execution and resident weight inventory resolve the same representation.

Packing preserves every weight and FLOP. Six compatible input projections read
the hidden tensor once and write the same six output slices once. A mixed
explicit operand representation keeps separate GEMMs and records
`separate_explicit_operand_formats`. The declaration adds no measured duration,
padding multiplier, or implicit contiguous copy. An explicitly selected MLA
source caller retains its own boundaries.

## Retained source and observed tensor contracts

The source mirror is vllm-ascend revision
`8c30e4f491f73cba737a1dacc522894e437cdf3d`. This is an inspectable implementation,
not proof that the captured binary has the same revision.

- `vllm_ascend/models/glm5next/kda.py` lines 136–142 disable projection
  quantization. Lines 164–182 merge Q/K/V/beta/f_a/g_a; f_a and g_a are replicated
  low-rank slices. Lines 233–242 construct g_b and the output projection; lines
  279–300 split views and execute f_b/g_b.
- `vllm_ascend/models/glm5next/model.py` line 318 supplies native MLA projections;
  `attention.py` lines 181–186 merge q_a and kv_a. Later loader code separately
  describes checkpoint conversion; runtime BF16 does not prove every source
  weight file was stored as BF16.
- The source mirror fuses BF16 index K and weighting projections. The delivered
  A3 capture instead contains a separate FP32 weighting GEMM. That discrepancy
  remains explicit: only the observed case declares `indexer.weights=fp32`.

The independent profiler receipt is
`profiler/examples/structure/acceptance-20261030-priority-operators/glm53flash-a3-b4-native-evidence.json`.
It identifies one q6/B4 forward in
`a3/glm5_3_flash_w8a8_high_conc_decode_livepre_tpot30_c4_20261030`, TP16, with
24 physical rows. The kernel CSV SHA256 is
`35c1ff6c9570a472ca26f1bfa293cd09709a1e4d0f2984b039de9e48d961c1e2`.

| Physical matrix (M,N,K) | Observed operands/output | New declaration |
|---|---|---|
| KDA (24,1796,4096), CSV line 11230 | BF16 | Q/K/V each512, beta4, f_a128, g_a128 |
| KDA (24,512,128), lines 11232 and 11234 | BF16 | separate f_b and g_b |
| KDA output (24,4096,512), line 11241 | BF16 | native output projection |
| MLA (24,2048,4096), line 11311 | BF16 | merged q_a1536 + kv_a512 |
| index weighting (24,32,4096), line 11323 | FP32 | explicit case role plus BF16→FP32 input cast |

The example `examples/cases/glm53flash-a3-projection-contract-q6-b4.json` retains
the complete 45-layer catalog and existing INT8 KV declaration. It does not
infer context from cache allocation or rewrite the historical input. The input
still records unresolved source/binary and KV scale contracts.

## Verification and integration boundary

`tests/test_projection_contracts.py` checks the actual 24×1796 geometry,
replicated low-rank capacity under TP16, packed/unpacked work conservation,
dependency readiness in the ordinary Engine, distinct key/value dimensions,
mixed parent/child precision, ordinary MLA frontend output, FP32 casts and
explicit MLA source precedence. The KDA helper returns raw beta; sigmoid,
gating and recurrent caller boundaries belong to the recurrent integration.

This commit provides the KDA helper and matching inventory. The main-thread
integration must replace its existing KDA projection construction with
`kda_projections`; the isolated branch does not modify the concurrently owned
`recurrent.py`. No whole-forward accuracy improvement is claimed from these
tensor-contract tests.

## Materialized KDA consumers

The ordinary recurrent frontend now preserves explicit contiguous consumer
contracts through `runtime.tensor_layouts`: `short_convolution.input` and
`kda_recurrent.beta`. Setting either to `contiguous` materializes only that
slice of a packed projection, with its source-derived offset and stride. The
copy is a real dependency of convolution or beta processing and has its own
device invocation. Without a contiguous requirement the same tensor remains a
view. Separate projections acquire no fictitious Slice kernel. The native q1
convolution adapter already owns its AutoContiguous call, so it is not copied
again here.

This covers the Q/K/V and beta consumer boundaries in the retained
`glm5next/kda.py:275–310` and `ops/kda.py` caller. It does not assign other
observed Slice events solely by matching a shape. Tests verify selected bytes,
offsets, consumer dependencies and delayed input readiness on the real Engine.
