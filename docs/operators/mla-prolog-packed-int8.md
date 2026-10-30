# Packed INT8 MLA Prolog native boundary

The new explicit branch supports INT8 token/projection operands, BF16 absorbed UK weights, BF16 query/query-RoPE outputs and a per-tile packed INT8 KV cache. It is separate from `mla_prolog_arch22_fullquant_a3925fd4`, which still requires BF16 cache and its own source domain.

```json
{
  "mla_prolog": {
    "implementation": "vllm_ascend_mla_prolog_v3",
    "options": {
      "cache_layout": "packed_int8_bf16_rope_fp32_scale",
      "cache_group_size": 128
    }
  },
  "native_fusion": {
    "implementation": "declared_native_fusion",
    "options": {
      "roles": ["mla_prolog"],
      "traffic_profile": "mla_prolog_packed_int8_tile128_analytic"
    }
  }
}
```

These are entries under `runtime.kernels.lowerings`; the case must also declare `quant.kv_dtype="int8"` and activation quantization. The currently supported packed geometry is latent512, RoPE64 and four FP32 scales for groups of128. No model name selects it.

## Source and representation

Pinned vllm-ascend `8c30e4f491f73cba737a1dacc522894e437cdf3d`, `vllm_ascend/attention/sfa_v1.py:1181–1289`, selects `kv_cache_quant_mode=3`, `ckvkr_repo_mode=1`, `quant_scale_repo_mode=1` and a128-element tile. Its INT8 full-quant branch passes one FP32 token scale per row and per-channel FP32 projection scales. UK remains BF16. `query_norm_flag` controls the optional INT8 normalized-query output and its FP32 row scale.

`attention/utils.py:69–80` specifies the physical row:512INT8 latent bytes at offset0,64BF16 RoPE values at offset512, then fourFP32 scales at offset640. The total is656bytes. It is not a656-element quantized latent vector and not two same-dtype cache tensors.

The independently retained ops-transformer source revision `a3925fd49cd256ad478d2faf983bf2961b8ae2c8`, `attention/mla_prolog/op_kernel/arch22/kernel_mla_prolog_split_n_arch22.h:1454–1556,1628–1645`, confirms the three writes: quantized latent, reinterpreted BF16 RoPE, and FP32 per-tile scales. This establishes representation and write boundaries, not capture-binary or dispatch identity.

## Execution and exported contract

The selected ordinary frontend includes a real packed-cache producer: BF16 semantic norm input, explicit widening/abs/max/clip/normalization/conversion work, INT64 slot reads, clip-alpha read, and656bytes written per valid row. The quantization operation count is analytic; it is not an instruction-accurate issue schedule. The cache capacity is excluded from per-call traffic.

Native fusion retains the existing three INT8 projections, BF16 UK multiplication, query normalization/quantization, RoPE and their workspaces. It groups them into one invocation with their real dependencies. No old arch22 BF16-cache schedule or arch35 local-reuse claim is silently applied. The complete group can receive one explicitly matched observable native-call response; this change assigns none.

The tensor view records token/projection scale shapes, optional query-norm scale shape, empty separate RoPE-cache output and empty query-nope scale. The active canonical cache contract is `[(1,656), int8, ND]` plus an absent separate RoPE cache. Cache allocation/page capacity and slot values are outside that active contract.

## Retained native comparison

`data/verification/mla-prolog-packed-int8-20261030.producer.py` builds each original78-layer GLM case through the public frontend. Each produces57ordinary Prolog calls and21calls with the normalized-query/scale outputs, matching the retained native ABI partitions. Only two native-group programs per case are priced; no full-body residual is used.

| Input rows | Capture | Predicted µs/call | Observed P50 µs, ordinary / query-norm | Signed relative gap |
|---:|---|---:|---:|---:|
|12|tpot30_c2|41.0004|40.601 / 40.441|+0.98% / +1.38%|
|12|tpot50_c2|41.0004|40.601 / 40.501|+0.98% / +1.23%|
|48|tpot88_c8|51.2631|47.901 / 52.661|+7.02% / −2.65%|

All six comparisons use unmodified hardware rates and no timing fit. Historical observations retain their `unassigned` role. This is conditional native-operator agreement, not held-out whole-model accuracy.

Verification:29tests passed,4existing optional profiler-fixture tests skipped,6.90s. Controls cover the78-call/57+21 boundary, exact656-byte writes, FP32 scale outputs, old BF16-cache profile rejection, one inclusive service per complete native call, late input readiness, released tasks, and0.5/1/2HBM scaling with preserved work.
