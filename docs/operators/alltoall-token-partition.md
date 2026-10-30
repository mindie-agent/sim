# ALLTOALL caller token ownership

The explicit `vllm_ascend_alltoall_tp_slice` caller keeps the router projection
and shared-expert matrices at full input rows. Hidden states and router logits
then become zero-copy TP views. Selection, packing, quantization and source
transport use those local rows. Received expert work retains its existing
global routing ledger; the caller does not divide expert FLOPs or traffic twice.

```json
{
  "moe_token_partition": {
    "kind": "vllm_ascend_alltoall_tp_slice",
    "layer_scopes": ["body"],
    "gather_output_materialization": "per_rank_copy"
  },
  "moe_transport": {
    "kind": "separate_quantized_payload",
    "payload_dtype": "int8",
    "scale_dtype": "fp32",
    "scale_granularity": "row",
    "histogram_dtype": "int32",
    "layer_scopes": ["body"]
  }
}
```

These are `runtime.kernels` declarations for a known ALLTOALL path. They do not
select a branch from model name, token threshold or timing. The current caller
domain is DP1, EP=TP>1, ETP=CP=PCP1, direct hidden rows, no sequence parallelism,
and a positive TP-divisible row count. `layer_scopes` separates body and draft
callers. The MC2 caller retains its separate name and padding contract.

`histogram_dtype` declares the stored count tensor independently of INT32 TopK
indices and INT64 count reductions. It accepts INT32 or INT64. The count
allgather transfers `(EP-1) * experts * sizeof(count)` bytes per rank, without
another TP division. Three all-to-all calls exchange FP32 row scales, INT8
activation rows and BF16 return rows. A final TP allgather reconstructs the
caller output after unpermutation. Thus the selected graph contains two
allgathers and three all-to-all calls per MoE layer.

`gather_output_materialization="per_rank_copy"` adds one typed TensorMove per
gathered rank slice; `"view"` retains the source's shared output allocation.
This choice is explicit because native materialization depends on the caller
and backend. It does not follow a batch-size threshold.

The independent source is `vllm-ascend` revision
`175ef4d49c2072f9233619636f0309f4f88b4e2c`,
`vllm_ascend/ops/fused_moe/prepare_finalize.py`,
`PrepareAndFinalizeWithAll2All` (blob SHA256
`88560c839172f5b433753c796d17a5d41278204f33b2e9ca9d4b2af9ccbe16d8`).
Its TP views and final gather are separate from
`TokenDispatcherWithAll2AllV` in `token_dispatcher.py` (blob SHA256
`6126304014959e8c52d87284ccac078f7622a6d72c7a7fc8be25929dfe644cfb`).
That dispatcher gathers `torch.histc` results, waits for scale dispatch before
payload dispatch, and transposes split counts for the return. This source
identity is not a claim about the binary used by a historical capture.

The retained A3 `2026-10-23/{v4f-prefill,v4pro-prefill}/raw/`
`pre4096.kernel_details.csv` files provide the paired tensor boundaries below.
Line numbers are physical CSV lines, including the header.

| Contract | V4F, device 8 | V4Pro, device 0 |
| --- | --- | --- |
| Full router rows | 4096, line 150 | 1024, line 152 |
| Local gating rows | 512, line 157 | 64, line 159 |
| Histogram result | INT32[256], line 159 | INT32[384], line 161 |
| Count allgather envelope | line 163 | line 165 |
| Following count tensor | INT32[8,256], line 164 | INT32[16,384], line 166 |
| Outgoing permuted rows | 3072, line 174 | 384, line 176 |
| Incoming routing rows | 3366, line 180 | 410, line 182 |
| All-to-all envelopes | lines 177, 179, 185 | lines 179, 181, 186 |
| Final TP allgather | line 189 | line 190 |
| Output TensorMoves | 8 × BF16[512,4096], lines 191–198 | 16 × BF16[64,7168], lines 194–209 |

The raw SHA256 values are
`b56b7439213cc9f76eaae63347393d4add6ae5c985bf28a6a7737f5783c81788`
and `5ffe5625942e4caf787ce047438576527b8513452d6dd6871b24682a19968abd`
respectively. HCOM envelopes and their `AivKernel` children are aliases of the
same calls and are not double counted. HCOM rows do not retain a tensor ABI, so
their exact scale-versus-payload identity follows the independent source order,
not an invented raw tensor descriptor.

Observed incoming rows differ from uniform routing expectations. This change
does not replace those expectations with fabricated histograms or inferred
per-peer matrices. Local received-row sorting, count-to-INT64 reductions and
host split coordination remain separate boundary/refinement questions. The
controls establish row, dtype, byte, call-count and dependency correctness;
they do not establish a whole-Prefill timing pass.
