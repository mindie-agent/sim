# Source scalar services in DSQ and MoE initialization

The existing source programs now consume optional owner-local services from
`device.compute_arch.instruction_services`. No hardware rates are installed by
this change. With no declarations, the original payload/register programs stay
active and the new inventories retain the unpriced counts explicitly.

For arch35 full-load `DequantSwigluQuant`, `scalar_gm_access` counts INT64
`GlobalTensor` scalar-index operations. The retained source visits every group
on every launched AIV, including zero groups and owners without data rows.
The total is `launched_AIV * group_count`. When priced, descriptor payload and
service occur before that owner's corresponding group processing; skipped
groups are combined into the next scan prefix and any trailing scan remains
after the owner's last computation. An idle owner executes the complete scan.
The existing allocated-capacity/valid-row split and group/tile UB dependencies
are unchanged. This does not count branch/address ISA or claim a cache miss for
each scalar access. Existing HBM payload is moved to those stages, never added
a second time.

For dropless full-range `MoeInitRoutingV3` COUNT_MODE, owner0 performs:

- `E` INT32 UB `GetValue` calls and `E` source scalar additions, in a serial
  load→add chain, using `scalar_ub_access` and `scalar_alu_issue`.
- One INT32 `SetValue` to publish the total-count cell, also counted as one
  `scalar_ub_access` operation.
- One source `PipeBarrier<PIPE_V>` after the existing count cast, using
  `pipe_barrier` when declared.

These services use the same physical owner and Engine as the surrounding
source stages. The accumulation is a lazy one-slot pipeline, so idle scalar
units cannot supply owner0, and a later count read cannot pass the preceding
addition. An effective source-operation interval is not an ISA latency claim.
The profile provenance must state its access/operation domain; a rate measured
for only UB loads does not automatically calibrate stores and mixed accesses.

Source anchors are `ops-nn`955f8b33985b6d72989ebff341bb9c2c564a51bc,
`dequant_swiglu_quant.h:244-320` (SHA-256
`3bf3fedad839383cee0e644dd3e07c0313e7abe9246092eb4ba3b1c3e997add0`), and
`ops-transformer`a3925fd49cd256ad478d2faf983bf2961b8ae2c8,
`moe_v3_expert_tokens_count_arch35.h:281-338` (SHA-256
`d0dfd8f977596357f37fddd1ea871789657b128e9c0d4bda03badbbd99ce7077`).

## Boundaries that remain unresolved

`Sort<float,true>` is one source library call with an input-block count. Its
retained caller does not prove exact `vbitsort32`/`vmrgsort4v2` expansion, and
Concat/Extract are separate source operations. They are not silently priced
as ordinary vector arithmetic or one ISA instruction per API call.

The two SIMT scans retain their logical INT32 access ledger. Cacheline128
transactions require warp coalescing and cache-policy information; the payload
count alone does not prove DCache misses or warp instruction issue. Inversion
also uses actual index addresses that are not recovered from a histogram.

Count publication uses `SetAtomicAdd<int32_t>()` plus one contiguous
`DataCopyPad` per active AIV (`expert_tokens_count_arch35.h:242-256`). It is an
atomic DMA write, not `E` SIMT atomic instructions. All owners target the same
count array, so a future128B transaction model must retain same-line hazards
and distinguish atomic issue, cache RMW and HBM traffic. No such rate or timing
is fabricated here.

## Verification

Seven new actual-Engine controls cover late7ms input, owner0 load/add ordering,
0.5/1/2 service scans, DSQ zero groups and idle owners, and unchanged payload,
register and quantization work. A synthetic E288 control declares1us per UB
access,2us per source addition and3us per barrier: the serial services consume
exactly868us, rather than dividing by64 idle AIVs or taking the maximum of load
and addition time. These are timing oracles, not measured hardware parameters.

With the existing DSQ, MoE-source and shared instruction-service tests, the
focused suite contains38 passing tests. Native accuracy calibration and the
broader Sort/SIMT/atomic mechanisms remain separate work.
