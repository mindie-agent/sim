# GLM routed/shared output reduction: source and capture reconciliation

The ten TP8/DP1/EP1 GLM-Flash cases now explicitly select `moe_recipe="tp_join_reduce"`.
This removes the extra expert-row TP reduction and moves the shared-path TP
sum after weighted routed unpermutation and shared+routed addition. The
previous 132-call evidence is preserved; this new arm has exactly 90 calls:
45 attention and 45 FFN reductions.

## Why this is the source path

Pinned Ascend revision `8c30e4f491f73cba737a1dacc522894e437cdf3d`:

| Evidence | Physical implication |
|---|---|
| `models/glm5next/model.py:104–124,197–205` | Shared up is column-parallel; down is row-parallel with `reduce_results=False`. |
| `ascend_forward_context.py:391–392` | Disabled EP selects the local ALLGATHER MoE strategy. This enum does not imply a DP1 network gather. |
| `ops/fused_moe/prepare_finalize.py:533–550` | With DP/PCP1, finalization does not reduce the routed output. |
| `ops/fused_moe/token_dispatcher.py:429–439` | Weighted unpermutation returns token rows before the caller joins outputs. |
| `ops/fused_moe/fused_moe.py:199–237,269–349` | Unreduced shared/routed outputs are added, then the final result is reduced. No latent output transform exists in these inputs. |
| `ops/register_custom_ops.py:127–155` | ALLGATHER without SP skips separate shared reduction and executes final TP reduction. |
| `ascend_config.py:597–604` | Shared-expert DP requires EP; these cases cannot silently use replicated shared weights. |

Upstream vLLM revision `cdc4824a21eaa986d4d1fee90a7e6465c9f706e6`
`fused_moe/config.py:1208–1234,1339–1344` keeps routed TP8 when EP is off
and partitions the intermediate dimension by TP. `routed_experts.py:485–509,
535–549` slices up weights on output and down weights on input. Ascend MXFP8
`quantization/methods/w8a8/w8a8_mxfp8.py:267–288` allocates those actual
partitioned weight/scale shapes.

## Actual retained capture agrees with the boundary

Both WS9 and the c1/c3/c96 serving logs declare TP8, resolve DP1, omit the EP
flag, and explicitly report FlashComm1 disabled at lines44/64. The only
additional configuration is `enable_force_eplb=true`. The c1 CSV provides
independent tensor/operation evidence:

| CSV line | Operation | Shape boundary |
|---:|---|---|
| 301 | MoeInitRoutingV3 | `[4,4096]`, 8 routes → `[32,4096]`, all288 expert groups |
| 304 | GroupedMatmulSwigluQuantV2 | `[32,4096]` → `[32,256]`; local256=`2048/TP8` |
| 305 | GroupedMatmul | `[32,256]` → `[32,4096]` |
| 307 | MoeFinalizeRoutingV2 | weighted32 route rows →4 token rows |
| 309/320 | shared up/down | `[4,4096]` →512 packed outputs →256 activation →`[4,4096]` |
| 321 | ForeachAddList | shared and routed `[4,4096]` →`[4,4096]` |
| 322 | hcom_allReduce_AicpuKernel | one reduction after the join |
| 324 | HcPost | consumes the joined/reduced output |

The nearby HcclLaunchAicpuKernel records are launch/control wrappers, not extra
device AllReduce calls. The 71 MB CSV was streamed; only the first330 rows
were inspected for this caller example. Source and capture hashes, precise
line ranges and selected fields are retained in the two audit JSON files.
The pinned source is not asserted byte-identical to the captured release.

## Validation and full-body outcomes

All ten ordinary target graphs retain 42 MoEInit and 42 fused gate-up calls,
with 42 joined reductions depending on both weighted unpermute and shared
down output. Matrix/vector/noncommunication HBM work is identical. HCCL
payload is `B × q × 4096 × 2` bytes, never expanded `top_k` rows.
All ten complete45-layer predictions completed, max22.752s and124.141MiB,
with no live tasks/programs. Source copy/reduce heuristic rates are unchanged
from the preceding controlled arm; no timing parameter was fitted.

| Case | Observed ms | Previous132-call ms | Corrected90-call ms | Error |
|---|---:|---:|---:|---:|
| `a5_glm53flash_tp8_c16_p4k` | 33.031 | 21.979 | 20.135 | -39.04% |
| `a5_glm53flash_tp8_c16_p8k` | 33.616 | 22.167 | 20.322 | -39.55% |
| `a5_glm53flash_tp8_c32_p4k` | 37.669 | 27.743 | 25.111 | -33.34% |
| `a5_glm53flash_tp8_c32_p8k` | 38.713 | 28.122 | 25.491 | -34.16% |
| `a5_glm53flash_tp8_c4_p8k` | 21.418 | 11.936 | 10.681 | -50.13% |
| `a5_glm53flash_tp8_c64_p4k` | 46.528 | 38.723 | 34.519 | -25.81% |
| `a5_glm53flash_tp8_c8_p4k` | 28.833 | 18.071 | 16.619 | -42.36% |
| `glm53flash-a5-tp8-c1` | 15.683 | 7.937 | 6.830 | -56.45% |
| `glm53flash-a5-tp8-c3` | 21.068 | 13.058 | 11.803 | -43.98% |
| `glm53flash-a5-tp8-c96` | 57.590 | 65.145 | 59.368 | +3.09% |

c96 improves from +13.12% to **+3.09%**. Low/mid-concurrency underestimates
become larger after removing the incorrect communication work. This is a
necessary structural correction, not a successful whole-campaign accuracy
claim: **1/10** remains within15%. HCCL control service and other genuine
operator residuals remain visible. The report separates actual body spans
from client TPOT and does not turn missing calibration into a simulation gate.

Artifacts:

- `glm-native-moe-reduction-audit-20261030.json`: pinned source hashes and ten-case structural/work checks.
- `glm-native-moe-capture-contract-20261030.json`: startup and bounded CSV evidence.
- `glm-hccl-join-reduce-20261030.json`: ten complete predictions, matching retained body populations.
- Cases: `examples/cases/high-concurrency-20261030/glm-hccl-join-reduce-conditional/`.
