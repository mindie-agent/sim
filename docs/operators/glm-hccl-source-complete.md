# GLM-Flash complete bodies with explicit HCCL and caller programs

These ten conditional A5 inputs compose the existing accepted-state KDA and
MXFP8 fused gate-up paths with three additional ordinary frontend bindings:

- TP AllReduce: `AICPU_TS / mesh_oneshot`, with separate input/scratch/output.
- KDA convolution caller: six-slot state gather and scatter.
- EP1 local MoE routing: `ascend_moe_init_routing_v3_mxfp8`.

Every original case remains TP8 / DP1 / EP1 with 45 layers and its original
query/context geometry. In particular, the captured `c3` case retains its
four-sequence graph geometry. No activation/weight precision, routing policy,
accepted-state assumption, pipeline bandwidth or invocation floor is fitted.

Cases are in
`examples/cases/high-concurrency-20261030/glm-hccl-source-complete-conditional/`.
The adjacent `hbm-only/` controls retain the same source programs without the
optional HCCL copy/reduce service declarations. Both are executable inputs,
not a delta applied to a previously measured duration.

## What the composition proves

The producer first builds a single actual MoE/KDA layer through `target_graphs`,
then checks every complete body. Each full body contains 90 source TP calls,
68 convolution state copies, 42 source MoE initialization calls, and 42 fused
MXFP8 gate-up calls. The full prediction uses `predict(scope="body")`, with
host submission, sampling and inter-step waits outside the measured boundary.

The 90 HCCL calls each have one original entry/exit interval. Their internal
local and wire phases are **not** added again to that call duration. A separate
42-call `expert_tensor_reduce` path on axis `etp` remains unchanged. Its work is
on expanded token-expert rows. Whether a captured FFN reduces those rows or
combines first is a distinct source/capture contract; this experiment does not
silently collapse or double-attribute that path to the 90 TP calls.

The observed reference is the profiler's retained device-body population in
`acceptance-20261030-runtime5ec8843a/decode-forward-glm-replay.json`, matched by
source case ID, query tokens, sequences and context. No client TPOT is used.
Each comparison remains conditional on its retained single-rank topology and
accepted-state assumptions. A captured AICPU operator name identifies an
executor class, not the exact OneShot template, current open-source revision
or private runtime implementation.

## HCCL source-selector service hypothesis

The pinned HCCL source is revision
`c2d6790464654ccf58aca39448de4b65fdc7b184`. The main case arm declares:

```json
{
  "rates": {"hccl_copy": 1400000000000, "hccl_reduce": 900000000000},
  "resource_units": {"hccl_copy": 1, "hccl_reduce": 1},
  "work_units": {"hccl_copy": "bytes", "hccl_reduce": "bytes"}
}
```

These values come from `src/ops/op_common/selector/cost_model.cc:117–118` and
`:561–583`. They are **source-selector heuristics, not measured A3/A5 rates**.
Their units match copied/reduced payload bytes. A copy additionally consumes
`2N` HBM endpoint bytes; a reduction consumes `3N` HBM read/modify/write bytes.
The Engine constrains both resources; it does not add two durations for the
same service. The controls omit only these two optional services, retaining
identical HBM, link traffic and dependencies.

The source's control coefficients do not identify a `hccl_control` rate:

| Source term | Unit and use | Treatment here |
|---|---|---|
| `CalcLatencyParams`, `cost_model.cc:587–598` | AICPU `2 us × kernelNum` in the selector | Recorded source heuristic; not charged per notification |
| `CalcLaunchParams`, `:612–622` | AICPU `0.5 us × taskNum` | Recorded source heuristic; not an independently measured dispatch service |
| OneShot `CalcCostCoeff`, `ins_temp_all_reduce_mesh_1D_one_shot.cc:23–55` | 15 abstract kernels; `10(R−1)` tasks | Distinct from the actual pre/post notify and wait counts |
| TwoShot `CalcCostCoeff`, `ins_temp_all_reduce_mesh_1D_two_shot.cc:22–50` | 20 kernels; `16(R−1)+10` tasks | Distinct from the actual RS/AG phase DAG |
| `cost_table.cc:402–410` | `max(composed segment cost, total task term)` | Never blindly sum both selector terms |

`hccl_control` remains explicitly unpriced in both arms. Consequently these
cases do not claim complete HCCL latency accuracy. The missing independent
control-service parameter is visible without preventing analytic simulation.
No remaining body residual is used to derive it.

## Reproduction and evidence

```sh
python data/verification/glm-hccl-composed-20261030.producer.py \
  --output data/verification/glm-hccl-composed-20261030.json
```

The producer runs serial, isolated complete-body children with a 60-second
simulation budget and 10-second receipt/cleanup allowance. It retains source
and package hashes, peak RSS, live-task/resource state, whole-call HCCL spans,
family activity and the unchanged observation identity. Scheduled family
activity can overlap and is not summed into whole-body latency. The runtime
baseline is `66e4c970a277f62d5d61fa8443e9879905801ad4`.

## Full 45-layer results

All 20 predictions completed. Peak child RSS was 125.391 MiB; the longest
child took 29.230 seconds. No live tasks or programs remained. The two arms
have equal physical HBM/link/matrix/vector work; only the optional HCCL
copy/reduce service demand differs.

| Case | Observed mean ms | HBM-only ms | Source service ms | Source error |
|---|---:|---:|---:|---:|
| `a5_glm53flash_tp8_c16_p4k` | 33.031 | 21.850 | 21.979 | -33.46% |
| `a5_glm53flash_tp8_c16_p8k` | 33.616 | 22.037 | 22.167 | -34.06% |
| `a5_glm53flash_tp8_c32_p4k` | 37.669 | 27.484 | 27.743 | -26.35% |
| `a5_glm53flash_tp8_c32_p8k` | 38.713 | 27.863 | 28.122 | -27.36% |
| `a5_glm53flash_tp8_c4_p8k` | 21.418 | 11.904 | 11.936 | -44.27% |
| `a5_glm53flash_tp8_c64_p4k` | 46.528 | 38.206 | 38.723 | -16.77% |
| `a5_glm53flash_tp8_c8_p4k` | 28.833 | 18.006 | 18.071 | -37.33% |
| `glm53flash-a5-tp8-c1` | 15.683 | 7.929 | 7.937 | -49.39% |
| `glm53flash-a5-tp8-c3` | 21.068 | 13.025 | 13.058 | -38.02% |
| `glm53flash-a5-tp8-c96` | 57.590 | 64.369 | 65.145 | +13.12% |

Only c96 is within 15% in each arm: **1/10**, not a completed accuracy campaign.
The source copy/reduce declarations add 0.008–0.776 ms/body. They do not explain
the remaining 7.7–11.4 ms low/mid-concurrency residual. The c96 overestimate
and the lower-batch underestimates are retained rather than flattened with a
constant floor. The source-local work and exact peer transport are now bound;
independent control-service parameters and the remaining operator/caller
contracts still require convergence.

The prior replay used a different runtime revision. Its prediction is retained
as historical context, not a one-factor experiment. Only the two current arms
form a controlled service-rate comparison.
