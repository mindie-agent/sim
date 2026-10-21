# Large-operator calibration and whole-network validation (2026-10-21, round 3)

This round targets the operators that dominate whole-network time — **MC2**
(MoE dispatch/combine + grouped matmul), **fused attention** (GQA and MLA,
prefill and decode) and **grouped matmul (GMM)** — and closes with the first
**measured end-to-end serving baseline** on A3. Small-operator gaps are
deliberately out of scope.

All operator fits are train-only (5-fold CV, heldout guard clean). The
end-to-end numbers are measured on real hardware; model-vs-measured end-to-end
figures are labelled predictions, never accuracy claims.

## 1. What was measured

| campaign | machine | cases / samples | notes |
| --- | --- | ---: | --- |
| fused attention + GMM | A3 `192.168.13.165` (910_9362, one die) | 24 + 27 cases / 3,060 samples | GQA prefill/decode, absorbed MLA (qk 576, v 512), GMM 32–384 experts |
| fused attention + GMM | A2 `173.125.1.2` (910B4, one die) | 50 cases / 4,500 samples | GQA + closest-GQA MLA, GMM 32–128 experts |
| MC2 dispatch/combine + GMM chain | both | 686 rows / 16 timed runs | ep 2/4/8, experts 64/128, 512–4096 global tokens |
| end-to-end serving | A3 one 910C die | 16 scenarios / 424 requests | Qwen2.5-7B, TP1, bf16, vLLM-Ascend, eager + ACLGraph |

Protocol: correctness canary per case (CPU reference; every GMM expert sampled),
device-event timing, 2–3 trials × 20–30 samples, clock attestation, frozen
runtime tree hashed, one HostQueue job at a time, leases released. All
campaigns checked against the declared heldout sets: **0 collisions**.

### Contract findings that shaped the model

* **Absorbed MLA is expressible** on A3 (`npu_fusion_attention`, qk 576 = 512 +
  64 rope, v 512, one shared KV head). On A2 the installed kernel requires
  `qk_dim == v_dim`, so the measured "MLA" is a closest-GQA proxy with v padded
  to 576 — recorded as a separate contract, never reused for A3.
* **GMM caps tensorList at 128 experts** on both machines (`EZ1001` /
  `aclnnGroupedMatmulV5`). 256/384-expert layers cost `ceil(E/128)` launches;
  A2's 256-expert case is recorded `not_run`.
* **A2 `npu_fusion_attention` silently ignores kernel-generated causal masks**:
  `sparse_mode` 2/3 and `sparse_mode=0 + mask=None` dispatch and return the
  unmasked result. Only an explicit bool mask + `sparse_mode=0` is correct. The
  A2 worker validates every contract against the CPU reference before timing.
* **MC2 dispatch/combine are callable standalone** (`npu_moe_distribute_dispatch_v2`
  / `combine_v2` with the HCCL world group as EP group; round trip exactly 0.0).
  Limits: `bs/rank ≤ 512` (A3) / `≤ 256` (A2), hidden ∈ [1024, 8192], and a
  `HCCL_BUFFSIZE` window formula that fails at the 200 MB default even for 256
  tokens. One dispatch is limited to 1024 global tokens at ep=2, so larger
  sizes are measured as repeated independent calls (per-call + amortised).
* The fused `npu_grouped_matmul_swiglu_quant` path and vLLM's single fused MC2
  kernel are **not expressible** on this build (FRACTAL_NZ int8 weight
  conversion fails). They are reported unmodeled, not faked.

### Throughput anchors

* A3 attention prefill: 110–118 TFLOPS for GQA and absorbed MLA; MHA-shaped MLA
  71–88 TFLOPS. A2 prefill: 71–103 TFLOPS.
* GMM: A3 249 TFLOPS (≈101 % of declared peak) at 4096 tokens/expert; A2
  180–228 TFLOPS at 32k–262k routed tokens, latency-limited below ~8k.
* Dispatch overhead: A3 16–21 µs per call (fresh/hot), 10.5–11.9 µs queued;
  the previous 24.5 µs figure is ~20 % high but safe. A2 per-call dispatch is
  ~60 µs, which pins small-shape A2 fits at the intercept bound.
* MC2 has **no compute/comm overlap**: the chain costs 1.0–1.2× the sum of its
  measured legs, never `max()`. A2 dispatch/combine is 3.7×/5.0× slower than A3.

## 2. Fitted operator accuracy (train-only CV)

| artifact | family | scope | n | CV WAPE |
| --- | --- | --- | ---: | ---: |
| A3 | attention_fused | device-event | 66 | **4.19 %** |
| A3 | attention_fused | block-interval | 66 | 4.98 % |
| A3 | moe_gate_up | per-call | 15 | 10.98 % |
| A3 | moe_down | per-call | 12 | 2.36 % |
| A3 | grouped_matmul (experts=8) | per-call | 10 | 1.65 % |
| A3 | gemm (per-call / batched) | — | 367 / 107 | **8.84 % / 11.50 %** |
| A3 | moe_gate_up / moe_down (MC2 legs, own recipe) | per-call | 14 / 14 | **3.71 / 4.18 %** |
| A2 | attention_fused | per-call | 28 | 13.61 % |
| A2 | attention_fused | operator/batched/block | 11 each | 20.76 / 8.14 / 37.45 % |
| A2 | grouped_matmul | per-call | 22 | 13.51 % |
| A2 | moe_gate_up / moe_down (MC2 legs) | per-call | 8 / 8 | 5.08 / 7.13 % |
| A2 | gemm (batched / per-call) | — | — | 7.40 % / 12.56 % |

The MC2 chunk's measured gate/up and down legs are ingested as their own recipe
(`npu_grouped_matmul_list_bf16_mc2`, 28 A3 + 16 A2 observations) next to the
prefill grouped rows; pooling them into the prefill row was rejected on a
measured counterfactual. Across the eight measured MC2 chain conditions the
chain error falls from 15.3 % to **7.6 % mean** and the compute leg from 18.8 %
to **7.7 %** (worst +35.7 % → +14.3 %; the eight rows are training rows, so this
is in-sample — the CVs above are the generalization estimate). The ep4/ep8 chain
residual (−18 / −26 %) is the declared chain factor plus the unpriced `swiglu`
leg, reported per condition.

The attention fits include a declared KV-cache **broadcast limit**:
`KV bytes = kv * max(kv_heads, ceil(q_heads / B)) * (qk_dim + v_dim)`, with `B`
selected on train-only CV (`None` = one read per KV head). A3 adopts `B = 4`
(`kv > 16k` band 64.4 % → 26.7 %), A2's big-operator rows `B = 8/4`
(`kv > 16k` 91.6 % → 6.4 %). The A2 attention-worker rows at
`device_event_warm_operator` (20.76 %) and `block_interval` (37.45 %) remain
intercept-bound; lowering the 60 µs bound makes them worse, so the limit is the
parameterization, not the bound.

The A3 GEMM improvement (16.05 → 8.84 %) comes from fixing an IRLS linearization
bug for `flat_additive` rows, not from tuning. The attention topology split
(GQA vs shared-KV/MLA) was tested and **rejected on CV** before the
KV-broadcast refit (6.10 % pooled vs 5.47 % single) and is recorded as a rejected
model-selection alternative. The A2 attention-worker rows at
`device_event_warm_operator` / `block_interval` remain intercept-bound (20.76 %
/ 37.45 %): the fitted efficiency sits below the measured 71–103 TFLOPS and the
intercept is pinned at A2's ~60 µs dispatch; these need more large-decode
conditions, not renaming.

## 3. Whole-network validation (A3, measured)

Model: **Qwen2.5-7B-Instruct**, `Qwen2ForCausalLM`, 28 layers, hidden 3584,
28 Q / 4 KV heads, bf16, **TP=1 on one 910C die**, vLLM 0.30.0 / vllm-ascend
`b64b4d71`, prefix cache off, greedy, exact output lengths. Eager
(`--enforce-eager`) and ACLGraph (captures 1/2/4/8/16). 424/424 requests
succeeded.

Measured, c=1 (ms):

| scenario | eager TTFT | eager TPOT | graph TTFT | graph TPOT |
| --- | ---: | ---: | ---: | ---: |
| 128/32 | 66.42 | 31.34 | 50.85 | 15.71 |
| 1024/128 | 93.26 | 30.14 | 101.15 | 15.24 |
| 4096/32 | 335.66 | 28.98 | 342.64 | 15.10 |

Marginal prefill is **78.9 µs/token** in both modes (graph does not change
prefill); the 128-token TTFT contains a **50–66 ms fixed client/scheduler/graph
floor**, not prefill. Graph halves decode cost (ACLGraph removes host launch:
effective 1.01 TB/s in graph vs ~0.53 TB/s in eager), and 4096-token prompts at
c=4 hit chunked-prefill serialization (`4×4096 > 8192`), which is a scheduler
artefact and is marked, not fitted.

After the attention and decode fixes, the model reproduces:

| metric (c=1), `device_event_warm_operator` | model | measured | error |
| --- | ---: | ---: | ---: |
| graph TPOT 128 / 1024 / 4096 | **14.64 / 14.69 / 14.84 ms** | 15.71 / 15.24 / 15.10 | −6.8 % / −3.6 % / −1.7 % |
| graph TTFT 4096 | 303.6 ms | 342.6 ms | −11.4 % (per-call scope; the batched scope gives 252.7 ms) |
| graph TTFT 1024 | 70.4 ms | 101.2 ms | floor is 50–66, so operator prefill is +8…+40 % |
| graph TTFT 128 | 23.6 ms | 50.9 ms | floor dominates; not a prefill error |

The batched scope (`device_event_warm_operator_batched_amortized`) brackets the
same measurements from the other side: graph TPOT 14.67 / 14.66 / 14.68 ms
(−6.6 % / −3.8 % / −2.8 %). Its prefill is under-costed (252.7 ms at 4096), which
is why the per-call/operator scope is the declared one.

Eager TPOT is **not** reproduced (model 18.9 / 19.3 / 20.7 ms vs measured
31.3 / 30.1 / 29.0 ms): eager is host-dispatch-bound (~227 kernels × ~60–70 µs per
token) and the artifact carries no eager-specific intercept above the measured
per-call one, so the model still charges ~20 µs. This is the main declared
residual; graph mode is the deployment mode the model targets.

### The bug that dominated the pre-fix whole-network error

The graph lowering charged attention as if the score/probability matrix were
materialised to HBM and KV were read once per query-key **pair**:

* Qwen 4096-token prefill: `attention_qk.hbm_bytes` = 18.15 GB/layer (real fused
  kernel: 0.305 GB), QK 15.15 ms/layer memory-bound, 424 ms QK + 424 ms PV.
* Pre-fix prefill prediction 1147.6 ms vs measured 343 ms (3.3× over).
* Fix: one `attention_fused` op with Q + tiled K/V (q_tile 128) + O bytes and
  on-chip scores; fused fit applied. Prefill 1147.6 → **225.3 ms** analytic /
  **323.1 ms** with the fitted per-call rows.

The pre-fix comparison (frozen separately) measured TTFT MAPE +76.2 %, TPOT
−10.1 %, E2E +17.2 % across all 16 scenarios; the post-fix table is appended to
`.workstreams/e2e-comparison/report.md`.

### Decode weight streaming and the replayed-graph launch

A decode step streams **13.55 GiB** of weights (14.21 GiB decimal) but the L2
blend credited **9 372 MiB of it (69 %) at 4.8 TB/s**, worth ~9.5 ms of the 15.1 ms
the stream takes: `min(bytes, l2_usable)` credits a 25–259 MiB per-op weight
tensor even though its next use is one token — 13.5 GiB of traffic — away, so L2
(192 MiB) can hold at most 1.4 % of it. The `≥8× L2` cap could not fire: no
*single* op streams 1.5 GiB.

The rule is now per op, declared and shape-only
(`a3_physical_calibration.weight_stream_basis`): a dense GEMM with
`m ≤ 128` (one or two activation row tiles), weights ≥ 4× its activation bytes and
weights ≥ 1/8 of the family's fitted resident prefix is a **weight stream** — no
L2 credit, priced at the part's measured streaming bound
(`STREAMING_BW_BYTES_PER_S`: A3 **1.01 TB/s**, A2 **471 GB/s**, the low end of its
measured 471–955 GB/s decode band). A family that never measured the regime takes
that bound; one that did keeps its fitted efficiency; a part with neither uses its
declared peak, flagged. `rmsnorm`'s 0.8 MiB working set keeps its blend. Every op
reports `memory_regime`, `l2_credit_bytes`, `eta_memory_effective`,
`streaming_rate_source` and a `streaming_basis` string.

That correction alone makes the model ~3.5 ms/step too *slow* (14.57 ms service
+ 4.75 ms launch = 19.3 ms vs 15.24 measured): the service term now implies
**0.98–1.00 TB/s**, exactly the measured stream, and the launch term is exposed.
A replayed bucket (`graph_bucket` set, ACLGraph) submits its 227 kernels from one
host call, so `a3_cost.GRAPH_KERNEL_LAUNCH_US = 2.5` now charges a declared
**device dispatch per kernel** (15.10 measured − 14.57 corrected service ≈ 0.53 ms
/ 227 ≈ 2.3 µs, declared 2.5 µs) instead of the ~20 µs host intercept, reporting
`launch_detail.composition = "graph_replay_device_launch"`, the replaced
`host_intercept_ms` and the per-op value. Eager decode keeps the serial host
intercept and prefill keeps the concurrent one; the MC2 fused chunk, launched by
the same graph, pays the same device dispatch.

The streaming regime has **0** training rows, so the rate is a declared measured
bound and not a CV-selected parameter; a separate `decode_gemv` family cannot
reach the 6-sample minimum and is not fitted, and the two GEMM rows are unchanged
(samples 367/107, `eta_memory 0.7076/0.9786`, `cv_wape 8.84/11.50 %`). Evidence and
before/after tables: `.workstreams/model-decode-stream/report.md`.

## 4. Wiring

* **Scope-family fallback**: artifacts carry per-call rows at
  `device_event_warm`, `_operator`, `_operator_per_call`, `_per_call` and
  `block_interval_operator`; the lookup prefers an exact scope, then the same
  family/recipe/protocol/dtype/topology inside the same scope family
  (device-event over block-interval, never batched→per-call) and records
  `scope_alias_from/_to`. This is what lets one whole-network case apply the GEMM,
  attention, rmsnorm and MoE rows simultaneously.
* **Fused attention** is emitted as a single op with on-chip score bytes; the
  graph no longer produces qk+softmax+pv for a fused kernel.
* **MC2**: `t = launch + (compute + comm) · chain_factor`, `chain_factor` bounded
  [1.0, 1.2] with default 1.0 and `max()` flagged as refuted by measurement;
  `ceil(E/128)` launch count; measured per-EP lane (A3 65/32 GB/s e64/e128, A2
  17.5 GB/s); dense-GEMM fallback disabled for MC2.
* Eager decode iterations use a **serial** launch composition so the engine cannot
  drop the per-kernel host dispatch; a **replayed** bucket charges the declared
  **device dispatch per kernel** (`graph_replay_device_launch`) because its host
  submit is amortized across the bucket.

## 5. Residual gaps (ranked)

1. Eager-mode host dispatch (~14 ms/token at 7B): not modeled; graph mode is.
2. Fixed 50–66 ms client/scheduler/graph floor at short prompts: outside the
   operator model; reported as a declared offset.
3. A3 MLA decode KV re-read at kv > 16k (64 % band), absorbed-MLA decode.
4. `moe_gate_up` effective bandwidth falls 1.37 → 0.74 TB/s with expert count
   (worst band 31 %); the A3 GMM expert axis is still unmodeled.
5. A3 attention at kv ≤ 512 (55 % band, dispatch noise); MHA topology (n=2).
6. A2 attention: fits pinned at the dispatch intercept; needs more large-decode
   conditions.
7. No A2 end-to-end serving measurement (A2 operator data only); no MoE model
   E2E, no TP≥2, no quantized/W8A8, no MindIE/ATB baseline (vLLM-Ascend only).

## 6. Evidence

* A3 bigops: `.workstreams/bigops-a3/` (report, 24 attention + 27 GMM, probes)
* A2 bigops: `.workstreams/bigops-a2/` (report, 50 cases, plan/amendments)
* MC2: `.workstreams/mc2-campaign/` (report, 686 rows, contract, raw both hosts)
* Fits: `.workstreams/model-bigops/` + `.workstreams/model-physical/*.json`
* Wiring/MC2 model: `.workstreams/model-mc2/` (report, validation)
* E2E measured: `.workstreams/e2e-benchmark/report.md` and
  `/Volumes/data/mindie-sim-data/a3/2026-10-20/e2e-20261021/e2e-20261021-a3-qwen25-7b-tp1/`
* Predict-vs-measured: `.workstreams/e2e-comparison/report.md`

Raw measurements live outside the repository; `.workstreams/` is working
evidence and git-ignored.
