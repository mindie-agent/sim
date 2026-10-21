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
| A3 | attention_fused | device-event | 66 | **5.47 %** |
| A3 | attention_fused | block-interval | 66 | 5.74 % |
| A3 | moe_gate_up | per-call | 15 | 10.98 % |
| A3 | moe_down | per-call | 12 | 2.36 % |
| A3 | grouped_matmul (experts=8) | per-call | 10 | 1.65 % |
| A3 | gemm (per-call / batched) | — | 367 / 107 | **8.84 % / 11.50 %** |
| A2 | attention_fused | per-call | 28 | 22.93 % |
| A2 | attention_fused | operator/batched/block | 11 each | 20.76 / 32.17 / 37.45 % |
| A2 | grouped_matmul | per-call | 22 | 13.51 % |
| A2 | gemm (batched / per-call) | — | — | 7.40 % / 12.56 % |

The A3 GEMM improvement (16.05 → 8.84 %) comes from fixing an IRLS linearization
bug for `flat_additive` rows, not from tuning. The attention topology split
(GQA vs shared-KV/MLA) was tested and **rejected on CV** (6.10 % pooled vs
5.47 % single) and is recorded as a rejected model-selection alternative.
A2 attention is weak and reported as such: the fitted efficiency sits below the
measured 71–103 TFLOPS and 3 of 4 rows are pinned at A2's ~60 µs dispatch
intercept; it needs more large-decode conditions, not renaming.

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

| metric (c=1) | model | measured | error |
| --- | ---: | ---: | ---: |
| graph TPOT 128 / 1024 / 4096 | 14.97 / 15.29 / 15.58 ms | 15.71 / 15.24 / 15.10 | −4.7 % / +0.3 % / +3.2 % |
| graph TTFT 4096 | 252.7 ms (+50–66 floor) | 342.6 ms | −7…−12 % after the floor |
| graph TTFT 1024 | 75.0 ms | 101.2 ms | floor is 50–66, so operator prefill is +25…+50 % |
| graph TTFT 128 | 16.6 ms | 50.9 ms | floor dominates; not a prefill error |

Eager TPOT is **not** reproduced (model 12–14 ms vs measured 29–31 ms): eager is
host-dispatch-bound (~227 kernels × ~60 µs per token) and the current artifact
does not carry an eager-specific intercept above the measured per-call one. This
is the main declared residual; graph mode is the deployment mode the model
targets.

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

### Decode bandwidth rule

The batched GEMM row had `eta_memory 0.979` (≈1.17 TB/s), over-optimistic for a
14.2 GiB weight sweep. The artifact now applies a declared streaming rule:
working set ≥8× L2 gets no L2 credit, and a family that never measured that
regime is priced at the measured **1.01 TB/s** (`eta_memory_effective 0.842`,
`streaming_measured: false`). Graph TPOT 12.66 → 15.58 ms at 4096, matching the
measured 15.10 ms. The regime has 0 training rows, so this is a declared
measured bound, not a CV-selected parameter; a separate `decode_gemv` family
cannot reach the 6-sample minimum and is not fitted.

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
* Decode iterations and replayed buckets use a **serial launch composition**, so
  the engine cannot drop the per-kernel dispatch.

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
