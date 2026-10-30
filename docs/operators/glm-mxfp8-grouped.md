# AIC-only MXFP8 NZ grouped matrix source program

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

`ascend_mxfp8_grouped_nz` adds an explicit source program for an integer-bank,
FP8 E4M3 × FP8 E4M3 → BF16 grouped matrix with MX32 E8M0 scales. It uses the
ordinary `RecipeBook` and Engine. There are no measured times, model names,
utilization factors or new invocation floors in the planner.

Select only a matching boundary, for example the existing ordinary frontend
expert-down recipe:

```json
{
  "runtime": {
    "kernels": {
      "recipes": {
        "balanced_grouped_down": {"kind": "ascend_mxfp8_grouped_nz"}
      }
    }
  }
}
```

This leaves the distinct gate-up/Swiglu/quantization producer chain intact. The
source program is optional; it does not silently replace another explicit
recipe or become a generic BF16/INT8 matrix implementation.

## Evidence and physical contract

The retained CANN `ops-transformer` revision is
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8`. Exact file hashes are in
`glm-mxfp8-grouped-20261030.json`.
The modeled branch is **CGMCT, `!IS_BLAZE`, MX NZ, nontransposed A/B, no bias**.
The current capture's binary/tiling version is not established; newer Blaze
and fused A8W4 implementations are different source domains.

| Source | Mechanism retained |
|---|---|
| `grouped_matmul_apt.cpp`, around 270/312–328 | MX NZ kernel is AIC-only; direct Fixpipe dequant path |
| `grouped_quant_matmul_tiling.cpp`, 1307–1337/1355–1367 | M256/N256 maximum source base blocks, M16/N128 alignment, K128 maximum MX base block, L0C double-buffer capacity |
| `gqmm_cgmct_mx_kernel.h`, 45–84 | CGMCT ASWT scheduler and empty vector epilogue |
| `kernel_qgmm_mx.h`, 180–231/309–326 | Cross-bank rolling core cursor, per-bank M balancing, only final declared bank may split its last core wave |
| `block_scheduler_gmm_aswt_with_tail_split.h`, 90–209/251–282 | Windowed M order, zigzag N order, actual tail coordinates and split ownership |
| `block_mmad_mx.h`, 145–183/194–267/341–413 | Two L1 buffers, separate E8M0 scale pairs, actual ND A and padded NZ B transfers |
| `block_mmad_mx.h`, 431–569/673–704 | Double L0 operand slots; A M16, transposed B N32 payload, separate M16/N16 scale loads; K64 MMAD padding |
| `block_mmad_mx.h`, 573–592/860–882 | Actual M/N MMAD dimensions; BF16 GM output through Fixpipe, not MTE3 |

The default full-K domain is `K <= 256`, subject to declared per-owner pool
capacities. Larger K accepts an explicit `l1_k` multiple of the base K tile;
that option declares equal A/B/scale refill geometry, rather than claiming to
reproduce every version of host dispatch. Unsupported precision/representation
or insufficient source buffers remains visible.

Source output tiles are modeled serially per AIC. Same-owner consecutive tiles
with identical M/N are compressed into a nominal dependency span while preserving
one initial load, one final store, every service byte/FLOP and ownership. Shared
HBM nominal supply is divided among active owner pipelines. All actual bytes
still compete on the original HBM resource. `compact: false` retains expanded K
and output-tile execution; `compact_outputs: false` compresses only K panels.

The compact middle is a fluid contention approximation, not a proof of equivalent
arbitration under arbitrary concurrent kernels. Scalar descriptor/flag latency,
ND→NZ transaction efficiency, small-M instruction issue cycles, cross-output
prefetch and capture binary identity remain unmodeled. GM reads are explicitly
assumed offchip; no hidden cache-hit factor is fitted. This is a conditional
source interpretation, not independently held-out accuracy.

## New 67ca7673 full-ABI comparison

The producer consumes the **new** profiler
`acceptance-20261030-glm-operators-67ca7673` family scoreboards and their retained
source locators. It does not reuse old prediction numbers. The baseline is held
at runtime `0163abc5`; later generic matrix candidate-domain changes are excluded
from this mechanism comparison.

All three cases have q=4, context=4096, TP8/DP1/EP1, 288 experts, top-8 routing,
and 42 expert-down calls per body. The captured native ABI is
`GroupedMatmul`, FP8 E4M3 inputs, E8M0 scales, NZ weights and BF16 output,
`Block Num=32`, `Mix Block Num=0`.

| Batch | Integer expert-row histogram | Observed µs/call | Generic µs/call | Explicit source µs/call | Source signed error | Signed 42-call gap, generic → source |
|---:|---|---:|---:|---:|---:|---:|
| 16 | 224×2 + 64×1 | 103.104 | 86.575 | 104.565 | +1.42% | −0.694 → +0.061 ms |
| 32 | 160×4 + 128×3 | 109.824 | 87.876 | 106.775 | −2.78% | −0.922 → −0.128 ms |
| 64 | 32×8 + 256×7 | 115.777 | 90.476 | 111.195 | −3.96% | −1.063 → −0.192 ms |

The histograms are the case's declared forced-balanced realization. Captured
`group_list` tensor values remain unavailable. All 288 banks are active, so
missing active banks do not explain these residuals. Native expert-down is
separate from `GroupedMatmulSwigluQuantV2`; adding that producer's SiLU or output
quantization here would double-count another kernel.

For batch 32, semantic matrix work stays **2,147,483,648 FLOPs** and private MTE2
traffic stays **315,752,448 bytes**. MTE1 changes from **320,077,824** to
**330,891,264 bytes** because L0 input tiles retain M16/N32 and MX scale-pair
access. GM traffic changes from **320,086,016** to **324,214,784 bytes** under the
explicit all-offchip read condition. The **8,388,608-byte BF16 output** moves
from MTE3 to Fixpipe. Rolling bank ownership and finite source load/compute/store
phases also replace the generic row-class schedule. No bandwidth/rate changes
are part of these three prediction arms.

The 42-call figures are isolated kernel activity sums. They are not complete
body errors, actual graph savings or TTFT/TPOT accuracy.

## Execution cost and checks

The expanded source geometry has 4,608 output tiles per call. The compact program
runs **524 Engine tasks per call**, versus **60,172 tasks** in the retained
expanded single-call control. The paired batch-32 graph takes 0.0105 s versus
0.4372 s wall time, with unchanged predicted span and physical service. In the retained producer, all 42 original
expert-down nodes execute as a clearly labelled cut-dependency diagnostic in
**21,926 tasks / approximately 0.4 s wall time**, with a 4.670174 ms simulated
span. This is a real complete set of 42 source calls, not a shortened model
presented as an entire body. One unmodified complete original layer is also run,
including its attention, FFN and communication nodes. All completed graphs leave
zero live tasks, programs and allocations.

The producer includes detailed-vs-compact controls for all three true ordinary
operators. Five independent physical axes (`hbm`, `mte1`, `mte2_aic`, `fixpipe`,
Cube peaks) are swept at 0.5×/1×/2× for batch 32, with unchanged work. These 15
paired controls agree within floating rounding; MTE2/Fixpipe acceleration can
legitimately plateau on HBM. Broader cross-kernel contention has not thereby
been proven equivalent.

Tests additionally check source-index arithmetic against an independent
per-block oracle, nonuniform/zero banks, final empty bank, cursor wrap, spatial
output coverage without duplication, N32 tail payload versus N16 scales, K tails,
late input readiness, per-owner buffer release and one inclusive invocation for
the complete kernel. The producer is bounded and local; it launches no NPU work.

Focused validation: 75 tests passed in 4.02 s across
`test_mxfp8_grouped_source.py`, `test_bf16_grouped_source.py`,
`test_balanced_experts.py` and `test_balanced_catalog_generalization.py`.
The final producer completes in 10.60 s with 94.55 MiB peak resident memory and
matching runtime package fingerprints at start/end.

## Independent review

The independent review retained in
`mxfp8-grouped-independent-review-20261030.json`
checked 1,000 nonuniform/zero/tail geometries and 144 actual Engine controls.
Source work, input readiness and released buffers hold. Under asynchronous
external contention, output-run compression differed from detailed execution
by up to +12.28% in those controls; K-only compression reached +11.95%. The
three captured-ABI comparisons and synchronous resource sweep are narrower
domains and do not establish exact scheduling under arbitrary contention.
