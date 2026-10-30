# A3 GLM Flash livepre GMM contract audit

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](verification.md).

The c16 comparison remains **FAIL, pending the input/rate contract**. This audit
changes no prediction, routing distribution, hardware rate or multiplier. The
report retains
CSV hashes, exact accepted tp0 windows, native cases/actuals, prediction runtime
identity and source hashes. Its times are native kernel activity, not additive
Body wall time or serving TPOT.

| Case | Complete ranks | Mean up/down across those ranks (µs) | tp0 observed up/down (µs) | Retained predicted up/down (µs) |
| --- | ---: | ---: | ---: | ---: |
| c4, 46 forwards | 14 | 77.825 / 41.587 | 78.892 / 42.138 | 149.536 / 75.440 |
| c16, 37 forwards | 14 | 137.312 / 70.247 | 142.415 / 71.927 | 281.297 / 141.905 |

The 32 rank captures were streamed once in 31.8 seconds with approximately
45 MiB peak RSS. Other ranks use tp0's absolute-time windows, conditional on
clock alignment. Fourteen ranks contain exactly 42 up/down calls per forward;
tp9/tp13 have incomplete counts and are excluded from the mean. Count equality
does not independently prove phase identity. Rank0 is not an unusually fast
rank that explains the discrepancy.

The captured input capacities are M=256 for c4 and M=768 for c16, with 18 weight
banks. They do not reveal the active-bank count. The retained c16 native sample
has router input `6×288` and DispatchV2 input `6×4096`, expert IDs `6×8`, and a
`BOOL[6]` active mask. Neither mask values nor GMM `group_list` values are in the
capture. Query tensor shapes therefore do not establish valid MoE token counts.
No timing or PMU activity is inverted into those missing values.

The original `RESTORE-wbn-flash-serve.txt` launch record identifies production
`wbn-flash`, image `sha256:50215278e467011c889987b5e80ad1eb44cef817bbfc40607a1f8930c3c3577a`,
DP1/TP16 with EP, natural `deepseek_mtp` with five speculative tokens, and
`FULL_DECODE_ONLY`. Its additional configuration has no force-EPLB override,
and its speculative configuration has no synthetic rejection override. The
campaign explicitly reports natural MTP and no forced EPLB for this population;
the forced/synthetic r21 relaunch is a different capture. The launch file SHA-256
is `469f4aa3b5ec691cd0ff12664949d7e1df2c976e83a8f194c18e74df2809f1eb`.

The inspected candidate caller at `ae923905` forms this chain:

1. `model_runner_v1.py:2435` passes `scheduler_output.total_num_scheduled_tokens`
   as `num_actual_tokens`, including scheduled draft verification queries.
2. `ascend_forward_context.py:209` makes the first N mask entries true and the
   padding tail false. `prepare_finalize.py:272` slices mask and inputs by TP;
   the already-SP branch preserves the local slice and pads false.
3. `token_dispatcher.py:174` passes that mask to DispatchV2; its returned
   `expert_token_nums` becomes GMM `group_list` after the EP exchange.
4. Sampling follows target forward. Accepted counts select/roll back KDA state
   (`models/glm5next/kda.py:393`); they do not construct the MC2 mask.

Thus, conditionally on N=96, TP16 produces six true entries per rank, while
global assignments remain `96×8=768` (48 expected local rows under uniform EP16
routing). Reducing global N to six would divide by TP again. In DP1 the general
constraint is `P=ceil(N_padded/TP)×TP`, `R=P/TP`,
`V_r=clamp(N-r×R,0,R)`, and `sum(V_r)=N`. Neither accepted length nor a kernel
duration supplies an alternative N. The JSON retains the complete source chain
and hashes. The launch uses a different installed source path, so its identity
with `ae923905` remains unestablished; exact scheduled N versus padded query
geometry and natural routing correlations remain unresolved.

The declared source ABI requires 16 MiB per up bank and 8 MiB per down bank.
ExpectedGroupedMatmul preserves these weight moments; its owner grid does not
duplicate the bank set. The selected source skips empty groups and disables
weight L2 caching for one M tile. Its source hash is retained, while equivalence
to the captured binary remains unestablished.

With the existing independent-token moments (8.845/16.796 active banks), the up
weight alone requires **123.666/234.820 µs on HBM** or **140.529/266.841 µs on the
declared AIC MTE2 resource**. Both exceed the observed 78.892/142.415 µs. Extra
activation, output and scale traffic can only increase these bounds. Changing
overlap alone cannot resolve the joint workload/rate mismatch.

The retained INI has separate `ddr_read_rate=32` and `l2_read_rate=110` fields;
its parser does not establish either as an independent MTE2 instruction cap.
The current profile promotes DDR 32 B/cycle to an AIC read cap and marks its
1.2 TB/s HBM declaration unverified. Retained profiler source distinguishes
local-pipe bandwidth from GM/L2 bandwidth, but supplies no verified 9362
replacement rate. This domain distinction does not prove 32 B/cycle numerically
wrong for cold DDR reads and does not justify increasing it to fit this capture.

The remaining discriminators are actual routing/mask values and independently
established hardware rate domains. The report preserves this unresolved boundary.

The committed report was compacted from the completed scan without reopening
raw CSVs. To explicitly reproduce the scan on a host retaining these sources:

```sh
python3 data/verification/glm-livepre-gmm-contract-20261030.producer.py \
  --scan --profiler-root ../profiler --vllm-root ../vllm-ascend \
  --caller-root ../vllm-ascend-v41-private \
  --output /tmp/glm-livepre-gmm-contract-reproduced.json
```

`--retained-scan /tmp/glm-livepre-rank-gmm-audit.json` instead compacts the original
scan result. That mode verifies metadata/source identities but does not rehash
the raw CSV captures; the report states which mode produced it.
