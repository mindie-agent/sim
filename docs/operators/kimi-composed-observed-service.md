# Complete Kimi source and observed-service composition

The complete 93-layer c32 cases combine the existing source programs with
explicit HCCL, MC2, small-M BF16 projection and two-bank BF16 grouped responses.
Each response retains its native-call assignment/validation population and
physical domain. No Body duration assigns a parameter, and no isolated cost
delta is added to the Body result.

| Context | Observed Body mean | Predicted Body | Signed gap | Relative error |
| --- | ---: | ---: | ---: | ---: |
| p4k | 40.018728 ms | 36.968860 ms | -3.049867 ms | -7.62% |
| p16k | 44.666193 ms | 42.155614 ms | -2.510579 ms | -5.62% |

Both values are actual `predict(scope="body", phase="decode")` executions,
with all 93 layers and dependencies. Wall times were 19.44 and 39.84 seconds
under a 60-second per-case budget. The receipt records rule query counts,
unchanged package identity across each run and zero live tasks/programs.

These are calibrated comparisons with the recorded source/capture contracts,
not an independent claim of accuracy for every model or workload. HCCL's
explicit AICPU executor remains a declared source condition; the communication
parent event alone does not identify its executor. The low-concurrency
grouped cases have a separately retained valid-SP-mask/active-expert
counterexample and are not covered by the two-bank rows8..16 parameter domain.

The projection service retains inverse HBM scaling as an explicit engineering
assumption. Fixed protocol synchronization is not relabeled as link bandwidth.
Physical source work and contention remain in the common Engine.

```sh
python data/verification/kimi-composed-service-20261030.producer.py
python tools/replay_service_regime_cases.py \
  data/verification/kimi-composed-service-cases-20261030.json \
  -o data/verification/kimi-composed-service-body-20261030.json
```
