# A5 conditional grouped effective-service regimes

This calibration uses observed **native operator completion** directly, under the user's explicit permission to calibrate small-workload regimes. It does not infer a new scalar ISA timing, add PMU counters together, or fit complete Body latency. Existing physical programs continue to execute with unchanged work and resource demand. The common Engine uses the inclusive maximum of their completion and the declared effective response.

## DSQ normal group loop

`data/observations/a5-grouped-service-regimes-20261030/parameters.json` contains the generated rule. Its producer reads only the c64 DequantSwigluQuant mean when assigning the coefficient:

`54.03106170886076 µs / 64 owner-group visits = 0.8442353392009494 µs/visit`.

The physical domain is Ascend950DT_9582, the explicit arch35 full-load INT32→INT8 DSQ program with FP32 scales, normal group-cut branch, output width512,64AIVs, valid per-group rows8–16, total valid rows512–1024 and allocated rows2048–4096. No model or concurrency name participates in selection. Group histograms remain declared forced-balanced inputs: the captures do not contain `group_list` values. Allocation capacity is not substituted for valid rows, and DSQ capacity does not close the upstream GroupedMatmul/MC2 allocation ABI.

The coefficient represents inclusive effective group completion, not an incremental per-group delay. The initial declared scaling assumption is inverse total AIV vector capacity relative to6.7584e12units/s. That extrapolation is a modeling hypothesis; these captures did not measure hardware throughput sweeps or isolate scalar versus DMA mechanisms.

| Cohort label | Assignment role | Observed µs | Effective prediction µs | Error | Decision |
|---:|---|---:|---:|---:|---|
| c64 | Parameter assignment,3,160calls | 54.0311 | 54.0311 | 0.00% | In-sample fit |
| c128 | Not used in this assignment,3,040calls | 56.5601 | 54.0311 | -4.47% | ≤15% |
| c4 | Different source branch,3,360calls | 7.2992 | 1.3120 | -5.9872µs | Unchanged; ≤15µs micro tolerance |

The c128 observations had already been inspected during earlier mechanism analysis. They are held out only from **this coefficient assignment**, not a blinded historical test set. The original raw observation roles remain unassigned. Cross-model validation for this same implementation and physical domain is still pending; no such result is claimed here.

`data/verification/a5-dsq-effective-service-20261030.json` records real ordinary-frontend→single-native-Engine runs with late7ms input, no service before input, released buffers and unchanged native work/service. The vector0.5/1/2declared sweep gives108.0621 /54.0311 /27.0155µs. Model/name changes leave output unchanged; changing capacity to select the source special branch leaves the regime unapplied. These are execution and applicability controls, not additional measured-rate validation. The report does not run or claim complete Body accuracy.

The separate condition cases are under `examples/cases/high-concurrency-20261030/qwen35-dsq-effective-service-conditional/`. Original source cases and global hardware profiles remain unchanged. Rules can be appended to other service rules by ID; callers must not replace unrelated MC2 or attention declarations when combining configurations.

## Small-M grouped bank service

The separate `grouped-parameters.json` and `grouped_producer.py` add four narrow physical domains. Q35's INT8→INT32 and INT8→BF16 paths have distinct output contracts. V4.1 selects the explicit MXA8W4 NZ source implementation with dynamic-buffer4 supply, FP8 activation, FP4 weights and BF16 output. Each response is `positive expert-bank count × calibrated bank service`; no model, concurrency label or allocated activation row count selects the rule. N/K, precision, source implementation, declared bank extent and maximum valid rows per bank delimit applicability.

| Physical domain | Calibration mean µs | Declared active banks | Effective µs/bank | c128 validation error |
|---|---:|---:|---:|---:|
| INT8→INT32, N1024 K2048, rows1–16 | 60.4423 | 64 | 0.9444111 | -2.23% |
| INT8→BF16, N2048 K512, rows1–16 | 42.2524 | 64 | 0.6601932 | -2.59% |
| MXA8W4→BF16, N4608 K5120, rows1–2 | 242.5993 | 48 | 5.0541521 | -3.01% |
| MXA8W4→BF16, N5120 K2304, rows1–2 | 119.8543 | 48 | 2.4969652 | -2.77% |

Only each c64 operator mean assigns its coefficient. The INT8 c4 shapes, with32declared active banks, are additional non-assignment controls: errors are-10.48% /-4.79%. All10operator cases satisfy the requested dual threshold, but four are in-sample and none establish unseen-model accuracy. These conditional physical domains can apply independently of model name; the captured group distributions remain unavailable.

`data/verification/a5-grouped-effective-service-20261030.json` records each original physical duration alongside the effective result, unchanged native work/service, released buffers and late-input controls. Rate scaling uses the declared private weight-input pipe: AIC MTE2 for INT8 and AIV MTE2 for FP4 supply, both relative to4.224e12B/s. This is an explicit effective-service extrapolation. Physical execution still wins when slower: at2×AIC MTE2, the INT8 up response would be30.221µs, but the original program takes35.234µs and that duration is retained. Likewise2×AIV MTE2 leaves the MXA8W4 up at215.642µs rather than forcing121.300µs. No work or bandwidth is overwritten to force the response.

## Complete Body composition

`grouped-mc2-effective-service-conditional/` preserves full40-layer inputs and merges rule IDs without replacing existing declarations. V4.1 also includes the independent MX MC2 dispatch/combine rules from `a1e0c8b2`, parameter SHA-256 `6b1ab467a10557627b00d39430e47767f559bc004aa56f886968caa1e2834912`. Q35 does not match those EP8/MX rules and receives only its grouped/DSQ regimes. No coefficients are adjusted after viewing Body results.

`data/verification/grouped-mc2-effective-body-20261030.json` retains three fresh-process attempts, each with a60sbudget:

| Input | Status | Complete Body prediction | Comparison / runtime |
|---|---|---:|---|
| Q35 c128, q4/B128 | Completed | 20.150964ms | vs28.584774ms observed: -29.50%;29.79s wall,116.19MiB RSS |
| V4.1 c64, q1/B8 on each of8DP peers | Timeout | unavailable | 60.008s; no retry or layer reduction |
| V4.1 c128, q1/B16 on each of8DP peers | Timeout | unavailable | 60.012s; no retry or layer reduction |

Q35 records exactly40applications for each grouped-up, DSQ and grouped-down rule,2,226,853completed Engine tasks and zero live tasks/programs/buffers. Its major Body residual remains despite successful isolated native-call comparisons. The prior DSQ-source-only result16.618497ms is a historical reference; no new control Body run was added here.

The V4.1 cases use the existing composed source input, including native MX MC2, prepared matrix representations and arch35 HC. They are not identical to earlier dynamic-grouped-only performance probes. An isolated grouped/MC2 service validation cannot replace these timed-out complete Body results; no V4.1 Body accuracy score is asserted. Declared symmetric peers and valid expert histograms remain conditional, and the report leaves the missing full-graph spans visible.
