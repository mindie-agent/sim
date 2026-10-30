# EP4 per-token INT8 MC2 observable response

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

This explicit candidate adds native DispatchV3 and CombineV3 completion responses
for Ascend950DT_9582, 256 experts, width 2048, top-8 and EP4. The response is
`max(existing physical native program, observed response)` once per kernel group.
It does not replace count/payload dependencies, quantization, local copies,
weighted combination, or real wire service, and it does not fit a Body total.

The parameter bundle is
[`int8-ep4-parameters.json`](../../data/observations/a5-mc2-service-regimes-20261030/int8-ep4-parameters.json).
The adjacent producer and retained receipt
`qwen35-mc2-service-regime-20261030.json`
contain capture hashes, source tensor ABI, PMU fields and the assignment split.
The three new cases preserve the parent grouped/DSQ rules exactly and append
two rules; their original cases remain unchanged.

## Observable domain and assignment

`M` denotes the native per-rank top-k input rows. The captured rows are 4, 64 and
128 after the explicit TP4 caller partition of full-query rows 16, 256 and 512.
The receiver capacities 128, 2048 and 4096 are not actual assignment counts.
The conditional cases retain ideal balanced routing; captured expert IDs and
peer readiness were not reconstructed. All observed calls use 64 AIV blocks.

| Kernel | Input / wire / output | Inclusive response, μs |
|---|---|---:|
| DispatchV3 | BF16 / per-token INT8 + FP32 scale / INT8 | 16.473949 + 0.193003965 M |
| CombineV3 | BF16 return and weighted output | 10.663718 + 0.253479455 M |

The implementation expresses the slope using external framed bytes. The
existing source frame layout yields 2560 B per dispatched assignment and 4608 B
per returned assignment. With top-8 and the EP4 external fraction 3/4 this is
15360 B/M and 27648 B/M respectively. Dispatch additionally exchanges 6144 B
of count/status per rank, independently represented by the physical program;
Combine has no analogous expert-count exchange. Its data flags are included
in its existing frame and receive phases.

The two endpoints M4 and M128 assign each line. M64 is withheld from these
coefficients. It is not a blind experiment and is not a whole-Body holdout of
the parent grouped/DSQ parameters, which have their own assignment population.

| M64 native kernel | Observed μs | Response μs | Signed μs | Relative |
|---|---:|---:|---:|---:|
| DispatchV3 | 28.795800 | 28.826203 | +0.030402 | +0.106% |
| CombineV3 | 27.371084 | 26.886403 | −0.484682 | −1.771% |

The rule domain is M4–M128 and matches hardware, implementation, EP, expert
count, width, top-k and dtype; it does not inspect a model name. Inputs outside
this domain continue on the analytic model. The observations identify an
effective completion response at this hardware configuration, not independent
HBM/AIV/network throughput. No guessed resource scaling coefficient is applied.
The underlying resource work remains active and can exceed the response as
hardware rates change. The explicit all-peer window is inherited from the
source-compatible parent, not inferred from these elapsed times.

## Execution and remaining Body error

`qwen35-mc2-service-execution-20261030.json`
was produced on runtime `43a23cb0`. All three paired native DAGs ran with input
readiness at 7 ms. Each response was applied once per native call. Physical
service before and after the response is unchanged apart from floating summation
roundoff up to 1.86e−8 bytes; every execution releases its tasks/programs.

The complete forty-layer target-body candidates retain q4, B4/64/128 and the
parent's declared context 4096. Actual per-step KV lengths are not observed from
cache capacity, so the comparison remains conditional. Each new rule is queried
exactly 40 times, not once per internal MC2 phase.

| B | Predicted Body ms | Observed Body mean ms | Gap ms | Relative | Local wall s | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 6.451378 | 11.551367 | −5.099989 | −44.15% | 6.03 | 102.14 |
| 64 | 16.945319 | 21.956845 | −5.011526 | −22.82% | 19.40 | 107.52 |
| 128 | 22.436580 | 28.584774 | −6.148194 | −21.51% | 30.53 | 115.59 |

All complete within the 60-second / 512-MiB budget, with zero live tasks and
programs. The native-call agreement does **not** establish Body accuracy: all
three Body comparisons still fail 15%. These remaining gaps are retained, not
added to MC2's response. Historical predictions in the observed-body source
report have a different runtime identity and are not used as this experiment's
same-runtime control.

## HCCL remains a separate boundary

The c128 HCCL ledger contains 9120 exact summary/device duplicate rows. Only
the concrete device tasks count as calls: 80 AllReduce and 40 AllGather per
forward, with 4.986733 ms/F activity. The 76 unlinked AICPU wrapper records total
0.128146 ms/F and are not extra device collectives. Neither sum enters the MC2
coefficients, nor may either activity sum be blindly added to Body elapsed time.
The parent AllGather caller and communication program remain intact.

Vector, scalar, MTE and wait PMU observations overlap. The retained native-call
mean is the response boundary; no PMU component difference is separately added
as network or control delay. Binary identity and actual route values remain
independent evidence questions, not reasons to disable the analytic path.
