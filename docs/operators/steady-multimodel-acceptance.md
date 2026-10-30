# Steady matrix complete-cohort acceptance, October 29

The subsequent owner-output compression at
`11fc9acf833faf1ac26dfbb347c29b37e7a5dc95` completed V4 prefill and all six
decode probes in a fresh 90-second campaign. It finished seven of seven in
84.753 seconds. V4 prefill completed in 16.206 host seconds with 130.125 MiB
peak RSS. The initial timeout campaign remains recorded below, followed by
the matched follow-up results.

The initial campaign executes the public `target_graphs` and `graph_costs` workflow,
including its isolated operator accounting. It uses committed runtime
`64b2846bccfd6a7f86dc727698fa0a759356abf7`, forced balanced routing, and explicit
`steady_matmul` for both ordinary matrices and the balanced expert gate/up and
down recipes. Model dimensions, precision and deployment declarations come
from the six catalog case files. No timings or measured residuals are inputs.

The prediction boundary is a target-only pure-model device graph for the first
independent DP cohort. All participants in that cohort execute: V4 and Kimi
each retain four replicas. These are not TTFT, TPOT, service throughput or
measured accuracy results. Isolated sums and graph spans remain separate.

## Reproducible source and probe boundaries

Concurrent profiler and scoreboard changes were excluded by exporting the
committed revision. The report retains the excluded worktree paths, input file
hashes, effective input hashes, driver hash and imported package hash:

```text
runtime revision: 64b2846bccfd6a7f86dc727698fa0a759356abf7
package SHA-256: fa6f7dcff434f07f50aae204fd58ea986fb6bea0d9eeec4c5e06a2fa99616936
driver SHA-256: 00f65e3bfdca0b844c3c9c5bb82255be816c8bcf8c7ed31c4c15ee8a85185dc0
```

All six successful workers observed that same package identity. The temporary
export path may subsequently be reused for newer validation; the retained
revision and content hash identify this campaign.

Decode uses one query token per sequence at context 2,048; GLM uses two
simultaneous sequences, and the other models use one. Prefill targets are
256 tokens except Kimi and GLM at 252, which preserve integer balanced expert
loads. Every probe uses full model scope. The declared budgets are 60 seconds
per worker and 180 seconds for the campaign. Explicit `--only` order places V4
first, then the remaining decode probes, before the remaining prefill probes.

`tools/model_convergence.py` now retains graph `completed_tasks`,
`peak_live_tasks`, `live_tasks`, `live_programs` and `live_allocated` directly
from the Engine summary. Worker RSS is the `getrusage(RUSAGE_SELF).ru_maxrss`
lifetime high-water mark with its native unit recorded; it includes frontend,
graph, isolated analysis and reporting. Worker wall time includes process
startup and teardown. These host execution measurements are different from
predicted device milliseconds.

## Initial bounded results at 64b2846b

The campaign ended as **partial** after 180.011 host seconds: six modeled
decode graphs, two interrupted prefill probes and four unstarted prefill
probes. No timeout budget was extended.

| Decode model | Cohort replicas | Device graph ms | Isolated sum ms | Worker seconds | Peak RSS MiB | Completed tasks | Peak live tasks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4.1 DP4 | 4 | 6.903340 | 27.500510 | 20.943 | 132.656 | 1,220,652 | 14,552 |
| Qwen3.8 | 1 | 10.806917 | 10.806917 | 3.777 | 48.359 | 275,183 | 2,116 |
| Qwen3.5 | 1 | 2.533396 | 2.533396 | 1.510 | 52.250 | 102,178 | 2,986 |
| Qwen3.6 | 1 | 6.256939 | 6.256939 | 1.635 | 52.328 | 115,228 | 2,964 |
| Kimi K3 | 4 | 49.979610 | 199.918440 | 39.073 | 278.766 | 1,994,888 | 32,496 |
| GLM5.3 | 1 | 24.655575 | 24.655575 | 2.577 | 61.156 | 111,452 | 4,088 |

All modeled graphs end with zero live tasks and programs. Qwen3.6 retains
`2.9802322387695312e-08` bytes of aggregate allocation arithmetic residue;
the other five report zero. The raw value remains visible in the report.
Successful worker peaks stay below 512 MiB. Interrupted workers do not have
a retained final RSS or graph result, so those values are unknown.

V4 prefill reached its 60-second per-probe limit. Qwen3.8 prefill was stopped
after 50.473 seconds when the remaining campaign budget expired; this does
not establish that it would fail a fresh 60-second budget. Qwen3.5, Qwen3.6,
Kimi and GLM prefill were not started. The result therefore does not establish
fast complete prefill execution across the six models.

No historical baseline is numerically paired with this campaign. Previous
reports used different runtime or effective hardware/recipe inputs. Lower
isolated totals from those reports cannot be claimed as measured improvement
or as a matched host-speed comparison.

## Separate bounded V4 prefill diagnosis

After acceptance finished, one independent 15-second cProfile window inspected
the same committed V4 prefill graph. It ended as
`diagnostic_budget_exhausted` after 15.005 seconds including interrupt handling.
It is not a completed prediction and does not change the acceptance timeout.
Profiler overhead prevents treating its time as an uninstrumented benchmark.

The sampled work was still inside the first complete-graph `Engine.advance`;
isolated operator analysis had not started:

| Function | Calls | Self seconds | Cumulative seconds |
| --- | ---: | ---: | ---: |
| `graph_costs` | 1 | 0.002 | 14.742 |
| `Engine.advance` | 1 | 0.350 | 13.726 |
| `Engine._settle` | 1,973 | 0.630 | 11.634 |
| `PipelineProgram.issue` | 39,383 | 0.130 | 9.137 |
| `SteadyMatmul.build` | 39,383 | 0.630 | 3.791 |
| `Engine.submit` | 381,110 | 2.414 | 4.479 |
| `coupled_rates` | 328 | 0.592 | 1.084 |
| `MappedMatmul.select` | 164 | 0.009 | 0.365 |

Cumulative times are nested and must not be added. This window identifies
per-output-tile construction and event turnover as the dominant remaining
host work. It does not support blaming the timeout on isolated mapping
screening or source attention. The compact K recurrence alone does not bound
the number of output tiles; further batching must preserve their work,
resource ownership, overlap and release dependencies.

Retained evidence:

- `data/verification/model-convergence-steady-20261029.json`
- `data/verification/steady-v4-prefill-diagnostic-20261029.json`

The census-driver checks passed 13 tests. Separate matrix acceptance covers
the 21/30.1-ms ranking counterexample, short-tail overlap, finite buffer
admission, owner-local contention and quantized grouped output traffic.

## Follow-up after compressing owner output tiles

The diagnostic led to bounded owner programs that count the at-most-four
output-tail geometries, preserve their nominal serial dependency spans and
carry their actual remaining work in one resource-coupled body per owner.
`compact_output_tiles=False` retains the individual serial tile control.
An independent 240-case acceptance compared the two: 160 single-owner cases
preserved nominal spans within `1.01e-11` ms, and 80 multi-owner cases preserved
physical service, input readiness, selected buffer capacities and zero live
tasks/programs. Contended multi-owner spans are still a fluid approximation.

The fresh campaign used the same case inputs, balanced routing and declared
recipes. It selected only V4 prefill and the six decode probes, with 60 seconds
per probe and 90 seconds total. No original timeout result was overwritten.
All seven workers imported package SHA-256
`9aa2fdb1a4ee46bbe3184ce565b72cfa38a49d3bb3672b2058d55d65c11d46b5`.

| Probe | Device graph ms | Isolated sum ms | Worker seconds | Peak RSS MiB | Completed tasks | Peak live tasks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V4.1 prefill, four replicas | 59.700025 | 209.910762 | 16.206 | 130.125 | 420,736 | 14,528 |
| V4.1 decode, four replicas | 6.903340 | 27.500510 | 14.308 | 130.641 | 378,600 | 14,496 |
| Qwen3.8 decode | 10.806917 | 10.806917 | 2.194 | 47.250 | 74,491 | 1,990 |
| Qwen3.5 decode | 2.533396 | 2.533396 | 1.416 | 52.359 | 56,535 | 2,944 |
| Qwen3.6 decode | 6.256939 | 6.256939 | 2.547 | 52.406 | 94,673 | 2,944 |
| Kimi K3 decode, four replicas | 49.979610 | 199.918440 | 45.410 | 228.719 | 1,099,968 | 32,032 |
| GLM5.3 decode | 24.655575 | 24.655575 | 2.659 | 61.797 | 110,832 | 4,088 |

All seven graphs complete with zero live tasks/programs. The Qwen3.6 allocation
arithmetic residue is unchanged; the other completed probes report zero.
Six decode inputs pair exactly with the initial campaign: semantic work and
executed service are unchanged within the retained arithmetic tolerances.
Isolated and graph durations differ only by floating-point summation.

V4 decode's completed tasks fall from 1,220,652 to 378,600. Kimi's fall from
1,994,888 to 1,099,968, but its observed worker wall time increases from 39.073
to 45.410 seconds on the active development host. The result supports a reduced
event count and successful bounded V4 prefill execution; it does not establish
a host-speed improvement for every model. Kimi remains a significant host-cost
case. The five other model prefill probes were not rerun in this follow-up.

The full follow-up, matched-input checks and independent mechanism acceptance
are retained in `data/verification/model-convergence-steady-owners-20261029.json`.
