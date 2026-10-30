# Matrix phase execution and full-graph cost

`phase_matmul` is an explicit resource-phase alternative to `mapped_matmul`.
Both select from the same finite mapping candidates using the same physical
geometry, resource rates and capacity screening. The phase recipe preserves
the selected mapping's per-owner work but deliberately changes its temporal
detail. It is useful when full-model screening would otherwise expand millions
of fine matrix events.

The existing default and detailed recipes are unchanged. Source-bound operators
retain their explicit planner precedence; this option does not replace a
selected native implementation or infer its dispatch.

## What the phase recipe models

Each active physical owner executes a finite sequence on the ordinary Engine:

1. Shared HBM and private operand-read service.
2. Inner operand transfer through MTE1 when that role exists.
3. Matrix work and any ordinary retained computation.
4. Declared final row/column scale fetch and vector epilogue, when present.
5. Output transfer and HBM store.

Independent owners may overlap and contend on the existing physical resources.
The phases within an owner are serial. This omits detailed K/micro-pipeline
overlap and output-tile interleaving. The resulting duration is an approximation,
not a reconstruction of a vendor schedule or a strict upper bound under
arbitrary external resource contention.

Macro/micro tile tails, equal independent group axes, ideal shared-HBM reuse
settings, private copies, widened inner operands and final scale/output work
are preserved by closed-form owner totals. No observation changes the work,
mapping or rates. The same mapping footprint screen checks declared tile
capacities. Aggregated phase traffic is never allocated as a fictitious giant
L1/L0 buffer, and phase reports explicitly identify that they do not simulate
dynamic tile residency.

For a single owner with six micro products, the independent control has 12 ms
of MTE1 service and 6 ms of matrix service. Detailed overlap finishes in 13 ms;
serial phases finish in 18 ms with identical work. That difference is expected,
and is tested rather than hidden by a coefficient.

## Why the explicit option was added

The retained V4 decode probe uses the complete first expert cohort:
DP=4, TP=1, EP=4, one query token per replica, context 2048, balanced integer
routing and model scope. Its frontend produces 6,592 operators in about 0.09 s.
The earlier full graph timed out at its 60 s budget.

A bounded cProfile run against the exact earlier runtime, revision `475a0b71`,
shows that the execution loop dominates. In 35 s it had submitted approximately
932,000 tasks; refill retirement/consumption, task submission and physical-unit
rate allocation are the major costs. A separate uninstrumented 55 s run had
completed 3,915,086 tasks but reached only 1.798002 ms of virtual time. Its
130.78 MiB peak RSS shows that this is event-expansion cost, not an unbounded
retained-history problem. The interrupted timestamp is not a graph prediction.

Arithmetic from the actual selected mappings establishes the size of the full
detailed matrix execution without expanding it:

| Selected matrix family | Occurrences | K refill panels | Base engine tasks |
| --- | ---: | ---: | ---: |
| Balanced expert gate/up | 160 | 1,382,400 | 5,564,160 |
| Balanced expert down | 160 | 691,200 | 2,803,200 |
| Q expansion | 160 | 204,800 | 1,454,080 |
| Grouped output B | 160 | 204,800 | 1,436,800 |
| Grouped output A | 160 | 163,840 | 1,152,000 |
| All selected mapped matrices | 2,060 | 3,154,736 | 15,215,768 |

The base task count includes loads, operand/matrix/retirement micro steps and
stores; wrappers and optional epilogues add more tasks. Expert gate/up and down
alone account for 55.0% of that lower bound. Merely increasing a wall budget
or optimizing a small Python helper would not address this modeling-detail
cost.

The exact owner-rate optimization is separate. Identical or disjoint ownership
partitions can perform independent max-min sharing without rebuilding a general
overlap constraint matrix. Partially overlapping and unpinned ownership still
uses the general progressive-fill calculation. Three representative 32-unit
arithmetic benchmarks improved by 2.66--3.37x; 1,000 deterministic mixed-cap and
overlap cases matched the previous arithmetic within relative tolerance 1e-12.
This is a helper benchmark, not a full-graph speedup claim.

## Explicit selection and retained execution

Set `runtime.kernels.programs.matmul.kind` to `phase_matmul`. Balanced experts
have explicit recipe bindings, so select their two phase recipes separately:

```json
{
  "programs": {
    "matmul": {"kind": "phase_matmul"}
  },
  "recipes": {
    "balanced_grouped_gate_up": {"kind": "phase_matmul"},
    "balanced_grouped_down": {"kind": "phase_matmul"}
  }
}
```

The parameters for candidates, inner tiles, slots and HBM reuse are the same as
`mapped_matmul`. Slot counts affect the selected mapping's footprint screen;
they do not imply that the phase execution retains a detailed buffer timeline.

The full unchanged V4 probe now completes when this phase candidate is
explicitly selected:

| Quantity | Result |
| --- | ---: |
| Complete device graph span | 9.633920909 ms |
| Sum of isolated programs across the four-rank cohort | 38.422834656 ms |
| Wall time, graph plus isolated analysis | 4.393 s |
| Peak RSS on the measured macOS host | 126.266 MiB |
| Completed engine tasks | 213,704 |
| Peak live tasks | 13,920 |
| Final live tasks / programs / allocated bytes | 0 / 0 / 0 |

The graph's executed service agrees with occurrence-weighted isolated service
to maximum relative delta 3.23e-14. Compared with the previously completed
detailed operator census, semantic work is exactly equal and physical service
agrees to relative delta 3.13e-14. The earlier detailed isolated sum was
27.287868240 ms. The phase sum is 40.8% higher because its per-owner phase
ordering omits pipeline overlap. This comparison measures a declared
approximation's effect; it is not measured performance accuracy. There is no
completed detailed full-graph span to compare against.

The census reproduction command is:

```sh
.venv/bin/python tools/model_convergence.py \
  --only v41-dp4-decode --level graph \
  --routing '{"policy":"balanced"}' \
  --matmul '{"kind":"phase_matmul"}' \
  --balanced-matmul '{"kind":"phase_matmul"}' \
  --probe-seconds 60 --total-seconds 65 \
  -o data/verification/model-convergence-20261029-v4-phase-graph.json
```

The retained execution used the existing committed-source export with six
explicitly hashed runtime overlays, excluding unrelated live profiler changes.
It is labeled as a working-source overlay, not falsely attributed to a commit.
The acceptance report contains those hashes and the full graph resource ledger.

Evidence:

- `data/verification/v4-graph-diagnosis-20261029.json`: interrupted profiles,
  finite expansion arithmetic, and owner-rate benchmarks.
- `data/verification/model-convergence-20261029-v4-phase-graph.json`: ordinary
  model-convergence workflow, declared overrides, mappings and cost breakdown.
- `data/verification/v4-phase-runtime-20261029.json`: complete graph work,
  finite execution state, wall time, RSS and source-overlay hashes.
- `tests/test_matrix_phase.py`: grouped/ragged arithmetic, private owners,
  output-scale ordering, explicit timing difference and K-independent task
  count. Together with ownership, engine, matrix mapping and epilogue controls,
  the affected scope passed 52 tests.

## Same-work six-model census

The subsequent census uses committed runtime `94ba5215`, with no working-source
overlay. It runs 12 operator probes in 15.647 s and compares them with the retained
`475a0b71` balanced mapped census; the expensive detailed side was not rerun.
Both sides use identical cases and integer balanced routing. Prefill is 256
tokens except Kimi and GLM at 252; decode is one query token at context 2048,
with two concurrent GLM sequences and one sequence for the others. Kimi and V4
retain their complete four-rank expert cohort.

| Probe | Detailed isolated ms | Phase isolated ms | Phase change |
| --- | ---: | ---: | ---: |
| Qwen3.8 prefill | 35.975729 | 66.979385 | +86.18% |
| Qwen3.8 decode | 10.757934 | 13.882690 | +29.05% |
| Qwen3.5 prefill | 12.435991 | 14.130999 | +13.63% |
| Qwen3.5 decode | 2.504517 | 2.665931 | +6.44% |
| Qwen3.6 prefill | 10.676194 | 14.646235 | +37.19% |
| Qwen3.6 decode | 6.165536 | 6.610140 | +7.21% |
| Kimi K3 prefill | 377.373892 | 571.253858 | +51.38% |
| Kimi K3 decode | 189.460492 | 208.389850 | +9.99% |
| GLM5.3 prefill | 48.978670 | 66.543365 | +35.86% |
| GLM5.3 two-sequence decode | 24.437411 | 25.931032 | +6.11% |
| V4 four-rank prefill | 206.138304 | 402.867840 | +95.44% |
| V4 four-rank decode | 27.287868 | 38.422835 | +40.81% |

All 12 pairs preserve semantic work exactly. Every physical resource's
occurrence-weighted service is also unchanged within the declared comparison
roundoff of relative 1e-12 / absolute 1e-9; the largest relative delta is
8.94e-13. The same-work test is distinct from a same-timing assertion.
`compare_runs` now reports physical-service deltas separately, so a mapping that
preserves FLOPs but reloads more bytes cannot be described as conserving physical
work. Its counterexample executes real changed-reuse programs.

The large prefill differences expose the cost of discarding overlap. Phase
analysis is therefore an explicit coarse screening choice. A decision sensitive
to overlap or a changed ranking still needs the detailed recipe. None of these
cost differences is measured accuracy, and the table is not a ranking of models
with different workloads.

`data/verification/model-convergence-20261029-phase.json` retains the 12
predictions, input fingerprints, source commit, physical service and comparison.
Its original execution-driver hash is preserved; the separately recorded
comparison-driver hash identifies the physical-work check added afterward
without rerunning predictions.

```sh
.venv/bin/python tools/model_convergence.py \
  --only qwen38 qwen35 qwen36 kimi-k3 glm53 v41-dp4 \
  --routing '{"policy":"balanced"}' \
  --matmul '{"kind":"phase_matmul"}' \
  --balanced-matmul '{"kind":"phase_matmul"}' \
  --model-prefill-tokens '{"kimi-k3":252,"glm53":252}' \
  --model-decode-sequences '{"glm53":2}' \
  --reference data/verification/model-convergence-20261029-balanced-mapped.json \
  --probe-seconds 60 --total-seconds 300 \
  -o data/verification/model-convergence-20261029-phase.json
```

The installed-wheel verifier additionally exercises `phase_matmul` through the
public `graph` CLI on the same quickstart geometry as its detailed mapped control.
It checks physical work equality, explicit approximation diagnostics, graph
bounds and complete execution cleanup. The final wheel from `c3da0893` passed
all 46 installed checks, including this phase check; its result is retained in
`data/verification/wheel-grouped-source-20261029.json`.

The final V4 run then imported that installed wheel under `python -I`, outside
the source checkout. It completed the same complete four-rank graph and isolated
analysis in **5.368 s**, with **125.125 MiB** peak RSS. Its graph span
**9.633920909 ms**, isolated sum **38.422834656 ms**, complete physical service
ledger and 213,704 completed tasks are unchanged from the earlier working-source
run. Effective case fingerprint, geometry and replicas also match. Peak live
tasks are 13,920; final tasks, programs and allocated bytes are all zero.

`data/verification/v4-phase-delivery-20261029.json` contains the final installed
import path, source commit, wheel/package hashes, RSS, actual graph resource
ledger and controls against the preserved earlier run. The package content hash
is `de79371d79ca086051b71be95ac206022fe2a5d4f4789b34b2d9a1a0f7d3ca68`;
the wheel SHA-256 is
`dfe2b70d00f7b679af7831b28eb81b8a002459c92f9b11ea7359f10880ada92f`.
This final packaging/execution acceptance does not change the explicit phase
approximation or establish measured device accuracy.
