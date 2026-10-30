# Deployment and bandwidth generalization

The local campaign uses the public serving API on nine complete catalog models:
Qwen3.5-35B-A3B, Qwen3.6-27B, Qwen3.8-27B, GLM-5.3, DeepSeek V4 Flash,
V4 Pro, V4.1 Flash, Kimi K3 and MiniMax-M3. Its two hardware declarations are
Ascend910_9362 and Ascend950DT_9582. Each is evaluated with colocated and
prefill/decode pools, on single-node and two/four-node placement candidates.
No model layers or vocabulary dimensions are reduced. MiniMax retains its
partial coverage and unmodeled features in every successful prediction.

The campaign is an execution and sensitivity test. It supplies no new
observations and does not establish measured forward or TPOT accuracy.
Decode body-forward comparison is a separate boundary; this campaign includes
request prefill, decode, admission, device commits and PD cache handoff.

## Run the campaign

```sh
.venv/bin/python tools/deployment_generalization.py \
  --budget 600 --run-timeout 30 \
  -o /tmp/deployment-generalization.json
```

The default workload is one request, 128 prompt tokens and two output tokens.
This small request exercises full model geometry and both phases; it is not a
high-concurrency throughput result. Use `--prompt-tokens` to change the prompt.
The ordinary probability routing scenario remains explicit; this serving
campaign does not assert forced equal expert counts for a single-token batch.

All 72 baseline schemes receive priority. Each completed baseline then receives
four independent `0.5x`, `1x`, `2x` scans; the baseline is reused. The complete
campaign contains up to 648 executions. Every child has a 30-second limit,
execution is serial, and the total campaign budget is 600 seconds by default.
Budget-exhausted rows remain `not_run`. Timeouts and exceptions retain their
status and details; the CLI returns exit 2 for incomplete or invalid campaigns.

Useful bounded subsets:

```sh
# Baselines across all 72 schemes, without resource sweeps.
.venv/bin/python tools/deployment_generalization.py --axes \
  -o /tmp/deployment-baselines.json

# Four-axis scans on a selected full model and actual cross-node deployment.
.venv/bin/python tools/deployment_generalization.py --models qwen35 \
  --hardware a3 a5 --layouts multi --deployments colocated pd \
  -o /tmp/qwen35-deployment-sensitivity.json
```

## Capacity and placement

A3 single-node has 16 dies; A5 single-node has eight. Multi-node candidates use
two or four such nodes. The planner checks native catalog geometry, declared
SKU memory and the ordinary weight/workspace/KV ledger. It tries W8A8, W4A8 and
BF16 declarations, selecting one feasible precision and parallel plan before
any bandwidth scan. The selected precision is a synthetic supported deployment
choice, not a claim about a historical checkpoint.

MoE candidates first distribute experts across DP/TP, then try pipeline
partitioning. Dense models use EP=1 and prefer pipeline partitioning so that
multi-node cases carry actual inter-stage traffic. Every candidate preserves
real SKU capacities and full layers. A model which cannot fit the declared
single-node capacity remains explicitly infeasible; the tool does not invent
additional device memory to make all combinations pass. PD uses two separately
accounted pools and reports cache handoff bytes and transfer resource service.

## Four independent resource axes

| Axis | Changed capacities | Preserved |
| --- | --- | --- |
| Network | Inter-node link capacity; PD connector bytes/s | Intra-node/package links, latency, tensor bytes |
| HBM | `rates.hbm` | Device memory capacity, copy-pipe capacity, arithmetic peaks |
| AIC | Every declared matrix precision peak; MTE1, MTE2_AIC, Fixpipe | HBM and AIV capacities |
| AIV | Vector, declared `reg_*` issue/lane resources, MTE2_AIV, MTE3 | Matrix peaks and HBM |

Inherited service modes follow their physical pipe's scaling through the
normal device-input merge. Explicitly supplied mode rates retain precedence.
The campaign records effective normalized device rates, modes, owners, capacities
and links for each run, rather than only recording requested factors.
Network overrides use unambiguous bytes/second convenient fields. Legacy
`transfer_channels.*_gbps` names retain their historical GB/s interpretation.

Comparison requires identical model/workload/runtime declarations, stable
runtime package identities, completed requests, released tasks/programs,
unchanged semantic work where exposed, and conserved PD handoff volume. The
reported timing is `makespan_ms` on the actual serving timeline. A faster axis
must give a nonincreasing time within floating-point tolerance. A plateau is
valid when that resource is unused or not on the critical path. `regression`
means a timing inversion requiring investigation; it is never suppressed.

Raw physical traffic can change when an explicit mapping selector selects a
different legal tile under different capacities. That is retained in executed
service; semantic work and tensor geometry must still agree. Resource service
expressed in matrix `peak_ms` or converted service-mode units is not compared as
if it were native FLOPs or bytes.

## Focused acceptance

`tests/test_deployment_and_bandwidth_generalization.py` executes in memory:

- Colocated and PD arithmetic, vector and HBM bottleneck controls respond strictly
  to all three factors while preserving semantic work and completing requests.
- A PD network-limited control preserves the exact 512-byte cache handoff.
- A real cross-node allreduce through ordinary `predict` responds to link
  capacity changes with conserved collective bytes.
- A single-device colocated network scan correctly plateaus.
- A3/A5 effective pipe rates, inherited service modes and inter-node capacities
  change on their selected axis while memory capacity and ownership remain fixed.

The focused set passes 12 tests in 0.12 seconds. Full-model campaign results are
retained separately with their actual source identity, duration, failures and
memory measurements. Focused success does not imply complete 648-run coverage
or a measured error below 15%.

## Completed full-model controls, 2026-10-30

Two stable campaigns execute package SHA-256
`b926386680ea1da8c9d24855b855a3a2451b6de8786300d45ff530ffdef72e80`
(214 Python/JSON package files). Each worker records equal before/after package
identities; all runs served the request and released live tasks/programs.

- `data/verification/deployment-generalization-dense-20261030.json`: complete
  Qwen3.6/Qwen3.8 x A3/A5 x single/multi x colocated/PD, **16/16 completed**
  in 52.512 seconds; maximum worker RSS 75.22 MiB.
- `data/verification/deployment-generalization-qwen35-axes-20261030.json`:
  complete Qwen3.5 x A3/A5 x multi-node x colocated/PD, four axes and three
  factors, **36/36 executions completed**, **16/16 strict-response sweeps**
  in 136.929 seconds; maximum worker RSS 100.19 MiB. Semantic work and PD
  handoff volume remain unchanged within each sweep.

These are ordinary public serving executions, with full native model geometry,
128 prompt tokens and two output tokens. The fixed W8A8 declaration is an
explicit deployment scenario. The results are not measured timing accuracy.

| Qwen3.5 deployment | Axis | 0.5x makespan ms | 1x makespan ms | 2x makespan ms |
| --- | --- | ---: | ---: | ---: |
| A3 multi, colocated | Network | 42.916 | 42.282 | 41.965 |
| A3 multi, colocated | HBM | 43.660 | 42.282 | 41.783 |
| A3 multi, colocated | AIC | 45.067 | 42.282 | 41.205 |
| A3 multi, colocated | AIV | 42.450 | 42.282 | 42.227 |
| A3 multi, PD | Network | 48.911 | 45.282 | 43.467 |
| A3 multi, PD | HBM | 46.660 | 45.282 | 44.783 |
| A3 multi, PD | AIC | 48.067 | 45.282 | 44.205 |
| A3 multi, PD | AIV | 45.450 | 45.282 | 45.227 |
| A5 multi, colocated | Network | 19.200 | 18.777 | 18.566 |
| A5 multi, colocated | HBM | 19.610 | 18.777 | 18.413 |
| A5 multi, colocated | AIC | 19.985 | 18.777 | 18.205 |
| A5 multi, colocated | AIV | 18.946 | 18.777 | 18.712 |
| A5 multi, PD | Network | 25.195 | 21.777 | 20.068 |
| A5 multi, PD | HBM | 22.610 | 21.777 | 21.413 |
| A5 multi, PD | AIC | 22.985 | 21.777 | 21.205 |
| A5 multi, PD | AIV | 21.946 | 21.777 | 21.712 |

A3 and A5 node sizes differ, and the selected ranks/parallel configuration are
recorded in each scheme. The table demonstrates within-scheme resource response;
it must not be read as an isolated A3-to-A5 chip speedup.

The earlier exploratory baseline campaign incorrectly assigned EP>1 to dense
models. The planner now derives EP=1 for them and retains cross-node PP; the
stable 16-run campaign above replaces those invalid planning rows. The original
failure remains in the exploratory report. Full 648-run verification has not
been completed.

### All 72 baseline declarations have a result

The consolidated index is
`data/verification/deployment-generalization-20261030.json`. It links the
original exploratory report, the corrected dense rows, the bounded completion
pass and the six successful early runs repeated to capture individual runtime
identities. Every original receipt is retained. All **54 completed predictions**
now have matching per-worker before/after identities for package `b9263866...`.
The other **12 predictions reached the 30-second worker limit**, and **six
single-node deployments do not fit their declared physical capacity**. There
are no remaining budget-unrun baseline rows.

| Full model | Modeled | 30 s timeout | Capacity infeasible |
| --- | ---: | ---: | ---: |
| Qwen3.5 | 8 | 0 | 0 |
| Qwen3.6 | 8 | 0 | 0 |
| Qwen3.8 | 8 | 0 | 0 |
| GLM-5.3 | 6 | 2 | 0 |
| DeepSeek V4 Flash | 8 | 0 | 0 |
| DeepSeek V4 Pro | 0 | 6 | 2 |
| DeepSeek V4.1 Flash | 8 | 0 | 0 |
| Kimi K3 | 2 | 2 | 4 |
| MiniMax-M3 | 6 | 2 | 0 |
| Total | 54 | 12 | 6 |

The six infeasible cases are V4 Pro on single-node A5, plus Kimi K3 on
single-node A3 and A5, each with colocated and PD forms. The planner tries
compatible representations and valid placements; it retains the unsuccessful
weight/workspace/cache checks. These capacity limits are not model-parser
rejections.

The timeouts are V4 Pro's six capacity-feasible cases, GLM A3 multi-node,
Kimi A3 multi-node and MiniMax A3 multi-node, each affected multi-node case in
both serving forms. The probability-routing grouped family still uses its
existing tiled aggregate program; directly assigning the mapped steady recipe
to that domain is unsupported. These CPU execution limits require a separate
program-domain and task-expansion investigation. They are not reported as
measured performance errors or successful simulation results.

The 15-run completion pass took 298.389 seconds. Across completed baseline
workers with memory telemetry, the largest peak RSS was 150.95 MiB. The initial
600-second exploratory campaign spans a runtime source change; it is preserved
as exploratory evidence. The final successful rows are covered by stable worker
identities and the independent reruns noted above. All four-axis sweeps remain
the stable, unchanged Qwen3.5 campaign; 648-run all-model sensitivity acceptance
is still outstanding.
