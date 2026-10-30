# MindIE Sim

Hardware-neutral causal performance modeling. Start with operator arithmetic,
tensor traffic, dependencies and isolated model cost. Add request scheduling to
evaluate throughput, capacity and service latency constraints such as TTFT/TPOT.

Simulation, primitive calibration and independent validation have separate APIs.
Prediction does not require measurements. A `modeled` result means the declared
simulation completed; it does not assert real-device accuracy.

```sh
uv venv
uv pip install -e '.[test]'
.venv/bin/mindie-sim predict examples/quickstart.json -o /tmp/prediction.json
.venv/bin/mindie-sim inspect examples/quickstart.json
.venv/bin/mindie-sim operators examples/quickstart.json --phase decode --tokens 4 --sequences 2 --scope model
.venv/bin/mindie-sim graph examples/quickstart.json --phase decode --tokens 1 --scope model
.venv/bin/mindie-sim study examples/studies/resource-tradeoff.json -o /tmp/study.json
.venv/bin/mindie-sim predict examples/disaggregated.json
.venv/bin/mindie-sim catalog
```

The quickstart uses illustrative device parameters. An Ascend SKU adapter is
included, while GPUs and other devices can declare the same neutral resource
schema directly. No weights, CUDA context or NPU context are needed to simulate.

The execution engine supports finite pipelines, resource sharing, allocation
lifetimes, dependencies and stream order. Serving adds arrivals, chunked prefill,
integer speculative commits, capacity admission, recomputation, PP microbatches
and causally scheduled P/D handoff. Model blocks and implementation recipes are
registries rather than model-name timing branches.

Default recipes are analytical assumptions: tiled GEMM with configurable reuse,
resource envelopes for unspecialized operators, and representative local shards
per pipeline stage. The retained measurements are independently indexed; they are
not silently reused as whole-model timing constants. Whole-model accuracy across
the catalog has not been established.

`operators` reports the sum of isolated device programs using the same engine.
The default `--scope body` includes the transformer body; `--scope model` also
includes embedding, final normalization, selection and the output head through
the public frontend. The sum excludes host submission, scheduling and draft
generation, and is distinct from a concurrent graph's critical path.

`graph` composes the device graph through that engine, including streams,
contention and complete cross-DP expert cohorts. It separates graph elapsed
time from isolated sums and precedence/resource lower bounds; those bounds are
not measured critical paths or achievable savings. Target capture scope is
inherited and media inputs are explicit. See [graph analysis](docs/graph-analysis.md).
[Family declarations](docs/operator-family-selection.md) connect selected
source implementations to ordinary model frontends and the same recipe book.

`study` evaluates explicit candidates across workload probes, exposes isolated
cost concentration, and optionally runs graph composition and actual-request
serving constraints. Only equal-work, equal-boundary successful candidates share
a ranking; failures remain visible. See the [study workflow](docs/studies.md).
Predicted ordering is separate from the public `accuracy` command's independent
measurement comparison. [Capture adapters](docs/capture-adapters.md) retain the
original call, replay and kernel-sum boundaries of historical measurements.

The [operator investigation](docs/operator-mechanisms.md) records source-derived
repairs and unresolved raw-device differences. The scalar-store queue has
independent address/shape measurements; its multi-unit error remains visible. The old checkout and raw captures
remain preserved while replacement accuracy is incomplete.

See the [architecture](docs/architecture.md), [input reference](docs/inputs.md),
[extension guide](docs/extending.md), [observation workflow](docs/observations.md),
[migration record](docs/migration.md) and [verification scope](docs/verification.md).
