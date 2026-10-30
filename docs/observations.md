# Observations, calibration and validation

A measurement is a source record. Whether it is appropriate for a particular
primitive or a service comparison is a separate, explicit operation. Prediction
never asks for an evidence-completeness certificate.

## Capture sources

Raw captures are read from their original external locations, including
`/Volumes/data/mindie-sim-data/`. Small regression inputs belong in
`tests/fixtures/`; active source contracts belong in `data/observations/`.
The repository does not retain a second historical implementation, migration
snapshot tree or obsolete calibration campaign. Source paths and SHA256
identities in current protocols refer to measurements, not latency overrides.

`ObservationIndex` streams hashes and metadata. The historical profiler database
is opened read-only; `ProfilerCatalogue` exposes signatures and raw event samples.
Its potentially unindexed event queries have a bounded progress-handler deadline.
An interrupted query is an error, not an empty measurement. Whole-file JSON reads
are not used for large profiling or service captures.

```sh
mindie-sim observations /Volumes/data/mindie-sim-index/observations.sqlite --find qwen
mindie-sim profiler-rows /Volumes/data/mindie-sim-data/profiling-reuse-20261023/kernels.sqlite --limit 10
```

## Operator comparisons

`observations.kernels.kernel_rows` streams native profiler CSV, preserving tensor
shapes, dtype/layout, block counts, core type and pipe activity. Epoch timestamps
are parsed with decimal arithmetic into integer nanoseconds. `N/A` shapes remain
unknown; serving batch size is not substituted for a missing operator shape.

`tools/compare_operators.py` compares selected line windows with independently
generated programs, checking matrix operand/scale contracts and reporting
unmapped rows. Recurrence sequence-length values and slice offsets are not
available in the selected CSV schema; matching shapes alone does not establish
those fields or vendor tiling. Neither timing nor a successful match enters
calibration automatically. See `docs/operator-mechanisms.md` for actual windows,
source provenance, independent captures and unresolved discrepancies.

`tools/modeling_priority.py` reads these reports and `operators` outputs without
rerunning prediction. It ranks repeated costs and conditional repeated residuals,
keeps signed and absolute totals separate, and excludes ambiguous tensor matches
from aggregation. Each report keeps its own denominator; a selected-family sum
is not silently normalized as a full-model result. Unpaired costs remain
unscored. See `docs/modeling-fidelity.md` for refinement policy and reproduction.

Native GDN tensor ABI also cannot identify the sequence-length array: two
partitions can have equal token/sequence/chunk counts while executing different
tail work. Reports retain declared sequence lengths separately. Native FN
convolution comparisons use active tensor descriptors, displaying the captured
full cache shape while excluding inactive cache capacity rows from matching.
Run mode, activation, cache-index values and initial-state flags remain declared
conditions unless separately observed.

The scalar-store investigation retains original compiler artifacts, a distinct
primitive, independent address/unit perturbations, and a frozen train/heldout
split. `fit_store_queue.py` identifies finite-queue service parameters only from
primitive training rows. `verify_store_queue.py` scores untouched shapes through
the production engine. Seven single-unit cases are within 3%; three 64-unit
cases still underpredict by 5-13%. These limits accompany the declared parameters;
no whole-GDN, model-body or serving time enters their fit. See
`docs/scalar-transfers.md` for exact sources and boundary limits.

## Primitive calibration

`PrimitiveKey` identifies hardware, implementation revision, kernel/recipe, phase,
precision, layout, semantic/local/padded/launch shape, physical work and metric
scope. The key never includes a model name or a serving case identifier. JSON
kernel bindings are included so that tile-policy changes do not reuse a timing
for a different implementation.

Use `prediction.cost.primitive_key(case, op, replay_scope)` to construct the
current identity from an inspected operator. Bind measurements only after
checking the raw capture's implementation and boundaries. A key produced by the
simulator is an identity to match, not proof that a raw row matches it.

`calibrate STORE SAMPLES_JSONL` accepts one record per line:

```json
{"key":{"hardware":{},"implementation":{},"kernel":"declared_kernel","dtype":"bf16","shape":{},"scope":"device_kernel"},"duration_ms":0.01,"run_id":"run-1","repeat_index":0,"source":"capture.csv#row=17","split":"train"}
```

The empty maps in this syntax example must be replaced with the complete observed
condition. The store computes a median of trial medians, preserves source rows,
and refuses conflicting repeats. Only `split: train` is eligible for prediction;
`heldout` samples remain separate. There is no neighboring-shape substitution.

`bind-profiler CATALOGUE STORE BINDINGS_JSONL` imports existing raw profiler events.
Each binding contains `variant` (the historical hex variant ID), `key`, `run_id`
and `split`. The old database's eligibility flags are not trusted as new bindings.
Bounded imports refuse oversized variant groups rather than silently truncating.

## Resource-rate calibration

`fit-rate` consumes `{ "samples": [...] }`, with each sample containing
`work_bytes`, `duration_ms` and `split`. It fits nonnegative setup plus work/rate
on training observations at distinct work sizes. Heldout rows are scored only
against the frozen fit. Put the resulting resource rate into the declared machine
input with provenance. Whole-model TTFT/TPOT and residuals are not resource rates.

The RTX 4090 experiment in `data/measurements/gpu/` measured a copy primitive using
the CUDA Driver API, with at most 256 MiB allocated and five trials per size.
64/96/128 MiB sizes were training inputs; 112 MiB was held out. The resulting
memory-service rate is 785.8399066142312 GB/s, counting read plus write traffic.
Five heldout engine predictions differ from measured times by -6.77% to +2.24%.
The experiment validates this primitive under this protocol only. It does not
validate GPU GEMM, draft kernels or end-to-end model serving.

`tools/measure/cuda_probe.py` and `copy.ptx` retain the actual bounded probe. No
PyTorch installation was changed after the host's existing Torch import failed.
The raw receipt records device/driver identity and preflight memory observations.

## Work-dependent primitive services

`fit-service` adds an optional one-dimensional work-response curve, with fixed
implementation, dtype, layout, ownership and measurement boundary. Only training
trials supply its knots/startup. Heldout timings are reported separately. The
selected `resource_service` recipe converts the response into an invocation
supply limit on the same physical engine; it does not overwrite global memory
bandwidth or calibrate unrelated kernels.

`resource-services.md` describes the packet, recipe binding, phase composition and
the retained 4090 replay. The 112 MiB heldout engine estimate is 0.300842663 ms
against median 0.301820000 ms; individual trial errors remain -6.162% to +2.904%.
This reuses the original capture and one heldout size. It is not a new hardware
run or evidence of full GPU/model accuracy. Analytical prediction remains
available without any curve.

## Independent service validation

`validate` reads the raw service capture as a stream, retaining only scalar metrics.
It compares matching statistics without feeding the observation into prediction.
The retained service capture format is measured at `client_stream`; the caller
cannot relabel it to `device_token_commit` through a command-line flag.

The historical `bench_stream` producer calls its mean nonempty SSE chunk interval
`tpot_ms`. A speculative chunk can carry several tokens. The importer preserves
that value under `reported_metrics` and computes per-token TPOT from each successful
request's `decode_ms / (usage.completion_tokens - 1)`. It never substitutes requested
output length or chunk count when token usage is absent. End-to-end latency uses
first-text latency plus first-to-last-text elapsed time, leaving the original
stream-finish statistic available under `reported_metrics`.

These normalized statistics use constant-memory count/mean/min/max accumulators;
chunk-interval medians are not reused as per-token medians. Output TPS uses actual
reported completion tokens over the captured wall window, only when every
successful request has usage. For the retained Qwen3.8 example, the original
25.243 ms chunk interval becomes 6.400 ms per token. This changes interpretation
of the historical measurement, not simulation parameters.

A boundary mismatch is explicit (`comparable: false`). Displayed differences then
include client transport/frontend contributions and must not be interpreted as a
like-for-like accuracy score. Unpaired values remain absent; missing observations
are never assigned zero error or used to suppress a simulation result.

## Historical operator-call events

`observations.historical.call_observations` reads the retained GDN measurement
capture through `data/observations/protocols/gdn-call-protocol.json`. The protocol names the
capture hash, producer hash, hardware metadata source and the complete timed
callback. Reading is streaming, with one active condition; the inventory retains
only bounded previews and counters. The original capture remains in place.

Each observation preserves its actual event samples and sequential trial/repeat
identities, conditions, raw environment, canary result and source-row identity.
It reports both the historical upper-central-order statistic and normalized
sample/trial medians. An event is one call within a trial, not an independently
restarted experiment. Failed calls remain failed observations without a zero
duration. Old intercepts, slopes and whole-model terms are never read by this
adapter, and historical fit splits are not reused as new train/heldout choices.

`compare_call` requires an explicit observation ID, identical declared identity
and the same call boundary. A mismatch produces no timing error score. In
particular, the old recurrent decode callback includes Q/K normalization and
metadata tensor construction/transfer before the recurrent call; its event
duration cannot score the recurrent kernel alone. The original prefill scan
canary failure remains visible even though a later separate canary corrected
the reference. These are observation interpretation limits, not simulation
admission rules. Reproduction and actual reuse counts are in
`operators/historical_absorption.md`.

## Activation and grouped-matrix protocols

`observations.captures.iter_observations` now imports the original activation
calls/bursts and grouped eager/graph replay events into the generic accuracy
sample contract. It also streams selected serving profiler traces into grouped
kernel sums, retaining missing route values instead of guessing them.
`observations.comparison.comparison_report` exposes descriptive same-boundary
residuals separately from independent heldout validation. Original unassigned
samples keep that role; no importer promotes historical rows to heldout.

The original captures supply 24,300 activation event/burst samples and 1,200
grouped probe events. Two serving captures yield 18 kernel-sum samples and three
explicitly excluded contaminated intervals. These counts describe imported
observations, not calibrated or accepted cases. Protocols, runnable import
commands, actual reports and remaining matching limits are in
`capture-adapters.md`.
