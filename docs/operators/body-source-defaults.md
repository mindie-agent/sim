# Body source and resource defaults

Body analysis now selects the existing rope-free indexed-attention pipeline
for BF16 latent512/rope0 and the packed INT8 adaptation for latent512/rope0.
INT8 requires the declared paired AIC/AIV owners and enough per-AIV scratch.
The rope64 BF16 path is unchanged. These remain analytical schedules; selecting
them does not identify a captured binary or establish its valid index values.
Explicit caller, recipe and program selections retain priority.

The GDN caller selects the existing `arch35_fp32` recurrence on A5 only inside
its FP32-state/BF16-input, q≤8, head and resident-UB domain. The shared caller
continues to work analytically outside that recurrence domain, with the choice
reported. There is no implemented `arch22_fp32` GDN recurrence to select on A3;
the A5 kernel and primitive parameters are not relabeled as A3 measurements.

`Ascend950DT_9582` now carries the independently identified BF16 scalar-store
service from `data/observations/gating-probe-20261028/native-queue-holdout/`:
88,397,790.055 stores/s/AIV, 8 outstanding requests, 128-byte address granules,
91.25 ns response and 7 ns row control. The original first24 training rows and
heldout comparison remain separate. These parameters describe the retained
scalar UB-to-GM program, not a kernel startup time or a model residual.
Other SKUs do not inherit this calibration.

The generic device `store_queues` declaration is keyed by the issue resource.
Its entry contains `element_bytes`, `outstanding`, `granule_bytes`,
`response_ms` and optional `row_control_ms`. Partial overrides merge fields
inside that queue. `scalar_store_issue` uses **stores**, never bytes. The GDN
default retains a hardware reference in its operator declaration and resolves
the queue from the current execution context. Repricing an already-built graph
therefore honors hardware response changes; explicit runtime queue parameters
remain explicit overrides. Structure export does not run the timing engine or
put queue response latency into declared physical work.

Independent Engine controls retained in
`data/verification/body-source-defaults-independent-review-20261030.json`
exercise issue-rate scaling, response overrides on a reused target, immutable
inputs, hardware roundtrip and explicit runtime priority. Issue-rate factors
0.01/1/100 produce 54.508/4.459/4.448 µs for the same 48 BF16 stores; changing
only response to 1 µs produces 48.079 µs on both a reused target and a newly
built graph. These are resource-response controls, not whole-body accuracy.
