# Compact matrix K pipeline

Select `{"kind": "steady_matmul"}` under `runtime.kernels.programs.matmul`.
Balanced expert groups select the same recipe explicitly through
`runtime.kernels.recipes.balanced_grouped_gate_up` and `balanced_grouped_down`.
Source-bound kernel planners retain precedence.

When a device declares separate AIC and AIV read pipes, matrix operands use
`mte2_aic` and final row/column scales use `mte2_aiv`. This applies consistently
to the tiled, hierarchical, mapped, phase and steady recipes and their mapping
bounds. An additional legacy `mte2` declaration does not override these physical
owners or receive a duplicate charge. A device with only `mte2` still uses that
shared pipe. HBM service remains shared and is charged once at its own boundary.

The coexistence control uses one INT8 A/B pair and two FP32 scales. With AIC
1 B/ms and AIV 2 B/ms, the dependent reads require 2 + 4 = 6 ms, regardless of
the unused generic pipe rate. Independent 0.5/1/2 scans produce 8/6/5 ms for
AIC and 10/6/4 ms for AIV while preserving bytes and releasing buffers. These
are mechanism controls, not device calibration or a whole-model accuracy pass.

The finite mapping, independent matrix groups, physical FLOPs, packed operands,
scales and output work come from `MappedMatmul`. Each owner executes its output
tiles serially. Inside one output tile, `matrix_cycle` solves the actual finite
refill/operand/cube recurrence:

```text
load[k]    = max(load[k-1], finish[k-refill_slots]) + load_service[k]
operand[t] = max(operand[t-1], load[panel(t)], cube[t-consumer_slots])
             + operand_service[t]
cube[t]    = max(cube[t-1], operand[t]) + cube_service[t]
```

It normalizes the complete finite state and jumps only after proving a repeated
state translated in time. Binary integer time avoids approximate cycle matching.
The last nonuniform panel is advanced separately. A bounded search reports an
unsupported recurrence instead of silently substituting a serial or roofline
estimate. For three 1 ms stages, two refill slots and four panels, completion is
7 ms; collapsing the three stages into an unconstrained maximum would give 6 ms.

The executed Engine program reserves the selected finite L1/L0 buffers, runs the
first load, a resource-coupled body, the last compute, and the scale/output
epilogue. Body caps derive from remaining work divided by the dependency span,
with no observed duration, fitted multiplier or residual. First operand work
belongs inside that body: pulling it out prevents the next load from overlapping
and fails short-tail cases.

The uncontended single-output-tile recurrence is exact for this declared
mapping. External contention uses the ordinary Engine's proportional,
resource-coupled fluid progress. It does not reconstruct arbitrary discrete
in-flight tile scheduling. Output-tile serialization is also explicit; this
recipe is not advertised as identical to detailed mapped execution under every
multi-tile or contended workload.

By default, repeated output tiles of one owner are also compressed. Their at-most
four M/N tail geometries supply exact occurrence counts, physical work, and the
sum of nominal serial tile spans. The executed run reserves one tile footprint,
runs its first load, a coupled body and its final store, then releases storage.
This retains the nominal serial-output schedule while extending the declared
fluid contention abstraction across the owner run. Set `compact_output_tiles`
to `false` to retain the individually scheduled output-tile control.

Engine task count is independent of both K-panel and output-tile counts per
owner. The geometry histogram uses constant storage and linear arithmetic in
output tiles. Live state is bounded by owners. Mapping-cycle arithmetic uses an LRU
of at most 1024 entries.
Finite allocations still constrain concurrent invocations and are released at
tile completion.

Validation covers the 21 ms / 30.1 ms hardware ranking counterexample,
load-heavy short tails, grouped geometry, quantized output scales, tight-buffer
serialization, owner-specific contention, and ten million K panels. Independent
checks compare the scalar recurrence against detailed `RefillPipeline` execution;
they establish the declared mechanism, not real-device accuracy.
