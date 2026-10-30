# Retained activation closures and independent call models

The activation producer is preserved at:

```text
/Users/maoxx241/code/mindie-agent/.workstreams/composability/scripts/bench_activation.py
SHA-256 ea76c64cd935e2d35c849f957f2179701d81d72c951a738a430260de4b251136
```

The reviewed functions are `make_cell`, `measure` and `measure_burst`.
`data/observations/protocols/activation-call-protocol.json` retains the producer identity,
both complete capture paths and their content hashes. Source was read as text;
the producer was not imported or executed.

## What the event actually encloses

The producer allocates and zeroes input tensors before timing. The single-call
method records an NPU event, invokes one closure, discards its return and records
the second event. It performs ten warmup calls outside the event interval.
The burst method uses three warmup calls and an event pair around the recorded
number of back-to-back invocations. It does not chain the previous output as
the next input. All calls use the same resident operands and device stream.

The API closures are:

| Variant | Timed API dataflow | BF16 logical traffic for `N=tokens*intermediate` |
| --- | --- | --- |
| `swiglu_fused` | `npu_swiglu([gate, up], dim=-1)` | Read `4N`, write `2N` bytes |
| `swiglu_separate` | `silu(gate)` followed by `temporary.mul(up)` | First API reads/writes `2N` each; second reads `4N`, writes `2N` |
| `geglu_separate` | `gelu(gate, approximate="tanh")` followed by multiplication | Same materialization and traffic as separate SwiGLU |
| `silu_standalone` | `silu(input)` | Read `2N`, write `2N` |
| `gelu_standalone` | `gelu(input, approximate="tanh")` | Read `2N`, write `2N` |

The separate closures materialize one BF16 intermediate tensor of `2N` bytes,
live from the activation return through multiplication. The producer's
`hidden` labels do not add tensor dimensions. Input initialization, correctness
canaries, copy drift canaries and warmup calls remain outside the modeled
timed region.

The burst invocation count is a recorded execution input. Although the old
producer selected it using an earlier timing, the simulator uses the resulting
protocol count; it does not repeat that timing-based selection or inspect the
recorded duration.

## Current implementation and boundary contract

`prediction/calls_activation.py` provides:

```python
from mindie_sim.prediction.calls_activation import (
    ActivationCallSpec, build_activation_call,
)
from mindie_sim.prediction.call import call_costs

call = build_activation_call(
    ActivationCallSpec("swiglu_separate", tokens=4, intermediate=3072)
)
device = call_costs(case, call.ops, recipes=call.recipes, invocations=100)
scenario = call_costs(
    case, call.ops, recipes=call.recipes, invocations=100,
    include_submission=True,
)
```

The builder creates one complete invocation. The common `call_costs` path
composes finite repetitions through the ordinary graph and execution engine.
It does not multiply an event duration or create another scheduling engine.
The builder supplies analytical API resource envelopes and real intermediate
read/write traffic. Its nominal lane-operation counts come from the SiLU and
tanh-GELU expressions; transcendental, divide and ordinary arithmetic costs are
not asserted to be identical in an actual binary.

API nodes are not confirmed native kernel launches. An unknown CANN dispatch
may use a different number of kernels or an internal implementation not
represented by this envelope. `coverage.native_kernel_count` is therefore
`null`. A declared submission scenario charges the chosen API-node schedule,
with host submission allowed to overlap earlier device service. It is not a
reconstruction of an unrecorded host timeline.

`observations/call_bindings_activation.py` verifies the audited producer hash,
API list, tensor layout, BF16 dtype and eager single/burst protocol:

```python
from mindie_sim.observations.call_bindings_activation import bind_activation_condition

binding = bind_activation_condition(condition)
result = call_costs(
    case, binding.call.ops, recipes=binding.call.recipes,
    invocations=binding.invocations, mode=binding.mode,
)
```

The binding retains `target_identity` separately from the result's
`device_graph` or `submitted_device_graph` boundary. It records
`strict_identity_equivalence=False`. A downstream call-alignment report may
show conditional descriptive residuals with the unresolved terms visible;
the binding does not create a matching strict accuracy prediction.

The builder never receives observations. The binding reads structural identity
and the condition identifier only: not `samples`, `duration_ms`, medians,
canaries, fitted coefficients, context labels or train/heldout roles.
Original samples remain unassigned.

## Actual retained-capture run

On 2026-10-29, both complete raw captures were read through the existing
source-checked adapter. All 1,080 conditions were structurally bound:
216 conditions per variant, 247 distinct physical/protocol target identities,
and 24,300 retained samples.

`examples/calls/activation-nominal.json` uses the existing registered
`Ascend910_9362` profile: nominal 1.2 TB/s HBM service and
2.112 trillion vector lane operations/s. Its optional 20-microsecond API-node
submission cost is explicitly nominal. These are not measurements of the
capture's acquisition run, and no rates were selected to minimize residuals.
The small model/workload context in this case satisfies the common case
schema; its transformer layers and serving requests are not executed.

`data/verification/activation-call-modeling-20261029.json` retains independent
predictions for all 247 identities under two scenarios:

- Device-only composition: complete invocation or whole burst graph.
- Declared submission: the same device work with the nominal API-node queue.

Across the full set, device-only spans range from 0.00001024 to
82.04058624 milliseconds; the declared-submission spans range from
0.02001024 to 82.06058624 milliseconds. These ranges mix tensor sizes and
invocation counts and are not performance comparisons. All 494 composed runs
finished with zero live tasks and programs. The report contains no residual,
tolerance result or accuracy judgment; the shared comparison tool owns those.

Focused checks:

```text
.venv/bin/python -m pytest -q tests/test_activation_calls.py tests/test_capture_adapters.py
15 passed
```

They verify independent tensor arithmetic, materialized dependencies, exact
recorded burst counts, host/device queue overlap, unchanged original sample
roles, and invariance under altered observation durations and experiment
labels. Changing a declared hardware rate changes the prediction.

## Explicit remaining limits

- The loaded kernel binary, actual internal dispatch and precise kernel count
  were not captured. The model makes no source-kernel identity claim.
- Python dispatch, allocator behavior, event insertion gaps, real launch
  scheduling, L2 reuse and backend tile mapping are unknown. Their cost is
  neither inferred from a residual nor quietly treated as an observed zero.
- Output allocation/destruction belongs to the timed closure. Its tensor
  lifetime is described, but the host allocator and device allocation policy
  are not simulated by the resource envelope.
- Resource envelopes do not establish warp/core occupancy or separate
  transcendental service rates. This is a useful bottom-up structural baseline,
  not final timing accuracy.
- The retained grouped-GEMM probe uses INT8 input/weights, `FRACTAL_NZ` weight
  layout and per-token/per-channel scales. The existing source-derived
  `ascendc_grouped_nd_bf16` planner is BF16/ND/no-quant. It is not bound to that
  capture merely because both are grouped GEMMs; that family remains a
  separate mechanism task.
