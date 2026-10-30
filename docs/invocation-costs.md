# Device invocation costs

`prediction.invocation.InvocationProgram` owns one explicit device invocation
boundary. It wraps an already-prepared finite `Program` body with a declared
startup delay and a declared completion floor. It is not a host launch queue, a
graph-replay submission, a message/transport path, or a `ResourceService` curve.

## Ownership and preparation

The root prepares the body and passes its enclosing context with
`minimum_ms=0`. The body must expose only the real executable kernel boundary:
root excludes join/barrier operations and does not spread this wrapper over inner
phases, tiles or repeated pipeline stages. A nested body invocation therefore
does not pay startup per phase. This module does not inspect or edit body
internals, strip a floor through reflection, or infer which operations are joins.

The wrapper keeps the body's engine, work, caps, buffers, queue resources and
owner. It does not create a global barrier or mutable ledger, so independent
invocations remain independent.

## Causal timing

For one invocation with original readiness `ready`:

```text
startup:  ready -> startup delay -> body expansion and service
minimum:  ready -> minimum floor                  (concurrent with startup+body)
duration: max(minimum_ms, startup_ms + body_duration)
```

Startup is a causal delay after the original readiness and before the body is
expanded or serviced. Deferring expansion keeps idle active state bounded and
honors delayed input readiness. The floor is anchored to the original readiness,
not to the end of startup, so it is not stacked as
`startup + max(minimum, body)`.

When `startup_ms` is zero the body starts as soon as original readiness completes,
without a virtual time delay. An unresolved input still defers body expansion.
An empty body is still a valid declared invocation; this wrapper does not infer
joins or residual work from its absence of resource service.

## Declared parameters

`InvocationTiming(startup_ms=0.0, minimum_ms=0.0, provenance={})` accepts only
finite nonnegative timings. `from_mapping` treats omitted timing entries as zero
and rejects unknown fields; explicit null is not a number. `to_dict` round-trips
the declared fields. Empty parameters mean zero; no value is inferred from an
observed grouped residual.

`provenance` is preserved as declared data. It can record a capture source,
revision or assumptions, but it is not admitted as proof of a measured
implementation identity and does not select a hardware/model branch. The
mechanism has no model-name/family-name logic, fitted scalar latency, historical
launch constant or 40 us grouped floor.

A device primitive floor may be supplied together with a startup when the root
has already removed the body's inner floor. The maximum semantics above still
apply. Physical service remains the body's work; startup and floor contribute no
resource service.

## Separate scopes

| Mechanism | Scope |
| --- | --- |
| Host launch FIFO | host submission ordering |
| Graph replay submission | whole-graph submission boundary |
| Message propagation / transport | communication path and frame timing |
| `ResourceService` startup | response curve for one resource invocation |
| `InvocationProgram` | declared startup/floor for one device kernel boundary |

These are composed explicitly by their owners; none substitutes for another.
The existing engine remains the only execution engine.

## Hardware execution-class parameters

`machine.device.kernel_invocations` supplies low-priority device parameters:

```json
{
  "kernel_invocations": {
    "vector": {
      "minimum_ms": 0.0025,
      "provenance": {"basis": "user_reported_experience", "reported_us": [2, 3]}
    },
    "cube": {
      "minimum_ms": 0.005,
      "provenance": {"basis": "user_reported_experience", "reported_us": [5, 5]}
    }
  }
}
```

These A3 values were reported by the user on October 29, 2026. They are now
declared on the six A3 SKU profiles, with provenance and the vector range.
They are not independently measured empty-kernel values. `launch_us` remains
host submission time and does not provide these device constraints.

Selection uses actual native cube/vector work after source planning. A mixed
kernel takes the maximum of applicable constraints once. Pure DMA, transport,
joins and nested resource phases receive no inferred vector charge. An explicit
`OpSpec.metadata.invocation_class` can identify a specialized class, including a
real source-bound empty kernel. TP/PCP representative execution takes the maximum
parameter across participating devices, as it already does for conservative
resource supply.

A matching runtime invocation rule replaces the hardware parameters. A zero rule
therefore disables the default for its declared scope. Per-device class maps
override inherited classes; to disable one class, declare its `minimum_ms` and
`startup_ms` as zero. The inclusive primitive constraint still combines by
maximum. An inclusive `ResourceService` rejects a second nonzero startup from
either source of parameters.

These constraints reach `operator_costs`, `graph_costs(include_submission=False)`
and serving through the same `Execution`. They do not charge each tile, create
synthetic resource work, or replace collective algorithms and graph dependencies.
The completion minimum is not added after a large kernel body.

The October30 `Ascend950DT_9582` profile adds two explicitly scoped defaults.
Ordinary vector kernels use an inclusive1.312us minimum, the minimum of the
independent BF16 `[1,4096]` library Add control. This is an **engineering
extrapolation**, not a universal hardware limit: the retained custom AscendC
FP32 tiny Add reaches0.876us. A specialized execution class or invocation rule
can replace the ordinary-library assumption. Ordinary nonempty Cube now uses
the independently observed1.578us BF16 ND MatMulV3 minimum. The earlier0.439us
one-block empty AIC control remains the explicit `empty_aic` class; the64-block
1.115us result is not assigned to every invocation. These declarations have zero startup.
Their source, observed scope, exclusions and vector counterexample are retained
in parameter provenance and ordinary prediction output. Other A5 SKUs do not
inherit these measurements automatically.

The separate32MiB MatMulV3 control takes11.471us while transferring useful
weights; that duration is not an empty Cube floor. Neither the new stream
cadence nor the roughly6–7us matrix residual sets a device startup. Such a
residual can identify a refinement priority without identifying a parameter.
See [the independent boundary audit](operators/a5-microbench-boundary-review.md).

The newer [tiny Cube audit](operators/a5-tiny-cube-boundary-review.md) also
provides `w8a8_nd_bf16`: an inclusive3.037us minimum for INT8 ND operands,
FP32 per-token and per-channel scales, and BF16 output in one MIX_AIC kernel.
The enclosing operator can select it through
`metadata.invocation_class="w8a8_nd_bf16"`. The class is explicit because
logical `int8_per_channel` metadata does not establish an ND library dispatch.
NZ weights, W4A8, MXFP8, expert groups and INT32 outputs retain their own
programs and ordinary execution-class constraints. Their analytic predictions
remain available without a matching calibration. A measured mixed kernel is
one inclusive constraint, not separate Cube and Vector startup charges.

## Public configuration

Ordinary `operators`, `graph`, `call` and `predict` execution reads the optional
`runtime.kernels.invocations` rule list. The following is a synthetic parameter
scenario, not a recommended hardware value:

```json
{
  "runtime": {
    "kernels": {
      "invocations": [
        {
          "match": {
            "recipe": "rmsnorm",
            "dtype": "bf16",
            "replay_scope": "full"
          },
          "startup_ms": 0.001,
          "minimum_ms": 0.004,
          "provenance": {
            "basis": "declared_scenario",
            "source": "independent parameter declaration"
          }
        }
      ]
    }
  }
}
```

Supported exact-match axes are `hardware`, `runtime`, `revision`, `format`,
`kernel`, `implementation`, `dtype`, `layout`, `replay_scope`, `recipe`, `family`
and `phase`. `hardware` matches the normalized hardware identity key;
`implementation` matches the complete declared implementation value. There are
no model-name, substring or wildcard selectors. An empty match is an explicit
all-invocation scenario. Prefer narrow implementation identities when using an
independently measured parameter.

Every specified axis must match. Among matching rules the rule with the most
axes wins; a tie is an explicit ambiguity error, rather than a list-order
override. Thus a more narrowly declared kernel/implementation condition can
specialize a broad family scenario. There is no implicit priority of a kernel
axis over several other axes. Matching rules and invocation counts appear under
`parameters.invocation_models`.

The source call is expanded before invocation selection. Each actual kernel can
match; the logical call join is not another kernel. Inner resource phases clear
the enclosing timing, so tiled/pipelined bodies pay it once. A cross-DP collective
rank still enters its device invocation when it sends no payload but receives
from peers. Its startup delays its own entry, while its completion floor delays
only its own completion and does not impose a cohort-wide final barrier.

For an in-domain `ResourceService`, a nonzero independent device startup is
rejected because the inclusive response already owns that boundary. A zero
startup with a completion constraint remains composable. When the service curve
chooses its explicit `analytic` behavior outside the work domain, its fitted
startup is absent; the separately declared invocation startup can then apply.
Other curve identity, ownership and work-domain errors remain visible. A
resource-phase response is not selected as a second device invocation.

The primitive store's enclosing completion constraint and a declared minimum
are combined by their maximum at original readiness. The body is lowered with
that enclosing floor cleared, preventing a second floor after startup. Neither
constraint creates synthetic bytes or compute work. Host launch/replay service
remains outside the device boundary and can overlap preceding device execution.

Public acceptance checks cover a delayed input across `operators`, `graph` and
`predict`; primitive and declared floors together; two eager submissions versus
one graph replay; a source MLA call with two kernels and one free join; an idle
cross-DP receiver; and both in-domain and analytic outside-domain resource
responses. These are causal controls, not observations used to set defaults.
