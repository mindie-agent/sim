# Extending the model

An extension belongs at the level where the underlying mechanism changes.

## Hardware

Supply the same `Hardware` schema for GPUs, NPUs or hypothetical hardware. Core
execution uses declared rates, capacities and links. Add a backend adapter only
when a native configuration needs translation. Keep SKU names and source-specific
conventions in that adapter. A declared rate is not a measurement unless its
provenance is independently recorded.

## Model semantics

A native family adapter produces `LayerPlan`, `QuantSpec`, `DraftSpec`, `VisionSpec`
and other structural records. A new checkpoint using existing blocks only needs
new configuration data. The computation must remain invariant when the model is
renamed.

For a new attention/FFN/residual mechanism, implement a builder and bind it through
`Frontend(attention={...}, ffn={...}, residuals={...})`. The callable receives
scheduled request views and structural records and returns real operators and
producer/consumer dependencies. `predict(case, lower=frontend)` uses that frontend.
Registering a name does not excuse missing capacity geometry: add a matching
storage formula for any new persistent tensors and verify it independently.

GQA/MLA, dense/MoE and recurrent state builders illustrate the contracts. Preserve
`OpSpec` semantic, local, padded and launch shapes, real dtype, physical work,
stream and dependencies. Padding is not an output token. Cache aliases identify
owners; cross-stage consumers require transfers, not just a renamed dependency.

## Implementation recipes

A `Recipe.lower(RecipeContext)` returns a `KernelProgram`, `PipelineProgram` or a
custom object implementing `start(engine, name, ready, owner=...)`. Stage work,
allocation, release and supply limits all use the same engine.

```python
from mindie_sim.prediction.program import KernelProgram, Stage, RecipeBook

class SerialLoadCompute:
    def lower(self, context):
        memory = context.resource_names["hbm"]
        load = {memory: context.work.get(memory, 0)}
        body = {name: amount for name, amount in context.work.items() if name != memory}
        return KernelProgram((
            Stage("load", load),
            Stage("compute", body, after=("load",)),
        ), minimum_ms=context.minimum_ms)

book = RecipeBook(recipes={"my_kernel": SerialLoadCompute()})
# predict(case, recipes=book)
```

This example demonstrates ordering only. A real implementation must preserve all
memory/compute/copy work and account for its own output stage and buffer lifetimes.
`ResourceEnvelope`, `TiledMatmul`, `TransferRecipe`, `RecurrentScan` and `RecurrentVector` provide
existing program patterns. JSON bindings can select built-in recipes by exact
recipe, family or program kind. Python bindings can supply new recipe objects
without changing the engine loop.

`TransferRecipe` serializes transactions inside each resource path and holds
each credit until its post-service `response_ms` delay ends. `outstanding=None`
means no credit constraint; a finite count creates bounded live work, including
fill and final response. This replaces the former average supply cap.
For framed links, the link declaration owns payload granules and wire headers;
the recipe consumes the resulting physical frames without rounding them again.
An explicit recipe granule must agree with the link payload granule. Unframed
paths use the recipe granule, or one byte when it is omitted. Paths are
independent; collective round and multi-hop forwarding dependencies are not
inferred. See `operators/historical-communication.md`.

`TiledMatmul.reuse_inputs/reuse_weights` describes ideal shared-cache HBM reuse.
It does not eliminate the operand copies required by each tile's private
buffers. Operator reports retain logical `declared_service` separately from
actual `serviced` work and the resulting `service_delta`; a repeated tile copy
is not a calibration error. See `open-source-mechanisms.md`.

To connect a source implementation to ordinary model inputs, register its
operator families or semantic blocks in `prediction.implementations`. One
`runtime.kernels.lowerings` declaration supplies frontend lowering and recipes,
including explicit implementation options and optional recipe selectors.
Preserve the semantic tensor contract and reject a selected implementation's
domain mismatch. Unselected analytical families remain usable without data.
See `operator-family-selection.md` for actual model-path examples.

When supplying Python-only recipes with a primitive store, assign a distinct
implementation identity in the case. The automatic key includes JSON kernel
bindings; it cannot introspect arbitrary Python code into a reproducible identity.

## Tensor contracts and lowering

Use `TensorLayout` and `TensorValue` for views; `TensorLowering.require` implements
consumer layout requirements and `reshape` emits a materialization when strides
cannot preserve addresses. Do not charge a full backing tensor for a slice copy
or turn every view into memory traffic. Transfer timing still needs an appropriate
recipe when transactions, copy instructions or synchronization dominate.

Representation/fusion passes are registered in `LOWERING_PASSES`, or supplied to
`lower_implementations(..., passes=...)`. Match dependencies, formats and shapes.
Preserve multi-output values consumed elsewhere. Do not identify an implementation
from a model label or use a capture's total time to choose a pass.

## Calibration and verification

Test small analytical cases before full-model smoke. Useful oracles include work
conservation, finite fill/drain, exact query-key enumeration, allocation lifetimes,
capacity admission, changes under independent hardware parameters and unchanged
results after renaming a model. Check the ordinary frontend and `graph_costs`
path, including real source expansion and repeated work, before serving metrics.
Graph bounds and structural completion are not measured accuracy.

Primitive measurements must match scope, shape, dtype, layout and implementation.
Heldout observations are never calibration candidates. A resource-rate fit must
be trained on primitive work, not on a model body's aggregate runtime. See
`docs/observations.md` for the data interfaces.
