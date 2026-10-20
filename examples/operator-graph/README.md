# A3 operator-graph examples

These files are predeclared inputs, not measurements. Manifests are identified
by `kind=a3_operator_graph_manifest`; results use
`kind=a3_operator_graph_timing_results`. No numeric schema-version field is used. No device work has been
run for them.

| File | Purpose | Counts |
| --- | --- | --- |
| `full-matrix.manifest.json` | Six train-side shapes, N=1/32/128, three timing modes | 54 conditions, 108 condition executions, 3240 raw blocks |
| `capability.manifest.json` | Minimal pilot: one small shape per operator, N=1, three modes | 9 conditions, 18 condition executions, 540 raw blocks |
| `manifest.schema.json` | Versioned manifest JSON Schema | — |
| `results.schema.json` | Versioned results JSON Schema | — |
| `results.example.json` | Non-measured setup-failure document showing result shape | no timing samples |

The capability pilot covers the three exact callables and both graph protocols
but is not the full matrix. The T512 widths and N=32/128 exist only in the full
manifest; neither manifest claims heldout data.

Validate without torch or a device:

```bash
python -m mindie_sim.a3_operator_graph \
  --manifest examples/operator-graph/full-matrix.manifest.json \
  --results /tmp/a3-operator-graph/full.json \
  --run-id example-dry-run --device npu:0 --dry-run
```

A real run requires an external coordinator lease visible as `npu:0`, a new
results path, and the outer supervisor window. See
`docs/a3-operator-graph.md`.
