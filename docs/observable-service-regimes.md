# Observable kernel service regimes

Small payloads, sparse expert groups, strided transfers and synchronization
protocols need not run at peak large-block throughput. An observed kernel or
phase response is a valid explicit model parameter; reconstructing its driver
instruction sequence is not a prerequisite. The original analytical program
remains useful for structure, physical work and behavior outside the observed
domain.

`runtime.kernels.service_regimes` binds such responses to prepared operators or
complete native kernel groups. A rule contains physical identity, numerical
coordinates, a domain, a response and observation provenance:

```json
{
  "id": "example-strided-copy",
  "match": {"recipe": "strided-copy", "dtype": "bf16"},
  "coordinates": {"rows": {"path": "local_shape.0"}},
  "domain": {"rows": {"min": 16, "max": 64}},
  "response": {
    "fixed_us": 3,
    "terms": [{"coordinate": "rows", "us_per_unit": 0.3,
      "scale_with": {"resource": "mte2_aiv", "reference_rate": 4000000000000}}]
  },
  "provenance": {"basis": "illustrative_parameters_not_measurements"}
}
```

The response is `fixed_us + sum(coordinate * us_per_unit)`. A term with
`scale_with` scales inversely with its current physical rate relative to the
declared reference rate, in native units per second. Cube terms additionally
declare their precision. A term without scaling is an explicitly fixed
observed response, not a recovered pure-link bandwidth or scalar instruction
rate. Fixed synchronization can dominate and produce a legitimate plateau in
a bandwidth scan; that is reported rather than replaced by a fabricated strict
speedup.

Paths address implementation metadata, tensor shapes, native work and owner
counts. Coordinates may reduce arrays with `max`, `min`, `sum`, `len`,
`product` or `positive_count`, then apply an explicit unit conversion `scale`.
Model labels, operation IDs and request IDs are not selector roots. Domain
misses keep analytical execution and record `analytic_outside_domain`;
unmatched declared rules remain visible in `regime_selection`. Overlapping
applicable rules are an error, not a choice of the more favorable timing.

The kernel finishes at the maximum of its physical program and its inclusive
response, both anchored at actual input readiness. This is not a delay appended
to already counted work. HCCL/MC2 source phase groups receive the response once
at native entry/exit; their phases do not each incur a new startup. Native bytes,
FLOPs, causal links, buffer lifetimes and contention are retained. Increased
contention can still exceed the observed isolated response. This first interface
models a response boundary, not the unobserved resource occupancy of every
instruction inside a mixed kernel.

Parameters may come from operator captures or PMU summaries in a declared
workload regime. Their provenance records which data assigned the parameters,
which data were not used in this assignment, and which records had already been
examined historically. Comparisons on fitted data are in-sample; a new split
does not retroactively make historical observations pristine heldout data.
Capture ABI, group-list and execution-mode uncertainties remain distinct from
permission to calibrate hardware service. A missing measurement never prevents
analytical simulation.

## Continuous physical responses

`response.kind: roofline` derives a response from native work and the current
physical supplies. It has no shape lookup or training knots:

```json
{
  "kind": "roofline",
  "fixed_us": 3,
  "services": [
    {"coordinate": "flops", "resource": "cube", "dtype": "bf16", "utilization": 0.5},
    {"coordinate": "memory_bytes", "resource": "hbm", "utilization": 0.25}
  ]
}
```

Here `flops` can address `work.cube` and `memory_bytes` can address `work.hbm`.
These example utilization and startup parameters are illustrative declarations,
not measured hardware values. Each service contributes
`coordinate / (current_rate * utilization)` seconds. Services sharing one physical
pipe add; distinct pipes overlap through a maximum, followed by `fixed_us` once.
This includes instruction modes sharing a vector pipe and different precisions
sharing the matrix pipe. The response records `physical_supply`; declaring it
alone does not mark the result as observed calibration.

TP changes reach this calculation through the frontend's real local M/N/K,
FLOPs and operand traffic. Prefill row counts and attention context lengths also
change native work. The same declaration can therefore predict unseen numerical
inputs without adding a per-shape rule. This establishes executable extrapolation
from stated physical assumptions, not verified accuracy at unmeasured shapes.
The tests in `test_service_response_geometry.py` cover TP2/TP8, M1/M4096,
context growth, current-rate perturbations and unchanged physical work.

The rule fixes `hardware`, `dtype`, `layout`, `implementation`, `replay_scope`
and either `recipe` or `kernel`. Numerical domain bounds remain optional explicit
limitations. Per-operator diagnostic coordinates may include M/N/K even when
the response uses derived FLOPs and bytes. A default response remains an inclusive
completion constraint. Selecting the separate `joint_native` execution policy
uses the declared native service queue; its internal occupancy is not recovered
from the response. See [joint native service](operators/joint-native-service.md).

## Empirical shape curves and surfaces

`piecewise_linear` interpolates one declared coordinate using ordered
`knots: [[coordinate, duration_us], ...]`. `multilinear` interpolates a complete
rectangular grid across several declared coordinates:

```json
{
  "kind": "multilinear",
  "grid": [
    {"coordinate": "m", "knots": [1, 4]},
    {"coordinate": "n", "knots": [2, 8]}
  ],
  "values_us": [3, 5, 6, 10],
  "outside": "analytic"
}
```

The ordered grid list defines row-major indexing: the final coordinate varies
fastest. The example values correspond to `(m,n) = (1,2), (1,8), (4,2), (4,8)`.
Every grid corner must be supplied; absent measurements are not synthesized.
Nonuniform knot spacing is supported. A one-coordinate grid is also valid.

Empirical rules additionally fix `runtime`, `revision` and `format`. An explicit
unknown identity remains unknown provenance; it does not establish matching
binaries. Other declared numerical coordinates must have equal minimum and
maximum domain bounds. For example, an M/N surface with fixed K declares K as a
coordinate with identical domain endpoints. The caller must also preserve the
remaining physical conditions such as expert occupancy, stride, padding policy,
context and graph mode. Changing a fixed categorical condition records a
`physical_identity_mismatch` and keeps analytical execution.

Exact knots record `anchor`; unseen in-grid points record `interpolated`.
Outside the knots, `outside: analytic` keeps the original physical program and
records `analytic_outside_knots`. Explicit `linear` extrapolation extends the
edge interval/cell and records `extrapolated`; explicit `error` rejects that
query. Negative extrapolated durations remain errors. A separate application
domain miss records `analytic_outside_domain` before interpolation is attempted.
Aggregated coordinate ranges and relation counts stay bounded by the rule list.

Prediction consumes only the supplied parameters. It neither reads validation
timings nor extends its training domain from a successful score. An M sweep at
one N/K cannot identify an empirical response to unseen N/K. Cross-TP accuracy
therefore needs independent physical supply assumptions or appropriate training
axes and separate held-out points; marking a domain miss is not evidence of
generalization accuracy.
