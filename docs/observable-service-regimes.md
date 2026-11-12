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

## Measured dual-load matrix service

The profiler C7 report carries one source-supported full-grid / tail pair:

```text
enrichment.pmu_observations.measured_supply.pair
```

Each condition is the median of its captured repeats, not the fastest event.
`physical_owners` comes from the declared AIC capacity (`--aic-count`), never
from the maximum sampled Block Num. The R64 QBMM condition also carries the
certified issued-work rule `2 * align16(M) * K * N` beside the logical `2MNK`;
the rate is issued work over measured active MAC time. The C7 carries
`source_identity` (`r64`/`r35`/`unpinned`/`absent`, deployment digest, basis)
from the same-capture status prepare or an explicit deployment image. R35 and
bare captures retain generic measured PMUs but cannot produce the R64 pair;
`import_profiler_supply` rejects their unavailable supply.

`mindie_sim.observations.calibration.import_profiler_supply` reads only that
fragment (streaming from a large C7), verifies the response with the normal
parser, requires an explicit hardware identity, and fixes its `match` to the
caller's selected recipe/kernel/hardware. `apply_profiler_supply` appends it to
a case-local `runtime.kernels.service_regimes` list but does not rewrite the
case's recipe/tiling. The existing CLI exposes the same path:

```sh
mindie-sim operators CASE.json --phase prefill --tokens 2 \
  --supply c7.json --supply-recipe output \
  --supply-hardware Ascend950DT_9582
```

The rule's `response.kind` is `measured_dual_load`; its coordinates are `m`,
`n` and `k`. The `domain` fixes the measured `n` and `k` exactly. The response
computes issued work from the certified alignment and selects the measured
issue class first: the exact M16 underfilled footprint or an integer multiple
of the saturated footprint. The active-owner count comes from that selected
measured load (24/32), not from a tile-derived macro guess. An explicit
`--supply-owners` value is only an optional consistency check; a mismatch falls
back to analytic:

```text
issued_flops = 2 * align16(m) * k * n
mac_us       = issued_flops / rate_flops_per_s * 1e6
```

`execution.kind: mac_service` replaces only the cube stage with that measured
per-owner MAC time. When the selected operator is the ordinary `TiledMatmul`,
the matched invocation uses the source rule's `tile_m`/`tile_n`/`tile_k`/slots;
the case, other operators and every off-domain invocation keep their original
recipe, geometry, MTE2 and service. HBM, MTE2, MTE3, vector, queues and overlap
remain ordinary; the measured MAC service is not a whole-kernel completion
floor. A missing owner, a different hardware/dtype/N/K identity, or an
unmeasured owner count returns an explicit mismatch/unknown and keeps the
untouched analytical execution; it is not reclassified to the nearest load.
Measured supply and pure theoretical/roofline supply remain available in the
same case.

For the genuine original piecewise route, `CompiledOperator.select_measured_mac`
matches the selected operator's `native_call`: query hardware, role/recipe,
wrapper source SHA/revision, kernel source SHA, `FRACTAL_NZ` weight layout,
dtype and K/N must all agree. It changes **only** `_elapsed_rate("cube")` to
the measured FLOP/s/1000 for the empirical static M16 underfilled class or the
full-wave saturated class. Physical `work`, `stage_work`, state, projections,
tiling, and every other pipe rate remain the source values; the ordinary
compiler recomputes stage/overlap timing from those unchanged values. The
optional normal entry is `z.predict_cell(spec, measured_mac=rule)`.

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
