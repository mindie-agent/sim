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
