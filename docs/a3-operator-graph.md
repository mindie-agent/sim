# A3 native-operator graph timing lane

Status: implementation and CPU-only validation only. No graph has been
instantiated, no device tensor allocated, and no NPU lease acquired here.
`sim_audit` / the external coordinator remains the sole owner of device work and
the shared queue. The plan's six shapes are preverified train-side shapes; this
lane neither reads nor relabels heldout/mixed measurements.

This lane is deliberately separate from the operator, core and catalog lanes.
It adds new files only and does not integrate into `mindie_sim.__main__` or any
existing worker/campaign/curve/adapter.

## Commands

Dry-run validates the actual schema, contiguous geometry, condition count,
trial/sample counts, budget ceilings and complete trial permutations using only
the Python standard library. It does not import torch or write results:

```bash
python -m mindie_sim.a3_operator_graph \
  --manifest examples/operator-graph/full-matrix.manifest.json \
  --results /tmp/a3-operator-graph/full.json \
  --run-id audit-dry-run --device npu:0 --dry-run
```

A real run is the same command without `--dry-run`. The results path must not
already exist unless `--overwrite` is passed. `--debug-dir` opts into
`NPUGraph.debug_dump()` outside timing; absence or failure of that optional dump
does not invalidate otherwise good replay timing.

```bash
python -m mindie_sim.a3_operator_graph \
  --manifest examples/operator-graph/capability.manifest.json \
  --results /tmp/a3-operator-graph/capability.json \
  --run-id a3-graph-capability-<id> --device npu:0
```

Exit codes: `0` only when every requested condition is measured and
measurement-complete, cleanup reports success, and the final results write is
durable; `1` for any partial/incomplete measurement, failed/pending condition,
cleanup failure or persistence failure; `2` invalid manifest/arguments or an
unsafe output path; `130` interrupt. A real run never accepts a device other
than `npu:0` and never acquires a lease, chooses a free card, SSHes or mutates a
service.

Current contracts use unversioned kinds only: the manifest is
`kind=a3_operator_graph_manifest`, the descriptive plan kind is
`a3_operator_graph`, and results are `kind=a3_operator_graph_timing_results`.
Identification uses only these unversioned kinds.

## Files

| File | Role |
| --- | --- |
| `mindie_sim/a3_operator_graph.py` | CLI, lazy device import, dry-run entry point |
| `a3_operator_graph_plan.py` | Versioned manifest validation, geometry, conditions, schedule, dry-run plan |
| `a3_operator_graph_results.py` | Result book, raw-clock/derived summaries, atomic partial persistence |
| `a3_operator_graph_reference.py` | Exact callables, deterministic buffers, actual tensor metadata, FP32 CPU checks |
| `a3_operator_graph_device_common.py` | Budget clock, failures, environment facts, condition artifact |
| `a3_operator_graph_device.py` | Stream, eager diagnostic, actual capture/replay preparation, stale-output check |
| `a3_operator_graph_timing.py` | Measured blocks, failure preservation, post-work cleanup |
| `a3_operator_graph_runtime.py` | Trial orchestration, partial writes, final status |
| `examples/operator-graph/manifest.schema.json` | Versioned manifest JSON Schema |
| `examples/operator-graph/results.schema.json` | Versioned results JSON Schema |
| `examples/operator-graph/full-matrix.manifest.json` | Predeclared 54-condition matrix |
| `examples/operator-graph/capability.manifest.json` | Minimal 9-condition pilot |
| `examples/operator-graph/results.example.json` | Non-measured schema example (no timings) |
| `tests/test_a3_operator_graph.py` | CPU-only schema/schedule/summary/failure tests |

## Full predeclared matrix

All cases use BF16 or uint8, contiguous row-major layout, `N in {1, 32, 128}`,
and all three timing modes. `warmup_iterations=10`, `trials=2`,
`samples_per_condition=30`, therefore 54 conditions, 108 condition executions
and 3240 raw blocks.

| Case | Operator | Physical declaration | Timing modes |
| --- | --- | --- | --- |
| `rmsnorm-bf16-t8-h5120` | RMSNorm | input `[8,5120]` BF16, gamma `[5120]` BF16, output `[8,5120]` BF16, rstd `[8,1]` FP32 | 3 |
| `rmsnorm-bf16-t512-h5120` | RMSNorm | input `[512,5120]` BF16, gamma `[5120]` BF16, output `[512,5120]` BF16, rstd `[512,1]` FP32 | 3 |
| `swiglu-bf16-t8-f2048` | SwiGLU | packed `[8,4096]` BF16, output `[8,2048]` | 3 |
| `swiglu-bf16-t512-f2048` | SwiGLU | packed `[512,4096]` BF16, output `[512,2048]` | 3 |
| `copy-uint8-rows8-row-bytes1024` | copy | source/target `[8,1024]` uint8, same target | 3 |
| `copy-uint8-rows512-row-bytes1024` | copy | source/target `[512,1024]` uint8, same target | 3 |

Counts are 9 conditions per case, 18 per mode and 18 per N. The minimal
capability example is one small shape per operator at `N=1` with all three
modes: 9 conditions and 540 raw blocks. It is explicitly not the T512 or
`N=32/128` matrix and does not claim those results.

## Exact callable math

```python
torch_npu.npu_rms_norm(x, gamma, epsilon=1e-6)
torch_npu.npu_swiglu(packed, -1)
target.copy_(source)
```

The installed schemas are `npu_rms_norm(Tensor, Tensor, float)->(Tensor,Tensor)`
and `npu_swiglu(Tensor, int)->Tensor`, each with only the default overload.
There is no `.out` overload and this lane does not fabricate one. RMSNorm and
SwiGLU outputs are native allocations; only copy uses a preallocated destination
(one same target for every block). Torch inference mode is active for the whole
run; nothing is inserted into the captured body except the exact callable and
host-side pointer bookkeeping.

## Stream, capture and lifetime

- One explicit `torch.npu.Stream` is created at setup and used for every mode
  and condition. Events are precreated on that stream for each condition.
- Graph modes use `torch.npu.NPUGraph()` and the actual
  `torch.npu.graph(g, stream=stream, auto_dispatch_capture=False,
  capture_error_mode="global")` context. The installed context performs
  synchronize and `empty_cache()` on entry; setup/capture are excluded from
  samples. `capture_error_mode` is never relaxed.
- `eager_block`: `N` eager calls per block. `graph_single_repeated`: capture one
  call, then `N` Python `replay()` calls per block. `graph_block`: capture `N`
  calls, then one `replay()` per block. `N` is the declared API invocation count,
  not an assumed hardware kernel count.
- Native outputs use retain-last-only lifetime, with different ownership for the
  two modes. For graphs, the captured final output is retained with the graph
  until post-work cleanup. For `eager_block`, only the current timed block's
  final output is retained; the previous owner is released before the next
  block, and diagnostic/correctness/warmup outputs are explicitly released.
  Copy's one preallocated destination remains a static buffer. Capture-time
  pointer sequences and unique-address counts are recorded, alongside an
  untimed eager allocator diagnostic. The graph stays in a private pool; no pool
  is passed to another graph. Address reuse differences are reported without
  claiming a pure host-dispatch causal tax.
- Capture failure becomes `failed` with the exact exception; unavailable API
  becomes `pending`. There is no eager fallback under a graph label.
- After capture, the graph is replayed once for correctness; inputs are mutated
  in place at the same address, replayed, checked for a stale output and restored
  in place before timing. Before **each** trial, the exact eager callable or
  graph replay is rewarmed for the declared 10 iterations outside all timing
  arrays, counted in case/global budgets, and recorded in that trial's warmup
  metadata. For graph modes this means 10 replays of the exact 1-call or N-call
  graph; for eager mode it means 10 calls of the exact callable, whose block
  invocation count N remains recorded separately.

## Verification

RMSNorm, SwiGLU and copy outputs are compared outside timing with deterministic
FP32 CPU references derived from the same seeded BF16/uint8 source bytes:
`rstd = rsqrt(mean(x^2)+1e-6)`, `y = x*rstd*gamma`, and
`silu(packed[...,:F]) * packed[...,F:]`. The returned RMS `rstd` is checked when
the installed tuple and element count make it feasible. Installed RMSNorm returns
the second output as FP32 `[T,1]`, so planned rstd geometry uses 4 bytes per
element (not the BF16 input dtype); this is an expected-contract correction, not
a relabel of captured measurements. Copy uses exact uint8
equality. NaN/infinite observations are preserved as failures, never repaired.

A condition is **measurement-complete** only when it is measured, all declared
trials and samples exist, correctness passed, and no timing anomaly was
recorded. Measurement completeness is not prediction validation, fit quality or
an accuracy claim; this lane performs no such ingestion or fit.
A failed correctness or stale-output check prevents timings for that condition.
Optional debug-dump unavailability/error is recorded separately and does not
invalidate otherwise good measured replay timing.

## Timing clocks

Each raw block records three independent values (milliseconds):

1. `cpu_submit_ms`: host `perf_counter_ns` around only the eager/replay loop;
   event record, elapsed-time and synchronization APIs are outside it.
2. `device_interval_ms`: precreated start/end `Event(enable_timing=True)`
   interval on the same measurement stream, read after end completion. It is an
   ordered device-marker interval, not proof of busy kernel execution.
3. `wall_complete_ms`: host interval from before start-event submission through
   the end-event completion wait.

`derived` contains `block/N` per-invocation values explicitly labelled
`derived`, with denominator kind
`declared_operator_api_invocations_not_measured_kernel_launches`. Raw values are
never replaced by summaries. No empty-event control is subtracted automatically,
and there is no `analytic_ms` field or analytic prediction in this lane. Blocks
are isolated with a measurement-stream synchronize outside all clocks.

## Results, budgets and cleanup

A `completed` status requires every requested condition to be both measured and
measurement-complete. All-measured evidence with an incomplete condition is
reported as `completed_with_incomplete_measurements`, never as success and
never with an invented accuracy label. Cleanup/persistence-specific completion
statuses are likewise non-success.

The results document keeps requested condition identities separate from
`progress` counts (attempted, terminal, measured, failed, pending, not_run,
measurement-complete, raw blocks). It is written atomically before setup and
after every trial or terminal condition, so partial evidence survives. A write
failure stops new device work immediately, triggers cleanup, makes one best
final write attempt and forces a nonzero result with the exact error on stderr
when durability is not established. Trial/sample records carry `run_id` and
`trial_index`; geometry, layout, strides, capture outcome, per-trial warmup,
output lifetime, allocation/pool facts and actual software/device facts are
preserved. No samples exist for failed/pending conditions.

Cooperative budgets are bounded at 24 s active case execution, 1200 s global
work, 60 s setup and 120 s cleanup; the external supervisor must allow at least
1380 s. Case time is accumulated active prepare, per-trial warmup and measurement
time per condition, not wall time between trial passes. Checks happen around
stages and samples; Python cannot preempt a hung native call, so the outer
supervisor is the hard bound. Cleanup synchronize and graph reset occur only
after the declared work loop and have their own 120 s budget. A failed
completion synchronize skips reset entirely, because resetting an unknown graph
state is unsafe; final synchronize failure also forces a non-success cleanup
status. Cleanup and persistence failures are reported separately from measured
condition results and force exit 1. The module implements no lease, SSH or queue
operation.

## Device assumptions still to be verified by `sim_audit`

- The installed torch/torch_npu pair can create the declared BF16/uint8 tensors
  on `npu:0`, capture the two native operator callables and copy with global
  capture error mode, and replay them.
- `NPUGraph.debug_dump(path)` support and its output shape are unverified; it is
  optional and never substituted for a measured task count.
- The exact native output addresses, graph private-pool reuse, memory queries and
  device property/clock fields observed on hardware may differ from the
  conservative metadata names used here. They are recorded as observed, not
  assumed.
- A cooperative budget check cannot interrupt a blocked driver call. The
  external supervisor must enforce the 1380 s outer window.

CPU tests validate schema, geometry, schedule/count arithmetic, summary
truthfulness, measurement-completeness/exit semantics, cleanup failure
propagation, persistence failure propagation, eager lifetime ownership and
per-trial warmup declaration. They do not claim device capture/replay behavior.
The optional debug path is exercised with a local writer only to validate path
sanitization and error separation, not graph support.
