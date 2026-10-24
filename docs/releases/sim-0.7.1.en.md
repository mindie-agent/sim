# mindie-sim 0.7.1: Profiled Kernel Cost Plans, Captured Duration Floors & Declared-Work Proxies

[中文](https://github.com/mindie-agent/sim/blob/v0.7.1/docs/releases/sim-0.7.1.md)

`mindie-sim 0.7.1` adds `ProfiledKernelProvider` (`backend="profiled_kernel"`), reusing `mindie-profiler` SQLite inventory observations through explicit operator `bindings` to build `KernelPlan` instances that apply captured median kernel durations as per-task completion floors while preserving declared Cube/Vector/HBM/MTE work proxies.

---

## 1. Key Features & Behavior Changes

### 1. Explicit `bindings` Matching & Strict `profiled_device_kernel` Scope (`mindie_sim/profiled_kernel.py`, PR `#16`)

- `ProfiledKernelProvider` takes `db`, `environment_key`, and configured `bindings`, querying `variants` joined with `costs` in the `mindie-profiler` SQLite store read-only (`mode=ro`, `busy_timeout=5000`, with a bounded 1024-entry lookup cache).
- Matching enforces an explicit contract with no shape, SKU, recipe, or environment wildcarding:
  - `(op.family, op.recipe, op.phase, op.launch_shape, op.dtype, op.layout)` (plus optional `op.metadata["profile_variant"]`) must match a unique entry in configured `bindings`;
  - The matched binding's `timing_scope` must equal `"profiled_device_kernel"` (`CAPTURED_TIMING_SCOPE`) exactly;
  - The SQLite observation must be `eligible=True` and match both `machine.hardware_key` and `self.environment_key`.

### 2. Captured Median Duration Floor & Declared-Work Proxy (`mindie_sim/profiled_kernel.py`, `operator_plan.py`, PR `#16`)

- Captured device kernel durations (`median_ns`) do not include per-pipe Cube, Vector, HBM, MTE2, or MTE3 occupancy breakdowns. To prevent concurrent operators on a shared stage from overlapping for free, `ProfiledKernelProvider` emits `ExecutionStage("captured_service", demands=demands, minimum_duration_ms=observation["median_ns"] / 1e6)`:
  - `minimum_duration_ms` sets a per-task completion floor on the parent stage so service time never finishes below the captured median duration;
  - The operator's declared `flops`, `vector_ops`, `hbm_bytes`, `mte2_bytes`, and `mte3_bytes` map onto shared stage demands (`cube`/`vector`/`hbm`/`mte2`/`mte3`), and operators declaring no work are refused (`profile_occupancy_undeclared_for_shared_stage`);
  - Plan `provenance` records `source_class="observed"`, `timing_scope="profiled_device_kernel"`, `resource_occupancy="not_captured"`, `occupancy_basis="declared_work_proxy"`, and `independent_accuracy_acceptance=False`.

### 3. Cost-Model Precedence & Observation Reuse (`mindie_sim/a3_cost.py`, `a3_measurement.py`, PR `#16`)

- **Calibration & Communication Precedence**: In `PredictiveCostModel.cost()`, dedicated measured calibrations and the communication topology model retain their normal precedence over a `profiled_kernel` plan; if the operator also specified `op.metadata["profile_variant"]`, the preserved record reports `superseded_profile_variant` rather than raising. `A3InputError` is raised only when `profile_variant` contradicts a non-executing witness, an existing stage plan, or an explicit serial cost (`PROFILE_VARIANT_CONTRADICTIONS`), or when an explicitly requested `profile_variant` is unavailable or substituted by an analytic backend (`analytic_backend_substituted`).
- **Skipping Redundant Calibration Requests**: Operators priced by an observed `profiled_kernel` plan receive `calibration.status = "profiled_kernel_observed"`, which `record_prediction_misses()` in `a3_measurement.py` treats alongside `"calibrated_exact"` and `"calibrated_interpolated"` so existing profiler inventory observations are reused without queuing duplicate measurement requests.

---

## 2. Verification & Evidence Boundaries

- **Targeted Test Suite**: Passed **`22 passed`** covering `ProfiledKernelProvider` binding lookups, completion-time floors, declared-work demands, precedence/contradiction handling, and miss-recording reuse.
- **Evidence Boundary**: `profiled_kernel` binds per-operator captured device duration floors with declared work proxies (`independent_accuracy_acceptance=False`); it does not claim measured per-pipe hardware occupancy or replace whole-network accuracy/performance acceptance.
