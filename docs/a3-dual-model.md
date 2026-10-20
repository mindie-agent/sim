# A3 predictive / normative dual model

This repository now has two deliberately separate answers:

| Purpose | Entry point | Question | Method |
| --- | --- | --- | --- |
| Predictive operational | `predict` / `mindie_sim.predict` | Given this machine, model, request case and config, what will actually happen? | Explicit framework schedule + shape lowering + event-driven shared resources + optional measurement calibration |
| Normative should-cost | `should` / `mindie_sim.should_cost` | Given the same semantic workload and hardware, what should a reasonable implementation cost? | Semantic required FLOPs/bytes/communication, optimistic lower bound **L**, finite legal plan family, feasible target **U** and witness timeline |

`U` is the best known feasible target in the enumerated family, not a global
lower bound. `L` is not a runnable schedule. Neither path adds a single global
"efficiency" multiplier to the other.

No A3 measurements are bundled in this repository. Declared examples are
explicitly labelled as declared inputs, never as `measured`.

## Predictive inputs

A case is one JSON object:

```json
{
  "machine": {
    "arch_family": "A3", "num_devices": 8, "dies_per_package": 2,
    "device": {
      "hbm_gib": 64, "cube_flops_per_s": 3.6e14,
      "vector_ops_per_s": 3.0e13, "hbm_bw_bytes_per_s": 1.6e12,
      "mte_bw_bytes_per_s": 3.2e12, "launch_us": 20
    },
    "compute_peaks": {"bf16": 3.6e14, "fp8": 7.2e14},
    "links": [{"name": "d2d", "capacity_gbps": 196, "members": [0,1,2,3,4,5,6,7]}]
  },
  "model": {
    "architecture": "gqa", "layers": 32, "hidden_size": 4096,
    "num_heads": 32, "num_kv_heads": 8, "head_dim": 128,
    "intermediate_size": 14336, "vocab_size": 152064,
    "activation_dtype": "bf16", "weight_dtype": "bf16", "compute_dtype": "bf16"
  },
  "workload": {"requests": [{"prompt_tokens": 1024, "output_tokens": 8}]},
  "runtime": {"tp": 8, "ep": 1, "dp": 1, "pp": 1,
              "max_num_batched_tokens": 16384, "max_model_len": 32768},
  "slo": {"ttft_ms": 2000, "tpot_ms": 50}
}
```

Key contracts:

* `hbm_bytes`/`hbm_gib` is **per die**. Package/node totals are never mixed in.
* World size is `dp * pp * tp`. EP is a MoE group inside the `dp * tp` domain;
  it is not multiplied again. `attention_tp` and `etp` are reported separately.
* GQA/MLA attention shapes use query length and KV length independently.
  `hidden_size` is the residual width; `num_heads * head_dim` may differ.
* If TP exceeds KV heads, KV heads/cache are replicated, not divided below one.
* `num_experts`, `top_k`, `moe_intermediate_size` and an explicit
  `expert_routing_histogram` or uniform distribution drive grouping/padding.
* `fp8`/`int4` peaks are used only if `compute_peaks` declares them; the model
  does not infer a 2x/4x speedup from a dtype name. Quantized weights add
  explicit dequant/accumulation work.
* `graph_buckets` never shrink work below the valid scheduled token count. A
  forced bucket that is too small falls back to eager with an explicit guard.
* Generic Dense and MoE FFN paths include the packed gate/up -> SwiGLU ->
  down activation as a real shape-changing vector op with its own work,
  input/output bytes and dependency; expert and shared-expert branches are not
  algebraically collapsed.
* KV transfer is a producer -> chunked transfer -> visibility -> consumer DAG;
  it shares the same D2D resource as collectives. The framework's first-token
  completion waits for full prompt KV visibility in this version.

## Predictive execution

`mindie_sim.a3_workload` derives semantic, local-shard, padded and launch shapes
for every op. `mindie_sim.a3_cost` converts work into resource demands with
per-family shape/tiling efficiency. `mindie_sim.a3_resources` advances a
dependency DAG on resources such as per-stage Cube/Vector/MTE2/MTE3/HBM, launch
and D2D link. Shared capacity is arbitrated (equal-share by default); event
arrival/completion recomputes the active set.

The request scheduler is a declared vLLM-like token-budget model:

* running requests first, then waiting requests;
* token and optional input budgets;
* chunked prefill; a partial prompt chunk never emits a first token;
* mixed batch: one-token continuation + waiting prefill can fill the same
  budget, so the target request waits for the whole batch;
* KV blocks and prefix-cache tokens; graph replay eligibility;
* iteration completion feeds request output/TTFT/ITL/TPOT/E2E.

Current support boundaries are reported in output:

* one in-flight iteration per DP replica; asynchronous overlapping batches and
  PP micro-batch overlap are not claimed by the request simulator (the finite
  PP microbatch solver is a separate public function);
* DP replicas are independent device groups. If EP spans DP, a representative
  `1/dp` D2D link share is used for cross-DP EP contention;
* framework KV transfer waits for full prompt KV visibility; chunk-level
  visibility is directly represented in the resource engine and is tested
  there.

## Calibration and the re-measurement loop

`SQLiteCalibrationStore` is a lightweight SQLite/WAL store (`busy_timeout=5000`
on every connection) with:

* raw samples: hardware, op family, phase, recipe, shape, dtype, layout,
  features, competition conditions, duration, analytic duration, source
  class, source, revision, run id, train/heldout split;
* deduplicated measurement requests: `pending`, `running`, `completed`,
  `failed`, priority, query count, predicted/error/extrapolation/target-impact
  fields and an external `run_id`;
* inventory and observed shape coverage;
* an executable manifest with a runner/ingest contract;
* CPU and managed `torch_npu` runner interfaces.

Source revision/commit labels are attribution metadata. They are excluded from
semantic sample/request keys and never gate modeling. The store has one current
business-table contract (no schema-version row, migration ladder or reflection
gate), and raw captures are imported by their source-format adapters. Only
`source_class in {measured, observed}` and non-synthetic samples calibrate;
synthetic claims of measured data are rejected. CPU samples are `derived` and
remain inventory evidence, not A3 calibration.

Calibration physical matching uses the shared operator canonical protocol and
exact equality: unknown is not eager, a missing invocation count/lifetime is
not a known one, and `graph_single_repeated` is not `graph_block`. Family,
hardware, recipe, precision, features and shape candidates are filtered before
protocol ambiguity, so an unrelated measurement cannot poison an exact match.
A caller that wants measured physical calibration declares the full protocol as
JSON in `runtime.metadata.calibration_protocol` (or `measurement_protocol`); a
partial runtime `graph_mode` is not treated as a measured block protocol. A DB
and a curve artifact can be supplied together without dropping either.

Generic curve binding preserves a `transition_unresolved` bracket instead of
silently returning no curve: the public op keeps the lower/upper bracket,
reason, protocol and concrete missing-condition geometry while retaining the
numeric analytic estimate. `--record-misses` materializes that missing physical
condition (including its protocol) as a pending request.

Native primitive transfer is opt-in and narrow. Set
`runtime.curve_allow_primitive_transfer` (or construct
`PredictiveCostModel(..., curve_allow_primitive_transfer=True)`) to allow a
plain BF16 RMSNorm curve whose hardware, input/output format, epsilon,
continuous layout and complete allocation/cache/timing protocol match the
actual op. Such a curve remains `production_match=false` and a candidate, never
production equivalence. GDN composite split/L2/mRoPE, wrong dtype/epsilon/
hardware/protocol and incomplete historical artifacts stay analytic. SwiGLU and
softmax feature expansion are separate work.

A typed whole-span communication TRAIN observation without stage work can be
used only as a scoped per-op completion envelope: the measured span calibrates
the operator/path completion while internal phase occupancy stays explicitly
unknown and no stage durations are fabricated. The envelope is a private
capacity-1 elapsed-time resource, not a shared physical link/serial demand, so
two independent opaque ops do not invent contention; known analytic
`op.link_bytes` remains a real link demand alongside the envelope. The caller
must declare the typed observed
execution (`runtime.communication_observed_backend`) and the usual
route/reduction/group descriptors. Outside the measured size support the result
is labelled `calibrated_affine_extrapolated` with measured sizes, singleton
status, unknown engine selection and an executable missing condition; it is a
numeric estimate, not in-envelope calibrated accuracy.

```bash
python -m mindie_sim predict --input examples/a3_dense_predict.json \
  --calibration-db /tmp/cal.sqlite --record-misses \
  --requests-output /tmp/requests.json
python -m mindie_sim calibration --db /tmp/cal.sqlite manifest --limit 32
# External managed runner (mindie-bench/coordinator) produces results.json
python -m mindie_sim calibration --db /tmp/cal.sqlite ingest --results results.json
python -m mindie_sim calibration --db /tmp/cal.sqlite inventory
python -m mindie_sim calibration --db /tmp/cal.sqlite run --runner cpu --limit 4
```

`torch-npu` is optional and imports only when that runner is selected. The
coordinator injects the visible device; the simulator never SSHes, leases or
selects a free card.

## Normative should-cost

`should_cost` derives required work directly from model semantics and request
token counts (no framework copy, graph padding, queue wait or observed
schedule). It computes:

* `L = hardware_lower_bound_ms` from raw declared peaks, unique weight/KV/IO
  bytes and required communication. This is an optimistic lower bound.
* A finite legal plan family over TP/PP/DP/EP, fusion, graph padding and MoE
  recipe. Invalid head/layer/expert divisibility and OOM plans are rejected.
* A witness timeline for each plan via the resource engine:
  compute and memory overlap, communication is dependent, launch is serialized.
* `U = feasible_target_ms` = best witness in the family; `gap = U - L`.
Unpinned candidates use automatic disjoint core placement; an explicit
`core_base` constraint is preserved, and the selected plan/witness exposes the
chosen placement.
* `optimal_in_family` and the evaluated plan count. `U` is not called a global
  lower bound.

The normalized algorithm fixtures are public for independent checks:

* `normalized_shared_cv_search()`: shared-memory C/V plans; lower bound 12 ms,
  enumerated feasible best 14 ms with witness.
* `normalized_chunked_pipeline_search()`: 8/6/4 ms stages with 0.2 ms
  per-stage/chunk overhead; optimum at 8 chunks.
* `simulate_microbatch_pipeline([3,5], 4, transfer_ms=1)`: first token 9 ms,
  last token 24 ms; finite schedule, never `total / PP`.

## CLI

```bash
python -m mindie_sim predict --input case.json
python -m mindie_sim should  --input case.json
python -m mindie_sim search  --input case.json --search-spec search.json
python -m mindie_sim pipeline-microbatches --stages 3,5 --microbatches 4 --transfer-ms 1
```

All commands emit one JSON document. Errors use a stable `reason` and non-zero
exit status; a partial or unmodeled result is not printed as a successful
numeric prediction.

## Tests

`tests/test_a3_modeling.py` contains the independent formula oracles, event
capacity/dynamic-set checks, 16K boundary scheduler checks, calibration store
round trip, normalized should-cost checks, search filtering and representative
Dense/MLA/MoE end-to-end runs. Live A3 probes remain opt-in and are not part of
the offline suite.
