# `tools/accuracy` schemas and contracts

Every artifact this package writes is declared here: the exact field names, what each field
means, which one is authoritative on a conflict, and which choices are declared rather than
derived. Read this before trusting a number in `dataset-report.md` or `acceptance.json`.

```
assemble_dataset.py ──> observations.json  (mindie-sim.observations/1)
                    ├─> dataset-manifest.json  (census, exclusions, split enforcement)
                    ├─> train-outside-store.json/.csv  (derived table + estimator mismatch)
                    └─> dataset-report.md
freeze_predictions.py ─> predictions.frozen.json (mindie-sim.frozen-predictions/1)
                      └> freeze-receipt.json     (mindie-sim.freeze-receipt/1)
score_frozen.py ──────> score.json (mindie-sim.score/1)
                    ├─> bootstrap.json
                    └─> acceptance.json (mindie-sim-accuracy-acceptance/2 + protocol projection)
sources.py             declared source registry + per-artifact readers (no output of its own)
heldout_manifests.py ─> manifests/<sku>--heldout-<lane>.json, heldout-manifests.json,
                        collisions.json, dryruns.json, heldout-runbook.md
metrics.py             the single definition of every identity key and metric
```

---

## 1. Observation dataset — `mindie-sim.observations/1`

```json
{"schema_version": "mindie-sim.observations/1",
 "observations": [ { ...one row per distinct physical condition... } ]}
```

### 1.1 The row

The eighteen keys below are emitted verbatim, in this order. `runs` and `diagnostics` are
the two additive extensions this round mandates (merged-run retention, pooled-median and
alias diagnostics). No other top-level key is produced; the assembler asserts the set.

| key | type | meaning |
| --- | --- | --- |
| `condition_key` | `"cond:"+24 hex` | Row identity. Hashes hardware, die, family, shape, recipe, execution kind, worker recipe, precision, layout, quantization, feature extras, protocol signature **and per-call amortisation**. Two rows with the same key are the same physical condition; a different die or a different protocol produces a different key. |
| `hardware_key` | string | SKU key **read from the artifact** (`Ascend910_9362`, `Ascend910B4`, …). Never defaulted, never hardcoded. |
| `die` | string | Physical die the measurement ran on: `phy1`, `phy0`, `sio:phy0+phy1`, `hccssw:phy0+phy2`, `phy2+phy3`, or `unknown`. |
| `arch_family` | `"A2" \| "A3" \| "A5"` | Declared by the source campaign, consistent with the split declaration. |
| `family` | string | Operator family as the artifact names it (`gemm`, `copy_bytes`, `attention_qk`, `graph_rmsnorm`, `comm_all_reduce_only`, …). |
| `recipe` | string \| null | Worker recipe / execution path (`torch_mm_preallocated`, `npu_rms_norm`, `graph_replay:graph_block`, `prefill_causal:qkv`, …). |
| `protocol_signature` | `"proto:"+16 hex` | Timing-protocol identity (see §2). |
| `run_id` | string \| null | Run that produced the **first** retained stream; all runs are in `runs`. |
| `split` | `"train" \| "heldout"` | Enforced split label (see §4). |
| `shape` | object | Physical shape with numeric values (`{"m":1,"n":7168,"k":14336}`). |
| `elements` | int \| null | Element count when the artifact declares an `elements` axis, else `null`. |
| `flops` | int \| null | Declared-work-model FLOPs for this shape (§5); `null` when no model covers the family. |
| `hbm_bytes` | int \| null | Declared-work-model bytes for this shape; `null` likewise. |
| `duration_ms` | float | **The scored target**: `median_t( median_e( raw_event_ms[t][e] ) )` over all retained trials, and over merged runs it is the median of *all* per-trial medians. |
| `raw_samples_ms` | float[] | Every retained event sample, in acquisition order (trials concatenated, runs concatenated). |
| `source_class` | `"measured" \| "observed"` | Only these two classes are emitted; `derived`/`synthetic` rows are excluded (§6). |
| `timing_scope` | string \| null | Declared timer scope (`device_event_warm`, `device_event_warm_operator`, `device_event_block_interval_inclusive_host_gap_native_allocation`, …). |
| `runs` | object[] | One entry per retained acquisition stream: `run_id`, `trials`, `trial_medians_ms`, `raw_events`, `raw_samples_sha256`, `pooled_median_ms`, `source_artifact`, `source_sha256`, `source_id`, `record_id`. |
| `diagnostics` | object | Identity and measurement diagnostics — see §1.2. |

### 1.2 `diagnostics`

| field | meaning |
| --- | --- |
| `geometry_key` | Bare geometry key, byte-identical to `geometry_key()` in `generate_splits.py` (verified against all 389 declared points). |
| `geometry_identity` / `geometry_identities` | Family-aware shape identity used for split matching and the prediction join, plus its declared family aliases (§3). |
| `family_class` | Family class used for identity (`copy` groups `copy`/`copy_bytes`/`graph_copy`). |
| `pooled_median_ms` | Diagnostic the **old calibration store** uses (median over all events pooled across trials). Never scored. |
| `pooled_minus_trial_median_ms` | Size of that estimator mismatch for this condition. |
| `trial_medians_ms`, `trials`, `raw_events`, `single_trial` | Trial structure actually recovered. |
| `estimator` | `median_of_per_trial_medians` (or `…_over_merged_runs`). |
| `protocol`, `protocol_origin` | Canonical protocol fields and the origin of each (`attested`, `record_level`, `declared_default`, `unknown`). |
| `die_attested`, `die_evidence` | Whether the artifact attests the physical die, and from which field. |
| `precision`, `layout`, `quantization`, `features`, `execution_kind`, `worker_recipe`, `backend` | Identity extras as the artifact attests them. |
| `analytic_ms_artifact` | Any `analytic_ms` the artifact carries — recorded, never used as a baseline (round-1 native files alias it to the measured median). |
| `source_artifact`, `source_sha256`, `source_id`, `record_id` | Provenance of the first stream. |
| `split_origin`, `split_hint` | Which field proposed a split label, and what it said. |
| `declared_heldout`, `heldout_point_id`, `heldout_test_set`, `heldout_declaration`, `heldout_matches` | Which declaration this condition satisfies. |
| `flags` | `train_row_relabelled_heldout`, `campaign_holdout_not_declared`, `trial_structure_not_recovered`, `correlated_streams_merged`, `same_record_conflicting_samples`, `no_work_model:…`. |
| `alias_group`, `alias_members` | Equal-payload alias group (§4.3). |

### 1.3 `dataset-manifest.json`

Census of every source scanned (id, role, files with sha256, record count, status), every
excluded record **individually** with its reason, every alias group with members, the split
enforcement section (relabels and campaign-local holdouts listed one by one, residual
train/heldout conflicts, which must be 0), the declarations in force with their hashes, the
band edges derived from the declaration, the estimator-mismatch summary, the round-2
discovery census, and `dataset_sha256` — the content hash of the observations array, which
is what the freeze receipt binds.

---

## 2. Protocol identity

`protocol_signature` hashes a canonical dict over:

```
timer, timing_scope, warmup, iterations, trials, preallocated,
cache_policy, measurement_scope, attention_scope
```

Rules:

1. **Absence never wildcards.** A field the artifact does not attest is the literal string
   `"unknown"`; the unknown value takes part in the hash. A silent side and an attested side
   therefore never collide.
2. **Lifts are declared, not guessed.** A value the artifact attests outside its protocol
   block is lifted and its origin recorded: nine-model attests `timing_scope` as a sibling of
   `protocol`; `trials` is the number of sample groups the artifact presents (proved by
   reconstructing the per-trial medians and comparing with the attested
   `trial_medians_ms`); the attention campaigns declare their timing scope and cache policy
   in the deployed protocol contract cited in `sources.py`.
3. **`warmup_scope` and per-call amortisation are excluded from the join signature** and kept
   in `diagnostics`. No round-1 measurement attests `warmup_scope`, so including it would
   turn an attestation gap into 100 % non-coverage rather than into a measurement. They *are*
   part of `condition_key`, so two runs that differ only in amortisation stay separate rows.
   `--strict-protocol` adds both fields to the signature for a round whose runners attest
   them.
4. A prediction joins an observation only when its protocol signature equals the one the
   observation attests. A mismatch is **`protocol_mismatch`**: the condition stays in the
   coverage denominator, contributes no error term, and is listed in `excluded`. It is never
   silently scored.

---

## 3. Geometry identity

`geometry_key = "geo:" + sha256(payload)[:24]` over
`{hardware_key, family, recipe, execution_kind, worker_recipe, precision, layout,
quantization, features, shape-as-floats}` — exactly the payload `generate_splits.py` uses, so
a declared point and a measurement of it produce the same key string.

`shape_identity` is the key used for **split matching and the prediction join** because it
normalises three real ambiguities that the artifacts carry:

| normalisation | why |
| --- | --- |
| `q_tokens→q`, `kv_tokens→kv`, `seq→tokens`, `dim→width` | the declared query manifests and the measuring artifacts name the same attention/RoPE axes differently. |
| copy → total payload bytes (`rows*row_bytes`, or `elements × itemsize`, or `bytes`) | “the same byte copy as rows × row_bytes and as elements” is one physical condition (leakage rule L7). |
| family aliases: `copy`/`copy_bytes`/`graph_copy` → `copy`; `masked_softmax` → `attention_softmax`; `grouped_matmul` → `moe_gate_up`/`moe_down` | round 1 declared these under different family labels than the artifacts emit. The **shape** decides the match; the label never does. |

`condition_key` additionally includes the die and every identity extra, so alias handling
never merges two rows.

---

## 4. Split enforcement and deduplication

### 4.1 Hard rules

1. A condition whose `(hardware_key, shape_identity)` matches any declared heldout key is
   `heldout` for the lifetime of the corpus — regardless of the label the campaign attached.
   Such a row is listed in `dataset-manifest.json › split_enforcement › relabelled_train_rows`
   with its heldout point id.
2. A campaign-declared holdout that the round never declared stays `heldout` and is listed as
   `campaign_holdout_not_declared`. It is never training data.
3. The assembler exits **3** if any residual `train` row still overlaps a declared heldout
   key (`train_heldout_conflicts` must be 0).
4. Declared heldout keys come from both split declarations (`new_heldout.points`,
   `families[].heldout`, gap-fill train) and the round-1 geometry-only query manifests
   (the 159 test-set-1 conditions).

### 4.2 Deduplication and merging

* Two records merge only when `condition_key` is identical — same die, same protocol, same
  recipe and shape. Runs accumulate in `runs`; all raw samples are retained; a re-measurement
  under the same protocol therefore widens the sample rather than creating a second condition.
* An exact duplicate (same run id, same record id, identical samples — e.g.
  `a3-gemm-expanded-train.json`, the three attention presentations, the raw-record mirrors)
  is collapsed and counted in `merge.exact_duplicate_rows`, never added as a second stream.
* The same record id under the same run with *different* samples is a data defect: retained,
  flagged `same_record_conflicting_samples`, and counted.
* The same run id with a different record id (the two ranks of one collective) is a
  correlated stream: retained, flagged `correlated_streams_merged`.

### 4.3 Alias groups

Copy conditions with the same payload but a different `(rows, row_bytes)` factorisation keep
separate rows (they are separately measured attempts at one payload) and are tagged with an
`alias_group` so no reader can count them as independent evidence. The group id hashes
hardware, die, family class, protocol signature and payload.

---

## 5. Work model and regime bands

`flops` / `hbm_bytes` and the regime band are computed from the **declaration's** work
model, not from code constants:

```
regime_rule.work_models = {"gemm": "flops=2*m*n*k; bytes=2*(m*k+k*n+m*n)", ...}
regime_rule.constants   = {"peak_bf16_ops_per_s": …, "hbm_bytes_per_s": …, "vector_ops_per_s": …}
analytic_lower_bound_ms = 1000 · max(flops/peak, ops/vector, bytes/hbm)
band                    = launch if < launch_edge, large if >= large_edge, else mid
```

The formula strings are evaluated by a restricted AST walker (numbers, names, `+ - * / **`
only — no calls, no attributes). An unknown variable, a family with no work model, or a
geometry the declared formula cannot describe produces `null` work terms, a
`no_work_model:…` flag, and — in the scorer — the `unbanded` regime with `band_reason`, never
a guessed band. Band edges are read from the declared band rule text; a rule that does not
parse makes every condition `unbanded` instead of silently mis-banded.

---

## 6. Exclusion reasons

| reason | meaning |
| --- | --- |
| `not_measured` | status is not `completed`/`measured`/`ok` (`pending`, `failed`, …). |
| `not_a3_measured` | the artifact declares `not_a3_measured` / synthetic / dry-run. |
| `not_a_device_measurement` | declared synthetic (`synthetic: true`, no NPU initialised). |
| `not_calibratable_source_class` | `source_class` outside `{measured, observed}`. |
| `no_shape` | the artifact attests no physical shape. |
| `no_family` | no family can be established for the record. |
| `no_hardware_key` | the artifact does not attest the SKU. |
| `no_usable_samples` | no positive finite device sample. |
| `protocol_unreadable` | the protocol block could not be canonicalised. |
| `shape_identity_failed` | a copy record declares no usable payload axis. |
| `not_queried` / `missing_outcome` / `join_ambiguous` / `protocol_mismatch` | scorer-side reasons (§7). |

Every excluded record appears individually in `dataset-manifest.json › exclusions` and in
`dataset-report.md`; nothing is dropped silently.

---

## 7. Frozen predictions and the join

### 7.1 `predictions.frozen.json` — `mindie-sim.frozen-predictions/1`

```json
{"schema_version": "...", "frozen_at": "...",
 "manifest": {"manifest_id", "sha256", "path", "entries"},
 "source": {"commit", "dirty", "dirty_entries", "package_files",
            "package_file_hashes_sha256", "python_version"},
 "protocol": {"declaration_files": {name: sha256}, "..._split_declaration_sha256", "combined_sha256"},
 "dataset": {"sha256", "file_sha256", "path", "training_conditions"},
 "scope": {"hardware_keys", "test_sets", "declared_heldout_conditions", "undeclared_queries"},
 "status": "completed|partial", "expected_entries", "completed_entries",
 "evaluation_outcomes_read": false, "outcomes_directory_listed": false,
 "strict_protocol": false, "automatic_fitting_performed": false, "scoring_performed": false,
 "predictions": [ {"condition_id", "geometry_key", "protocol_key", "protocol_core", "kind",
                   "family", "shape", "hardware_key", "query_sha256",
                   "variants": {"analytic|calibrated_db|curve|db_and_curve": {
                       "status", "prediction_ms", "region", "lower_bound_ms",
                       "upper_bound_ms", "product_result_sha256", "product_result"}},
                   "status", "error"?} ]}
```

The manifest is checked for outcome fields (`OUTCOME_KEYS`) before anything is called; a hit
is a refusal. `--import-predictions` re-keys an existing frozen document (round-1 format or
this one) into the receipt schema; entries sharing a `condition_id` are merged, with the
direct curve applied last so the `curve` variant means the direct curve query.

### 7.2 `freeze-receipt.json` — `mindie-sim.freeze-receipt/1`

The receipt carries the prediction file's sha256, source commit/dirty flag, package hashes,
declaration hashes, the dataset content and file hashes, the training-inventory hash (over
the *set* of calibratable conditions, drawn from the dataset's `train` rows), scope,
completeness and the two false claims. The scorer re-verifies every one of them.

### 7.3 The join

```
join key   = hardware_key + shape_identity (family-aware, aliased)
assertion  = observation.protocol_signature == prediction.protocol_key
```

* No observation → `missing_outcome`; no prediction → `not_queried`; more than one candidate
  after preferring the protocol-matching one → `join_ambiguous`; signatures differ →
  `protocol_mismatch`. All four count in the coverage denominator and appear in `excluded`.
* The join never reads a prediction's stored `observed` value (there is none) and never opens
  a measurement file: outcomes come from the frozen dataset only.

---

## 8. Scoring contract

Unit = one physical condition. Target = `duration_ms` (§1).

| metric | definition | denominator |
| --- | --- | --- |
| `wape_pct` | `100 · Σ|pred−obs| / Σ obs` | covered conditions only (duration-weighted) |
| `wape_ci95` | cluster bootstrap over conditions, B=10000, seed=20261021, percentile 2.5/97.5 | covered, n ≥ 5 |
| `median_ape_pct` | `median(100·|pred−obs|/obs)` | covered conditions (count-weighted) |
| `p90_ape_pct` | nearest rank: `sorted[ceil(0.9·n)−1]` | covered |
| `max_ape_pct` | `sorted[-1]` | covered |
| `mae_ms` | `mean|pred−obs|` | covered |
| `median_signed_bias_pct` | `median(100·(pred−obs)/obs)` — sign is what does not average out in a whole-model sum | covered |
| `spearman_rho` | rank correlation of prediction vs observation | covered, n ≥ 3 |
| `coverage` | `numeric / declared` | **every declared condition in scope**, including unsupported, unresolved, unqueried and unmeasured |
| `coverage_denominator` | constant string naming that denominator | — |
| `wape_penalized_pct` | `100·(Σ_covered|err| + Σ_uncovered obs)/Σ_all obs` — an abstention costs what a 100 %-wrong prediction costs | all declared |
| `wape_in_envelope_pct` / `wape_extrapolated_pct` | WAPE restricted to the in-envelope region class (`exact`, `interpolation`, `interpolation_2d`, …) vs `extrapolation*`; a variant with no attested region reports `null`, never a guess | covered with an attested region |
| `containment_rate` | inclusive `lower ≤ obs ≤ upper` | conditions reporting both bounds |
| `insufficient_n` | `numeric < 5` — no CI is reported for such a cell | — |

Every metric block states `weighting` (`duration_weighted_over_covered_conditions` for WAPE,
`count_weighted_over_conditions` for APE) and its `n`, per leakage rule L8.

**Primary path.** Exactly one variant per condition, chosen from a declared chain:
`curve_first` = direct curve if numeric, else `calibrated_db`, else `analytic` (this round's
headline); `calibrated_first` = `calibrated_db` → `curve` → `analytic` (the residual
analysis's policy, kept so its 13.59 % is reproducible). `db_and_curve` is never a primary
path: in round 1 it is numerically identical to `calibrated_db` on 83/83 conditions, so using
it would triple-count the generic rows. The all-variant roll-up is always published beside
the primary roll-up.

**Aggregation cells.** Pooled primary, every variant, per family, per regime band
(`launch`/`mid`/`large`/`unbanded`), per die, per protocol signature, plus per-family bands,
per-family variants, paired ΔWAPE versus the analytic baseline (paired bootstrap on the same
conditions) and per-family outlier lists (APE > 2× the family median, kept in every metric).

---

## 9. `acceptance.json`

Top-level contract of `mindie-sim-accuracy-acceptance/2`:

```
status, hardware_key, arch_family, test_sets, conditions, numeric, coverage,
wape_pct, wape_ci95, wape_penalized_pct, wape_in_envelope_pct,
median_ape_pct, median_ape_ci95, p90_ape_pct, max_ape_pct, mae_ms,
median_signed_bias_pct, spearman_rho, weighting, coverage_denominator,
primary_path{policy, chain, counts, per_condition},
variants{...}, per_family{...}, per_regime{...}, per_die{...}, per_protocol{...},
gates[G1..G12], excluded[], limitations[], repeat_noise_floor,
provenance{...}, artifacts{...},
protocol_acceptance{...}, protocol_acceptance_validation{...}
```

* `status` is `scored` (no threshold enforced), `accepted` (all enforced thresholds met) or
  `rejected` (an enforced threshold was violated, CLI exit 2).
* `gates` always carries all twelve entries; a gate that this run cannot evaluate is
  `not_evaluated`, `insufficient_n` or `report_only` with the reason in `note` — never
  silently omitted and never reported as passed.
* `protocol_acceptance` is the same numbers projected onto
  `acceptance-schema.json` (`mindie-sim-accuracy-acceptance/1`), validated in-process with
  `jsonschema` when it is installed; the result is in `protocol_acceptance_validation`. Two
  schema rules can legitimately fail on this round and are reported rather than patched: the
  schema requires `source.dirty: false` (this worktree is dirty by construction while sibling
  workstreams are in flight) and `complete: true` (only a completed freeze is scoreable).
  `--emit-protocol-acceptance` writes the projection as `acceptance.protocol.json`.
* `provenance` records both hashes, both scoring-module hashes, the bootstrap parameters, the
  declared join key and the protocol/source identity, so a score can be reproduced or refused.

---

## 10. CLI and exit codes

```
python -m tools.accuracy.assemble_dataset [--data-root DIR] [--a2-root DIR]
    [--a3-round2-root DIR] [--declarations DIR] [--round1-evidence DIR] [--out DIR]
    [--strict-protocol] [--keep-excluded] [--no-derived-table]

python -m tools.accuracy.freeze_predictions --manifest FILE --out DIR [--repo DIR]
    [--dataset DIR] [--declarations DIR] [--curve-artifact FILE] [--calibration-db FILE]
    [--import-predictions FILE] [--round-id ID] [--allow-dirty] [--allow-undeclared]
    [--strict-protocol]

python -m tools.accuracy.score_frozen --predictions FILE --receipt FILE --out DIR
    [--dataset DIR] [--declarations DIR] [--round1-evidence DIR]
    [--primary-policy curve_first|calibrated_first] [--bootstrap-b N] [--seed N]
    [--allow-dirty] [--strict-protocol] [--emit-protocol-acceptance] [--enforce]
    [--max-wape PCT] [--max-median-ape PCT] [--max-p90-ape PCT] [--max-max-ape PCT]
    [--min-coverage F]
```

| exit | meaning |
| ---: | --- |
| 0 | success (assembly complete, freeze complete, or scoring without an enforced threshold) |
| 2 | acceptance failure: an enforced threshold was violated (`--enforce` + limit flags), or a freeze is partial |
| 3 | integrity violation: a hash changed after freeze, a dirty tree without `--allow-dirty`, an outcome field in the manifest, a training condition overlapping a heldout key, a query that is not declared heldout, a malformed artifact, or a train/heldout conflict |

---

## 11. Known limitations of this schema

1. `warmup_scope` and per-call amortisation are **not** in the join signature (§2.3). They
   are in `condition_key` and in `diagnostics`, and `--strict-protocol` promotes them.
2. Family aliases (§3) are declared, not discovered: a new campaign that renames a family
   must add an alias, or its conditions stay separate rows (which is safe, not silent).
3. The declaration's work model cannot describe two geometries that round 1 measured — a
   generic copy expressed as `elements` and the NCHW masked-softmax `(q, kv, heads)` shape.
   Their `flops`/`hbm_bytes` are `null` and their regime is `unbanded` with a reason.
4. The calibration-store sources (production loop, paired steady) keep repeats but not trial
   membership; when the protocol declares more than one trial the row is flagged
   `trial_structure_not_recovered` and its target equals the pooled median.
5. `die` for `graph_replay` is declared `phy1` from the inventory because the artifact attests
   only a logical `npu:0` inside a one-device lease; `die_attested` is `false` there.

---

## 12. Heldout measurement manifests — `mindie-sim.heldout-manifests/1`

`heldout-manifests.json` indexes every generated manifest with its sha256, lane (campaign /
attention), condition count, split histogram, recipes and families; it also carries the
declaration hashes, the dataset hash the collision check ran against, the collision report,
the dry-run evidence and the guard result.

* Campaign lane (`a3_operator_worker`): `{"campaign_id", "protocol", "requests": [...],
  "run_scope", "status"}`. Each request has `condition_key`, `request_key`, `measurement_id`,
  `point` (shape + `split: "heldout"`), a full `condition` block (`worker_recipe`,
  `execution_kind`, `precision`, `protocol`, `shape`, `status: "executable"`, …),
  `lineage_condition_keys`, `analytic_ms: null`, `predicted_ms: null` and a `declaration`
  block recording the point id, geometry key, regime and the declared lower bound. No model
  prediction is written into a measurement order.
* Attention lane (`a3_operator_worker_attention`): `{"kind": "a3_attention_rope_manifest",
  "manifest_id", "protocol", "source_contract", "expectations", "cases": [...]}`; fusion
  attention cases carry `mask`, `scale = 1/sqrt(qk_dim)` and the `source` block the worker
  validates, RoPE cases carry `rotary_mode`, `positions` and `inv_freq_base`.
* Keys are deterministic: `condition_key = cond:sha256({hardware, family, shape,
  worker_recipe, protocol-core})[:24]`, `request_key = req:sha256({condition_key,
  protocol_key})[:24]`, `measurement_id = ho:sha256({hardware, condition_key})[:20]`.
* The generator refuses to emit a manifest when any heldout condition appears in the training
  dataset (family-aware shape identity, per hardware).
* The guard (`--guard`) exits 3 unless the frozen-predictions receipt exists, is schema-valid,
  is complete, declares `evaluation_outcomes_read: false`, and is bound to the current dataset
  content hash.
