# Operator shape-linearity campaign (A3)

This document describes the reusable operator measurement campaign added for
all nine archived model configurations.  It is a measurement-support package,
not a claim that any NPU job has been run here.  `sim_audit` remains the sole
device-work owner and executes a frozen manifest under the existing
`defaultHostQueue`; `mindie_sim` never acquires a lease, selects a free card,
SSHes or mutates a service.

## Components

| File | Role |
| --- | --- |
| `mindie_sim/a3_operator_campaign.py` | Public config/build/write/CLI orchestration; re-exports the public APIs |
| `a3_operator_campaign_conditions.py` | Actual-OpSpec classification, precision/state/feature/lineage normalization and condition identity |
| `a3_operator_campaign_sweeps.py` | Legal axis points, physical point identity, precommitted splits, request materialization |
| `a3_operator_campaign_selection.py` | Budget selection and family/recipe inventory aggregation |
| `a3_operator_campaign_report.py` | Pending coverage, coverage markdown and artifact writing/loading |
| `a3_operator_worker.py` | Manifest runner, CPU dry-run, result schema and CLI |
| `a3_operator_worker_recipes.py` | Primitive/reference recipes and CPU correctness checks |
| `a3_operator_worker_native.py` | Probed native fused recipes (RMSNorm/SwiGLU/softmax/grouped list/quant) |
| `a3_operator_curves.py` | Join raw samples, train-only fits, heldout evaluation, secants/residuals and coverage |
| `a3_operator_curves_report.py` | Coverage/markdown rendering |
| `a3_operator_curves_adapter.py` | Measured production linearity adapter and reviewer-normalized export |
| `examples/operator-campaign/` | Checked-in configuration and small capability examples; commands generate the full nine-model plan and CPU dry-run locally |
| `tests/test_a3_operator_campaign.py` | Real catalog extraction, axis legality/boundaries, physical split dedup, budget/pending coverage, worker dry-run, synthetic regime recovery, overlap/replay/correctness guards |

## Condition extraction

The campaign uses the public `derive_iteration_work(plan, model, runtime,
machine)` path, which dispatches to `infer_catalog_iteration` for catalog
models.  It expands representative prefill/decode iterations for every model
and representative TP/EP/PP topology from the checked-in configs.  Model
dimensions are loaded through `model_catalog.load_model`; no model is shrunk.

For every emitted `OpSpec` the campaign preserves:

- model, topology, layer, stage, op id/template and request lineage;
- semantic, local, padded and launch shapes;
- family, recipe, execution kind, worker recipe/backend and target match;
- input/weight/output/compute/accumulation precision, quantization/packing,
  weight representation and layout;
- feature/state facts (phase, graph bucket, routing, heads/dim, expert counts,
  cache/recurrent state) and the fixed measurement protocol;
- generic core-owned `ai_cpu`, `aiv` and `communication` expansion fields
  when present.  They are never merged into operator timing; A5 collection is
  a separate root-owned TODO.

A **condition identity** excludes model origin and source revision, but
includes family, recipe, phase, execution kind, physical shapes, precision,
layout, quantization, relevant feature/state facts and protocol.  Equivalent
conditions are merged and retain complete compressed origin summaries
(model/topology/phase/demand/op template/layer sets).  `conditions.json` is a
complete gzip+base64 table of every extracted condition (selected, executable
not selected, conditional, pending, unsupported) with exact shape, precision,
protocol, state and source mapping, plus a plain sorted key index.  Selected
requests retain their full source case list. First generate the condition table
from the checked-in configuration:

```bash
python -m mindie_sim.a3_operator_campaign plan \
  --config examples/operator-campaign/campaign-config.json \
  --output-dir examples/operator-campaign --budget 320
```

The generated plan, condition table, inventory and dry-run outputs are excluded
from Git. Once `conditions.json` exists, a missing condition can be materialized
without re-running lowering:

```bash
python -m mindie_sim.a3_operator_campaign materialize \
  --conditions examples/operator-campaign/conditions.json \
  --condition-key cond:... --output materialized.json
```

Pending conditions now materialize explicit points (not `[]`) with
`expected_status`/`required_backend`; `axis_rejections` preserves OOM/illegal
shape evidence.

### Execution kinds are not interchangeable

| `execution_kind` | Meaning |
| --- | --- |
| `primitive` | One declared primitive recipe (for example preallocated `torch.mm`, `torch.softmax`, QK/PV `torch.bmm`) |
| `fused_kernel` | A probed native fused candidate (for example `torch_npu.npu_rms_norm`), or a pending target route |
| `reference_decomposition` | A separately named, real executable reference (block-FP8 dequant + BF16 mm, RMSNorm multi-op, SiLU+mul, residual add, cache row copy).  Its time is **never** labelled or fitted as the target fused kernel |

FP8 block GEMMs therefore produce two conditions where applicable: a real
block-dequant + BF16 mm reference and a pending native blockwise FP8 target.
RMSNorm and SwiGLU produce a probed native target and an explicit reference,
but only with explicit `tokens,width`; missing width is pending, never
"executable".  Qwen GDN `qk_norm` (`gdn_split_qk_l2norm_mrope`, explicitly
`not_rmsnorm`) is a pending native composite, not RMSNorm.  Router
`sigmoid`/`sqrtsoftplus` + top-k is pending actual scoring/top-k composition,
not plain softmax.  DSv4 index-K int8 per-row quantize has a named real
`int8_per_row_quant_reference` plus a separate pending native target; it is not
mapped to `npu_dynamic_quant`.  Unsupported exact recipes (packed grouped MoE,
recurrent KDA/GDN, rotary, fused attention, etc.) remain concrete pending
inventory entries with a reason and required backend.

## Finite sweeps and budget

Default demand cases (configurable, deliberately not a full Cartesian product):

- prefill tokens: `1, 32, 128, 512, 4096, 16384`;
- decode: `(batch=1, context=1024)`, `(8, 8192)`, `(32, 32768)`,
  `(128, 1024)`.

Around each actual demand shape the generator varies one axis at a time:

- GEMM/MLA projection: `M`, `N`, `K` independently;
- attention: query length with context held fixed, KV/context, heads, head dim;
- norm/softmax/elementwise: rows/tokens and width, or explicit elements;
- router softmax: tokens and experts;
- MoE grouped: routed tokens, experts, N/K (pending native formats retain
  unmeasured axis examples);
- cache: rows and row bytes.

`x0/4`, `x0`, `4*x0` are train anchors; `x0/2`, `2*x0` are interpolation
holdouts.  Boundary probes come only from observed lowering padding, declared
weight/expert blocks, or a declared KV block.  For a boundary `T`,
`T-2, T, T+2` are train and `T-1, T+1` are boundary holdouts; wider points are
also train.  One outer interval (`x0*8`) is an extrapolation holdout when
feasible.  The simulator's heuristic tile constants are not asserted as
hardware boundaries.

The manifest has an explicit `budget` (`--budget`, default 320).  Selected,
omitted and pending conditions are all reported; a budget shortfall is never
silently treated as coverage.

## Splits and cross-model leakage

Splits are frozen in the manifest **before timing**.  The physical point
identity used for global dedup contains only execution kind/recipe, physical
shape, dtype/layout/quantization and protocol — not model, layer or semantic
family name.  Therefore a shape that appeared in train for one model/layer can
never become a holdout for another.  Duplicate train anchors win.  The analysis
keeps train and heldout partitions separate, rejects any overlap instead of
merging it, and never uses heldout outcomes to choose a fit regime.  Repeated
identical trials/results are idempotent and do not increase independent N.
Correction-failed results are excluded from fitting and counted as failed.

## Worker protocol and recipes

`python -m mindie_sim.a3_operator_worker --manifest … --results …` runs on the
coordinator-injected device (`npu:0` by default).  The default protocol is:

- warmup 10, iterations 30, trials 2 (all configurable);
- device events, preallocated inputs, same buffers reused, no cache flush;
- correctness check against a CPU sample **before** timing where output
  exists; a failed check is a failed result, never a sample;
- raw event values for every trial, per-trial medians, timer/scope/cache/device
  versions and both `query_context` (pre-use request facts) and
  `observed_context` (the actual API/backend called).  No AICPU/AIV claim is
  inferred from an environment preference.

Supported worker recipes include:

- plain BF16/FP16 preallocated GEMM (`torch.mm`);
- block-FP8 and int8 explicit dequant + BF16 mm reference;
- `torch.softmax`, QK/PV `torch.bmm`, residual add, row copy as explicit
  reference where applicable;
- probed `torch_npu.npu_rms_norm`, `torch_npu.npu_swiglu` (packed `[T,2F]`
  input, `dim=-1`), `torch_npu.npu_scaled_masked_softmax` with official NCHW
  `[1,H,q,kv]` H/W constraints, `torch_npu.npu_grouped_matmul` A3 list API
  (`group_list=None`, `group_type=-1`) using the actual point routed-token/
  expert histogram and validating every non-empty expert, and
  `torch_npu.npu_dynamic_quant` with the returned `[T]` scale broadcast by
  `unsqueeze(-1)`.  The grouped list recipe is explicitly named and is not a
  packed single-X `group_type=0` claim.  FA and ordinary RoPE have no product
  builders: they stay pending with concrete geometry/backend, never become
  SUPPORTED by assertion and never use an add/GEMM surrogate.

Native availability is probed from the installed runtime; a missing symbol or
a dispatch rejection returns `pending` with a required backend, not a fake
success.  Native-output allocation is reported truthfully; no in-region copy
is hidden under a `preallocated=True` label.  Packed MXFP4/FP8 grouped work is
not substituted by a BF16 group.

`--dry-run` validates the manifest/recipe wiring without importing torch.  It
writes only `dry_run_validated`/`pending_backend` rows, `source_class=derived`
and `not_a3_measured=true`; it never writes a measured sample.

## Analysis

`python -m mindie_sim.a3_operator_curves --manifest … --results …` writes
`analysis.json`, `analysis.md`, `measurement-index.json` and
`normalized-conditions.json` (the independent reviewer's pre-use format).  It
reports per point:

- independent trial count, raw sample count, median/MAD/min/max/P90;
- local secants and residual magnitudes;
- train-only piecewise affine segments (BIC-selected, bounded segment count),
  transition brackets, and train residuals;
- launch-floor/constant-cost and saturation/plateau **hints**, explicitly not
  proof of launch latency or hardware saturation;
- calibration export provenance: per-case source class, hardware, run id and
  protocol are preserved; derived/CPU/non-A3 rows are isolated as
  non-calibratable, never promoted to `measured`; native fused semantics are
  not silently imported as legacy matmul/add/copy rows;
- `normalized_conditions(..., predictions=...)` accepts caller-frozen public
  analytic/calibrated predictions by measurement or condition key; the adapter
  never fits or invents a prediction itself;
- local per-op quant metadata (input/output dtype, scale format/bytes/axis/
  shape) takes precedence over global model quantization defaults;
- heldout interpolation/extrapolation errors separately; root owns acceptance;
- work/time rates with an explicit work unit and logical-traffic provenance;
- pending, unsupported, failed and unselected coverage with reasons/backends.

Groups keep condition, axis, full condition shape, protocol and observed
backend separate.  There is no pooled average across dtype/layout/recipe/
backend/protocol.  No source SHA or version is an admission gate.

## Commands

Start with `plan` on a fresh checkout. Its generated `manifest.json` is required
by the worker, and its `conditions.json` is required by `materialize`. Analysis
also requires the corresponding worker results. The checked-in minimal native
capability manifest can be validated independently.

```bash
# Generate the nine-model manifest, conditions, inventory, coverage and
# unmeasured/pending index under examples/operator-campaign.
python -m mindie_sim.a3_operator_campaign plan \
  --config examples/operator-campaign/campaign-config.json \
  --output-dir examples/operator-campaign --budget 320

# CPU-only validation; never a measured claim.
python -m mindie_sim.a3_operator_worker \
  --manifest examples/operator-campaign/manifest.json \
  --results examples/operator-campaign/results/dry-run-sample.json --dry-run

# Minimal native capability dry-run (softmax/masked-softmax/quant/grouped list).
python -m mindie_sim.a3_operator_worker \
  --manifest examples/operator-campaign/native-next-wave-capability.json \
  --results examples/operator-campaign/results/native-next-wave-capability-dry-run.json --dry-run

# sim_audit runs the real manifest under the existing defaultHostQueue with
# device injection.  This repository does not launch that job.
python -m mindie_sim.a3_operator_worker \
  --manifest manifest.json --results results.json --device npu:0

# Analysis (train fit, heldout evaluation, pending coverage, normalized export).
python -m mindie_sim.a3_operator_curves \
  --manifest manifest.json --results results.json --output-dir analysis

# Analyze an existing measured linearity split file (train or heldout) without
# using heldout values to choose a fit.  The 30 event repeats are one independent
# measurement, and the file's analytic_ms alias is deliberately ignored.
# Also writes adapted-manifest/results, calibration-groups.json,
# calibration-excluded-cases.json, normalized-conditions.json,
# analysis.json/md and measurement-index.json; no first-group shim.
python -m mindie_sim.a3_operator_curves --adapter linearity \
  --input train-results.json --output-dir analysis-train

# Native wave2 manifest/results use predeclared origins[].axis/anchor_width.
# The top-level CLI auto-routes the native schema to the campaign adapter:
python -m mindie_sim campaign analyze \
  --manifest train-manifest.json --results train-results.json \
  --output-dir analysis-native-wave2
# Equivalent module-level form:
python -m mindie_sim.a3_operator_curves --adapter campaign \
  --manifest train-manifest.json --results train-results.json \
  --output-dir analysis-native-wave2

# After plan: materialize a specific missing/pending condition.
python -m mindie_sim.a3_operator_campaign materialize \
  --conditions examples/operator-campaign/conditions.json \
  --condition-key cond:... --output materialized.json
```

## Scope and non-goals

- The campaign measures operator points; it does not measure communication,
  A5, or multi-rack behavior.  Communication expansion remains core-owned.
- The analytic `analytic_ms` field is a declared roofline baseline; it is never
  replaced by the observed duration.
- Pending/unsupported exact recipes are durable coverage, not zero samples.
- Reference decomposition times are useful for shape behavior but are not a
  substitute for the target fused/native kernel.
- No commits, pushes, releases, remote mutations or lease acquisition are
  performed by these modules.
- Modules are split under `a3_operator_*` ownership so each cohesive unit stays
  near the requested size; no existing core/catalog file is edited.

## Public train-only curve forecast

`mindie_sim.a3_operator_curve_forecast` exports a serializable artifact from
the existing train-only piecewise fits and answers pre-use physical queries;
the shared positive prediction policy lives in
`mindie_sim.a3_operator_curve_query`:

```bash
python -m mindie_sim.a3_operator_curve_forecast export \
  --adapter campaign --manifest train-manifest.json --results train-results.json \
  --output native-artifact.json
python -m mindie_sim.a3_operator_curve_forecast query \
  --artifact native-artifact.json --query query.json --output prediction.json
```

`export_curve_artifact` filters to train requests/results before fitting and
never copies heldout observations into support or parameters.  Export selects
the request/protocol from each analysis group's observed signature, so protocol
variants remain distinct curve IDs.  Query identity includes hardware, backend,
full precision, layout, quantization, physical features, recipe/execution kind,
target scope and protocol; missing/incompatible fields are unsupported, not
wildcards.  `target_match=false` yields a physical
candidate, never an exact production match.  Hardware/placement is always
strict; `allow_primitive_transfer` is retained only as a caller acknowledgement
and never permits cross-hardware copying of elapsed milliseconds.

Interpolation inside a fitted segment is positive linear prediction.
Transition brackets between segments return `prediction_ms=null` with explicit
endpoint bounds and `status="transition_unresolved"`; they are not silently
back-extrapolated or clamped.  Extrapolation uses the nearest segment, is
labelled, and is missing if the result is non-positive/non-finite.  Analysis
and the operational query call the same `_predict_query` policy.

Graph/eager protocol identity is canonicalized by
`mindie_sim.a3_operator_curve_protocol`: `mode`, block N,
capture/host/replay counts, output lifetime, pool sharing/output policy and
declared interval semantics are preserved; address counts, run IDs and timing
values are provenance only.  Eager and graph protocols never wildcard each
other, and a legacy artifact without `mode` cannot answer a graph-mode query.

The public export and query APIs are `export_curve_artifact`,
`load_curve_artifact` and `query_curve_artifact` in
`mindie_sim/a3_operator_curve_forecast.py`. Build queries from pre-use requests
with `campaign_query_from_request` in `a3_operator_curves_adapter.py`, or
`attention_query_from_request` in `a3_operator_attention_adapter.py`. Queries
carry the physical hardware, primitive, precision, features, shape and protocol;
observed timing values are not query inputs.

## Actual-kernel geometry, precision and warmup

- Grouped MoE conditions use the actual launched expert group from op metadata:
  `group_size`/`expert_tokens` drive `shape.experts`/`routed_tokens` and the
  point histogram.  Whole-rank routing is retained separately as
  `rank_experts`, `rank_routed_tokens` and `rank_local_expert_counts`; global
  expert count is separate again.  Point materialization carries exact
  `expert_token_counts` summing to the point's routed tokens, and the worker
  prefers that exact histogram.
- Primitive precision is resolved from the actual op.  A homogeneous FP32
  softmax keeps FP32 compute and no model MX/FP8 weight or scale facts;
  genuine INT8/FP8/MX quantize/dequant/packed GEMM metadata is preserved.
- New generic worker collections warm up 10 calls before each of the 2x30
  trials, record `warmup_scope=before_each_trial` and per-trial warmup counts.
  Historical once-warmup data is not rewritten or relabelled.
- `mindie_sim.a3_operator_attention_adapter` reuses the accepted standalone
  native FA/RoPE worker via its manifest/plan/run API.  It preserves three
  clocks, native allocating outputs, per-trial raw samples and explicit
  inclusive event/host-gap scope; generic campaign conditions lacking exact
  FA/RoPE fields return concrete pending fields. See
  [`examples/operator-attention/README.md`](../examples/operator-attention/README.md)
  for the standalone worker and campaign adapter APIs.

## TRAIN protocol/scope identity repair

- Curve artifacts use the matched result group's canonical protocol
  (including actual preallocation/cache/timing scope), retaining the pre-use
  request protocol as evidence. Missing fields are never inferred.
- Generic `coverage_scope` is part of physical condition identity, so distinct
  plain-BF16 probe descriptions are separate curves; original scope text stays
  evidence. Source queries match their own physical scope, and input order does
  not change curve identity.
- Current source-query TRAIN acceptance: generic 457/457 and native 180/180
  `ok`; graph/N/lifetime/hardware/precision/recipe/protocol negatives remain
  unsupported.

## Derived-axis TRAIN series and public queries

Declared native series axes are projected through
`a3_operator_curve_axis` before analysis/export/query:

- `q_kv` → `q == kv == x`
- `tokens_per_expert` → `routed_tokens == x * experts`, histogram
  `[x]*experts`; unbalanced patterns remain distinct
- `kv` decode keeps q fixed and varies kv
- RoPE source `seq` maps to canonical `tokens`, retaining source axis evidence
- `dtype_probe` is non-numeric and explicitly insufficient, never q fallback

`campaign_query_from_request` and `attention_query_from_request` construct
source-derived queries from original requests; queries are never copied from
artifact rows.  Current wave3 TRAIN exports yield 10 generic curves (50/52
source queries ok; two dtype probes explicitly insufficient) and 7 attention
curves (42/42 source queries ok, three clocks preserved).  Attention precision
keeps attested BF16 input/output and backend-default/unattested native compute;
CPU FP32 reference is not claimed as native compute.

`campaign_query_from_request` now shares the adapter's explicit pre-use
normalization: coverage scope from the source query context, primitive
precision quantization (not model-level labels), and the same normalized
grouped histogram pattern.  Original wave3 TRAIN manifests yield 50/52 generic
`ok` (two dtype probes explicit insufficient) and 42/42 attention `ok`; all
physical coupling/hardware/precision/mode/mask/RoPE/routing controls remain
unsupported.  No query fact is taken from a result protocol or curve row.

Measured native/generic artifacts use the actual result-environment hardware as
the curve physical key; the request placeholder is retained only as
`requested_hardware_key`.  `campaign_query_from_request` accepts an explicit
known pre-use `hardware_key`; explicit `Ascend910_9362` matches the wave3 TRAIN
artifact, while the declared placeholder or another NPU is unsupported.  No
cross-hardware transfer or requested-to-observed rewriting is performed.
