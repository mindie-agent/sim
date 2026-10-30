# Inputs and commands

## Case

`examples/quickstart.json` is a complete runnable input. Top-level fields are
`machine`, `model`, `workload`, optional `runtime` and optional `slo`. Parsing does
not load weights or contact devices.

| Field | Unit / meaning |
| --- | --- |
| `machine.device.memory_bytes` | Bytes per device |
| `device.compute_peaks` | FLOP/s by explicit compute precision; 2*M*N*K counts multiply and add |
| `device.rates.memory` | Aggregate memory bytes/s; translated to the internal `hbm` role |
| `device.rates.vector` | Declared elementary vector operations/s |
| `device.compute_units` | Matrix scheduling units used by the tile-wave recipe |
| `device.resource_units.vector` | Vector units for independent recurrence/head tasks |
| `device.work_units` | Native units for custom resources/service modes; standard FLOPs/bytes/ops retain their defined units |
| `device.buffers` | Named bytes of storage per representative compute unit |
| `device.launch_us` | Host submission service per ordinary kernel |
| `links[].capacity_bytes_per_ms` | Physical link bytes/ms |
| `links[].latency_us` | Per declared collective step/setup latency |
| `links[].semantics` | Shared, source-port or pair resource semantics |

`architecture` is an identity label, not a dispatch branch. Declare a `devices`
list for heterogeneous devices or `num_devices` plus `device` for replicated
geometry. `backend: "ascend"` with a concrete `sku` materializes a profile; explicit
device rates and capacities override profile parameters. Nominal assumptions are
included in `machine.assumptions`. Link routing is shortest-hop, not optimized for
bandwidth. Source-port resources do not implement packet congestion or duplex NIC
queuing in detail.

Device inheritance is profile, machine-wide `device`, then each `devices`
record. Each layer normalizes native fields before merging, so an explicit
`mte1_bw_bytes_per_s`, `aic_count` or `hbm_gib` cannot disappear behind an
inherited neutral value. Within one layer, neutral fields (`rates`, `buffers`,
`compute_units`, `memory_bytes`, `compute_peaks`) take precedence over native
aliases. Partial resource/precision maps preserve other inherited entries.
The legacy `cube_flops_per_s` supplies BF16/FP16 peaks only; it does not infer
low-precision throughput. Unit counts do not rescale aggregate rates.

Earlier versions ignored several native overrides after SKU expansion and
allowed `hbm_bw_bytes_per_s` to overwrite a same-layer `rates.hbm`. Correcting
this input behavior can change predictions from unchanged case files. Compare
normalized hardware declarations before attributing a change to an execution
mechanism; use an explicit neutral declaration to reproduce an earlier
effective configuration.

A generic dense model declares `layers`, `hidden_size`, `num_heads`, `num_kv_heads`,
`head_dim`, `intermediate_size`, `vocab_size` and optional precision/FFN settings.
Native catalog defaults can be selected with `model.catalog_key` and changed using
`layer_overrides` keyed by layer index. Fully explicit `layer_plans` provide
heterogeneous schedules and semantic fields without a catalog identity.

`quant` declares packed formats, block shapes, scale formats and category exclusions.
A catalog `quantization` preset changes representation; it does not quantize all
attention/residual arithmetic. Use `quant` for non-preset representation geometry.
Source metadata is not an execution override. Editing model geometry must keep the
layer and optional draft declarations coherent; replacing a complex attention
family generally requires complete layer plans.

Activation scales use `activation_group_size` (0 means one group per row),
`activation_scale_format`, and `activation_scale_pack`. These are independent of
the weight's block/scale declaration. The MXFP8 preset uses group32 E8M0 scales
packed in pairs; it does not impose that geometry on every FP8 format.

## Requests and software

Each request declares `id` (or `request_id`), `prompt_tokens`, `output_tokens` and
optional `arrival_ms`, `prefix_tokens`, `priority`, `chain_id`, `gap_ms`.
`text_tokens` plus `media_tokens` can declare text/processed media positions.
`media_encoded` skips the tower while retaining sequence/cache positions. Raw image
pixels are not accepted as a replacement for processor geometry.

Runtime mechanisms include:

- `max_num_batched_tokens`, `max_num_input_tokens`, `max_num_seqs`, `max_model_len`,
  `enable_chunked_prefill`, `kv_block_size`, `gpu_memory_utilization`.
- `parallel` (`dp`, `tp`, `ep`, `pp`, `cp`, `pcp`, `cp_plan`); flat axis names are also
  accepted. EP is inside DP*TP, and CP is inside TP. PCP adds physical devices.
- `pp_layer_partition`, `pipeline_microbatches`, `async_scheduling`.
- `graph_mode`, `graph_capture_mode`, `graph_capture_keys`, `graph_buckets`,
  `piecewise_capture_keys`, `mixed_mode`, `splitting_ops`, `invalid_full_modes`.
- `moe_recipe` (`alltoall` or `mc2`), `mc2_chunks`,
  `multistream_overlap_shared_expert`, `deepep_high_throughput` (eager graph policy).
- `routing.realizations`: optional integer expert-row counts keyed by scope,
  layer, phase and token count. Counts are execution inputs, independent of
  implementation choice or calibration. The first domain is EP=1, PCP=1,
  unpadded all-to-all with unrestricted expert groups; probability moments
  remain the default. See `operator-family-selection.md`.
- `submission` (`scheduler_ms`, `graph_replay_ms`, `completion_ms`),
- `serving_io` (explicit host preparation/results, H2D/D2H payloads and capacities,
  optional client delivery; see [Serving I/O](serving-io.md)),
  `attention_policy` (`query_tile`, KV reuse, fused batch, recurrent chunk), `kernels`.
- `lowering_passes`: ordered pass names, default `["activation_quantization"]`;
  optional `norm_quant` and `residual_norm` declare implementation fusion.
- `tensor_layouts`: consumer/argument layout requirements (`strided` or `contiguous`).
  For example, `{"gdn_gating":{"alpha":"contiguous","beta":"contiguous"},
  "short_convolution":{"input":"contiguous"}}`. Unspecified consumers accept
  strided views. A reshape that cannot preserve addresses still requires a copy.
- `final_norm_scope`: `all` (default) or `selected`, specifying normalization
  relative to hidden-row selection.
- `memory`: explicit workspace, graph, communication and runtime reserves, optional
  `cache_page_groups`, and `derive_workspace`. Byte values are per stage rank.
- `engram_cpu_offload`: requires declared host memory/vector/H2D resources.
- `vllm_impl`, `vllm_version`, `kernel_format`: implementation identity for exact
  primitive reuse. They do not supply latency.

Speculation uses optional model `auxiliary` structure plus `runtime.metadata`:
`draft_mode`, `draft_proposal_tokens`, `draft_accept_trace`, `draft_accept_tokens`,
`speculative`, and `random_seed`. `mtp_acceptance_length` and
`mtp_acceptance_by_output_tokens` are mean committed token counts; they do not
identify proposal width. Distributions are sampled reproducibly; means use an
integer-credit policy. `runtime.dspark` declares acceptance/confidence only.

These metadata keys affect behavior; other metadata is provenance. Never put a
runtime parameter only in a free-text `recipe` and expect `predict` to interpret
it. Declare proposal width explicitly; an acceptance mean does not identify it.

`pools.prefill` and `pools.decode` each declare a machine and runtime for a P/D
case. `model`, `workload` and `slo` are shared. `connector.bytes_per_s` and optional
`latency_us` describe the inter-pool transfer. The flat form keeps its `pd_link`
resource identity and also accepts the physical `Link` framing and switch fields.
For an explicit physical declaration, use `connector.link` with
`capacity_bytes_per_ms`, optional `name`, `semantics`, `latency_us`,
`transaction_bytes`, `alignment_bytes`, `protocol_overhead_bytes`, `switch_hops`,
`switch_hop_latency_us` and `switch_capacity_bytes_per_ms`. The nested link is
the complete connector declaration; it does not inherit outer transfer fields.
Its endpoint members are `[0, 1]` (prefill and decode pools), independent of model
TP/EP/PP ranks. The default semantics is one shared capacity. `per_port` charges
the source Tx and destination Rx capacities concurrently; a switch plane is an
additional concurrent capacity constraint, not another transmitted payload.

Each completed destination-sharded cache is one logical message. Framing rounds
that message once; propagation precedes wire service, and decode waits for wire
and both pools' memory work. Each pool charges one local shard per replica/stage
to its representative HBM resource. Packet streaming, per-rank connector maps
and heterogeneous KV-layout conversion remain outside this declaration.
`execution.handoff_bytes` reports useful destination payload;
`execution.connector` reports physical wire bytes, transactions and the resolved
link. See `examples/disaggregated.json`.

## Public commands

`fit-service packet.json -o fitted.json` trains a one-resource work-response curve
and reports heldout estimates separately. Bind its `model` through a
`resource_service` recipe in `runtime.kernels`; ordinary uncalibrated recipes
remain analytical. See `resource-services.md` for complete conditions, units,
phase composition and the executable retained-copy example.

```sh
mindie-sim predict CASE -o OUTPUT --trace-limit 2 --event-limit 256
mindie-sim inspect CASE -o OPERATORS
mindie-sim operators CASE --phase decode --tokens 4 --sequences 4 --context 16384 --scope model
mindie-sim graph CASE --phase decode --tokens 1 --sequences 4 --context 16384 --scope model
mindie-sim study MANIFEST --primitives STORE -o RESULTS
mindie-sim accuracy PACKET -o REPORT
mindie-sim catalog [NAME]
mindie-sim observations INDEX --scan DIRECTORY --attach-profiler PROFILER_DB
mindie-sim observations INDEX --find SEARCH_TEXT
mindie-sim profiler-rows SOURCE --limit 50
mindie-sim calibrate STORE SAMPLES_JSONL
mindie-sim bind-profiler CATALOGUE STORE BINDINGS_JSONL
mindie-sim predict CASE --primitives STORE
mindie-sim fit-rate OBSERVATIONS_JSON
mindie-sim validate PREDICTION RAW_SERVICE_OBSERVATION --boundary client_stream
```

`inspect` builds the first colocated iteration with the normal frontend; it does
not run the complete service timeline. For PD, inspect each pool as a colocated
case. `catalog NAME` is available for structural draft/component inspection.
Invalid inputs and unavailable mechanisms are errors, not empty success results.

`operators` treats `--tokens` as query tokens **per sequence**. `body` scope
uses the requested geometry before graph-bucket padding; `model` scope uses the
public target frontend, including graph padding, embedding, normalization,
selection and logits, with no draft proposals or sampling. Both use the same
kernel recipes and execution engine. The returned `operator_sum_ms` sums isolated
device programs; it is not TTFT, TPOT or a concurrent graph's duration.

`graph` executes dependencies, streams and shared resources for the first
independent DP cohort, including all participants when EP spans DP. `sequences`
is per participating replica. It preserves the frontend's chosen capture scope.
`--include-submission` adds host costs; `--replay-scope` explicitly overrides
capture submission while keeping lowered shapes. `--skip-isolated` omits the
independent-cost baseline. Prefill model analysis of a case containing media
requires `--media-tokens-per-sequence`, with 0 explicitly excluding the tower;
this also applies to `operators`. See `graph-analysis.md` for the API, external
inputs, report boundaries and limits of the precedence/resource bounds.

`study` enumerates explicit candidates across named geometry probes and reuses
these public paths. Its default is isolated operator screening, with optional
graph execution and separately evaluated original-case serving constraints.
See `studies.md` for overrides, comparison domains, budgets and failure behavior.
`accuracy` exposes independent heldout comparison from `accuracy-regression.md`;
neither missing observations nor an unassessed accuracy report gates simulation.
