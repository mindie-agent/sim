# Sharing declared structure with the profiler

The simulator owns model configuration adapters, layer plans, graph builders and
implementation recipes. The profiler owns raw capture parsing, observed tensor
contracts and measured event attribution. Sharing structure must reuse these
owners rather than introduce another model catalog or raw CSV/DB parser.

## Structure export

`mindie_sim.structure.export_structure` reuses `target_graphs` and
`RecipeBook.expand` to produce `mindie-model-structure/1`. The manifest describes
the selected graph, source declaration, phase/scope, node identity, layer/role,
tensor contracts, dependencies and implementation. It contains no predicted or
measured durations and does not run the cost engine.

Export is metadata and graph construction only. It neither loads weights nor
initializes devices. Implementation and workload declarations can select a
different graph; measurement availability and observed durations cannot.

The catalog's optional MTP, DSpark or media declarations do not establish that a
particular capture executed those paths. The consumer must retain that distinction.
Repeated layers can have identical tensor signatures while remaining different
declared nodes.

Each node and subgraph also exports `physical_work` and `physical_signature`.
These contain declared resource quantities and implementation-sensitive identity,
not latency. Occurrence IDs, layer numbers and structure-only membership
annotations do not split identical work. Precision, resource demand, physical
recipe and internal subgraph dependencies remain significant. Changing physical
work changes `graph_id`, even if the tensor projection remains equal.
Explicit recipe, family and program bindings also participate in graph and
physical identity. Reusing a recipe name with a different implementation cannot
reuse an old mapping silently.
An unbound matrix's semantic recipe/family labels resolve through its actual
fallback program before fingerprinting. Explicit family/recipe bindings,
invocation rules and resource services retain those selection axes. This lets
equal Q/K or output projections form a physical cohort without erasing a real
implementation difference.

The profiler uses these declarations for multiplicity-level `windows[].cohorts`.
Every candidate must have the same contract and physical work on one declared
stage/phase. For one captured device, complete symmetric DP candidate sets may
share a cohort: all target IDs remain, `replica` is unknown, and
`declared_candidate_instance_count` differs from the per-device
`declared_instance_count`. Multiplicity divides by the per-device count.
Incomplete DP symmetry, different physical hardware, or missing device identity
cannot establish that comparison. An explicit occurrence binding retains its
known replica. Conflicting evidence, overlapping candidate boundaries and
unknown multi-launch stages are excluded. This preserves instance ambiguity
while providing actual invocation counts, full-event Mean/P50 durations,
activity sum and interval union. It never allocates group timing to individual
layers. A window's step label is not proof of the number of complete iterations.

The public call is:

```python
export_structure(
    case, phase="prefill", tokens=16384, context=0, sequences=1,
    scope="body", source={"kind": "caller_declaration"}, expand_calls=True,
)
```

`case` uses the existing complete Case schema. Its machine fields satisfy that
schema without being evaluated as rates or timing. The CLI accepts the same JSON:

```bash
python -m mindie_sim.structure case.json \
  --phase prefill --tokens 16384 --scope body --output structure.json
```

The selected scope is `body` or `model`, and phase is `prefill` or `decode`.
Selected native resource plans use the same pure `RecipeBook.prepare` and device
ownership geometry as execution. Their prepared work is exported; relevant
core counts and buffers appear in `execution.resource_geometry`. Source tiling
can therefore change declared work without invoking a cost engine or fitting
measured timing.
Export covers the first independent DP cohort. Each node retains its replica,
stage, layer, block, role, family, dependencies, operator, contract, contract scope,
boundary and lineage. Expanded calls retain `source_calls`; dependencies outside
the selected scope remain in `external_inputs`.

The frontend marks the known attention and FFN builder boundaries with
`metadata.structure_block`; their normalizations receive the same block.
`metadata.structure_role` distinguishes `body` operations from the plain,
hyper-connection or attention-residual builder's `residual` operations.
Inserted quantizers inherit their consuming operation's scope. Expanded
source-call kernels and their join inherit the caller's explicit scope.
Generic residual/norm fusion keeps the target normalization's scope and retains
the residual in `fused_from`. `fused_origin_scopes` retains the original operator
IDs, layers, stages, blocks and roles, including a preceding FFN residual fused
with the following Transformer block's attention normalization.
These labels do not change work, tensor geometry, dependencies or timing recipes,
and do not infer a block from an operator's name. Standalone declared graphs
without these metadata fields may leave the role unspecified.

Profiler rejects an explicitly supplied analysis `--phase`/`scene.phase` that
conflicts with the manifest's execution phase. An omitted phase stays unspecified.
The mapping retains `declared_execution` and caller-supplied `observed_phase`;
the latter is not inferred from the trace. Bound nodes retain `contract_scope`
for subsequent comparison.

`graph_id` hashes canonical manifest content excluding `graph_id` and caller
`source`. The profiler verifies this identity when accepting the graph. This
checks a declaration, not the identity or contents of a raw capture.

## Declared operator graphs

`export_op_structure(ops, *, model, execution, kernels=None, source=None,
expand_calls=True)` exports an existing `OpSpec` DAG through the same node,
contract, call expansion and hash implementation. A sequence selects replica 0;
a mapping supplies explicit replica-to-DAG assignments. It does not need a
hardware Case, model weights, a capture, or the profiler package.

`model` and `execution` describe the caller's declared structure. Execution
requires a `phase` matching each operator; scope and parallel placement are
explicit caller declarations. The exporter adds the actual `replicas` and
`expand_calls` values. It does not apply the LLM exporter's representative-shard
or first-DP-cohort assumptions to a general graph. `kernels` is the existing
`RecipeBook.from_mapping` input when source calls require expansion.

For example, `profiler/tests/live_framework_capture_workload.py` declares
`Linear(256, 512) -> ReLU -> Linear(512, 256)` for an FP32 input `[8, 256]`.
Both Linear modules have bias because the source uses PyTorch's default.
The following declaration comes from that workload definition and its arguments,
without reading any observed event, shape or duration:

```python
from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.structure import export_op_structure

quant = QuantSpec("fp32", weight_format="fp32", activation_format="fp32")

def linear(module_id, deps, in_features, out_features):
    return gemm_op(
        op_id=module_id, deps=deps, stage=0, layer=-1, requests=(),
        m=8, n=out_features, k=in_features, quant=quant,
        category="dense_mlp", family="linear", phase="eager_mlp",
        metadata=(
            ("module_id", module_id), ("module_type", "Linear"),
            ("input_shape", (8, in_features)),
            ("weight_shape", (out_features, in_features)),
            ("output_shape", (8, out_features)),
            ("bias", True), ("bias_shape", (out_features,)), ("bias_dtype", "fp32"),
        ),
    )

ops = (
    linear("0", ("input",), 256, 512),
    vector_op(
        op_id="1", deps=("0",), stage=0, layer=-1, requests=(),
        elements=8 * 512, ops_per_element=1, dtype="fp32",
        family="relu", phase="eager_mlp", recipe="relu", shape=(8, 512),
        metadata=(("module_id", "1"), ("module_type", "ReLU")),
    ),
    linear("2", ("1",), 512, 256),
)
manifest = export_op_structure(
    ops,
    model={"kind": "sequential", "input_shape": [8, 256], "dtype": "fp32"},
    execution={
        "phase": "eager_mlp", "scope": "model",
        "replica_axis": "data_parallel", "shard_scope": "unsharded",
    },
    source={"path": "profiler/tests/live_framework_capture_workload.py"},
)
```

The node IDs are `rank:0/0`, `rank:0/1`, and `rank:0/2`; module IDs remain explicit
metadata. These Sequential module IDs do not declare Transformer layer numbers,
so all nodes use `layer=-1`. `rank:0/input` is the external input, and the three
dependencies preserve the declared execution order.

The Linear contracts are `matrix_bias` with M/N/K, activation/weight/output
dtypes, bias shape and bias dtype. They cannot match a bias-free `matrix`
contract. ReLU keeps its complete shape and dtype. A scaled activation with bias
has no contract until its additional scale descriptors are represented; this
exporter never drops those fields to produce a match.

These are semantic declarations, not a claim about physical kernel boundaries
or a complete timing model for bias epilogues. All exported nodes retain the
`semantic` boundary. The capture must independently establish which kernels ran
inside each module scope. Changing the workload activation or dimensions requires
changing the declaration; profiling observations never rewrite it.

## Attention, cache and fused body boundaries

The existing GQA and recurrent builders supply tensor metadata alongside their
work records. This metadata does not change FLOPs, bytes, dependencies or engine
recipes. `structure_native_contracts` projects that independent declaration onto
the supported observation boundary:

| Declared operation | Compared fields | Excluded or unsupported fields |
| --- | --- | --- |
| Fresh-prefill GQA attention | Q/K/V and primary output shapes, dtypes and ND descriptors | Mask and auxiliary slots, sequence-array values, layout scalar, causal/dispatch flags and selected implementation |
| GQA cache write | Active K/V element counts and dtypes, slot count | Native storage layout, cache capacity, slot dtype/values, scalar flags and selected implementation |
| Indexed latent attention | Q, Q-rope and output descriptors; K/V/K-rope cache head dimensions and dtypes | Cache capacity/page size, sequence/index tensors and dispatch flags |
| Shared-KV attention | Q/output descriptors; original and optional compressed cache head dimensions and dtypes | Cache capacity/page size, sequence/index/sink tensors and dispatch flags |
| Gated RMS output norm | Three input and two output tensor slots, with declared weight, gate, group width and no bias | Epsilon, gate activation and normalization order are not observed |
| KDA gated RMS output norm | Flattened active rows, feature width and weight descriptors | Leading tensor axes, epsilon, gate activation, bias/residual flags and auxiliary statistics |
| Gemma RMS norm | Native input/weight/output/statistics slots, with model-declared weight offset | Epsilon, weight values and captured binary implementation |

The attention projection is `attention_active_qkv` for compatible
`FusedInferAttentionScore` observations. The builder supplies it only for
fresh prefill with no prefix and no CP/PCP partition. Prefix-prefill, decode
and paged-cache operands remain unsupported by this projection; the exporter
does not infer a dense TND view from cache capacity. An implementation-specific
source call also remains unsupported as one tensor contract. Its expanded
kernels retain their actual boundaries, dependencies and `source_call_id`.

The cache projection is `kv_cache_store_active_elements` for compatible
`ScatterPaKvCache` observations. Its active element counts come from valid
new tokens, local KV heads and head width, including during prefix-prefill or
decode. They do not include the existing context or padded query rows. Current
declarations require CP=PCP=1 and equal activation/KV dtypes; quantized cache
conversion and distributed storage need their own declared interfaces.
The profiler first checks the supported native ABI before projecting a raw row.
Matching element counts do not establish equal native tensor layouts, head
factorization, cache ownership or contents. The native descriptors remain in
the observed record.

The MLA/DSA builder supplies `sparse_attention_active` projections for compatible
`SparseFlashAttention` or `SparseAttnSharedkv` observations. Latent dimensions,
local query heads and rotary width come from the model plan; compressed-cache
presence comes from the plan's compression branch. Native operator families stay
separate even when Q and output shapes coincide. A missing compressed operand
does not match a declaration that consumes one. CP/PCP partitioning and a selected
source implementation do not inherit this generic projection.

`LightningIndexer` combines score calculation and top-k. The semantic graph
keeps those as two separate nodes. An explicit `lightning_indexer` subgraph
retains both nodes and provides the candidate interface described below;
neither child independently claims the fused native contract.

The GDN fused norm contract names `_layer_norm_fwd_1pass_kernel_npu` and retains
ordered `(x, weight, gate) -> (output, fp32 statistics)` slots. The builder
declares the RMS/grouping/weight conditions and keeps `fused_from` lineage.
KDA uses a distinct `gated_norm_active_rows` contract for
`layer_norm_gated_fwd_kernel`: its source wrapper flattens x and gate by their
final feature dimension, restores the output shape and discards statistics.
The model plan's existing KDA/GDN identity selects the declaration; it is not
inferred from an observed kernel name.

The Qwen configuration adapter retains the model class's Gemma RMS semantics:
weight offset 1 and multiplication before the output dtype cast. These catalog
facts preserve the independent local source revision, file hashes and locations;
they do not identify the captured binary. Body normalization inherits these
facts and can compare the `GemmaRmsNorm` native ABI. It never aliases that ABI
to ordinary RMS. The current A3 Qwen cases do not select residual-norm fusion;
even when explicitly enabled, the generic fusion does not establish that
`AddRmsNormBias` ran. That native implementation remains a declaration gap:
an empty bias slot cannot reveal whether the caller passed weight or weight+1.

The captured
`split_qkv_rmsnorm_mrope_kernel` rows expose only one of the source's four
outputs, with Q and gate sharing a shape. These rows cannot prove a complete
fusion interface or attach the fusion to a standalone QK-norm/rotary node.

The source-backed offline checks currently cover these independent declarations:

| Existing case | Declared prefill tokens | GQA Q / K,V shape | Gated norm shape |
| --- | ---: | --- | --- |
| `qwen35-a3-tp4` | 8192 | `(8192,4,256)` / `(8192,1,256)` | `(65536,128)` |
| `qwen36-a3-tp4` | 8192 | `(8192,6,256)` / `(8192,1,256)` | `(98304,128)` |
| `qwen38-a5` | 16384 | `(16384,24,256)` / `(16384,4,256)` | `(786432,128)` |

The same tests compare `glm53-a3-w4a8`, tokens 4096, with real A3 sparse
attention: Q/output `(4096,8,512)`, Q-rope `(4096,8,64)` and cache head widths
512/64. For `deepseek-v4-flash-0731`, tokens 4096, the retained A3 shared-KV row
has Q/output `(4096,8,512)` and no compressed operand. It matches the two
uncompressed layer declarations and rejects the 41 compressed layer declarations.
Two additional unchanged compressed rows match those 41 compressed declarations
and reject the two uncompressed layers. Their differing index/block-table
descriptors remain visible, but this projection does not distinguish compression
ratios. The identical Q/output shape does not match GLM's different native family.

The Kimi A5 source recipe sets `max_num_batched_tokens=8192` for a 32768-token
request. Declaring that one 8192-token chunk through `kimi-k3-a5` produces KDA
norm-gate rows `(98304,128)` and weight `(128,)`, compatible with the retained
complete raw row. The retained decode row with all tensor fields `N/A` remains
unknown. Kimi's observed fresh-prefill FIA has unabsorbed Q/K width 192 and V/O
width 128; the current generic MLA graph absorbs Q into latent width 512 before
attention. This is an implementation-boundary gap, so no FIA contract is attached
to that latent node. The KDA check does not establish full Kimi attention coverage.

`tests/test_structure_native_contracts.py` obtains these shapes through the
production builders from the existing cases and explicit token arguments. It
compares them with unchanged retained profiler CSV rows in
`tests/fixtures/attention_cache_contracts/` and `tests/fixtures/body_fused/` in the
profiler source tree. Their provenance records retain original capture paths,
hashes and line locations. The tests also preserve real paged-versus-contiguous
attention differences and cache-capacity differences across original rows.

Qwen3.5 and Qwen3.6 above share the cache activity projection; cache alone cannot
identify either model. Their attention and gated norm geometry differ. Repeated
layers within each model still have equal contracts and remain distinct
candidates. None of these descriptor checks establishes Host/module occurrence
binding, numerical equivalence, full-model coverage, new live capture or timing
accuracy. Other model families and native cache/attention interfaces remain
separate support gaps until their declarations and observed boundaries agree.

## Explicit semantic subgraphs

The additive `subgraphs` field in `mindie-model-structure/1` describes a native
interface over existing semantic nodes. It never inserts another execution
node or duplicates member work. Each group includes:

- `subgraph_id`, `node_ids`, `role`, `family`, `replica`, `stage`, `layer`,
  `phase`, `block` and `structure`;
- `external_inputs` and `external_outputs`, calculated from the final lowered
  DAG, after quantization, generic fusion and `RecipeBook` source-call expansion;
- `contract`, `contract_scope`, independent implementation conditions, source,
  and `coverage.kind`/`coverage.limitations`;
- `lineage.declared_op_ids` and `lineage.source_call_ids`, retaining the original
  declaration and expanded source boundaries.

Both nodes and subgraphs carry the actual layer plan's `structure`: attention
kind/architecture, cache kind, window/compression branch, indexer execution and
top-k ownership, FFN kind and residual kind. This allows the consumer to group
real declared GQA/MLA/DSA/recurrent, MoE/dense and residual branches. It does not
assert kernel coverage for every structure name. Shared owner identities remain
per-layer values rather than a synthetic common owner. `structure` describes the
Transformer block configuration, whereas a node's `block` and `role` identify the operation's
own builder scope. An FFN node can therefore share a layer's GQA configuration
without being classified as an attention operation.
MoE and dense/MLP describe the FFN sublayer, not the entire Transformer block.
Groups can span layers or attention/FFN blocks within one replica, pipeline
stage and phase. Such groups use `layer=null` or `block=null`, retain `layer_ids`
and the actual `owner_scopes`, and keep only common `structure` fields. The
exporter derives these scopes from member nodes and preserved fusion origins.

The existing MLA builder declares `lightning_indexer` over its score and top-k
nodes when CP=PCP=1 and no candidate filter is selected. The active contract
compares query, weighting tensor, cache head and selected-index shape/dtype.
It excludes page capacity, sequence/index values, scores, scales and scalar
flags. Quantized indexer variants do not inherit the BF16/FP16 contract.
`index_topk.dtype` remains the FP32 score-comparison compute dtype;
`metadata.selection_output_dtype` separately declares INT32 selected indices.
This subgraph does not require the MLA prolog selection.

An MLA prolog candidate additionally requires the explicit software selection:

```json
{
  "runtime": {
    "kernels": {
      "lowerings": {
        "mla_prolog": "vllm_ascend_mla_prolog_v3"
      }
    }
  }
}
```

This selection applies to latent MLA with dense cache, CP=PCP=1, BF16 activation
and cache, and declared BF16/INT8/FP8-E4M3 projection weights. It groups existing
query/KV projections, normalization, rotary, cache store and absorbed query
operations. The independent source requires an unquantized `kv_b` checkpoint:
the selection marks the existing quantization category `mla_absorb` as BF16.
The same category governs the `kv_b` weight ledger, source and resident capacity
bytes, and `q_absorb`/`v_expand` execution. Without this selection it inherits
the ordinary `mla` precision rules, including any existing MLA exclusion.
Projection dimensions/FLOPs stay unchanged; the selected representation changes
its weight/activation bytes and removes unnecessary absorption quantization.
No fused timing recipe is inferred.
Quantized prolog operands require the `activation_quantization` lowering pass.
Disabling that pass is a visible implementation-domain error; the exporter
cannot declare an INT8/FP8 boundary while leaving a BF16 producer in the DAG.
BF16 operands do not require a conversion, and supported custom pass orderings
retain the actual quantized producer boundary.
Full indexer layers in the quantized selected domain must consume that same
normalized-query activation format. An independent `indexer_fp8` quantization
exclusion that changes the consumer to BF16 is an implementation-domain error,
because it would expose a BF16 `q_norm` instead of the declared shared quantized
output. Declaring both projection and indexer operands as BF16 remains valid.

The prolog contract retains active token rows, six logical parameter
shape/dtype pairs, cache head dimensions, Q/Q-rope and optional normalized-query
outputs. It excludes native parameter NZ storage, leading-axis grouping, rope
input values, scale tensors, cache allocation, slot values and scalar flags.
`contract_scope` makes these exclusions explicit. Source provenance records
the independently inspected vLLM Ascend revision and file hash; selecting it
does not prove that the captured binary used that revision.

The final DAG determines the output boundary. In the selected GLM INT8 path,
`q_a.quantize` is outside the prolog. The shared `q_b.quantize` is inside it and
is an externally visible producer for full-layer `index_q`, including its
activation scale output. It is not relabeled as a BF16 `q_norm` output. Rotary
is an explicit attention input, so the Q-rope producer remains externally
visible alongside the cache store and absorbed query. A generic norm/quant
fusion or source-call expansion retains the group membership and entry boundary.

The GLM adapter obtains full/shared indexer ownership from the independent
configuration. Full layers execute the indexer and expose normalized query;
shared layers depend on their owner's top-k result and do not fabricate a
per-layer indexer. Retained GLM A3 prefill/decode slices match both declared
prolog interfaces and the LightningIndexer interface. Kimi's current independent
case declares BF16 dense attention while its retained A5 prolog reports FP8;
selecting the prolog does not rewrite that precision and the mismatch stays
visible. This is distinct from successfully parsing the observed native ABI.

`tests/test_structure_subgraphs.py` verifies these real capture comparisons,
shared-owner dependencies, quantized/rotary external outputs, source-call
children and joins, generic norm/quant fusion, and weight/execution capacity
agreement. Tests do not replace the captured shapes or load captures into the
declaration.

The profiler may match one event to a subgraph candidate while preserving its
member node IDs. Host/module evidence must contain the complete declared group,
or an explicit binding must name the subgraph. Group timing remains an inclusive
measurement of that boundary: it is not divided among child nodes or added to
their timings as another execution. Descriptor agreement alone is still a
candidate, including when only one subgraph has that contract.

## Further native fusion correspondences

The generic `prediction.structure_groups.declare_group` records an existing
builder's members and entry operators. It is used by attention, FFN and residual
builders; the exporter does not identify a sublayer by parsing operator names.
The following declarations distinguish source correspondence from a comparable
observed tensor contract:

| Native boundary | Declared members | Current comparison boundary |
| --- | --- | --- |
| `VllmAscendQuantLightningIndexerV2` | Index scores and top-k | INT8 query/cache, FP16 weighting tensor, declared query/cache scale precision and INT32 selected indices; candidate-filter branch excluded |
| `Compressor` | Value/gate projections, position-weighted state pooling, RMS norm and rotary | Input/parameter descriptors, state channel width, rope width and output width/dtype; allocated output rows and state/page capacity excluded |
| `HcPre` | FP32 input conversion, projection, inverse RMS, scale/bias/gates, Sinkhorn and residual mixture | Canonical input/primary-output descriptors; epsilon, iteration count, optional premix and extra pre-coefficient output excluded |
| `HcPost` | Sublayer output and residual post-mixing | Canonical input/output descriptors; leading token grouping excluded |
| `AttnResFwd` | FP32 residual packing, normalization, score projection, softmax, mixture and output cast | Independent semantic correspondence; retained backing buffers do not identify native active views |
| `GroupedMatmulSwigluQuant` / `GroupedMatmulSwigluQuantV2` | Expert gate-up, SwiGLU and the actual inserted output quantizer | Local expert-bank count, input/intermediate widths and operand precisions; routing rows, group values, packing and scale representation excluded |
| `ChunkKdaFwd` | Chunk products, triangular solve, WY transform, scan and output; gating only for the explicitly selected older wrapper | `multi_launch_stage_unknown`, no tensor contract: several launches share native kind and descriptors |
| `RecurrentKda` | QK normalization, in-kernel gating and recurrent update | With the explicit older wrapper selection, gate/beta/A-log/dt-bias descriptors and operand precisions; Q/K/V and state backing allocations do not establish active shapes |

The FFN fused group excludes attention, the expert down projection and
dispatch/combine communication. MC2 chunks retain distinct groups. The
quantizer feeding the down projection belongs to the gate-up fusion, whereas
the down GEMM remains an external consumer. V1 and V2 comparisons retain their
native kinds and INT8 versus INT4 weight domains; they do not establish which
custom operator the captured binary selected.

The compressor helper is shared by main-attention and index-key compression.
Its valid output work is the number of completed groups per request. The native
output allocation may reserve one additional row, so that allocation is not
used as the number of computed rows. Index compression preserves its independent
head width and overlap factor. Cache storage, Hadamard conversion and index
quantization are external producers/consumers. The ordinary latent indexer
instead uses mean/variance LayerNorm with affine weight and bias, followed by
rotary. It is not aliased to the compressor's RMS normalization.

`LayerPlan.compressor_norm_dtype` defaults to BF16. The independently inspected
source supports an FP32 norm-parameter branch; callers can explicitly declare
`compressor_norm_dtype="fp32"` on the relevant plans. That declaration changes
parameter read/residency bytes while retaining the BF16 checkpoint representation
through the existing runtime category. It is never selected from an observed
dtype or a device name. The retained A3 sample has FP32 norm parameters, so the
default BF16 declaration deliberately remains different. A replay override must
retain its base case and source-branch justification as a caller declaration.

The retained older KDA wrapper can be declared with
`runtime.kernels.lowerings.kda="vllm_ascend_kda_in_kernel_gates"`. Its independent
source revision and file hash are recorded. Beta sigmoid produces FP32 outside
the native group; gate transformation and QK normalization are internal for
recurrent decode. The current prefill wrapper performs gate-prefix work outside
the chunk operator, and the default group preserves that external boundary.
Selecting the older wrapper includes its internal gate work but still cannot
identify which chunk stage a captured event executed.

`tests/test_structure_fusions.py` compares independently built case declarations
with retained A3/A5 rows, tests the compressor precision and valid-row distinction,
checks real preprocessing dependencies, and preserves cross-layer fusion scopes.
Source references distinguish committed files from local worktree content and
retain file hashes. They are source evidence, not capture binary identities.
These tests establish semantic and descriptor agreement; they do not establish
layer occurrence binding, new live capture, numerical accuracy, fused resource
costs or service throughput.

## Dependency direction

The profiler consumes the manifest as data and does not import `mindie_sim`.
The simulator's explicit profiling adapter imports `mindie_profiler` lazily to
consume the shared observation parser and tensor contract. Ordinary prediction,
graph/operator analysis and structure export do not require the profiler package.
A missing optional installation is an error on the profiling path; it does not
select a copied legacy parser.

The shared observation APIs currently require the matching development source.
Released `mindie-profiler 0.5.0` lacks them, so the optional dependency declaration
`mindie-profiler>=0.5.0` alone does not establish compatibility. With adjacent
source checkouts, install the matching profiler from the sim directory:

```bash
python -m pip install -e ../profiler
```

This development workflow does not claim a coordinated formal release.

This is one contract between the existing components, not a new service or
registry. Keep the structure artifact small enough for the selected graph and
leave large captures at their original location.

## Reading the attribution

The profiler's `analyze --structure MANIFEST` returns `structure_mapping`.
Explicit event relationships may be supplied with `--bindings BINDINGS`.

- Matching shape, dtype, layout and role narrows candidates. Even a unique tensor
  candidate does not establish the actual module call, cache state or vendor
  implementation.
- Repeated layers remain ambiguous without event-level evidence. Neither event
  order nor predicted duration can assign a layer.
- Exact relationships require a concrete event/locator binding or a supported
  Host call-chain association. A reused graph task ID alone is insufficient.
- A node can correspond to multiple kernels, and a fused kernel can correspond
  to multiple nodes. Shared event intervals are counted once in parent totals.
- Missing observed fields stay unknown. A declaration is never copied into an
  observed tensor field to make a match succeed.

Automatic Host binding requires `globalTaskId` to be unique in the complete TASK
table, with a unique direct Task-to-Host connection or a unique Task-to-CANN
connection and an unambiguous Host chain. The CANN-to-Host edge may use a unique
connection or strict same-thread containment; equal or crossing Host ancestors
are not a unique chain. Device interval containment alone cannot prove binding.

When multiple correlation reports provide scopes for the same occurrence, every
scope must support the same unique tensor candidate. Conflicting scopes remain
ambiguous rather than discarding the incompatible part of the evidence.

Capture-rank assignments to simulator replicas and PP stages are explicit
`rank_replicas` and `rank_stages` binding fields. Unspecified assignments retain
applicable candidates; physical rank indices are not presumed to be DP replicas.
Layer/block totals retain replica and stage. One fused event cannot cross those
boundaries.

Independent operator comparisons consume the shared raw observations and tensor
contracts. `mindie_sim.cohort_comparison.compare_cohorts(case, mapping_report)`
also consumes `mindie-profile-structure-mapping/1` directly, without importing
profiler or rereading the capture:

```bash
python -m mindie_sim.cohort_comparison case.json mapping.json -o comparison.json
```

The consumer verifies the regenerated graph, complete candidate set, tensor
contract and physical work, then prices one representative declared program.
Physical fingerprints normalize named routing producers while retaining their
alias relationships, so layer-specific node IDs do not split equal work.
It compares per-call prediction with complete-event Mean/P50 and multiplies by
observed call count for activity-sum error. Instance counts remain independent
of step counts. These are conditional comparisons under the declared workload
and contract scope, not proof of native dispatch or per-layer occurrence.

Selected [native fusion resources](native-fusion-resources.md) provide one
physical program for a compound boundary. Unselected multi-program subgraphs
are `reference_only`, with no fused-kernel error score; clipped windows remain
unscored. Attribution does not calibrate a primitive or assign a train/heldout
role automatically; those remain explicit operations under
[the observation contract](observations.md).

Comparison has no native-operator allowlist. Dense and grouped matrices keep
distinct contracts; grouped expert-bank size does not establish the observed
per-expert row histogram. Source-declared INT8 V3, MXFP8 V3 and FP8/FP4 V4
interfaces preserve their different scale order and NZ axes. Where declared
work is incomplete, `contract_scope.prediction_coverage` exposes the gap:
W4A8 grouped programs without correction/rescale work produce
`reference_only / quantized_epilogue_unmodeled`, retaining their measured
statistics and partial prediction without a full-native error score.
Compound boundaries inherit incomplete member work even after a generic
single-launch fusion. The explicit `ascend_w4a8_msd` gate/up program and
`ascend_w4a8_msd_grouped_down` source program replace their corresponding
correction/rescale work; only those implemented gaps are removed. Their
integer routing and resource-rate declarations remain independent conditions,
not a recovered captured expert histogram or dispatch identity.
Likewise, an MXFP4 program that expands packed operands to BF16 without modeling
unpack, scale conversion and the native matrix path remains
`reference_only / mxfp4_native_execution_unmodeled`. A verified tensor ABI or
complete semantic boundary does not by itself make that physical program
complete.

### Per-call accuracy fields

Comparison reports declare `accuracy_policy.version=mindie-sim.cohort-accuracy/1`
at the top level.
Each cohort also carries its policy version and `observed_per_call_us`,
`predicted_per_call_us`, signed `per_call_diff_us` (prediction minus observation),
`abs_error_per_call_us`, `accuracy_tier`, `accuracy_status`, `accuracy_pass`
and `accuracy_tolerance`. The observed statistic is the complete-event mean.
Above 50 us, `relative_15pct` accepts absolute
relative error at most 0.15. At or below 50 us, `absolute_15us` accepts absolute
per-call difference at most 15 us. Both tolerance boundaries are inclusive.
Passing rows use `within_relative_tolerance` and `within_launch_floor_tolerance`,
respectively; a failing row uses `outside_tolerance`. The launch-floor label
names the absolute acceptance tolerance, not a prediction adjustment: no floor,
rounding or timing adjustment is applied.

Only `status=compared` is eligible. Reference-only and unscored rows retain
available per-call values and differences, but their pass and tolerance are
null. Their tiers are `reference_only` and `unscored`, respectively; a failed
prediction is also unscored. Their accuracy statuses use those same labels.
Accuracy acceptance remains conditional on the
declared boundary and does not establish native dispatch or layer identity.

The pure `mindie_sim.cohort_accuracy.accuracy_fields(row)` function recomputes
these additive fields from a saved comparison row, ignoring older accuracy
fields and residuals. It neither predicts nor reads captures and does not
modify its input. `accuracy_policy()` supplies the report-level definition.
For mixed-family summaries, count pass/fail separately by tier and report both
eligible cohort counts and their observed event counts. Exclude reference and
unscored rows from those denominators while retaining separate counts for them.
Event weighting repeats the cohort-mean decision; it is not a per-event
residual test.
Do not apply the policy to an averaged family duration, allow signed errors to
cancel, or combine overlapping declarations/rank windows into one acceptance
denominator.

## Verification boundary

An offline integration case should read an unchanged real CSV or DB slice, export
a graph through the production builders, and attribute those original events
through the profiler. Keep source locators and observed durations inspectable.
This establishes the component path, not new hardware performance.

Discriminating cases cover repeated-layer ambiguity, many-event and fused-event
relationships, repeated replay IDs, missing tensor fields, and a duration-only
perturbation that changes timing but not graph identity or mapping candidates.
Controlled perturbations test the algorithm; they are not new hardware captures.
Standing NPU capture and complete service accuracy require their own completed
acceptance records.

The profiler source test `tests/test_structure_sim_acceptance.py` exercises the
actual export/parser/mapping path with this repository's provenance-backed A5
CSV slice and the profiler's GLM45 replay DB slice. It requires both source trees
on `PYTHONPATH`; an installation without the sim source fixture explicitly skips
the cross-repository acceptance. The controlled model declaration and explicit
bindings are algorithm evidence, not claims about the capture's model layers.
