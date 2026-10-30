# Operator-family implementation selection

Ordinary model cases can select source implementations through
`runtime.kernels.lowerings`. The same declaration drives semantic lowering and
the `RecipeBook` used by serving, isolated operators and composed graphs.
No Python-only recipe injection or measurement lookup is needed.

An implementation selection declares software behavior. It does not establish
that a hardware capture ran that implementation, or that its timing is accurate.
Unselected families retain their analytical recipes.

## Declaration

```json
{
  "runtime": {
    "kernels": {
      "lowerings": {
        "gqa_attention": {
          "implementation": "vllm_triton_unified_attention_2d_pointer",
          "options": {"compute_capability": [8, 9]}
        },
        "rmsnorm": {
          "implementation": "vllm_cuda_rms_norm",
          "recipes": ["rmsnorm"]
        }
      }
    }
  }
}
```

The short string form remains supported, including the existing
`"gdn_prefill": "gdn_prefill_arch35"` binding. The object form separates:

- `implementation`: a registered implementation, with supported semantic selectors;
- `options`: implementation-specific software dispatch inputs;
- `recipes`: optional exact semantic recipe names, for operator-family selectors.
  Omitting this field selects every operator in that family. Block selectors
  such as `gqa_attention` do not accept a recipe filter.

The model label and hardware name never select implementations. Compute
capability is an explicit input to the pointer implementation's source tile
dispatch; it does not select a physical performance correction.

## Connected domains

| Selector | Implementation and supported domain | Explicit limits |
| --- | --- | --- |
| `gdn_prefill` | Existing `gdn_prefill_arch35` caller, convolution and Prepare/H/O plans | Its existing projection, dtype, PCP and mixed-mode restrictions remain |
| `gqa_attention` | `vllm_triton_unified_attention_2d_pointer`; one sequence per launch; BF16/FP16 Q/K/V; causal or noncausal; ordinary prefill or explicitly selected 2D decode | No window, quantized KV, CP/PCP, fragmented query ranges, tensor descriptors or segmented 3D decode |
| `rmsnorm` | `vllm_cuda_rms_norm`; ordinary weighted or explicitly weightless RMSNorm; BF16/FP16/FP32 | Requires the semantic reduction width and weight-presence contract; no quantized output or additional returns |
| `add_rmsnorm` | Same implementation, source fused residual-add RMSNorm kernel | Requires an actual fused semantic operator and preserves its residual output |
| `grouped_gemm` | `ascendc_grouped_nd_bf16`; ordinary MoE gate/up and down with explicitly realized per-expert integer rows; unquantized BF16 ND operands | EP=1, PCP=1, unpadded tokens, `alltoall` recipe, unrestricted expert groups; no probability-to-histogram conversion, MC2 chunk reconstruction or quantized experts |
| `mla_decode` | `vllm_triton_mla_decode_tuple`; one batched call for single-token decode; absorbed BF16 query and dense BF16 latent/rotary cache | Explicit `num_kv_splits`; latent/rope dimensions `(256,32)` or `(512,64)`, at least two local heads, page size divisible by 16; no CP/PCP, padding, sparse/compressed cache or mixed prefill/decode |
| `quantize` | `cuda_dynamic_int8_quant_symmetric_no_azp`; ordinary BF16 activation chain, one INT8 payload and one FP32 scale per integer row | Contiguous per-row symmetric conversion only; no block scales, fractional MoE rows, asymmetric zero points, fused norm or FP8 output |

The GQA bridge uses the model's local head geometry and the runtime's KV page
size. The declaration must provide `compute_capability`. Its absence does not
silently assume a GPU. `fused_batch=true` accepts one sequence; multiple
sequences require explicitly setting
`runtime.attention_policy.fused_batch=false`. This produces one declared source
launch per sequence, preserving dependencies and actual launch count.
The bridge does not split a declared fused batch merely to fit an existing
one-sequence planner.

Standard block normalization now carries `reduction_width`, `norm_has_weight`
and epsilon. The bridge preserves the graph identity, dependencies, stream,
semantic shape, phase and layer while replacing the physical recipe and work.
The `recipes: ["rmsnorm"]` filter in the example deliberately selects standard
block norms. Combined Q/K normalization, flattened MLA norms and other
normalization contracts need their own semantic description before selecting
this implementation. They are not inferred from a family name or a flattened
element count.

## Dense latent MLA decode

```json
{
  "runtime": {
    "kernels": {
      "lowerings": {
        "mla_decode": {
          "implementation": "vllm_triton_mla_decode_tuple",
          "options": {"num_kv_splits": 3}
        }
      }
    }
  }
}
```

This selector covers decode only. Prefill continues through the declared
analytical MLA path. A mixed batch or speculative multi-token decode does not
silently become a collection of single-token source launches. The adapter uses
all actual decode requests in one source invocation, retaining each context
length and the locally sharded head count.

The selected caller has separate absorbed latent and rotated query components.
It materializes their concatenation and initializes both output and LSE before
the attention call. These are actual operations in the pinned
`vllm/v1/attention/backends/mla/triton_mla.py:199-213`: tuple input takes
`torch.cat`, and the two following `torch.zeros` calls are unconditional.
Caller metadata retains that file's SHA-256 and source range. Each
materialization uses a compact memory-work envelope; this does not assert a
particular PyTorch fill/copy kernel implementation.

The core call reuses `triton_decode_grouped_mla`, expanded into stage 1 and the
dependent split merge. Its call join feeds `v_expand`; it cannot be treated as
one launch or skip the merge. The BF16 cache aliases its latent columns as V
instead of reading another value buffer. FP32 partial output/LSE scratch traffic,
empty splits and padded matrix lanes remain in the source plan. Logical
attention pairs stay in semantic metadata and are not substituted for the
source's padded work.

`num_kv_splits` is the declared result of caller dispatch, not an automatic
SM/length heuristic or a measurement lookup. Increasing it can restart key
tiles and increase scratch work even at the same logical pair count. Caller
zero fills are separate from those scratch stores; the scratch allocation is
not itself charged as a memory fill.

## Routing scenarios and grouped GEMM

`model.expert_routing_histogram` describes a statistical distribution. Its
normalization and expected rows/active experts remain the default analytical
path. Those expectations do not identify integer expert rows or tile boundaries.

The independent `runtime.routing` declaration supplies an actual per-forward
assignment histogram to either the analytical or source path:

```json
{
  "runtime": {
    "routing": {
      "realizations": [
        {"scope": "body", "layer": 0, "phase": "prefill", "tokens": 4,
         "expert_rows": [3, 0, 3, 2]},
        {"scope": "body", "layer": 0, "phase": "decode", "tokens": 1,
         "expert_rows": [1, 0, 1, 0]}
      ]
    },
    "kernels": {
      "lowerings": {"grouped_gemm": "ascendc_grouped_nd_bf16"}
    }
  }
}
```

Each declaration is a reused scenario for the given model scope, layer, phase
and token count. Every matching forward uses those same counts, including
successive decode iterations. It is not an iteration trace, a measured-shape
timing table, or a prediction of which experts the router will choose.
`expert_rows[i]` belongs to expert `i`; its length must equal the layer's expert
count. Counts must be integers from zero through `tokens` and sum to
`tokens * top_k`. The zero entries remain in the source group list while the
source planner skips their matrix work. The analytical path also reads weights
only for the explicitly active experts.

A declared scope/layer requires exactly one matching phase/token scenario.
Chunked prefill or a changed batch shape therefore needs its own declaration;
it cannot reuse a full-prompt histogram or fall back to probability moments.
Undeclared scope/layers retain their statistical route. Graph padding, PCP,
EP>1 and MC2 currently require richer per-rank/per-chunk realization contracts
and are rejected for declared scenarios. Tensor and expert-tensor parallel
matrix dimensions continue to use the existing sharding rules.

Group-restricted routing also needs an assignment feasibility contract beyond
the aggregate histogram. For example, with two groups of two experts,
`top_k=2` and one selected group per token, `[4, 0, 4, 0]` conserves all eight
assignments for four tokens but is impossible: the two experts in each chosen
group must receive equal counts. This initial contract therefore accepts
ungrouped routing or selection over all groups; partial group selection remains
unsupported for integer scenarios.

The two recipes use the same logical FLOPs and actual active expert identities.
Their physical traffic can differ: the analytical recipe assumes ideal matrix
reuse, while the source plan accounts for source M/N/K tiles and repeated loads.
Removing only the kernel binding compares these models under the same declared
integer assignment. Selecting the source implementation without an integer
realization is an error; normalized probabilities are never rounded into counts.

Selected domain misses raise a modeling error. Missing **measurements** never
disable either source or analytical simulation. Detailed source programs do
require declared service rates for their typed work; an absent resource rate
is a missing physical input, not a demand for a timing measurement.

## Fusion and quantization

Explicit source operators bind before generic representation/fusion passes and
after each pass creates new operators. The passes preserve their source
boundaries. Selecting ordinary
RMSNorm therefore keeps dynamic quantization as a separate dependent operator,
even when `norm_quant` is requested. A source-specific fused norm/quant
implementation is required to combine them.

To select residual-add RMSNorm, declare `add_rmsnorm` and request the
`residual_norm` lowering pass. Binding occurs after the pass produces the fused
family. Its live residual output and every downstream residual consumer remain
in the graph. Selecting ordinary `rmsnorm` as well binds those norms first and
preserves their separate source calls; it does not implicitly authorize fusion.

The `quantize` selector can bind the dynamic producers inserted by
`activation_quantization`:

```json
{
  "runtime": {
    "kernels": {
      "lowerings": {
        "quantize": "cuda_dynamic_int8_quant_symmetric_no_azp"
      }
    }
  }
}
```

The producer must declare an INT8 output, complete integer rows, one FP32 scale
per row and the matching payload/scale byte counts. The existing CUDA planner
then accounts for two input passes and preserves the separate scale output
consumed by each matrix. No block scales or probability-expected expert rows
are rounded into this contract. Binding immediately after the conversion pass
also prevents `norm_quant` from replacing that selected source boundary.

This ordinary-model binding currently requires BF16 activations throughout its
input chain. The standalone CUDA planner also supports FP16 and FP32 input
tensors, but that does not establish those ordinary model paths: quantized
matrix builders currently return BF16, and the frontend does not insert a cast
merely because `model.activation_dtype` differs. Such a declaration is rejected
by this binding until the full producer and consumer representation contract is
modeled; changing byte counts alone would not provide the missing conversion.

Options `input_byte_offset` and `output_byte_offset` declare pointer alignment;
the default is zero offset. `zero_absmax=false` declares the ordinary nonzero
row branch. An explicit `true` declares all selected rows have zero absolute
maximum; the adapter does not infer this data property from dimensions.

## Runnable model control and verification

`examples/source-families/gqa-norm-control.json` is a complete ordinary model
case with explicit implementation declarations and synthetic resource rates.
It is a CPU structural/execution control, not a 4090 profile or an accuracy
result. It can be passed to the normal `predict`, `operators` and `inspect`
commands.

`examples/source-families/grouped-gemm-control.json` is an ordinary MoE case
using the integer scenarios above and the existing BF16 grouped GEMM source
planner. It uses synthetic rates for structural/execution verification. Its
full `predict` path includes one prefill and two repeated decode forwards;
removing `runtime.kernels` retains the same integer assignments in the
analytical model.

`examples/source-families/mla-decode-control.json` runs a normal prompt followed
by source MLA decode. `examples/source-families/dynamic-int8-control.json` runs
an ordinary INT8 model whose activation producers use the selected CUDA
conversion. Both are synthetic CPU controls usable with `predict`, `operators`
and `graph`; neither is device accuracy evidence.

`tests/test_family_selection.py` covers:

- normal model frontend selection through public runtime declarations;
- mathematical attention pairs versus padded source work;
- identical physical work in isolated and full-model execution, including
  delayed input readiness and released allocations;
- per-sequence launch boundaries and explicit unsupported-domain errors;
- typed-rate sensitivity without changing work;
- BF16/FP16/FP32 norm traffic, separate quantization and live residual outputs.

`tests/test_grouped_family_selection.py` covers actual normal-model calls with
both route interpretations, zero-expert compaction and identities, exact work
conservation, distinct source scheduling, repeated decode scenarios, tensor
shards and scenario/domain boundaries.

`tests/test_mla_family_selection.py` checks caller materializations, batch and
TP geometry, real two-launch dependencies, independent split-tile/scratch
ledgers, empty splits and rejection of incompatible decode branches.
`tests/test_quant_family_selection.py` checks real producer/consumer scale
contracts, two input passes, fusion boundaries and incompatible scale/routing
domains.

The existing source-expression and numerical-oracle tests continue to validate
the underlying planners. These tests establish selection and structural
composition; they do not add hardware timing acceptance.

## Remaining families

Grouped GEMM source selection covers the explicit realized domain above.
Expected routing moments remain analytical; EP and PCP deployment mappings,
MC2 chunk histograms, restricted-group feasibility, padding ownership and
quantized expert implementations still require their own semantic contracts.

MLA prefill, compressed/indexed MLA, MoE align/combine and other quantization
implementations still require their own semantic-domain adapters.
Register a module and its supported
operator or block selectors in `prediction.implementations`; the module supplies
the semantic adapter and the existing physical recipes. The engine requires no
new model-name branches.
