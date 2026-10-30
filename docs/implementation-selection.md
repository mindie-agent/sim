# Explicit implementation domains

`runtime.kernels.lowerings` can select a different implementation for each
target or draft call. A conditional declaration contains `cases`, each with
`domain` and `binding`. A binding uses the existing implementation/options/recipes
format or the explicit string `"analytical"`.

```json
{
  "kda": {
    "cases": [
      {
        "domain": {"phases": ["decode"], "max_query_tokens": {"max": 8}},
        "binding": "vllm_ascend_recurrent_kda_arch22"
      },
      {
        "domain": {"phases": ["prefill"]},
        "binding": "vllm_ascend_kda_chunk_state"
      },
      {
        "domain": {"phases": ["prefill", "decode"]},
        "binding": "vllm_ascend_kda_chunk_state"
      }
    ],
    "otherwise": "error"
  }
}
```

This example declares source choices; it does not certify every tensor geometry
for either implementation. Each selected implementation still validates its
own inputs. For example, chunk-state KDA requires equal key/value head counts.

`phases` matches the complete phase set exactly. `['decode']` matches only a
pure decode batch; `['prefill', 'decode']` matches only a mixed batch. Numeric
domains use inclusive `min` and/or `max` bounds:

| Field | Meaning |
| --- | --- |
| `valid_tokens` | Actual query rows on this replica; target includes verification rows and draft uses its own query block |
| `padded_tokens` | Target rows after graph-bucket padding; draft retains its actual caller allocation |
| `sequences` | Active requests on this replica |
| `max_query_tokens` | Largest target query length for one request |
| `max_context_tokens` | Largest incoming context length for one request |
| `replay_scopes` | Allowed `none`, `full`, or `piecewise` capture scopes |
| `model_scopes` | Allowed `target` and/or `draft` calls |

These dimensions are evaluated before PCP/SP partitions. They describe one
model call, not a per-operator shape or a TP-summed model dimension. Serial MTP
calls resolve separately with q1 rows and advancing contexts; target verification
still includes every proposal. Draft preparation resolves as draft prefill.
The replay scope belongs to the enclosing scheduled graph. An idle EP participant
uses the active cohort's phase set and retains zero local rows.

For example, a target MXFP8 routing source can explicitly retain BF16 draft work:

```json
{
  "moe_init_routing": {
    "cases": [{
      "domain": {"model_scopes": ["target"]},
      "binding": "ascend_moe_init_routing_v3_mxfp8"
    }],
    "otherwise": "analytical"
  }
}
```

Omitting `model_scopes` makes a case eligible for both models, subject to its
other domains. An unconditional binding still applies to both. Each selected
call undergoes its own source validation and transforms; a target transform
cannot rewrite draft operators. Equal native-fusion declarations share their
required-role check across the complete selected graph, preserving the existing
unscoped declaration contract. Capture segments are assigned once before fusion.

Cases must be disjoint for the current geometry. An overlap is an error; order
does not imply priority. An unmatched domain raises an error by default.
`otherwise` may explicitly name another binding or `"analytical"`. Analytical
selection removes only that lowering and retains ordinary semantic work and
other declared recipe choices. It does not assert a native implementation.

Selection happens before lowering in both Serving and body/model probes. Source
validation or planner failures propagate directly; they never trigger another
case. All declared source recipe providers are registered before execution.
Bindings that change persistent cache/weight representation (`mla_prolog`,
`mx_matrix`) remain unconditional. Conditional `mla_head_padding` and
`causal_conv_update` reserve their prepared weights once if any branch selects
the source. Original weights remain resident for prefill/analytical paths.
The per-batch decision selects execution without freeing or reallocating that
prepared inventory; all source branches must share its preparation options.

`IterationWork.implementation_selections`, graph-root metadata and serving traces
record the selected case, model scope, coverage and actual geometry. The execution
`step_summary.implementation_selections` counts affected microbatches as `steps`
and selected target/draft invocations as `calls`, even when trace retention is
zero, using only declared cases and model scopes as aggregate keys.
Batch traces retain decisions from each constituent microbatch. Selecting an
analytical branch is visible as `analytical_semantic_work`; this label describes
the selected lowering, not the coverage of unrelated source operators.

For `ascend_grouped_int32_dequant_swiglu` with `dequant_program=arch35_full_load`,
`input_capacity_rows` may be an explicit integer allocation or `"valid_rows"`.
The latter declares a variable-size allocation equal to the actual received
expert-row total for that operator (minimum one allocation row for an empty
group). Resolution happens after routing produces the integer expert histogram.
The resulting numeric tensor extent drives the existing source tiling; it does
not reuse a decode capture's capacity for a larger prefill batch. It is a caller
allocation assumption, not proof of captured native allocation behavior.
