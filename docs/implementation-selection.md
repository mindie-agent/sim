# Explicit implementation domains

`runtime.kernels.lowerings` can select a different implementation for each
scheduled target batch. A conditional declaration contains `cases`, each with
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
| `valid_tokens` | Actual target query rows on this replica, including verification rows |
| `padded_tokens` | Target rows after graph-bucket padding |
| `sequences` | Active requests on this replica |
| `max_query_tokens` | Largest target query length for one request |
| `max_context_tokens` | Largest incoming context length for one request |
| `replay_scopes` | Allowed `none`, `full`, or `piecewise` capture scopes |

These dimensions are evaluated before PCP/SP partitions. They describe the
scheduled target batch, not a per-operator shape or a TP-summed model dimension.
Draft subgraphs inherit the target step's selection. An idle EP participant uses
the cohort's phase set and retains zero local rows. Declarations that require
different draft or per-layer policies need separate explicit support.

Cases must be disjoint for the current geometry. An overlap is an error; order
does not imply priority. An unmatched domain raises an error by default.
`otherwise` may explicitly name another binding or `"analytical"`. Analytical
selection removes only that lowering and retains ordinary semantic work and
other declared recipe choices. It does not assert a native implementation.

Selection happens before lowering in both Serving and body/model probes. Source
validation or planner failures propagate directly; they never trigger another
case. All declared source recipe providers are registered before execution.
Bindings that change persistent model preparation (`mla_prolog`,
`mla_head_padding`, `causal_conv_update`, `mx_matrix`) remain unconditional.

`IterationWork.implementation_selections`, graph-root metadata and serving traces
record the selected case, coverage and actual geometry. The execution
`step_summary.implementation_selections` counts every microbatch, even when trace
retention is zero, using only the finite set of declared cases as aggregate keys.
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
