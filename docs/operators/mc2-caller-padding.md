# MC2 caller padding is not whole-body graph padding

The GLM-5.3 TP16 captures distinguish two physical tensors. For q6/B2,
`MlaPrologV3` receives INT8 `[12,6144]` (78 calls per forward), and the router and
shared projections also retain 12 rows. Only the routed MoE caller pads hidden
states and logits to 16 rows before TP slicing. There are 75 hidden-state
`ConcatD([12,6144],[4,6144])` calls and 75 logits
`ConcatD([12,256],[4,256])` calls per forward. The q6/B8 control already has 48
rows and requires no additional caller padding.

The historical input copied raw launch capture sizes `[1,2,4,8,16,24,32,48,64]`
straight into resolved token buckets. The generic GraphPolicy therefore selected
16 for M12. This was a case declaration error, not an MLA arithmetic or TP
matrix rule. Raw launch sizes need not equal the final speculative decode keys:
local vLLM's `config/compilation.py::adjust_cudagraph_sizes_for_spec_decode`
rounds keys to the uniform decode length. The capture's native tensor rows are
stronger evidence than guessing the exact runtime version's resolution.

New independent cases in `examples/cases/high-concurrency-20261030/glm78-capture-rows/`
declare resolved body keys 12/48 and an explicit caller:

```json
"moe_token_partition": {
  "kind": "vllm_ascend_mc2_tp_slice",
  "padding": "right_to_tp",
  "gather_output_materialization": "view"
}
```

This uses the existing Engine and ordinary frontend. Full-row router and shared
work precede two real concat copies; TopK sees one allocated row per TP rank for
M12/TP16. MC2 payload uses the valid source mask. Final allgather reconstructs
16 rows, followed by a zero-copy view of the original 12. Explicit per-rank
materialization, if selected, copies the allocated rows before trimming.

The selected source is vllm-ascend `8c30e4f491f73cba737a1dacc522894e437cdf3d`:

- `ascend_forward_context.py:217–226`: padded length is rounded to TP and the
  valid mask excludes padding.
- `ops/fused_moe/prepare_finalize.py:288–307`: hidden/logits concat, then TP split.
- `prepare_finalize.py:196–228,316–325`: final allgather then slice original rows.
- `routed_experts.py:656–675`: preparation precedes TopK selection.

This is an explicit source-compatible wrapper; it does not identify the capture
binary. A3 uses the generic transport timing path here, not the A5 V3 program.
`native_kind`, `input_shapes` and `output_shape` expose the ConcatD ABI for the
profiler's contract adapter without changing its owned adapter files.

Generic balanced routing retains its existing integer expert histogram for the
valid body rows. A known source mask plus uniform destination expectation gives
a directed communication matrix with zero payload from invalid ranks; it does
not invent captured expert IDs. The separately explicit
`force_eplb_source={"kind":"vllm_ascend_mc2_tp","global_bs":0}` option instead
uses the pinned source cursor and contiguous expert placement. The new capture
cases do not enable that additional assumption.

`data/verification/glm78-capture-rows-20261030.json` records all three complete
78-layer graphs, including the historical global-padding control. The 12-row
cases retain 78 prologs (57 ordinary, 21 query-norm), 150 concat calls and 75
allgathers. Every MoE layer routes 96 valid assignments, not 128. The 48-row case
retains 384 assignments and no added concats. A separate explicit bucket16 test
still computes all projections at M16 and routes 128 assignments; genuine
whole-body capture padding has not been removed. Timing accuracy is not claimed
by this structural audit, and no hardware rate or invocation floor changes.
