# Recurrent KDA previous-state selection

Both explicit `vllm_ascend_recurrent_kda_arch22` and
`vllm_ascend_recurrent_kda_arch35` lowerings accept `options.state_selection`.
This extends the existing source program; it does not introduce a second
recurrence implementation or a fitted timing term.

The retained source at `8c30e4f491f73cba737a1dacc522894e437cdf3d` establishes:

- `recurrent_kda.h:325–372`: packed metadata uses the absolute token index;
  speculative metadata uses `batch * row_stride + local_token`. Accepted
  counts must be in `[1,q_b]`; initial state comes from index `accepted[b]-1`.
- `recurrent_kda.h:815–882`: each head loads its selected initial state once
  across value tiles, executes all current query rows, and writes the state
  checkpoint for every query to `indices[b,t]`.
- `recurrent_kda_torch_adpt.h:116–153`: `final_state` aliases `initial_state`,
  `output_final_state=false`, and `inplace_final_state=true`.
  `op_kernel/recurrent_kda.cpp:34` chooses one state output address. The
  three low-level output descriptors do not imply three state payload copies.

The corresponding arch35 header uses the same state selection. Its existing
per-head beta load differs from arch22's per-owner/sequence reuse; those
programs and their ownership remain separate.

## Exact addresses

For two sequences with current query lengths `[2,4]`:

```json
{
  "kind": "exact",
  "prefix": [0, 2, 6],
  "index_layout": "speculative",
  "index_row_stride": 4,
  "state_indices": [[17, 4, -1, -1], [31, 11, 27, 2]],
  "accepted_tokens": [1, 4],
  "state_capacity": 32
}
```

Initial slots are `[17,2]`; checkpoint slots are `[17,4]` and `[31,11,27,2]`.
Changing accepted counts to `[2,1]` changes initial slots to `[4,31]`, not
the six executed queries or six checkpoint writes. Metadata cells outside
each sequence's current query extent are not read. Packed indices use a
flat array, `index_layout="packed"` and zero `index_row_stride`.

Exact mode validates all queried addresses and accepts repeated checkpoint
slots within one sequence. Cross-sequence slot aliasing is outside this
independent-sequence program: the native kernel could then race read/write
operations, which this adapter does not turn into a fabricated serial order.

The `accepted <= current q` restriction is from both retained kernel headers,
not inferred from index capacity: arch22 lines 348–351 and arch35 lines
345–348 explicitly reject larger counts in `ValidateStateSlots`. Thus
current `q=1`, row stride 8 and accepted count 7 are outside this source domain
even though slot column 6 exists. A deployed kernel supporting that combination
would require its own verified state-selection semantics; this extension does
not silently substitute table width for the pinned kernel's condition.

## Symbolic valid-range domain

For shape-only inputs, the caller can explicitly declare this conditional
domain instead of manufacturing state indices or acceptance counts:

```json
{
  "kind": "symbolic_valid_selection",
  "index_layout": "speculative",
  "index_row_stride": 6,
  "validity": {
    "prefix": true,
    "state_indices": true,
    "accepted_tokens": true,
    "independent_sequences": true
  }
}
```

The declaration means the prefix matches the supplied current query lengths,
every queried slot lies inside its state pool, accepted counts lie in
`[1,q_b]`, and different sequences do not share state slots. Actual addresses,
accepted counts and state capacity remain unknown. The output labels this
as `caller_declared_valid_ranges; values_not_inferred`, not a verified
capture contract. Capacity is not a substitute for valid query rows.

Both modes may supply `dependencies`, a list of external state or metadata
producer IDs. Those become real kernel dependencies; ordinary caller QKV,
gate and beta producer dependencies are retained. Without separate IDs,
the existing caller input-ready boundary applies.

## Physical work and evidence

Address selection does not change data payload or arithmetic in this program.
The same state residency, finite UB/output queues, active AIV ownership and
read/write pipes remain. There is one enclosing kernel invocation. Scalar
accepted-array reads are counted separately: one during each state validity
check and another during each initial-slot resolution. Their GetValue
cache/issue latency remains unpriced, alongside the existing scalar ledger;
the observed kernel duration is not a scalar service parameter.

The retained GLM B4/q6 evidence has prefix extent `[5]`, index shape `[4,6]`
and an accepted-count tensor `[4]`. It contains no accepted/index values;
Q/K/V and state expose backing extents rather than complete logical strides.
This extension therefore supports a conditional source program for that
geometry, but does not certify its exact binary, gate flags or addresses.
The separate profiler comparison guard is unchanged. Original cases are not
rewritten or automatically switched to either declaration.

Tests execute both source architectures with one/ragged/many AIV owners,
different accepted values, exact and symbolic modes, and a late state input.
They check unchanged resource service, output payload, finite UB residency,
active owners and duration, plus the selected addresses and additional
scalar-read ledger. Invalid queried metadata cannot silently turn a skipped
source sequence into a successful modeled invocation.

## Independent source and integration review

The `kda-accepted-independent-review-20261030` receipt and producer retain
18 actual Engine controls across both source architectures, exact packed and
strided values, symbolic validity, AIV counts and value widths. An independent
formula checks one initial state read per sequence/head and a checkpoint write
for every query. Six unselected-acceptance controls match the parent work and
trace exactly. A separate ordinary caller waits for accepted metadata at 17 ms
before its single native invocation; no state work precedes that event.
The six pinned source objects were independently hashed. No measured duration
or acceptance rate supplied missing state addresses.
