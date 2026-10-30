# Explicit peer traffic

`metadata["peer_traffic"]` declares the directed useful bytes exchanged by one
physical collective group. It is consumed by `Topology.accounting`/`messages`
and by causal `PairwiseGroup` transport. It adds no timing multiplier, measured
fit, device execution or runtime dispatch.

## Public descriptor

```python
# Uniform forward approximation: source_bytes[s] is source s's total
# off-diagonal forward payload.
{"kind": "uniform_peers", "source_bytes": [0, 4, 8], "direction": "forward"}

# Explicit cells: payload_bytes[source][destination].
{"kind": "matrix",
 "payload_bytes": [[0, 0, 0], [2, 0, 2], [4, 4, 0]],
 "direction": "forward"}
```

`direction` is optional and defaults to `forward`. Indices are positions in the
physical group's rank order, not global rank ids and not DP indices. The same
descriptor is interpreted independently inside each physical group and may be
repeated in independent PCP groups. No bytes cross group boundaries.

For `uniform_peers`, source `s` sends `source_bytes[s] / (N - 1)` to each other
peer. It is a declared uniform-peer approximation, not a hidden measured split.
`row_total(s)` is the declared off-diagonal forward total. For `matrix`, only the
supplied cells exist; `payload_bytes[s][d]` is the useful payload of the one
directed message. The diagonal must be zero because local work is not network
traffic. A zero cell emits no message, wire bytes, transaction framing or setup
latency, while its source and destination still participate in the round
schedule.

`transposed()` flips the direction and reads the reversed original cell:
`payload_t(s, d) = payload(d, s)`. It does not reuse or swap stored row totals.
For a uniform declaration the transpose stays O(N); no dense matrix is
materialized. The physical counterexample is:

```text
forward   [[0,0,0],        transpose [[0,2,4],       forward row totals (0,4,8)
           [2,0,2],                   [0,0,4],       transpose row totals (6,4,2)
           [4,4,0]]                   [0,2,0]]
```

Both directions carry 12 useful bytes. In a three-rank topology with independent
directed 1 B/ms links, zero initial delays and the declared pairwise round order, forward
completion is 8 ms and transpose completion is 6 ms. Reusing source totals
would lose this difference. Transposing twice returns the original declaration.

## Validation and joint plans

`PeerTraffic.from_mapping(raw, size)` validates the group size, shape, finite
nonnegative values and zero diagonal once. Row-total comparisons use
`isclose(abs_tol=1e-9, rel_tol=1e-12)`, preserving analytic fractional bytes
without integer rounding.

`Topology.messages` uses the descriptor for `alltoall` and `alltoallv` even when
the locally declared volume is zero. It requires every local physical source's
row total to match that replica's one `OpSpec` volume. A zero-row source
still participates in a joint plan and can receive. `Topology.joint_plan`
resolves complete physical groups with `_joint_groups`, checks the descriptor's
group size, and verifies every physical source's volume against its row total.
`PairwiseGroup.volumes` remains the actual row totals; its optional
`peer_traffic` field selects the directed cell for each round. Round `r` still
sends from source to `(source + r + 1) % N`; framing and per-hop store-forward are
charged per actual message, with no collective-wide barrier.

`alltoallv` without an explicit peer descriptor fails in both local accounting
and joint-plan binding instead of silently claiming uniform semantics. Ring collectives do not
accept this descriptor. `alltoall` without a descriptor preserves the previous
uniform behavior. Arbitrary unequal per-rank volumes within one replica's single
`OpSpec` scalar are rejected, not averaged.

## Boundaries

The model describes declared transport payloads and causal peer rounds only. It
does not infer a vendor algorithm, execute devices or networks, model reduction
math, or claim measured accuracy. Explicit matrices cost O(N^2) because the
caller supplied them; uniform declarations retain O(N) state. See
`tests/test_peer_traffic.py` for the physical counterexample, transpose and
framing controls, TP2/EP4 stage-relative mapping and public graph accounting.

The independent acceptance review corrected singleton zero-traffic accounting
and transpose row sums beside an overwhelmingly larger local source. Small
remote flows are summed directly; subtracting the large local total from the
whole-group sum can erase them. The review also compared 240 directed matrix
controls against a separate scalar recurrence, including two to six ranks,
zero cells, staggered input readiness, both directions and per-message framing.
This is mechanism verification, not measured transport accuracy.

## Dispatch and combine in the ordinary frontend

Dispatch carries a graph-scoped `collective_id`; combine explicitly references
it with `reverse_of_collective`. `bind_cohort_traffic` joins physical source
ranks in the complete active cohort and supplies the same descriptor to every
participant. Combine transposes the directed cells and uses the resulting row
totals. It does not copy each rank's dispatch byte total.

Binding runs before target graph signatures, mapped isolated analysis, graph
execution and serving execution. Already bound standalone calls can be priced
independently. An unbound inverse call spanning DP requires the complete cohort.
Joint declarations must agree, including zero senders. Peer cells participate
in primitive identity; graph reference labels do not change physical identity.
The binder is pure and idempotent and retains no iteration history.

Uniform expert routing remains an explicit approximation even when the source
DP batches differ. Different row totals within a single representative TP
shard cannot be averaged into its scalar volume. MC2 across DP currently
requires divisible source/cohort chunk partitions; tail partitions need an
explicit per-source chunk contract. Otherwise splitting the total first can
create expert computation in a chunk with no dispatched input, so that domain
is rejected visibly.
