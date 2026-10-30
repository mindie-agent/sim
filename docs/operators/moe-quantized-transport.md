# MoE quantized transport

`runtime.kernels.moe_transport` selects one explicit ALLTOALL expert transport
protocol. It is a physical software branch, not a model-name rule. When the key
is absent, dense and MoE graphs retain their prior producers, dispatch and
combine behavior.

The only supported initial declaration is:

```json
{
  "kind": "separate_quantized_payload",
  "payload_dtype": "int8",
  "scale_dtype": "fp32",
  "scale_granularity": "row",
  "histogram_dtype": "int64"
}
```

Any missing field, unknown field or different value raises `ModelingError`.
There is no hidden fallback to the default transport. The selected protocol is
for the ordinary ALLTOALL (and its deferred TP-reduce form) path; selecting it
with an MC2 recipe raises rather than silently reinterpreting that path.

## Ordinary frontend sequence

For each active MoE layer the ordinary frontend emits:

1. `topk` ids -> local INT64 expert histogram. It reads one INT32 id per
   physical-source token assignment, writes one INT64 entry for every expert (including
   zero-count experts), and charges one increment per assignment.
2. Histogram -> EP allgather. Every physical source contributes its local
   `E * 8` count bytes; a ring participant's useful remote volume is
   `(EP - 1) * E * 8`. This local histogram is not divided by TP.
3. Histogram gather and routed input -> payload pack -> exactly one dynamic per-row INT8 payload plus FP32
   row-scale producer. Its read/write traffic and vector work use the generic
   dynamic quantizer formula. Source assignment rows are
   `tokens * top_k / min(TP, EP)`, split once.
4. Scale alltoall waits, then INT8 payload alltoall waits. The payload
   declaration includes its row byte width.
5. Expert gate-up consumes the received INT8 payload and row scales directly.
   The ordinary `activation_quantization` pass preserves this producer and
   does not insert a second quantizer for the same expert input. Other
   consumers, including a quantized down input where required, still receive
   their ordinary quantizer.
6. Down output is BF16 for this initial branch. The BF16 return is transposed
   from dispatch peer **rows**, then encoded at BF16 row width. It is not the
   INT8 byte total with a timing multiplier, and no timing rate is changed by
   the representation ratio.

With `tokens == 0`, the rank still emits the histogram, count allgather,
payload/scale events and return-event identity so a complete cross-DP cohort
can bind. The histogram writes its zero-count entries; no phantom expert rows
are executed.

## Bytes

For one physical source with `R = tokens * top_k / min(TP, EP)` rows and latent
width `K`:

- histogram read: `R * 4` bytes (physical-source assignment ids);
- histogram write: `E * 8` bytes;
- histogram allgather useful volume: `(EP - 1) * E * 8` bytes;
- scale alltoall: `R * 4 * (EP - 1) / EP` bytes;
- INT8 payload alltoall: `R * K * 1 * (EP - 1) / EP` bytes.

The remote fractions declare uniform peer routing. Declared probability routing
remains an expectation. A forced-balanced realization supplies global integer expert counts; these do
not identify a per-source histogram. The source `counts` field therefore stays
null while `global_expert_counts` retains the declared totals. Only an explicit
EP1 source realization provides known local count values. No observed token
permutation or per-source count vector is synthesized.

`cohort_traffic` transposes a complete forward peer ledger. When an op declares
`peer_traffic_row_format = {"elements": K, "forward_dtype": "int8",
"reverse_dtype": "bf16"}`, the binder derives the `K`-byte forward row and
`2*K`-byte return row from dtype widths, recovers row counts, then transposes
the return ledger. Every cell obeys `reverse[i,j] / 2 = forward[j,i]`. Without that metadata the historical byte-for-byte
transpose is unchanged. The initial branch asserts a BF16 return rather than
silently accepting FP16 or FP32.

## Supported representation boundary

The selector requires an INT8 expert activation contract with one FP32 scale
per complete row: dynamic INT8 (W8A8) and W4A8 expert weights with INT8
activations are compatible because their communicated expert input is INT8.
An FP8 block-scale profile, a block width smaller than the expert row, a
non-FP32 activation scale, non-BF16 model activations, or a non-BF16 expert
down output raises. The model's quantization declaration is never rewritten
to make a profile fit. Source-bound quantizer planners keep precedence.

## Source boundary

The semantics are from official vLLM-Ascend revision
`8fd0bc4885cb52e58791aaa8efacc571b9bd115d`, source
`vllm_ascend/ops/fused_moe/token_dispatcher.py`, SHA256
`1dcfd8181cb2d34c584d8336c3379dbeefba75c0bf664f924063701b93295cbb`:
the ALLTOALL path gathers per-expert counts in
`_preprocess` lines616-641, issues/waits for the dynamic-scale alltoall at
lines507-516 before the quantized payload at518-522, then returns expert
output at559-576 using swapped input/output split counts. LoRA can disable
this branch. The chosen INT64 histogram representation is explicit; the
source call alone does not identify the capture histogram tensor dtype. This establishes
mechanism and ordering only. It does not establish a captured binary,
instruction schedule, communication algorithm, or measured timing identity.

The graph uses the existing causal `Engine`, ordinary operator resources and
explicit `ring_collective`/`pairwise_collective` bindings when selected. It adds
no floors, fitted rates or model-specific timing constants.

## Example

`examples/source-families/moe-quantized-transport.json` is a two-replica
balanced INT8 case. Focused checks live in
`tests/test_moe_quant_transport.py`.

## Independent acceptance

The initial DSH implementation is `d23e3c85`. Independent review corrected an
invalid source-row substitution for destination expert rows: an idle source
can receive tokens and must still compute its assigned experts. The transport
selector now preserves the original destination expert FLOPs and active weight
banks. It changes representation producers and communication, not routing.

The reviewer also removed synthesized source histograms from balanced global
counts, partitioned histogram ID reads consistently with source rows while
preserving all `E*8` count bytes, retained explicit gather-before-pack readiness,
and made the gate's payload producer point to its received payload. Scale
collectives declare FP32 payload independently of the INT8 tensor message.

Tests include actual public graph/predict calls, TP8 equal expert counts, an
unequal three-DP cohort with an idle source that still computes four expert
rows, delayed input readiness, and an independently supplied directed peer
matrix whose BF16 return reverses every INT8 cell. Rebinding preserves bytes.
No residual time is introduced. CPU split extraction, source-specific local
re-permutation and native HCCL algorithm selection remain separate mechanisms.
The existing analytic pack/unpack work is retained; this protocol does not
claim a complete source-accurate local permutation kernel.

## V4 Flash target-only control

`examples/source-families/v4flash-quantized-transport.json` retains the previous
V4 Flash case and adds only this selector. A bounded 4,096-token target-only
operator run completed for both cases in about six seconds each. It excludes
the separate MTP forward and serving; it is not the captured 44-layer request
window. `data/verification/v4flash-quantized-transport-20261029.json` records
case and changed-source hashes and the unmodified resource declarations.

The 43 target layers keep exactly 86 expert matrices and
6,648,609,374,208 expert FLOPs on the representative rank. Dynamic quantizer
count remains344: each expert-input quantizer moved before transport rather
than being duplicated. Histogram calls increase by43, allgathers from1 to44,
and the43 old BF16 dispatches become43 FP32 scale plus43 INT8 payload calls.
Return calls remain43 BF16 exchanges.

The isolated target operator sum changes from294.311702 to293.858966 ms.
Useful per-representative-rank link bytes change from6,943,896,512 to
6,471,543,232. The lower predicted cost is retained: the narrower forward
payload more than offsets the additional declared exchanges at these resource
parameters. This source-supported correction does not close the historical
communication residual and provides no justification for fitting its remainder.
The original default envelope collective recipes are unchanged in this paired
control; explicit causal ring/pairwise execution is covered by separate tests.
