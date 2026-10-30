# Kimi K3 AttnRes sequence parallel

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

`mindie_sim.prediction.sequence_parallel` plans the source-selected Kimi K3
AttnRes sequence-parallel region and emits ordinary collective/view `OpSpec`
records. It does not create a second execution engine and it does not rescale a
completed graph after construction.

## Source boundary

Evidence is the retained `vllm_ascend/models/kimi_k3.py` image file:

- source file SHA-256
  `be9adc792774f919f443caf483eb2e04e4453617465381f47a482dd4736ec63f`
- container image
  `sha256:883670fcc91cbd75a7070b00501be0ecd81a266b81366a21bf02139d8e6a33e1`
- lines 877-880 select
  `use_sequence_parallel = (PP == 1 or v2runner) and enable_expert_parallel
  and TP > 1`. There is no prefill/decode conditional.
- lines 239-260: dense MLP allgathers the local rows, runs full-token TP
  matrices, then reduce-scatters.
- lines 786-794: attention allgathers to full tokens, runs attention, and
  reduce-scatters when the fused output-projection path is not selected.
- lines 1010 and 1033-1036: the first stage embeds full tokens and the body
  starts its TP shard; residual and auxiliary rows use the same shard boundary.
- lines 1089-1095 and 1116-1130: a PP boundary and the final/source-stage
  close-output path allgather back to full rows, including packed auxiliary
  channels.
- The MoE router and latent transforms consume the local token shard. Expert
  routing group totals remain cohort-global and are not divided by TP.
  AttnRes and normalization consume local rows.

This module implements the declared `PP == 1` branch. It does not infer the
`v2runner` alternative, a padding-skip branch, a fused attention output, or a
different shard materialization. The explicit `right_to_tp` extension below
uses separately pinned helper and AllToAll sources. Source line numbers are provenance only; no
model-name or measured-timing rule selects a path.

The retained image caller alone does not expose the primitive `sp_shard` helper. `shard_materialization: "view"` is therefore an explicit
input declaration, not an inferred zero-copy kernel. `input_shard` declares
that view and records zero physical payload. No copy or primitive payload is
invented.

## Exact declaration

The plan is enabled only when `runtime.kernels["sequence_parallel"]` is
present. Its value must be exactly this mapping; extra keys, missing keys and
unsupported option values are rejected. `padding` accepts `none` or the
explicit `right_to_tp` AllToAll branch:

```json
{
  "implementation": "vllm_ascend_kimi_sequence_parallel",
  "enable_expert_parallel": true,
  "padding": "none",
  "attention_output": "separate_reduce_scatter",
  "shard_materialization": "view",
  "shared_expert_layout": "tp_sharded"
}
```

The selected domain additionally requires:

- `PP == 1`, `CP == 1`, `PCP == 1`, `TP > 1`;
- `EP >= TP` and `EP % TP == 0`;
- every model layer plan has source AttnRes `residual_block_size > 0`;
- every model layer plan is in the body scope and outside the
  hyper-connection branch; draft layers are not supported by this selector;
- no padded views and no speculative proposal/accepted/advanced fields;
- With `padding="none"`, `tokens == sum(view.q_len)`, every `q_len > 0`, and aggregate `tokens`
  is divisible by TP (individual sequences need not be divisible). An idle
  DP replica may have empty views and zero local rows while the complete
  EP cohort has positive rows;
- when EP spans DP (`EP > TP`), a complete `global_tokens` total for the EP
  cohort. It must be at least this replica's full rows and divisible by TP.
  It is not assumed to equal `tokens * (EP / TP)` and it is not a total over
  unrelated DP replicas outside the cohort.

Decode uses the same source branch. Under `padding="none"`, Decode rows that
are not divisible by TP raise an explicit unsupported no-padding error; the declaration is never
silently disabled. A standalone `target_operators` call that cannot supply the
complete cross-DP cohort total also raises instead of inventing equality across
DP replicas. `target_graphs`/serving callers that build complete cohorts must
pass the complete total.

An unsupported CP/PCP/PP axis, incomplete cross-DP total, tail that is not the
exact TP reduce point, or malformed declaration is an error, not a fallback.

## Public API

```python
select(runtime, model, views, *, tokens, global_tokens=None) -> SequenceParallelPlan | None

input_shard(*, plan, op_id, deps, stage, layer, requests, phase, width, dtype,
            row_multiplier=1, channel_multiplier=1) -> OpSpec
allgather(*, plan, op_id, deps, stage, layer, requests, phase, width, dtype,
          row_multiplier=1, channel_multiplier=1) -> OpSpec
reduce_scatter(*, plan, op_id, deps, stage, layer, requests, phase, width, dtype,
               row_multiplier=1, channel_multiplier=1) -> OpSpec
close_output(ops, last, *, plan, width, dtype, op_id=None,
             row_multiplier=1, channel_multiplier=1) -> (tuple[OpSpec, ...], str)
```

`select` returns `None` only when the declaration key is absent. The returned
plan is frozen and carries:

- `valid_global`: the original sequence query rows;
- `input_rows`: physical input rows after any graph-capture padding;
- `tokens_global` / `physical_global`: this replica's physical rows after SP padding;
- `valid_per_tp`: exact valid rows in each contiguous TP shard, diagnostic only;
- `tokens_local = tokens_global // TP`;
- `cohort_tokens`: physical router rows over the EP cohort; includes declared
  graph/SP padding for the declared unmasked AllToAll routing branch;
- `source_partitions = 1`: the incoming tensor is already TP-sharded, so
  outgoing source assignments must not be divided by TP a second time.
  Collective/view metadata also carries the existing
  `source_assignment_partitions = 1` spelling for the same instruction;
- `tp`, `ep`, `dp`, observed `phases`;
- mechanism-level `source` metadata and explicit `assumptions`.

`plan.ffn_tokens(layer)` returns `tokens_global` for a dense layer (full-token
TP matrices) and `tokens_local` for a routed MoE layer. `attention_tokens`
equals `input_rows`, before SP padding but after any graph positions padding;
`attn_res_tokens`, `normalize_tokens` and `router_tokens` are local.
Expert counts remain cohort-global; TP does not divide them.

## Wire and shape contract

For both allgather and reduce-scatter:

```text
per-physical-rank useful payload =
    tokens_global * width * dtype_bytes * (TP - 1) / TP
```

`row_multiplier` preserves an explicitly repeated row domain;
`channel_multiplier` preserves final packed auxiliary channels
(`1 + number_of_aux_hidden_states`). The default is one for both. Metadata
keeps the true tensor shapes:

- allgather: `input_shape = (tokens_local, width)`,
  `output_shape = (tokens_global, width)`;
- reduce-scatter: `input_shape = (tokens_global, width)`,
  `output_shape = (tokens_local, width)`;
- `input_shard`: full input rows and local output rows, with
  `shard_materialization=view`, `physical_payload_bytes=0` and
  `zero_payload_reason=declared_view_no_padding`.

Collectives declare `group_axis="tp"`, `steps=TP-1` and the source mechanism
metadata. They are ordinary communication records with nonzero `link_bytes`;
they are not zero-work barriers. The existing `Topology` and `Engine` account
and execute those records for the normal physical group, including all ranks.

`close_output` accepts the operator sequence and the exact current `last`
identifier. It replaces only the final record when that record is an
`allreduce` on `group_axis="tp"` with a sole producer dependency, and keeps
that producer as the new reduce-scatter's dependency. It does not scan id
prefixes, does not accept an allreduce from the middle of the graph, and does
not leave both an allreduce and a reduce-scatter in the graph. The replacement
preserves the original stream. The root may use it or perform the same single
replacement itself.

## Ordinary frontend integration

`Frontend.body` implements these boundaries before implementation lowering:

1. call `select` once per iteration/replica, before building the body;
2. create the input shard view from the full embedding output and keep its
   dependency explicit;
3. feed local rows to router, latent transforms, AttnRes and normalization;
4. slice the allgather output to the incoming positions rows before attention;
   pad its output back to TP for the separate reduce-scatter;
5. give dense FFN full-token TP matrices; give routed MoE local rows and pass
   `cohort_tokens` to the global expert-count allocation;
6. keep `source_partitions=1` when constructing source assignments so an
   existing source division is not applied twice;
7. close the final/source region with one allgather to full rows, including
   any packed auxiliary channels.

An idle source replica has no input shard, attention, allgather or
reduce-scatter. It can still receive and compute expert assignments from its
nonempty EP cohort. The collective helpers reject zero local rows so an
integration cannot create an empty TP launch while suppressing real expert
work. The frontend, not the token planner, builds that receive/compute path.

Shared expert placement is explicit: `shared_expert_layout="tp_sharded"`
selects gather of the local input, TP-sharded matrices over the full rows, then
reduce-scatter before joining the local routed result. The source factory can
also select replicated shared weights; that branch is not inferred from a
checkpoint or the presence of shared experts. No extra allreduce remains.

The ordinary model graph retains global embedding rows, local residual/norm
rows, and global attention rows. At TP8 an8192-token prefill therefore uses1024
rows for latent down/up, without shrinking attention or the total expert
assignments. Probability routing and a forced integer-balanced scenario remain
different declarations. The latter requires genuinely equal integer expert
groups; a nondivisible8192-token scenario cannot be relabeled as balanced.

AttnRes source-state selection is orthogonal. Without its explicit selector,
the existing residual builder uses local rows but is not claimed to reproduce
the native prefix/bank state transitions.

## Verification boundary

`tests/test_sequence_parallel.py` independently recomputes TP 2/4/8 allgather
and reduce-scatter payloads, checks global/local matrix and normalization
dimensions through the plan, exercises an aggregate-divisible multi-request
batch whose individual sequences are not divisible, runs the records through
the ordinary graph/`Engine`/`Topology` path with external input readiness, and
checks closing replacement and the truly unsupported PP/CP/PCP and
non-divisible decode domains.

These are synthetic structural/semantic controls. They establish graph and
communication invariants, not measured accuracy or a hardware claim. Review
also covers FP32 and INT8 collective payloads, an idle DP source with a positive
expert cohort, preserved stream identity, and the actual retained Kimi
TP8/DP4/EP32 case with unequal DP totals `(8192, 16384, 8192, 0)`.

## Retained descriptor correspondence

The profiler's independent structural review at clean revision `1aeb52ad`
is preserved byte for byte in
`data/verification/kimi-sp-descriptor-review-20261029.json`. Its original
producer text is retained alongside it as `.producer.txt`; this archived
source records the historical procedure and is not a new runtime tool.
The scoreboard input records both artifact hashes and the fixture/provenance
hashes. Importing this evidence did not rerun simulation or rescan the
110 MB original capture.

The two retained GroupedMatmul descriptors from source CSV rows 136 and 174
have the following one-bank geometry:

| Projection candidate | G | M | N | K | Activation scale shape |
| --- | ---: | ---: | ---: | ---: | --- |
| Latent down | 1 | 1024 | 3584 | 7168 | `[1024, 112, 2]` |
| Latent up | 1 | 1024 | 7168 | 3584 | `[1024, 56, 2]` |

Both use FP8 E4M3 operands, BF16 output and E8M0 activation scales. The
declared SP prefill graph has 368 candidates for each descriptor: 92 MoE
layers across four DP representative graphs. Without SP, its latent rows
are 8192 and neither descriptor matches.

The earlier streamed capture counts also contain 368 events per descriptor,
but those counts cover the historical Device0 capture, not one declared
four-DP graph. Numerical equality does not establish an event-to-layer,
step or replica assignment. In particular, 736 is not a validated numerator
over the existing 19,982-event single-bank reference domain.

Group-list contents, weight-scale storage, active weight reads, native
dispatch and tiling remain outside this correspondence. This evidence adds
structural candidates; it adds no timing pass or occurrence binding.


## Explicit right-padding and AllToAll routing

Set only `padding` to `right_to_tp`; the other declaration fields stay the same.
This branch implements zero right-padding to `ceil(input_rows / TP) * TP`,
followed by contiguous equal shards. The helper source is independently pinned
at vLLM `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`,
`vllm/models/common/ops/sequence_parallel.py`, SHA256
`699fde98360fc8d604055e2987d93108ee489e87cc5563949d2a3dcd838eb8bd`.
The retained image caller imports this helper interface; identity of that
image's helper binary has not been established.

At TP8, q1/B1 has eight physical global rows and one row per TP rank, with
valid rows `(1,0,0,0,0,0,0,0)`. q4/B1 has the same physical shapes and valid
rows `(1,1,1,1,0,0,0,0)`. Router scores, local norms/AttnRes and AG/RS use
physical shapes. Attention instead slices gathered rows to
`positions.shape[0]` before its projections and convolution. The original
sequence views still determine attention pairs, recurrent query loops and
state work. The initial pad reads existing input rows and writes the full
zero-padded tensor; its vector work is the added zero elements. The TP slice
and attention input slice are views. The helper repeats an actual pad copy
on each attention output before reduce-scatter when positions rows are not
TP-divisible. FFN matrices and routing keep their original physical shapes.
Final allgather output is sliced back to `input_rows`, as in caller lines
1093–1095; that contiguous prefix is another view. Pad copies model physical
payloads, not recovered vendor launch grids. The current source anchors and
full-depth row ledgers are in [the attention-row repair](kimi-sp-active-attention-rows.md).

For this supported AllToAll branch, both the ordinary Ascend router and forced
routing process the padded rows. The ordinary-router chain is specified below.
Local vLLM-Ascend `ae923905948f360f4b7ab06df52d0789a0c9316e` establishes:

- `force_eplb.py`, SHA256 `bd85c5f499f04a6e15aa23fca8b653d1c527cd35651c80da4504efae8cdd5f2a`,
  generates IDs for all `topk_ids.shape[0]` rows.
- `routed_experts.py:601–602`, SHA256
  `aaea89c2d83ebbb4f19f0162ef3076c33e359de0017355f9fb6d831c64d716fb`,
  replaces selected IDs after ordinary router selection.
- `prepare_finalize.py:128–176`, SHA256
  `19d7d17ccff6ff1d55c391142dbe1f4c5e849a6e0132fa7ad0872999ebb7aa53`,
  returns `mc2_mask=None` for AllToAll.
- `token_dispatcher.py:575–615`, SHA256
  `86d316b1daa1b5cd633b9e74717b883eb562c4318c047a9e7110d620accee06c`,
  builds the histogram and permutation from all IDs; `num_out_tokens` is their
  full element count, with no second validity mask.

The independent ordinary-router audit is retained in
`kimi-sp-ordinary-router-audit-20261030.json`,
including the hashes of all twelve inspected files. In the same pinned Ascend
revision, `patch_fused_moe.py:148–173` installs `AscendFusedTopKRouter` or
`AscendGroupedTopKRouter`. Their inherited `BaseRouter._select_experts` calls
`_compute_routing`, optional EPLB mapping and dtype conversion without reading
`forward_context.is_padding`. The Ascend fused branch calls
`DeviceOperator.moe_gating_top_k` with no padding-mask argument; its fallback
runs softmax/sigmoid and top-k over every physical row. Both produce valid
expert IDs for the zero-padded rows even when forced EPLB is disabled.
The Ascend EPLB mapper also returns physical IDs for all rows;
`num_unpadded_tokens` limits load-statistics recording, not execution IDs.

Upstream GPU `FusedTopKRouter` does pass `_get_padding_mask` to its routing
kernel, but the inspected Ascend factory replaces that router. This audit
therefore establishes the explicit Ascend AllToAll branch, not a universal
SP rule. Custom routers, MC2 and other source revisions remain outside it.
The retained image includes the Kimi caller, not this exact factory/router
helper chain; the conditional case still cannot establish the old capture's
binary identity or forced-routing settings.

Executing padded rows does not establish their expert distribution. Zero
logits plus correction bias may concentrate them on a small set of experts.
A probability-routing case continues to declare a statistical distribution;
it must not be labeled as forced balancing, an observed histogram, or a
measured reproduction solely because the physical row counts now align.

Accordingly the balanced q/r scenario uses the sum of independently padded DP
populations. `(1,4,9,0)` at TP8 yields `(8,8,16,0)` physical rows and 32 cohort
rows, not the next multiple of the 14 valid queries. AllToAll traffic preserves
the corresponding per-source rows; combine still transposes that ledger. An
idle DP source sends zero but can receive expert work. The existing ideal q/r
scenario remains distinct from the source's exact rank-shifted ID table.

MC2's separate `x_active_mask` mechanism is outside this selector. Upstream
`sp_padding_mask` alone cannot justify masking the AllToAll forced-ID path.
No automatic SP declaration is installed; `none` retains its former domain.
The source revisions above are mechanism evidence, not measured capture ABI
identity or accuracy evidence.

`tests/test_sequence_parallel_padding.py` checks the real 93-layer, 896-expert
Kimi catalog at q1/q4; it also executes the real first two layer subgraph over
four unequal/idle DP sources, verifies physical traffic and q/r conservation,
late input readiness and released Engine work. This Kimi checkpoint is distinct
from the 40-layer/384-expert V4.1 checkpoint; no Kimi 384 config was used.

The explicit checkpoint control is
`examples/cases/kimi-k3-a5-sp-right-pad-conditional.json`, derived from the
profiler's current-checkpoint second arm with its role quantization and native
norm/Situ selection retained. The producer and compact report are
`data/verification/kimi-sp-padding-20261030.producer.py` and the adjacent JSON.
Both q1/q4 graphs retain all 93 layers and four DP participants; the reported
Engine span separately covers only the first two layers and excludes the
7 ms injected input delay. The report records source hashes and does not
claim full-model error convergence.
