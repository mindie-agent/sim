# AttnRes cross-layer prefix

[Role] Document the explicit, source-compatible Kimi AttnResFwd point and its
cross-layer prefix state. This is an implementation note, not evidence that the
retained capture loaded this library or used these scalar flags.

## Selection

```json
{
  "runtime": {
    "kernels": {
      "lowerings": {
        "attention_residual": {
          "implementation": "vllm_ascend_attn_res_prefix",
          "options": {"optimize_prefill": true}
        }
      }
    }
  }
}
```

`optimize_prefill` defaults to `false`. The retained caller sets it only for an
explicit pure-prefill A5 request (`_use_attn_res_prefill_cache`); this simulator
does not infer a hardware name, so enable it from the caller. The model frontend
sets the flag only in prefill; the standalone point honors its explicit flag.

The implementation is registered as a block selector over the existing residual
builder. When it is not selected, the ordinary semantic `attention_residual`
mixture, `norm1`, `norm2` and residual additions are unchanged. When selected,
the frontend does not emit those nodes at all; the old six-node
`normalized_learned_residual_mixture` group is never relabeled as a full native
AttnResFwd call.

The frontend's `tokens` is the local row count supplied by its composing caller.
This module does not split SP or edit request views; all-gather/reduce-scatter
belongs to the shared sequence-parallel caller. The standalone point accepts
`tokens_local` and physical `block_token_stride` explicitly. The complete model
caller allocates the source's contiguous bank, with stride `capacity * H`.
Capacity and stride remain separate point fields.

Engram, hyper-connection, mixed `residual_block_size`, or selected layers with
no block size raise an explicit modeling error instead of silently dropping
semantics. The supported native input/output type is BF16. The complete caller
currently supports PP1; cross-stage bank allocation/copy is not approximated by
adding a false native point.

## Caller state

For layer `i`, block size `B`:

```text
boundary  = (i % B == 0)
slot      = i // B
prev_valid = ceil(i / B)
```

One `attn_res_point(...)` is one native AttnResFwd launch:

```text
attn_res_point(context, prefix_id, addend_id, bank_ids, valid_blocks,
               write_slot, output_norm, mix, materialize, tokens_local,
               bank_capacity, block_token_stride, optimize_prefill)
    -> ops, hidden_id, raw_prefix_id, materialized_id, updated_bank_ids
```

`bank_ids` has exactly `bank_capacity` entries. Only the first `valid_blocks`
rows enter the mixture; the current raw prefix is the extra row. The graph node
exports `hidden_states`, `prefix_out` and `materialized`. `prefix_out` aliases
the input prefix when there is no addend; `materialized` aliases the hidden
output when saving is false. A bank slot written by the point is represented by
the point's node id, so later readers depend on the actual write.

The caller sequence is:

| Point | Prefix/addend | `valid_blocks` | `write_slot` | Output norm | Pending MLP |
| --- | --- | --- | --- | --- | --- |
| prepare attention | prefix + previous MLP (BF16 round) | `ceil(i/B)` | `i//B` on boundary | input norm | absorbed, then cleared |
| prepare FFN | boundary: attention output, no add; else prefix + attention | `ceil(i/B)+boundary` | none | post-attention norm | stays in FFN output |
| final | prefix + final MLP | `ceil(end/B)` | none | none | absorbed |

Layer 0 has no pending MLP; it writes the embedding into slot 0. A boundary
logically clears the live prefix after prepare, so the FFN point starts from the
attention output with no addend. Every FFN output is pending and is absorbed by
exactly the next prepare or final point. It is not an
ordinary residual add.

### B=2, three layers

| Point | valid | write | addend | bank state |
| --- | ---: | ---: | --- | --- |
| L0 attention | 0 | slot 0 | none | `[embedding, -]` |
| L0 FFN | 1 | none | none (cleared) | `[embedding, -]` |
| L1 attention | 1 | none | L0 MLP | `[embedding, -]` |
| L1 FFN | 1 | none | attention | `[embedding, -]` |
| L2 attention | 1 | slot 1 | L1 MLP | `[embedding, L2 prefix]` |
| L2 FFN | 2 | none | none (cleared) | `[embedding, L2 prefix]` |
| final | 2 | none | L2 MLP | both slots mixed, no output norm |

A final-norm operator remains separate at the logits boundary; the final point
deliberately has no `output_norm`. Independent committed upstream versions
`58d3918e3ea0a544ffedadad2ba84559e9c51d8f` and
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` perform `self.model.norm` inside
`KimiLinearForCausalLM.compute_logits`. The image's precise upstream revision
and runner-selected logits rows are not established by this source bundle.

The retained non-last PP source uses an ordinary add, not an AttnRes launch.
The receiving stage also copies old bank slots into its larger bank. These
cross-stage mechanics are outside the PP1 implementation.

## Physical model

The selected kernel is AIV-only: no Cube GEMM, no FP32-v user workspace. The
16 MiB source system reservation is metadata, not user tensor traffic. BF16 is
the stored/input/output boundary; FP32 casts, score reduction, softmax and the
weighted sum are local working arithmetic, and the add/mixture round back to
BF16.

Ownership follows the retained tiling:

```text
used = min(T, AIV)
tokensPerCore = ceil(T / used)
ranges = contiguous [start, min(tokensPerCore, T - start)]
zero-token tail owners are absent
```

`T = tokens_local`, `H = hidden_size`, `N = valid_blocks`, `C = bank_capacity`.
Resident selection uses the retained `CalcMaxResidentRows` with the declared
device `ub` buffer and the AttnResFwd add-buffer reservation:

```text
work = 4*H*4 + 2*H*2 + 32KiB + H*2
resident iff N + 1 <= floor((ub - work) / (H*2 + 4))
```

History traffic:

```text
resident, mixing: 2*T*N*H bytes
reload,   mixing: 4*T*N*H bytes
mix=false or N=0: 0 bank-row bytes
```

Mandatory conditional I/O:

```text
prefix read                 2*T*H
addend read                 2*T*H iff fuse_add
prefix_out write            2*T*H iff fuse_add (otherwise aliases prefix)
write_slot bank write       2*T*H iff block_write_idx >= 0
hidden output write         2*T*H always
materialized write          2*T*H iff save_materialized
score weights (proj + norm) 2*H + 2*H per active core, only when mixing
output-norm gamma           2*H per active core when the prefill cache fits,
                            otherwise 2*H per token; none when output_norm=false
```

The gamma cache additionally requires the explicit source flag,
`tokensPerCore > 1`, and the retained UB estimate plus the aligned gamma row
fitting in the declared UB. Hardware-dependent work and diagnostics are bound
in `AttnResRecipe.prepare`, after semantic graph construction.

Vector work includes the RMS square/reduction, score normalization and dot
reduction, the current-prefix row in the weighted sum, all BF16/FP32 casts, and
the once-per-owner projection/norm-weight product. Reductions count logical
additions; exp, sqrt and division count one semantic operation each. This
retains the main arithmetic without claiming instruction issue or special
function latency. No timing residual enters this count.

The executable program has one serial load/compute/store phase chain per active
owner. The vector and copy stages are pinned to their physical owner; reads use
`mte2_aiv` on a split topology and `mte2` on a shared topology. HBM remains a
shared chip resource. The per-owner UB allocation spans the phase chain and is
released on store completion. This is an explicit serial-owner approximation,
not a reconstruction of the kernel's queue overlap or compiler schedule.

For example, `T=65, AIV=64` has `tokensPerCore=2` and only 33 active owners.
For `T=1`, increasing the number of AIVs at fixed aggregate vector throughput
reduces the one owner's available throughput; idle AIVs cannot lend capacity.

## Metadata and group

Every point carries:

- `native_kind = AttnResFwd`, one submitted kernel, complete source call;
- `attn_res_view` with `T`, `H`, `tokens`, `valid_blocks`, `bank_capacity`,
  `block_token_stride`, `fuse_add`, `block_write_idx`, `mix`, `output_norm`,
  `save_materialized`, `optimize_prefill`, resident/reload, active-core count,
  per-core token ranges, cache eligibility, stored/working dtype and rounding;
- `attn_res_bank_ids`, input prefix/addend ids, exported raw-prefix and
  materialized ids, and the bank rows actually read;
- source image/caller/operator hashes and explicit limitations;
- per-core read, write and vector work.

The declared subgraph role is `attn_res_prefix_point` (family
`attn_res_prefix`), not the old `attention_residual` role. Its contract remains
`None`: the strict descriptor machinery cannot recover `valid_blocks`, `mix`,
`fuse_add`, write slot or materialization flags from the CSV extents, and this
module does not broaden capture matching.

## Source identity

```text
image sha256   883670fcc91cbd75a7070b00501be0ecd81a266b81366a21bf02139d8e6a33e1
caller sha256  be9adc792774f919f443caf483eb2e04e4453617465381f47a482dd4736ec63f
common sha256  f8abf2be43c0e0a70a19df4d4d1cdab518dc939fee6612aabcaf1f6e090e53a2
tiling sha256  3249e4f252af0c68187b8caa7a00f63592a42c231fefecc28b7b1e7609b60669
resident sha256 bb5a40a14e3d5637ab9deff26b77e8e3a305500f9c582a3274d7bf0b23840d94
reload sha256  e346cd0db6eeedcfffaf911a07f6ddf4e34475c69677f73cb9e279f39641f070
```

These are a source-compatible declaration, not proof of the loaded capture
library, per-event tiling identity or scalar flags.

## Verification

```sh
PYTHONPATH=/Users/maoxx241/code/sim-kimi-attnres-20261029 \
  /Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python \
  -m pytest -q tests/test_attnres_prefix.py
```

A compact public control is `examples/source-families/attnres-prefix-control.json`:

```sh
PYTHONPATH=. /Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python \
  -m mindie_sim graph examples/source-families/attnres-prefix-control.json \
  --phase prefill --tokens 64 --scope body
```

The independent review is retained in
`data/verification/attnres-independent-review-20261029.json`. It includes the
original missing-owner/read-pipe counterexamples and the corrected actual
Engine controls. The affected suite passes 66 tests, with three pre-existing
optional skips; the earlier DSH full-suite result predates these corrections.

`tools/verify_attnres.py` prices a source caller histogram and retains candidate
scalar combinations for the historical descriptors:

```sh
PYTHONPATH=. python tools/verify_attnres.py \
  --case examples/cases/kimi-k3-a5-current-checkpoint-conditional.json \
  --tokens-local 1 1024 --optimize-prefill \
  --capture-evidence /Users/maoxx241/code/profiler/examples/structure/acceptance-20261029/kimi-attnres-boundary.json \
  -o data/verification/attnres-source-candidates-20261029.json
```

The 93-layer, block-size-12 declaration has 187 native points per forward.
Its primary declaration has no materialized auxiliary captures and enables the
explicit cache only in prefill. The isolated point sums are 1.408802 ms at one
local row and 20.522043 ms at 1024 local rows. These are predictions on the
declared current-checkpoint hardware, not complete model spans or accuracy.
Cache-disabled and source-allowed materialization alternatives remain separate
candidates with unknown observed occurrence counts; they do not enter the
primary histogram sum. No duration selects a history depth or scalar flag.

The historical prefill descriptors total 748 events, compatible with four
187-point source forwards. Its 97.622454 ms activity sum can therefore be
compared conditionally with 82.088172 ms under the complete-four-forward,
active-view and source-caller assumptions. Event/layer/step identity is still
unproven, so this is not a strict accuracy result or a step critical-path gap.
The decode inventory has 14020 events, five fewer than 75 complete forwards;
it is retained as an incomplete source-count comparison.

The focused tests exercise the public Frontend/target paths and actual Engine
runs: unselected graph identity, B=2 transitions, pending-MLP absorption once,
bank slot/valid progression, no-mixture bypass, resident/reload traffic,
capacity versus physical stride, token tails, no Cube service, input-ready
delay, final norm separation, explicit PP1 support, and structure
metadata.

Known limits: capture flags are not proven; score/softmax work is analytical
rather than a scalar schedule; SP AG/RS is external; the complete source caller
currently supports PP1; aux materialization is only explicit; no measured
duration, fitted bandwidth or hardware floor is used.

## Composition with sequence parallelism

`examples/cases/kimi-k3-a5-sp-attnres-conditional.json` combines the explicit
SP and AttnRes selectors. The source point receives the body's local rows:
8192 rows at TP8 become 1024 for the residual bank and latent projections.
Attention still receives 8192 rows after allgather; routed experts retain
their complete DP cohort. The first point depends on the input shard, each
FFN point waits for attention reduce-scatter, and the terminal point precedes
the final output allgather. There is no second norm1/norm2 or residual add.
The existing separate logits norm remains a caller assumption.

An idle DP participant has no local point or latent projection while it
continues to receive and compute expert rows. Lowering receives its actual
zero local-token count, so the explicit projection wrapper can be absent
there. A missing selected projection on a nonidle or unspecified caller
continues to report an error.
