# Ordinary MoE routing frontend selection

Status: structural integration of the existing `operators/moe_align.py` and
`operators/moe_combine.py` planners with the normal MoE frontend. This is not a
hardware measurement, capture-dispatch claim or accuracy result.

## Declarations

The selectors are the existing semantic recipes `route_pack` and `route_unpack`:

```json
{
  "runtime": {
    "kernels": {
      "lowerings": {
        "route_pack": {
          "implementation": "cuda_moe_align_block_size",
          "options": {"block_size": 16, "pad_sorted_ids": true}
        },
        "route_unpack": {
          "implementation": "triton_moe_fused_mul_sum",
          "options": {"compute_capability": 89, "weight_dtype": "fp32"}
        }
      }
    }
  }
}
```

`implementation`, not a model or hardware label, chooses the path. The same
declaration supplies the lowering and the physical `RecipeBook` used by
`target_graphs` -> `graph_costs`, `operators` and `predict`.

`route_pack` options:

| Option | Domain |
| --- | --- |
| `block_size` | Required positive integer source tile geometry, not a fitted timing knob |
| `pad_sorted_ids` | Explicit boolean; selects the source wrapper's buffer rounding |
| `ignore_invalid_experts` | Explicit boolean; the ordinary frontend supplies no expert map |
| `topk_ids_dtype` | Must match the frontend's declared integer id representation (`int32`) |

`route_unpack` options:

| Option | Domain |
| --- | --- |
| `compute_capability` | Required; exactly `89` selects the source's SM89 tile heuristic |
| `weight_dtype` | Required declared router-weight dtype; must match the frontend's FP32 top-k weights |

## Alignment and the real pack copy

`moe_align` produces `sorted_token_ids`, `expert_ids` and related padding/index
state. It does not move activation payload. The binding therefore inserts a new
dependent `moe_align` call between the top-k producer and `route_pack`; it does
not replace `route_pack`. Its activation payload reads and compact payload
writes remain. The selected adapter scans the alignment's padded
`sorted_token_ids`, used `expert_ids` blocks and `num_tokens_post_pad`.
Sentinels contribute index reads and one declared scan operation, but no
activation reads or output rows. The output includes the inverse token/slot
index beside each valid compact expert row. This is an explicit analytical
aligned-to-compact copy; no source pack kernel is claimed.

The source call expands into its actual launch children: one small-batch kernel,
or the large two-kernel align/sort pair with the sort depending on align. Stream,
phase, layer, stage, requests and the original pack dependencies are retained.

`RoutingRealization` may now carry explicit optional `topk_ids`. The ids must
match the declared token count, top-k, expert range, uniqueness per token and
exact per-expert histogram. They are preserved in metadata. A histogram is never
converted into a synthetic token mapping: a selected alignment without explicit
ids fails visibly. Histogram-only grouped GEMM behavior is unchanged.

Hand counterexample for the supplied four-token control (`topk=2`, latent `32`,
`int32` ids, histogram `[3,3,2,0]`, block size 16): the unaligned pack reads
288 bytes and writes 544 bytes. Alignment produces 48 used index slots across
three expert blocks. The adapter reads 256 activation bytes plus
`48*4 + 3*4 + 4 = 208` index/count bytes, and still writes only 544 bytes.
Its 40 sentinel slots never become padded GEMM rows or payload copies.
Substituting alignment for pack would delete real payload work; retaining the
old 32-byte index read would omit the producer's actual padded index scan.

## Combination domain

`route_unpack` selects `triton_moe_fused_mul_sum` only when all of the following
hold:

- its semantic input is explicitly `[tokens, topk, hidden]`;
- output layout is contiguous and the source writes `tokens*hidden` values of
  the declared activation dtype;
- router weights have a declared dtype matching the binding option;
- the source's no-expert-map branch is selected;
- weights are applied by the kernel, never already on the input
  (`apply_router_weight_on_input=false`);
- the explicit source option `compute_capability=89` is selected;
- the ordinary caller is EP=1 local rows.

The normal compacted expert path produces expert-grouped rows, not the source's
token-major tensor. The binding represents the required mapping with an explicit
dependent `route_unpack.materialize` copy: it reads the compacted rows plus their
inverse token/slot indices and writes the full `[tokens, topk, hidden]` payload. The source
combine then consumes that materialized tensor and the top-k weights through a
real dependency. It reads the pack-produced inverse token/slot index, rather
than reconstructing a possibly nondeterministic within-expert sort order from
top-k IDs or counts. The materialization is an analytical copy abstraction, not a
claimed source kernel, and is not a fallback for a failed combine selection.
Unsupported layouts, missing ids, missing weights, missing capability or a
different capability fail with a modeling error.

Hash routing currently produces integer IDs without a floating-point weight
tensor. It can select alignment but cannot select this weighted combine.
Top-k normalization retains separate producers for integer IDs and normalized
weights; the alignment does not consume the normalization output as IDs.

EP>1 distributed combine maps and expert-map branches are intentionally
unsupported by this ordinary binding. They need their own ownership/layout
contract rather than a relabeled compacted tensor.

The source grid still uses real M/K tile tails. For `tokens=5, hidden=130,
topk=3`, the SM89 branch selects `BLOCK_M=2`, `BLOCK_K=256`, grid `(1,3)` and
writes only the `5*130` valid output values; masked arithmetic remains charged.

## Known limits

- The current alignment sort ledger splits routing ids by flat CTA index. Two
  equal-numel mappings with the same histogram therefore have the same supplied
  sort-CTA work even though their token identities differ. No alternative CTA
  distribution is invented. The explicit ids are still required and retained
  for the source histogram/scatter inputs.
- The ordinary binding fixes integer top-k ids to `int32`; the standalone
  alignment planner continues to expose other integral input widths.
- `route_unpack.materialize` is an explicit analytical transfer. It is not a
  measured unpermute kernel and does not claim instruction-level fidelity.
- No historical measurement for these source kernels is used or claimed.

## Verification

`tests/test_moe_family_selection.py` covers the retained pack work and
dependencies, small and two-kernel alignment expansion, explicit-id agreement
and zero experts, same-histogram/different mappings, combine tail tiles,
declared capability/weight dtype, and public `graph_costs`/`predict` execution.
`examples/source-families/moe-routing-control.json` is a synthetic CPU control.

```sh
/Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python -m pytest \
  tests/test_moe_family_selection.py tests/operators/test_moe_align.py \
  tests/operators/test_moe_combine.py tests/test_grouped_family_selection.py -q
```
