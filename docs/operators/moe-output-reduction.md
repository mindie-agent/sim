# TP expert output reduction

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

`runtime.moe_recipe = "tp_join_reduce"` explicitly selects a TP-only,
non-sequence-parallel schedule. Routed experts retain their partial TP outputs
through weighted unpermute. TP-sharded shared experts also retain their local
output. One all-reduce follows their token-wise sum.

The payload is `tokens * hidden * activation_bytes`, with the declared ring
algorithm's `2 * (TP - 1) / TP` per-rank traffic factor. It is not
`tokens * top_k * hidden`, and shared output does not incur a second reduction.
This is a dependency and tensor-geometry change; no measured duration or
communication event count chooses the policy.

The current supported declaration requires EP=DP=CP=PCP=1, TP-sharded shared
weights, replicated router/shared gates, and no routed latent output transform.
An output transform can contain nonlinear normalization and requires completed
TP output before the transform. Shared-expert DP and sequence-parallel schedules
also have different reduction ownership. Those paths keep their separate
recipes; this declaration does not silently reinterpret them.

The source basis is official `vllm-project/vllm-ascend` at
`8fd0bc4885cb52e58791aaa8efacc571b9bd115d`:

- `vllm_ascend/ops/fused_moe/fused_moe.py`: shared/routed join at lines 203-244,
  early-reduction exceptions at 304-312, final reduction at 343-356.
- `vllm_ascend/ops/register_custom_ops.py`: the non-SP ALLGATHER branch leaves
  shared output local and reduces the final result; MC2/ALLTOALL/FUSED_MC2
  already-reduced routed outputs follow another path (lines 123-156).

This version is declared by the retained GLM capture configuration. The retained
image digest and version declaration do not independently prove that the
installed binary exactly matches those source files. Historical comparisons
therefore retain that assumption.

The old GLM graph contains 231 reduction nodes: 78 attention, 78 dense/shared,
and 75 expert. Selecting this policy produces 156 body reductions (78 attention,
3 dense, 75 joined MoE). The capture's 157 native all-reduce rows do not establish
157 unique invocations: the available native export lacks stream/task/correlation
identities. An extra captured row can belong to the complete caller. Node counts
are structural diagnostics, not a count-fitting target.

Tests verify token-row payload, dependency placement after both weighted routed
and shared output, gated shared output, conserved non-communication work, complete
graph execution, and the algebraic reason linear combine can move before sum
while a nonlinear output transform cannot.

## Independent acceptance

The 2026-10-29 review executed 24 complete graph controls: balanced integer
routing or nonuniform expected probabilities, softmax or hash routing, no
shared expert or a shared expert with/without its gate, and shared overlap
enabled/disabled. At TP=4, 32 tokens and hidden width 32, every control retained
one 3072-byte per-source token-row reduction, removed the separate expert/shared
reductions, and preserved non-communication FLOPs, vector work and HBM bytes.
Every graph completed with zero live tasks/programs and a lower bound no larger
than its executed span. These are mechanism controls, not measured accuracy.

A separate source-planner control used explicit top-k IDs
`[[0, 1], [0, 2], [1, 2], [0, 1]]`, giving unequal expert counts
`[3, 3, 2, 0]`, with source alignment, source weighted combine and normalized
router weights. The complete graph succeeded, and its late reduction depended
on the actual source combine. The weight producer remained
`s0.l0.route_normalize`. Changing that declaration to hash routing, which
supplies IDs without the required FP32 weights, correctly failed with
`moe_combine_requires_declared_router_weight_dtype`; IDs never became weights.

Nonuniform routing is covered here only inside the declared EP=DP=1 scope:
TP shards see the same routing decision, so weighted unpermute and the shared
join can precede their sum. This acceptance does not cover cross-EP inverse
traffic, shared-expert DP, sequence-parallel ownership, routed latent transforms,
or automatic recognition of the captured native branch.

## Qwen397 TP16 retained caller correction

The two retained TP16/EP1/DP1 cases now explicitly select `tp_join_reduce`.
The independently retained Qwen3Next shared MLP sets `reduce_results=False`;
the Ascend non-SP ALLGATHER path restores routed token rows, joins the gated
shared output, then reduces that sum. Source hashes and line locations are in
the control receipt.
The captured order corroborates this boundary: grouped down produces 6400 rows,
unpermute restores 640 rows, shared work follows, then HCOM precedes the next
normalization. HCOM tensor fields are unavailable; exact binary identity remains
conditional. The recipe is selected by these input declarations, not model name.

With q=4, context=4104 and unchanged hardware, the complete 60-layer body gives:

| Captured query batch | Observed body mean ms | Before: expert-row reduce ms | After: token-row reduce ms | Signed gap | Relative gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| B64 / M256 | 76.089008 | 84.849228 | 68.312003 | -7.777005 | -10.22% |
| B160 / M640 | 119.054350 | 137.372823 | 102.779761 | -16.274589 | -13.67% |

Both satisfy the strict 15% body criterion under the original comparison
assumptions. Non-communication FLOPs, vector operations and HBM bytes are equal;
only reduction payload and its dependency boundary change. These are conditional
retained-body comparisons, not heldout accuracy or client TPOT. The physical
correction is kept even though it changes the residual from positive to negative.
