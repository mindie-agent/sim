# Explicit SP mask and source force-EPLB routing

The generic `routing: {policy: balanced}` scenario is unchanged. The new source declaration selects the pinned vllm-ascend `8c30e4f491f73cba737a1dacc522894e437cdf3d` caller:

```json
{
  "runtime": {
    "routing": {"policy": "balanced"},
    "kernels": {
      "force_eplb_source": {"kind": "vllm_ascend_sp_mc2", "global_bs": 0},
      "native_moe_protocol": {"kind": "arch35_fullmesh", "max_inflight_per_rank": 8}
    }
  }
}
```

An explicit sequence-parallel declaration is also required. The supported caller domain is DP1, EP=TP>1, CP=PCP=PP=1, contiguous global valid-token prefix, and native MC2. This is an explicit source scenario: the binary and group-list contents of the historical captures have not been established. No profile, rate, invocation floor or observed service parameter changes here.

## Source contract

The source checkout is `/Users/maoxx241/code/vllm-ascend-v4-default-comm`. At that revision:

- `ascend_forward_context.py:217–226` separates allocated/padded mask capacity from the actual valid token prefix.
- `ops/fused_moe/prepare_finalize.py:273–284` slices the mask using TP rank and the already-sharded tensor's allocated rows.
- `ops/fused_moe/force_eplb.py:130–142` constructs an allocated-row cursor. For assignment `i` on source rank `s`, `cursor = i + allocated_local_rows * top_k * s + s`; destination rank is `cursor % EP`, and local expert is `(cursor // EP) % (E / EP)`.
- `ops/fused_moe/token_dispatcher.py:127–144,177–186` passes the active mask for `global_bs=0`. The other branch is not selected implicitly. The returned expert-token counts are the valid grouped input.

The mapping uses **contiguous EP ownership**, unlike the generic scenario's strided IDs. Router and TopK retain allocated tensor shapes; masks filter dispatched payload, receiver expert rows and return payload. Count/status participation remains present on idle sources. Transport explicitly sends the exact off-diagonal integer assignment matrix, then its transpose at the return frame width. Local cells remain local physical work and do not consume wire bandwidth.

The ledger uses bounded `O(EP × E)` residue arithmetic and a 128-entry geometry cache. It never creates a task per token or expert assignment. Existing MC2 phases and the same Engine consume the result.

## Rank boundary and examples

The ordinary TP representative graph models **physical EP rank zero's compute**, with full physical-peer transport. It retains every rank's expert counts and clearly labels this conditional boundary. Different ranks can have different active bank counts; the rank-zero compute must not be called a maximum-rank bound.

For E=16, EP=TP=8, top_k=8:

| Global valid tokens | Allocated local rows | Rank-zero expert rows |
|---:|---:|---|
| 1 | 1 | (1, 0) |
| 2 | 1 | (2, 0) |
| 4 | 1 | (3, 1) |
| 8 | 1 | (5, 3) |
| 16 | 2 | (8, 8) |
| 32 | 4 | (16, 16) |

The prior generic scenario has (4, 4) for each of the first four rows because it deliberately routes padded rows. These are distinct declarations, not two names for the same input.

## Verification

`tests/test_source_force_eplb.py` compares closed arithmetic with a token-by-token independent oracle across 3 EP sizes, 3 expert-bank counts, 3 top-k choices, 3 allocated row sizes and valid-prefix tails. Real two-layer frontend controls cover all six table entries, preserve allocated router/TopK work and active-bank weight identities, and conserve grouped FLOPs. Three actual Engine controls verify full framed wire bytes, transpose direction, late input readiness and released tasks/buffers. A 0.5/1/2 link sweep preserves work and changes duration monotonically.

The focused regression group passed **63 tests in 6.60 s**: the new controls plus generic balanced routing, SP padding, native MC2, MX dispatch, per-token INT8 and token-partition callers.

`data/verification/source-force-eplb-20261030.producer.py` records 12 source scenarios and 24 actual grouped programs against the retained native operator medians. All programs release tasks. It adds no timing fit. Low-row analytic predictions remain below the observations (e.g. rank-zero up: 14.27 vs 21.90 µs at c1 and 29.14 vs 41.27 µs at c4); the source mask resolves the bank-count mechanism without presenting that structural correction as complete timing convergence. Whole-body replay belongs to the combined mainline acceptance, not an extrapolated activity sum here.
