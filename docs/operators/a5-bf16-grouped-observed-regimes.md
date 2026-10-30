# A5 BF16 two-bank grouped call regime

The reusable rules in
`examples/service-regimes/a5-bf16-grouped-two-bank-20261030.json` bind complete
native BF16 `GroupedMatmul` calls with **two weight banks, 8–16 declared valid rows
in each bank**, BF16 output and these two N/K contracts:

| N × K | c32 p4k median used to assign service |
|---|---:|
| 6144 × 3584 | 47.053 µs |
| 3584 × 3072 | 22.110 µs |

Selection uses physical bank count, per-bank row extrema, dimensions and dtypes.
There is no model/case matcher, batch threshold, or piecewise interpolation
through a Body residual. The response is inclusive: the unchanged source program,
weight traffic, buffer dependencies and hardware contention still execute.

The fitting native ABI has input capacity 64 rows, two BF16 weight banks and
`Block Num=32`; the forced-balanced scenario declares valid rows `[16,16]`.
Capacity is **not** substituted for valid rows. Native `group_list` contents are
not captured, so this remains an explicit scenario-to-call pairing. c16 uses
capacity 32 and declared valid rows `[8,8]`.

Only c32 p4k native call medians assign parameters. Across c16/c32 × p4k/p16k,
**8 of 8 real frontend/Engine comparisons are within 15%**, worst **6.20%**.
The remaining 16 low-row comparisons retain analytic source execution and remain
in the report; historical source roles stay `unassigned` throughout.

## Low-row counterexample retained

A bounded streaming CSV audit finds the same two native tensor ABIs and
`Block Num=32` at c1/c2/c4/c8: capacity `[16,K]`, weight banks `[2,K,N]`.
The current SP caller declares `[4,4]` valid rows in all four cases, although
`valid_local_tokens` differs and padded tokens are eight. Native medians differ:

| Client concurrency (observation label only) | Up µs | Down µs |
|---|---:|---:|
| 1 | 21.895 | 16.026 |
| 2 | 22.406 | 16.211 |
| 4 | 41.265 | 20.058 |
| 8 | 42.813 | 20.584 |

This is evidence of an unresolved native group-list, padding/mask or dispatch
condition. Equal allocated tensor sizes do not establish equal actual expert
work. Assigning 47.053 µs to every `[4,4]` scenario would more than double the
c1 prediction; selecting a fee by client concurrency would hide that mismatch.
The high-row rule therefore does not cover this ambiguous domain. Simulation
continues analytically, with `analytic_outside_domain` reported.

The producer preserves native ABI samples, original capture identities and
complete-forward statistics. Its CSV probes stream only until 200 grouped calls;
no large capture is read into memory. Three targeted Engine tests retain valid
payload/bank work, late readiness, released resources and the low-row fallback.
The main integration task combines these rules with small-M matrices, MC2 and
HCCL for one full Body replay.

```sh
python data/verification/a5-bf16-grouped-regimes-20261030.producer.py \
  data/verification/a5-bf16-grouped-regimes-20261030.json
python -m pytest -q tests/test_a5_bf16_grouped_regimes.py
```

A retained-source oracle now explains a concrete candidate for the low-row
transition. At pinned vllm-ascend `8c30e4f`, `ascend_forward_context.py:217–226`
sets the first actual token bits true and padding false;
`prepare_finalize.py:273–284` slices those bits per SP rank.
`token_dispatcher.py:177–186` supplies the active mask in `global_bs=0` mode.
`force_eplb.py:130–142` uses `cursor=local_rows*topk*source_rank+source_rank+i`,
then maps column and row into **contiguous**, not strided, expert ownership.
For E16/EP8/topk8 this yields rank0 bank rows **[1,0], [2,0], [3,1], [5,3],
[8,8], [16,16]** at valid token counts 1/2/4/8/16/32. Total assignments and
all directed peer counts are conserved in the retained pure-integer oracle.
This explains why one bank can be active at the two smallest populations while
the allocated tensor remains unchanged. It is an explicit source hypothesis,
not a claim to have recovered unrecorded `group_list` values. The next change
must carry this mask and cursor consistently into both expert work and peer
traffic; changing only matrix counts would be incomplete.
