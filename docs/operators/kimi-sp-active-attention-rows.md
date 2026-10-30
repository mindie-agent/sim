# Sequence-parallel attention rows

The selected Kimi SP caller gathers the physical shard allocation and then
slices it to `positions.shape[0]` before invoking attention. That is the tensor
row count before SP padding, not the TP-rounded allocation. The slice is a view.
The source's `sp_reduce_scatter` subsequently copies those output rows into a
right-padded tensor and zeroes the tail before communication.

The retained caller is
`vllm-ascend-v41-private/vllm_ascend/models/kimi_k3.py`, lines 499–507,
SHA256 `c069c65685a8f40ad97624e23554c1c9235abf31bbc36f39b908e5074f23ce73`.
The helper is `vllm/models/common/ops/sequence_parallel.py`, lines 30–39,
SHA256 `699fde98360fc8d604055e2987d93108ee489e87cc5563949d2a3dcd838eb8bd`.
These are source mechanism identities; the capture's loaded helper binary is
not independently identified.

The frontend now passes `SequenceParallelPlan.input_rows` into attention and
records the gathered slice as a zero-payload dependency. A separate pad copy
precedes the attention reduce-scatter when needed. It reads `B * hidden * 2`
BF16 bytes and writes `ceil(B/TP) * TP * hidden * 2` bytes, zeroing only the new
rows. No such copy is added when B is already divisible by TP.

For TP8 and B1/2/4, Q/K/V projections, MLA prolog and short convolution now
consume 1/2/4 rows. Local residual and router tensors still have one row;
FFN allgather tensors and collective payloads still have eight physical rows.
Separately declared graph-capture padding is preserved: valid B1 with incoming
positions rows4 and SP capacity8 uses four projection rows, one recurrent
query and four output pad rows. This change does not equate a graph bucket
with valid requests.

The two `kimi-sp-attention-rows-20261030-{before,after}.json` ledgers use identical
inputs and all 93 layers at B1/2/4/8/32. Source hashes remained unchanged during
each run. FFN executable work signatures, all communication work signatures
and graph-exterior work signatures are identical for all five cases. At B8
and B32, the complete attention work signatures also remain identical. At
B1/2/4, removing the incorrect projection/conv rows and adding the real output
padding changes attention work as follows:

| Batch | FLOPs delta | HBM bytes delta | FFN / communication changed |
|---|---:|---:|---|
| 1 | -69025259520 | -91243296 | No |
| 2 | -59164508160 | -74655552 | No |
| 4 | -39443005440 | -42659712 | No |

The 64 focused controls include the source graphs at all five batches,
input-readiness and allocation cleanup through the actual Engine, existing
unequal/idle DP and graph-bucket behavior, native KDA and paged MLA selection.
The change has no fitted rate, launch floor or measured latency input.

Two complete 93-layer source-host body runs at runtime `dbcb30eb` completed
with the original 90-second worker budget and package SHA256
`5b0313089714668f8b607326f05ddd1ad86ea9cc69ca0beabc82191c3a24248f` unchanged:

| Batch / context | Body ms | Wall / CPU s | Peak MiB | Engine tasks |
|---|---:|---:|---:|---:|
| 1 / 4096 | 23.30537 | 7.65 / 7.60 | 150.8 | 577929 |
| 4 / 4096 | 23.59918 | 9.28 / 9.19 | 153.3 | 680913 |

Both released all tasks, programs and allocations. The previous identical
B1 source-host input predicted 23.58040 ms; the source row correction lowers
it by 0.27504 ms. This is a mechanism correction, not a claim of improved
measured accuracy. Receipts are `kimi-sp-active-body-20261030*.json`; reproduce
them with the existing source-body producer, `--host-dispatch`, and explicit
`--run` selections for the B1/B4 cases. Original receipts remain unchanged.

## Independent integration review

The retained `kimi-sp-independent-review-20261030.json` and producer compare
30 complete 93-layer frontend graphs across TP2/4/8, B1/2/4/8/32 and both
positions=B and positions=B+3. Attention rows, view alias and reduce-scatter
padding agree with independent source formulas. FFN/outside work and every
communication payload match the parent. Three unselected-SP controls remain
identical; PCP/CP source support is unchanged. A new actual two-layer Engine
control with valid3/positions5/physical8 starts after an 11 ms input and
completes at 11.461717 ms with no live work or allocation. These checks
establish caller geometry and dependencies, not measured timing accuracy.
