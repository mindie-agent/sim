# Independent review: arch35 full-load DequantSwigluQuant

The reviewed implementation preserves the pinned source's host-capacity branch,
valid expert rows, owner assignment and finite queue dependencies. No blocking
runtime issue remains in the inspected revision. This is mechanism acceptance,
not a claim that DSQ duration now matches a particular capture.

Review target: `sim-mc2-int8-dispatch-20261030`; exact file hashes are in
`data/verification/dsq-arch35-independent-review-20261030.json`. The adjacent
producer accepts `--source-root` and does not modify that checkout.

## Source boundary

Official `cann/ops-nn` revision `955f8b33985b6d72989ebff341bb9c2c564a51bc`:

- Host `quant/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling_arch35.cpp`,
  lines 756–768: allocated input rows and group count select special splitting.
  Lines 939–1022 and 1042–1043 define full-load UB rows and launched/max cores.
- Kernel `quant/dequant_swiglu_quant/op_kernel/arch35/dequant_swiglu_quant.h`,
  lines 196–223: two input/output queues, one group-scale queue, one output-scale
  queue and one temporary buffer. Lines 240–317 select each source group;
  lines 318–843 load scales once per owner/group and execute row tiles.

Their file hashes match the source declarations. The captured binary's choice
of this template remains an explicit conditional assumption.

## Controls and findings

**378 owner-geometry controls and 21 actual Engine executions passed.**
The controls cover group counts 1–96, 1/4/64 physical AIVs, zero and positive
expert banks, small rows, tails, multiple UB tiles and unchanged-work bandwidth
scaling. All Engine runs respect input release at 7 ms and release every task,
program and UB allocation. Peak task count remains bounded by active owners.

The decisive branch counterexample holds 32 valid rows and 32 expert groups
constant while increasing input capacity from **543 to 544**:

- `543 // 32 == 16`: special group splitting, 64 launched AIVs.
- `544 // 32 == 17`: ordinary per-group row splitting, 13 launched AIVs.
- Both retain 32 valid rows, 65,536 bytes of weight-scale reads, and unchanged
  valid tensor payload. Only the source-selected owners and metadata scans
  change. Padding is not converted into extra dequantization work.

The multi-tile controls verify that the next dequantization waits for the prior
quantization and scale write; MTE3 scale output waits for the preceding output;
and a new group's scale load cannot overtake the prior vector consumer/read.
The retained report records **2,484 cross-tile dependency checks** across rate
controls. Scale reads occur once per actual owner/group, not once per UB tile.
Zero experts have no payload work or separate kernel invocation.

Two zero-group-hole counterexamples reproduce the pinned source's unusual
`continue` before advancing the special-group cursor: later positive work on
the same owner would be skipped. The implementation reports this unsupported
source domain rather than silently dropping rows. The selected contiguous
balanced Qwen cases do not encounter this hole.

Review found an export contract issue: native tensor ABI used valid rows while
host dispatch used allocated capacity. The developer corrected the source ABI
for input, row scales, output and output scales to allocated capacity. Valid
logical rows and typed downstream scale producers remain separate. The ordinary
analytical path retains its prior valid-view contract.

The synthetic rates in this review only test resource conservation and
monotonicity. They are not hardware calibrations. Scalar descriptor issue and
the source's ambiguous repeated inScale FreeTensor lifetime remain disclosed;
no residual-derived startup delay was introduced.
