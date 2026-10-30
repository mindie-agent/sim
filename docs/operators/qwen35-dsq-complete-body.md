# Qwen3.5 complete Body with explicit arch35 DSQ

Three new cases preserve the source-complete TP4/EP4 W8A8 caller, native GDN, INT8 MC2, TP token slicing and HCCL AR/AG selections. They add only the explicit `grouped_int8_caller.options` pair `dequant_program: arch35_full_load` and `input_capacity_rows: 128 / 2048 / 4096`. The existing cases are unchanged.

Cases: `examples/cases/high-concurrency-20261030/qwen35-mc2-tp-dsq-source-conditional/qwen35-35b-a5-tp4-c{4,64,128}-mc2-tp-dsq-source.json`.

The new run uses runtime Git revision `287f932bd4318d2b0698aa708bd44a5f17da2191`, including MC2 assignment-prefix accounting `06bc2a48` and DSQ source program `8eda7e91`. The package SHA-256 is `cffe6ac02eada08e469067b4a90ba38efa5083b0c11bebcca0f55b6564cc511c`; all three child start/end package identities match. The full report and reproducer are `data/verification/qwen35-dsq-complete-body-20261030.{json,producer.py}`.

## Complete 40-layer results

All three calls use decode q4, B4/B64/B128 and declared context4096, without Host submission or inter-step waiting. The selected observed populations are retained from the established complete-forward report. The actual active cache lengths and captured binary dispatch remain unverified.

| B | Observed Body ms | Historical control ms | New source Body ms | New relative gap | Wall / CPU s | Peak RSS MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 11.551367 | 4.705846 | 4.490392 | -61.13% | 5.562 / 5.544 | 102.77 |
| 64 | 21.956845 | 11.723498 | 11.706520 | -46.68% | 18.810 / 18.528 | 107.73 |
| 128 | 28.584774 | 16.621713 | 16.618497 | -41.86% | 29.743 / 29.631 | 116.83 |

The historical control comes unchanged from `qwen35-mc2-tp-source-complete-20261030.json`, package `0c36081a75ff95a6b1a0478e6b52bbba740df8116592823e1469b3fa8c596782`. It predates the MC2 prefix update, so these Body differences are **not a single-variable measurement of the DSQ change**. The matching old configuration is rebuilt structurally on the new package, but is not executed again. The producer asserts that every non-DSQ operation's geometry, dependency edges and logical work are unchanged across those two configurations.

Three fresh child processes completed within their independent60s budgets (54.45s total campaign wall). Completed tasks were507,683 /1,457,033 /2,226,853; peak live tasks10,280 /11,527 /11,927. Each finished with zero live tasks/programs and released buffers. No retry or budget extension was used.

## Capacity is local to the DSQ contract

The DSQ native input/output ABI and host dispatch use allocated rows128/2048/4096. The execution uses only valid rows32/512/1024 from the unchanged declared balanced expert histogram. The special host branch uses32 active owners at B4; the normal group loop reuses8/16 active owners at B64/B128. All64 launched AIVs reserve the declared finite UB buffers.

This does **not** close the allocation ABI of the entire grouped chain. Up `GroupedMatmul` and MC2 receiver metadata still expose valid rows32/512/1024. Their allocation capacity and captured `group_list` values have not been inferred from DSQ's output shape. The whole grouping therefore remains conditional; a matching DSQ capacity alone does not make a strict full-chain measurement pair.

The40 DSQ caller intervals total0.052480 /0.513821 /0.513821ms. These are activity sums, not independent additions to the Body critical path. High-batch source predictions remain around12.846µs/call versus observed54–57µs/call. The source branch and ownership are now explicit, but the major residual is unresolved. Scalar descriptor issue latency, unknown polling/compiler latency and PMU overlaps remain unpriced rather than being replaced by a measured per-group constant or residual floor. See [the source program and limitations](grouped-dequant-arch35.md).
