# Qwen3.5 A3 q4 frozen FFN response transfer

Two complete long-context populations now use the unchanged older A3 count-prefix,
fused gate/up and expert-down responses. Both improve, but neither meets the
15% Body criterion. This delivery changes input declarations and retains evidence;
it does not change the runtime or fit a new coefficient.

| Actual batch / query tokens | Complete Bodies | Prior ee3 Body ms | Frozen FFN Body ms | Observed Body ms | New error |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 / q4 | 169 | 9.582281 | 14.084175 | 17.505444 | −19.54% |
| B4 / q4 | 138 | 16.947766 | 19.048784 | 22.870630 | −16.71% |

The candidate starts from the Q35 FAI source case in commit `1c5939cd`. Its shared
expert remains BF16 and its reduction remains `tp_join_reduce`. It adds only
`group_list_prefix=vllm_ascend_group_list_prefix`, the existing
`declared_native_fusion` for `grouped_gate_up_activation_quant`, and three verbatim
rules from `data/observations/parameters/a3-native-decode-20261030/parameters.json`.
The packet's original 4K assignment and 16K validation provenance remains intact.
These new 126K-prompt observations were inspected for this diagnosis and are not
blind holdouts; none assigned or changed the response coefficients.

For both new populations, the full exported gate/up and down ABI equals older
retained ABI at the same routed rows (32 and 128). This includes the INT8 operands,
FP32 input scales, packed NZ weights, BF16 down output, and Block20/Mix40 or
Block20/Mix20 declarations. The gate/up and down response domain stays 16–128
routed rows. No extension to B6 or larger is implied.

Cumsum has no tensor descriptor in either capture. Its 256-entry INT64 contract
is therefore a conditional source/consumer inference, backed by the same fused
consumer's 256-expert ABI and the pinned source conversion from count-list type1
to cumulative type0. It is a real `AI_CPU` native operation, distinct from the
`allreduceAicpuKernel` scheduler wrapper. The unchanged 49.131-us response remains
at exactly 256 entries; the new native means are 47.121 and 45.964 us.

The local ledger streams each original CSV after verifying its full SHA256 and
uses the retained complete Body windows without new segmentation. All selected
Bodies have exactly 40 calls of each FFN operator. Every HCOM interval is retained
once; only known communication scheduler wrappers are excluded. Native activity
sums equal the existing Body activity union for these captures: 16.895238 and
22.259926 ms. The separate uncovered gaps are 0.610205 and 0.610704 ms. This equality
is observed for these captures, not a general rule permitting activity or modeled
scheduled intervals to be added as a Body correction. Only three retained forwards
per source contain detailed layer windows; unavailable layer occurrences are not
invented.

Each ledger took 2.1–2.6 seconds and under 47 MB RSS. Serial ordinary `predict`
Body runs took 3.73 and 7.30 seconds with 82.7 MB process peak; each queried all
three transferred responses 40 times and drained all tasks, programs and buffers.
The prefix is a dependency of fused gate/up, while expert-down consumes that fused
INT8 output. Target binary identity and actual routed values remain unobserved.

Evidence is in `data/observations/a3-qwen35-frozen-ffn-20261030/`:

- `native-b1.json` and `native-b4.json`: full source identities, typed native
  groups, count ranges, locations and observed activity.
- `validation.json` and `prediction-b*.json`: exact older ABI matches, unchanged
  rules, graph checks, complete prediction output and the remaining failed scores.
- `cases.json`: standard Profiler manifest for exactly the two inspected source
  hashes, with the independent candidate case path. It does not edit current
  Profiler source declarations or infer bindings for other captures.

```sh
python3 data/observations/a3-qwen35-frozen-ffn-20261030/bind_validate.py --predict
python3 data/observations/a3-qwen35-frozen-ffn-20261030/native_ledger.py \
  --batch 1 --comparison /path/to/retained-single-source-body-report.json \
  --output /tmp/qwen35-q4-b1-native-ledger.json
```

The metadata-only command without `--predict` rebuilds the candidate and exact
source manifest. Remaining work is to inspect communication and typed GDN/helper
caller boundaries; the remaining 3–4 ms is not used as a fitting target.
