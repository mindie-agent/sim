# Qwen3.5 A3 q4 frozen FFN and helper response transfer

Two complete long-context populations now use six unchanged older A3 responses.
B4 meets the 15% Body criterion; B1 narrowly fails at −15.0039%, without rounding
the score into a pass. This delivery changes input declarations and retains
evidence; it does not change the runtime or fit a new coefficient.

| Actual batch / query tokens | Complete Bodies | Prior ee3 Body ms | Three-rule FFN Body ms | Six-rule Body ms | Observed Body ms | Six-rule error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 / q4 | 169 | 9.582281 | 14.084175 | 14.878945 | 17.505444 | −15.0039% FAIL |
| B4 / q4 | 138 | 16.947766 | 19.048784 | 19.847313 | 22.870630 | −13.2192% PASS |

The candidate starts from the Q35 FAI source case in commit `1c5939cd`. Its shared
expert remains BF16 and its reduction remains `tp_join_reduce`. It adds only
`group_list_prefix=vllm_ascend_group_list_prefix`, the existing
`declared_native_fusion` for `grouped_gate_up_activation_quant`, and six verbatim
rules from `data/observations/parameters/a3-native-decode-20261030/parameters.json`.
Commit `c4e91818` retains the three-rule count-prefix/gate-up/down baseline. The
next transfer adds only `expert_add`, GDN gating and output norm-gate responses.
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

Each ledger took 2.1–2.6 seconds and under 47 MB RSS. The six-rule serial ordinary
`predict` Body runs took 2.57 and 4.88 seconds with 82.8 MB process peak. Each queried
the four FFN responses 40 times and the two GDN helper responses 30 times, and
drained all tasks, programs and buffers.
The prefix is a dependency of fused gate/up, while expert-down consumes that fused
INT8 output. Target binary identity and actual routed values remain unobserved.

Evidence is in `data/observations/a3-qwen35-frozen-ffn-20261030/`:

- `native-b1.json` and `native-b4.json`: full source identities, typed native
  groups, count ranges, locations and observed activity.
- `validation.json` and `prediction-b*.json`: exact older ABI matches, unchanged
  rules, graph checks, complete prediction output and the remaining B1 failure.
- `helper-source-contract.json`: exported helper ABI comparisons, paired native
  recurrence descriptors, source hashes and the limits of inferred tensor fields.
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
source manifest. The source check reads the pinned local vLLM-Ascend revision
`8c30e4f491f73cba737a1dacc522894e437cdf3d`; its path and per-file hashes are retained
in the helper contract. No remote run or additional capture is required.

## Three additional helper boundaries

`ForeachAddList` has a complete identical old/new typed ABI at T4 and T16:
two BF16 ND `[T,2048]` operands, scalar FLOAT, BF16 ND output, AIV Block40/Mix0.
Its unchanged response is `15.9517368421 + 0.00376973684211*T` us.

Gating and output norm-gate retain `N/A` tensor descriptors. Their transfer is
conditional on source geometry, corroborated by the same exported native type,
core and launch counts. It is not a claim of fully exported typed ABI or matching
capture binaries. The old q2 and new q4 bodies have the same flattened T4/T16
helper geometry:

- Gating's wrapper takes contiguous `[T,8]` a/b, sets `seq_len=1`, and launches
  `(40,1)` with 64-row, 8-head tiles. It has no request-query, context, history or
  checkpoint argument. The paired recurrence descriptors independently agree
  on BF16 beta `[T,8]` and FP32 g `[T,8]`. The unchanged response is
  `8.81040526316 + 0.00541710526316*T` us.
- The caller reshapes BF16 recurrence output and z to `[8*T,128]` before norm.
  The retained norm wrapper uses a single 128-wide group with 64 rows per program,
  producing Block1 at T4 and Block2 at T16, as seen in both populations. It has no
  request-query, context, history or checkpoint argument. Its unchanged response
  is 4.7045 us, with the original T2–T16 domain and caller selector.

The three added helper means are within 15% at both targets. Their errors for
B1/B4 are −0.042%/+3.332% for expert-add, −7.954%/−10.271% for gating and
−3.185%/−2.844% for norm-gate. The earlier B4 gate/up and expert-down responses
still fail their operator checks at −20.969% and −23.668%, despite the new Body
pass. These are operator checks, not additive Body corrections. The remaining
Body discrepancy is not used as a fitting target.

Convolution and recurrence responses are excluded: q2→q4 changes history slots
4→6 and checkpoint counts 2→4. The old route-pack response is also excluded
because its quantization boundary has not been closed. Shared BF16 precision,
TP join reduction, FAI, communication and all other baseline declarations stay
unchanged.
