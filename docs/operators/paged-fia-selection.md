# Paged FIA recipe selection audit

The retained comparison isolates recipe selection on runtime `3387f5c2`.
Both arms use the same parsed case, rates, query/context declaration and paged
storage. The selected arm additionally binds `flash_attention_decode` to
`staged_gqa_decode`; no timing parameter comes from the observations.

`data/verification/paged-fia-selection-20261029.json` retains the twelve actual
public `target_operators` / `operator_costs` executions, source hashes, case
hashes, raw capture identities, service work and zero-live-state checks.
The probe is the first ordinary body attention operator, with host submission
excluded. Each execution took less than 0.061 seconds locally.

| Declared case / query | Envelope us | Staged us | Observed mean us | Staged difference |
| --- | ---: | ---: | ---: | ---: |
| A3 Qwen3.5 / 2 | 6.999 | 12.097 | 53.574 | -77.42% |
| A3 Qwen3.6 / 1 | 6.996 | 10.820 | 54.717 | -80.22% |
| A3 Qwen3.6 / 4 | 7.014 | 26.479 | 58.192 | -54.50% |
| A5 Qwen3.8 / 1 | 16.784 | 134.640 | 135.013 | -0.28% |
| A5 Qwen3.8 / 4 | 16.806 | 157.259 | 143.916 | +9.27% |

The A5 result establishes that selecting the existing finite-owner phase model
removes most of this conditional discrepancy. It does not identify the captured
binary, dispatch, active owner count or native tiling. The analytic recipe uses
query tile 64, key tile 128, one KV split and two pipeline slots; these values
were already its defaults.

The complete selected A5 case is
`examples/source-families/paged-fia-a5-decode.json`. It preserves the original
case and adds only paged storage plus the explicit recipe binding. Reproduce
the isolated attention comparison from the repository root:

```python
import json
from pathlib import Path
from mindie_sim import operator_costs, target_operators

case = json.loads(Path(
    "examples/source-families/paged-fia-a5-decode.json").read_text())
for query in (1, 4):
    ops = target_operators(case, phase="decode", tokens=query,
                           context=16384, scope="body")
    attention = next(op for op in ops if op.family == "attention")
    print(operator_costs(case, (attention,)))
```

A3 retains a distinct hardware-coverage issue: its declared profile exposes
aggregate memory and compute but no private MTE1/MTE2/MTE3 service. The selected
recipe therefore cannot enforce those private transfer limits. That omission
does not identify a startup duration. At this audit revision, the two A3
query-8192 prefill controls used `ResourceEnvelope`: 508.463 and 762.694 us
against conditional observed means 2476.504 and 3678.352 us. The subsequent
explicit prefill recipe is described below.

The observation source is profiler's `paged-gdn-coverage.json`. Tensor matches
exclude context length, causal flags, mask values, cache page indices, KV-head
factorization and verify/draft role. Those remain caller declarations.
Observed counts differ from declared layer counts, and no event is bound to an
individual occurrence. All roles remain `unassigned`. Consequently these are
descriptive single-program comparisons, not formal heldout accuracy, a measured
whole-model gap, or a justification to multiply event counts into a step total.

## Explicit A3 prefill cases and SP compatibility

`examples/source-families/paged-fia-a3-prefill-qwen35.json` and
`examples/source-families/paged-fia-a3-prefill-qwen36.json` preserve the respective
profiler reference cases and add only
`runtime.kernels.recipes.flash_attention_prefill` with
`kind=staged_gqa_prefill`, `query_tile=64`, `key_tile=128`, and `slots=2`.
They retain the original default `kv_storage=auto`. Fresh context-zero prefill
therefore exposes contiguous TND operands; the filenames do not declare a
paged-cache ABI or identify a native dispatch. A paged comparison must declare
`attention_policy.kv_storage=paged` explicitly in both comparison arms.

Inspect the selected geometry without replaying a full model, or execute the
first attention operator, from the repository root:

```python
import json
from pathlib import Path
from mindie_sim import operator_costs, target_operators

for model in ("qwen35", "qwen36"):
    case = json.loads(Path(
        f"examples/source-families/paged-fia-a3-prefill-{model}.json").read_text())
    ops = target_operators(case, phase="prefill", tokens=8192,
                           context=0, scope="model")
    attention = [op for op in ops if op.recipe == "flash_attention_prefill"]
    print(model, len(attention), dict(attention[0].metadata)["gqa_prefill"])
    # Optional isolated execution; the compatibility review did not rerun it.
    # print(operator_costs(case, (attention[0],)))
```

The compatibility review on revision `1aeb52ad` checked 10 Qwen3.5 and 16
Qwen3.6 selected prefill operators: query8192, context0, respectively 4 and 6
local query heads, one KV head, head dimension256, and BF16 operands. This was
geometry construction only. The previously executed isolated query8192
results remain in `data/verification/paged-fia-prefill-20261029.json`; no full
query8192 graph was rerun.

`data/verification/sequence-parallel-kda-compatibility-20261029.json` also records
the independent SP review. With KDA Prepare/FWD_H explicitly selected, all four
Kimi DP replicas retain 69 Prepare and 69 FWD_H operators, sequence length8192,
unchanged physical signatures, and full attention projection rows. A TP2/EP2
query64 KDA graph and a DP2/TP2/EP4 query4 GQA graph with the selected staged
prefill recipe actually executed. Both released all tasks, programs and buffers.
These are selection, geometry and execution checks, not measured accuracy.
