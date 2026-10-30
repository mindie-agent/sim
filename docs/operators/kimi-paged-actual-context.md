# Kimi paged MLA host metadata at retained prompt contexts

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The earlier `kimi-source-paged-host` inputs bind native CPU SplitCore metadata
for nominal contexts 4096 and 16384. The profiler replay instead supplies 4104
and 16392 from its retained geometry/client prompt references. All 12 inputs
were rejected by the existing exact-context guard. The new inputs under
[`kimi-source-paged-actual-context`](../../examples/cases/high-concurrency-20261030/kimi-source-paged-actual-context/)
use freshly executed CPU SplitCore metadata for those declared contexts.
The old inputs, old oracle results, runtime and guard are unchanged.

The input/evidence index
records the original profiler manifest and failed-replay hashes, all 12
matching original case hashes, each captured source reference, new input
hashes and the local reproduction of `paged_mla_dispatch_geometry_mismatch`.
No raw CSV was rescanned. B/q come from the retained primary geometry, not
client concurrency or tensor capacity. The context is the initial prompt
reference; active per-forward KV growth remains unobserved.

The new CPU oracle
records the retained C++ source/header and harness hashes. It evaluates
contexts 4104/16392 and batches 1/2/4/8/16/32 with actual current model heads
(16 physical heads from 12 logical heads) and 32 AIC owners. It retains the
original C++ scheduling arithmetic, including FD metadata. It neither chooses
arbitrary KV split counts nor uses measured timings. This identifies a
retained source algorithm, not the deployed binary's dispatch metadata.

Current SP behavior is verified in all 12 actual four-layer graphs:
attention projections and FIA receive B input rows, while SP all-gather and
FFN retain `max(8,B)` physical rows. Source KDA still receives B independent
q1 sequences. Every graph waits for the declared 7ms input and releases all
tasks, programs and buffers. The new metadata partitions the actual KV tile
extent exactly once through the unchanged runtime parser.

Four complete 93-layer endpoint predictions also completed:

| Batch | Declared context | Body prediction ms | Worker wall seconds |
| ---: | ---: | ---: | ---: |
| 1 | 4104 | 23.30537 | 7.55 |
| 1 | 16392 | 23.39219 | 8.04 |
| 32 | 4104 | 27.66017 | 11.70 |
| 32 | 16392 | 30.37815 | 21.09 |

These are conditional predictions, not new measured accuracy results. Each
full-body result retains source bindings, resource service and lifecycle
evidence. Runtime commit is `294bcd2c`; package SHA is
`bce73c42fec043e680e6ad2e1399bef9bcbff358577611c11aa0b457ce965fdc`.

The four workers had already started with 90s outer limits when the shared
60s acceptance budget was clarified. All finished within 22s. Their original
budget is explicitly retained, no completed run was relabeled or rerun to
obtain a pass, and the reproducible producer now uses 60s.

Reproduce the input metadata and bounded graphs locally:

```sh
python data/verification/paged-mla-split-oracle-20261030.producer.py \
  --source-root /path/to/pinned/ops-transformer \
  --contexts 4104 16392 --batches 1 2 4 8 16 32 \
  -o data/verification/paged-mla-split-actual-context-20261030.json
python data/verification/kimi-paged-actual-context-20261030.producer.py \
  --manifest /path/to/profiler/acceptance-20261030-paged-dynamic/decode-forward-kimi-glm-cases.json \
  --failed-replay /path/to/profiler/acceptance-20261030-paged-dynamic/decode-forward-kimi-glm-replay.json \
  --run-endpoints -o data/verification/kimi-paged-actual-context-20261030.json
```

The CPU oracle now accepts explicit `--contexts`, `--batches`,
`--physical-heads` and `--aic-cores`. Omitting them preserves the original
nominal geometry defaults. The shared host-dispatch converter accepts an
explicit receipt path; its default still points to the original oracle.
