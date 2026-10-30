# Explicit GLM attention and expert-down alternatives

The retained replay manifest has 26 source populations and 22 distinct model
inputs. `data/verification/glm-compatible-variants-20261030.json` maps each input
to its unchanged baseline and explicit alternatives, including hashes, source
identity and the existing profiler geometry. All 26 input hashes match the
manifest. This is a local mechanism experiment, not a new capture or a new
accuracy result.

| Input domain | Populations | Alternative |
|---|---:|---|
| A3 GLM78, BF16 latent512/rope64 | 5 | Existing pipeline unchanged |
| A3 GLM-Flash, INT8 latent512/rope0 | 8 | Explicit packed528 analytic attention |
| A5 GLM-Flash, BF16 latent512/rope0 | 10 | Explicit rope0 analytic attention; separate attention-plus-MXFP8-down arm |
| A3 GLM78, INT8 latent512/rope64 | 3 | Unchanged; outside packed528 domain |

There are 28 new cases under
`examples/cases/high-concurrency-20261030/glm-compatible-variants/`. Names end in
`--attention-only.json` or `--attention-plus-down.json`. The original cases are
unchanged. The two A5 arms let a replay isolate attention from the additional
source-down change. Both leave hardware, precision, routing, topology and the
original `programs.matmul` declaration unchanged. An originally absent matrix
program uses the current body policy's fixed 128×128 output grid; an explicitly
declared steady program retains its own mapping policy.

The BF16 extension is described in [indexed-rope0-analytic.md](indexed-rope0-analytic.md).
It shares the existing query-owner, finite-slot attention pipeline; it is not a
native rope-free kernel binding. The INT8 alternative is the existing explicit
`indexed_attention_packed_int8` program: a 528-byte row holds 512 INT8 elements
and four FP32 scales. It must not be described as the retained native arch22
672-byte padded layout. Prior c32/c64 comparisons remain approximately +56%
and +59%; this work does not change those residuals or declare them accepted.

Only `balanced_grouped_down` uses `ascend_mxfp8_grouped_nz`. Its ordinary
frontend operands are FP8 E4M3 with block32 E8M0 scales, integer expert rows and
BF16 direct Fixpipe output. The fused gate/up boundary is left on its existing
analytic decomposition. The original KDA programs also remain: nonempty
accepted-token state selection/checkpoint ABI is not covered by the source
adapter. These structural limitations are in each variant's model metadata
and emerge as `model_coverage=partial`. Unverified captured binary/dispatch
identity and geometry-extrapolation provenance live separately in
`runtime.metadata.compatible_variant.evidence_assumptions`; they do not become
model structural gaps. Baseline unknown config fields remain unchanged.

The producer's `--verify` run completed all 28 variant and 18 baseline witnesses
in under eight seconds. It preserved the original first four layer geometries and
executed only layer3 attention and expert-down operators, using the public body
declaration carrier. Every witness released tasks and buffers and observed the
7 ms input dependency. This is **isolated operator work**, not a four-layer or
complete-model Forward prediction. For example, A5 c32/q4 attention is
295.174 us in the rope0 arm; source expert down is 106.775 us versus the
baseline's 87.876 us. These are predicted program differences, not proof of
captured dispatch or accuracy. Full replay should use the original 45-layer
variant and the receipt's q/B/context, not client concurrency as a batch guess.

`tests/test_glm_compatible_variants.py` additionally executes complete four-layer
graphs for A3 INT8 and both A5 BF16 alternatives, checking ordinary frontend
selection, unchanged matrix and KDA paths, source-down output bytes, dependency
completion and partial coverage. The immutable input receipt retains the full
source model; shortening to four layers is only a bounded test witness.

Rebuild the cases and bounded witnesses with:

```sh
python data/verification/glm-compatible-variants-20261030.producer.py \
  --manifest /path/to/profiler/decode-forward-glm-cases.json --verify
```
