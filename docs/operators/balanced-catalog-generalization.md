# Balanced routing across catalog models

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The 2026-10-30 integration is covered by
[`test_balanced_catalog_generalization.py`](../../tests/test_balanced_catalog_generalization.py)
and the bounded full-model report.
The tests use real catalog `LayerPlan` dimensions and the public parser, ordinary
FFN lowerings and shared Engine. They do not replace catalog layers with mocked
objects or derive timings from observations.

## Routine tests

The nine catalog inputs include Qwen3.5/3.6/3.8, GLM-5.3, V4 Flash/Pro/4.1,
Kimi K3 and MiniMax-M3. Their 408 routed layers reduce to nine identical-FFN
structure classes, retaining every original layer's class membership. The
sweep covers every integer batch from 1 through 128 and q=1,4,5,8: 4,608
class/scenario combinations. It verifies top-k conservation, integer q/r row
counts, empty expert banks, maximum rank-zero load and bounded per-expert rows.
The two dense siblings execute dense FFN work without manufacturing experts.

| Catalog | Routed layers | Experts / top-k | FFN classes |
|---|---:|---:|---:|
| Qwen3.5-35B-A3B | 40 | 256 / 8 | 1 |
| Qwen3.6-27B | 0 | dense | 0 |
| Qwen3.8-27B | 0 | dense | 0 |
| GLM-5.3 | 75 | 256 / 8 | 1 |
| V4 Flash | 43 | 256 / 6 | 2 |
| V4 Pro | 61 | 384 / 6 | 2 |
| V4.1 | 40 | 384 / 6 | 1 |
| Kimi K3 | 92 | 896 / 16 | 1 |
| MiniMax-M3 | 57 | 64 / 8 | 1 |

Sparse, remainder and populated FFN operators retain real latent widths,
ETP partitioning, active weight banks, quantization producers and row scales.
Eighteen actual FFN component graphs cover all nine inputs on both A3 and A5,
using `programs.matmul=steady_matmul` without balanced-specific recipe overrides.
All 34 tests pass in 1.49s; full-model runs are deliberately kept out of these
routine unit controls.

The retained catalog currently declares no restricted group routing: GLM uses
one selected group of one and the other inputs declare zero groups. Therefore
this catalog sweep does not establish a captured group-limited router branch.
The separate integer-row tests exercise explicit restricted groups. Hash versus
score routing and the other real FFN fields remain unchanged in this sweep.

## Full-model API controls

The producer
uses full catalog layer counts, TP8/EP8 and explicit W8A8 on both devices. Body
geometry is one sequence, q=5 and context2048. It excludes admission and starts
with input/KV ready. Each query row executes; no speculative acceptance discount
is applied. These configurations test portability, not checkpoint identity.

| Model | Full layers | A3 body ms | A5 body ms |
|---|---:|---:|---:|
| V4 Flash | 43 | 23.684 | 8.077 |
| V4 Pro | 61 | 53.715 | 17.964 |
| V4.1 | 40 | 21.218 | 7.416 |
| MiniMax-M3 | 60 | 22.774 | 8.617 |

Two additional A5 ordinary `predict()` runs retain memory admission, request
scheduling and both prompt/decode iterations. V4 Flash and MiniMax each serve
one prompt1/output2 request completely; their makespans are 13.908ms and
12.785ms. These serving numbers are different boundaries and geometries from
the body table and must not be subtracted from its spans.

All ten runs complete with zero live tasks/programs. Total campaign wall time
is 33.296s, longest worker boundary 5.051s, peak RSS 101.078MiB. The per-worker
45s and campaign450s budgets are retained. Runtime package SHA256 is
`a63569b206ab9eed46599c47dd21080cc01b6eb6add6bf2740c1a250e96b8210`
at test/evidence revision `96bded01`; package identity stays unchanged across
every run. Runtime mechanisms are the stable `089bbcf5` integration.

All four full-model controls still report partial model coverage. V4 attention
geometry stays outside the reviewed default indexed-MLA source domain, so its
existing declared path is preserved and identified in `default_mechanisms`.
MiniMax receives the supported GQA decode pipeline while retaining its broader
partial architecture declaration. The report contains those limitations and
unmodeled features rather than equating successful execution with complete
modeling or <=15% measured accuracy.
