# Compact matrix default for body probes

When `scope="body"` has no explicit `runtime.kernels.programs.matmul`, the shared effective body declaration installs `steady_matmul` with candidates `[[128,128,128],[128,128,64]]`. The M/N output-owner grid remains128×128. K128 is preferred when feasible; K64 supports the same grid when double-buffered operands exceed a declared L0 capacity. This closes a public-path gap: ordinary high-batch body probes previously selected `TiledMatmul`, expanding every K panel even though a compact matrix mechanism was already available.

The selected program is the existing finite tile/mapping implementation. It retains physical work, owner placement, resident buffers and the nominal K-loop overlap; contended execution uses its documented coupled-fluid approximation. This change does not fit a timing coefficient or add a new matrix model. The hierarchical program adds explicit MTE1/L0 mechanisms that the older flat tiled program lacked, so exact equality of predicted time is not promised.

The first body-default revision used unrestricted `SteadyMatmul` candidates. That unintentionally enabled a more optimistic output-owner mapping as part of a CPU execution optimization. The corrected default separates these decisions. When no grouped family is declared, a separate `grouped_gemm: steady_matmul` family retains the already-existing grouped candidate domain; narrowing ordinary matrices must not narrow independent expert banks. Explicit ordinary programs leave their existing grouped inheritance unchanged.

`predict(scope="body")`, `target_graphs`, `target_operators`, their graph/operator costing paths, structure export and cohort replay share the same declaration carrier. The selection is recorded under `default_mechanisms.selections` as `family="matmul"`, with the chosen program and priority. Rebuild historical body structure/cohort mappings against this declaration rather than silently reusing an older graph identity.

Priority remains named/source recipe → family recipe → program default. Explicit tiled/mapped programs, explicit family choices and source kernels retain their execution. An explicitly supplied `RecipeBook` is preserved. `scope="model"`, serving prediction and `apply_body_defaults=False` retain their prior behavior. To request the old detailed matrix control while keeping body attention defaults, declare `programs.matmul={"kind":"tiled_matmul"}`.

Two retained cases with absent raw kernel declarations completed full local body execution at **q4, B96, context4104**, each within a separate90-second budget:

| Retained case | Full model wall time | Predicted body | Engine tasks | Live tasks/programs |
|---|---:|---:|---:|---:|
| Qwen3.6 A5 TP2 c96 | 28.318 s | 45.409890 ms | 1,952,611 | 0 / 0 |
| GLM-Flash A5 TP8 c96 | 6.290 s | 55.026659 ms | 216,061 | 0 / 0 |

Qwen executed304 ordinary steady matrix calls; GLM executed446 plus84 integer grouped calls. All device allocations were released. Qwen still has substantial recurrent-program task count; this change fixes the missing matrix default without claiming every remaining program is compact. These are declared-geometry execution checks, not capture matching or new body-accuracy results.

Current evidence: `data/verification/body-matrix-fixed-grid-20261030.json`, produced by the adjacent original body-matrix producer. Package hashes match before/after both executions. The34 focused checks exercise actual prediction, graph, operator, structure and cohort paths, explicit priority, source grouped selection, FP32 capacity, unchanged grouped mappings, body opt-out and unchanged serving execution. `body-matrix-default-20261030.json` retains the superseded unrestricted-candidate run; its times are not the current default.

## Why the mapping policy changed

The retained full-size GLM-5.3 A3 layer3 control is in `data/verification/glm-body-matrix-default-diagnosis-20261030.json`. The model still has78 layers; only one complete MoE layer is executed with its true dimensions at q1/B1/context4096. All arms freeze the old grouped-family recipes and change only the ordinary matrix program.

| Ordinary matrix arm | Router `[1,256,6144]` | Router owners | Complete layer |
|---|---:|---:|---:|
| Original Tiled128 | 60.056 µs | 2 | 420.279 µs |
| Mapped current domain | 7.956 µs | 16 | 368.570 µs |
| Steady current domain | 7.956 µs | 16 | 368.761 µs |
| Steady fixed128 / K64 / two operand slots | 60.207 µs | 2 | 420.734 µs |
| Steady fixed128 / K128 / one operand slot | 60.368 µs | 2 | 421.173 µs |
| Steady current / individual output tiles | 7.956 µs | 16 | 368.782 µs |

The unrestricted selector chose16×16 output tiles for this FP32 router. HBM remains6,317,056 bytes; private operand reads rise from6,340,608 to6,684,672 bytes because more owners repeat the input row, while16 owners can process them instead of2. This accounts for the timing decrease. K compression changes the whole layer by only+0.190 µs relative to the same mapped domain; output compaction changes it by−0.021 µs. Grouped gate/up and down remain15.196/8.064 µs in every arm.

Fixed128×128×128 with two FP32 operand slots is correctly rejected: L0B would require131,072 bytes but the A3 profile declares65,536. The old flat tiled program did not represent this L0 capacity. K64 keeps the original output-owner grid while fitting the declared storage. This policy is derived from the old default and capacity contract, not from selecting the arm closest to an observed time.
