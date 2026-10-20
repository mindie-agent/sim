# Model reference packages

These cards are implementation references for the nine archived se-tools model configs. They are source-attributed but not measured A3 performance claims. Each card records the exact body schedule, tensor geometry, quantization representation, cache/state ownership, optional draft/media/Engram branches, optimizer assumptions and a reproducible public case.

Source snapshot for provenance only: `maoxx241/se-tools` at `ae8779fdd5a602bd3a4358bca93b2161d134a47b`. Revisions are attribution, never runtime admission gates.

| Model | Body layers | H | Attention | FFN | Optional branch | Vision |
|---|---:|---:|---|---|---|---|
| `deepseek-v4-flash-0731` | 43 | 4096 | swa_only:2, dsa_compressed_4:21, dsa_compressed_128:20 | moe:43 | MTP+DSpark (one native mode) | — |
| `deepseek-v4-pro-0813` | 61 | 7168 | dsa_compressed_128:31, dsa_compressed_4:30 | moe:61 | MTP+DSpark (one native mode) | — |
| `deepseek-v4-10t` | 61 | 7168 | dsa_compressed_128:31, dsa_compressed_4:30 | moe:61 | MTP+DSpark (one native mode) | — |
| `deepseek-v4.1-flash` | 40 | 5120 | swa_only:2, dsa_shared_compressed_2:18, dsa_shared_compressed_1:20 | moe:40 | MTP+DSpark (one native mode) | deepseek_v41 |
| `glm-5.3` | 78 | 6144 | mla_dsa:78 | dense:3, moe:75 | MTP | — |
| `kimi-k3` | 93 | 7168 | kda:69, mla:24 | dense:1, moe:92 | DSpark | kimi_k3 |
| `kimi-k3-dspark` | 5 | 7168 | gqa:5 | dense:5 | standalone DSpark | — |
| `qwen3.8-2.4t-a95b` | 92 | 8192 | gdn:69, gated_gqa:23 | moe:92 | MTP | — |
| `qwen3.8-27b` | 64 | 5120 | gdn:48, gated_gqa:16 | dense:64 | MTP | qwen3_5 |

## Scope boundary

- Implemented: body-layer schedules, projection/attention/FFN shapes, packed weight/scale bytes, quantization exclusions, per-rank cache/state allocation, owner/shared-producer dependencies, optional draft/media/Engram and prepare-eligible work.
- Not claimed: measured end-to-end latency, NPU kernels, HCCL physical link bytes, processor resize/crop semantics beyond supplied grid metadata, and learned routing/candidate values.
- Missing measurements become analytic estimates and measurement gaps; no synthetic row is labeled measured.


## Current reference execution

- `examples/model-reference/generate_reference.py` regenerates all nine
  structure/case/index packages from the archived full model configs.
- All nine full-model public `predict`/`should`/`search` runs and the Qwen media
  reference were executed. Results are stored under
  `examples/model-reference/results/`; all nine predict and search statuses are
  `modeled`, all nine should statuses are `modeled`, and each search attempted
  two finite candidates (unfused and fused). No status was suppressed.
- Numeric TTFT/U values are token-ready analytic estimates, not NPU latency.
  Model-derived preparation is enabled for several cases; stored result rows
  keep preparation-task counts and engine-drain timestamps separate from the
  token-ready figures, so startup/preparation is explicit rather than hidden.
- Cases declare the accepted A3 `Ascend910_9391` per-core/capacity facts
  (24 AIC, 48 AIV, 512KiB L1, 64/64/128KiB L0A/L0B/L0C, 192KiB UB, 192MiB L2)
  so finite fusion capacity evaluation is physical. Compute/HBM peaks are not
  altered. The one-link fabric across 8–128 dies is declared and synthetic for
  >8-die topologies; it is not field-validated multi-node networking.
- `examples/model-reference/pending-measurement-inventory.json` records the
  current pending operator inventory derived from real OpSpec lowering
  (11,386 conditions, 318 selected requests, 4,902 pending conditions, 82
  pending family/recipe groups) and executable campaign commands. It contains
  no measured samples or heldout outcomes.
