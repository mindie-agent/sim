> **Historical Archive (Non-Gate Snapshot)** — Preserved from `sim` baseline commit `6800d9d7`.
> Original path: `docs/issue34-features.md` · Point-in-time wave/audit/review record excluded from active gates.
> Retained for historical modeling provenance; consult active `docs/` guides for current interfaces.
> Current living specification: [`docs/architecture.md`](../architecture.md).

# Issue #34: canonical features, units and boundaries

This note documents the v0.9.14 additions behind
[mindie-agent/sim#34](https://github.com/mindie-agent/sim/issues/34). All
features are opt-in except the correctness fixes for runtime-aware speculative
state capacity, draft-cache geometry and eligible generic MXA8W4 execution
precision. Disabled configurations keep their previous byte and timing
identities.

## Canonical inputs

| Owner | Field | Default | Meaning |
| --- | --- | --- | --- |
| `ParallelSpec` | `o_proj_tp` | `1` | Node-local TP width for attention/MLA output projections |
| `ParallelSpec` | `linear_proj_tp` | `1` | Local TP width for linear-attention input/gate/output matrices |
| input alias | `kda_proj_tp` | - | Alias of `linear_proj_tp`; only the canonical name is serialized |
| `ParallelSpec` | `lm_head_tp` | `1` | Local TP width for logits and independent draft heads |
| `ParallelSpec` | `redundant_experts_per_rank` | `0` | Extra resident routed-expert slots per EP owner |
| `Software` | `collective_startup_mode` | `legacy` | `legacy`, `ring`, `log2`, `concurrent` (`direct` normalizes) or `topology` for AllToAll/AllToAllV |
| `Software` | `matrix_compute_policy` | `auto` | Resolve eligible generic packed FP4/FP8 work on native FP8 hardware; `declared` pins it |
| `Software` | `recover_ssm` | `false` | Compact trajectory records plus one acceptance-bound commit |
| `Software` | `prefix_cache_mode` | `private` | `shared` asserts equal keys denote identical immutable prefixes |
| `Software` | `prefix_read_mode` | `auto` | `per_sequence` or `batch_shared` selected kernel reuse |
| `Software` | `homogeneous_graph_reuse` | `off` | `auto`/`require` template reuse after a structural proof |
| `RequestSpec`/`RequestView` | `prefix_cache_key` | `""` | Physical prefix-pool identity; request IDs are not prefix identities |
| `Hardware` | `collective_protocols` | `{}` | Declared `alltoall`/`alltoallv` startup protocol |
| `Device` | `compute_peak_aliases` | `{}` | Report-only supply aliases such as `mxfp4 -> fp8_e4m3` |

Validation rejects booleans as integers, non-integral floats, non-finite values,
negative widths/counts and unknown modes. Equal duplicate declarations are
accepted; unequal ones raise `runtime_feature_conflict:<field>`. The narrow
top-level `software: {recover_ssm: ...}` input alias merges into `runtime` and
rejects every other key.

## Physical assumptions

- Local TP requires global `tp=1` and no CP/DCP subdivision. Groups are confined
  to one node, PP stage and PCP lane and partition the eligible ranks. Column
  projections gather activation rows and exchange output slices; row projections
  exchange input-feature shards and reduce-scatter partial rows. Per-rank matrix
  FLOPs remain `2*M*N*K`. Weights and operators use the same padded local shape.
- `R` extra expert slots add `R*W_expert_local` resident bytes per EP owner and
  distinct active weight reads. Logical expert/token work is unchanged.
- ReplaySSM keeps `S` recurrent slots plus `q*R` compact records for a verify
  window `q`; without replay it keeps `q*S` checkpoints. Convolution history uses
  a compact rollback buffer. Commit tokens are integers from scheduling or the
  declared body input, never a rounded acceptance mean.
- A shared prefix is reserved once per DP replica. Dense blocks share only
  complete immutable prefix blocks; a partial block receiving private tokens is
  a private COW tail. SWA keeps only live window intervals and compressed caches
  share only complete compression groups.
- AllToAll startup rounds are `G-1` (legacy/ring), `ceil(log2(G))` (log2) or one
  (concurrent); explicit `steps` is an integer in `[1, G-1]`. Peer messages,
  framing and wire bytes are independent of the startup model.
- Generic packed FP4 weights with an FP8 activation execute on native E4M3/E5M2
  only when every executing die declares the native mode and a finite positive
  supply. Weight, activation and output representations are unchanged. An
  `mxfp4` peak alias is reporting only and is never admitted as a native operand.
- Hardware derivation scales AIC-owned peaks, owner counts and supplies by
  `s_c`; vector/AIV-owned supplies and UB units by `s_v`. Per-core buffers, HBM
  bandwidth and host/network/fixed control do not auto-scale. HBM capacity is an
  explicit override.
- `exposed_breakdown_ms` is ordered max-plus critical-span elimination:
  computation first, memory second, communication third. Percentages conserve
  `100` within `1e-9`; a positive step boundary is required. An operator-only
  report returns `available=false` instead of inventing a step percentage.

## CLI examples

```sh
mindie-sim compose --model glm-5.3-flash --hardware Ascend950DT_9582 \
  --tp 1 --dp 8 --ep 8 --num-devices 8 --ranks-per-node 8 \
  --o-proj-tp 8 --linear-proj-tp 8 --lm-head-tp 8 \
  --recover-ssm --mtp-tokens 5 --mtp-accept 3.5 \
  --prompt-tokens 131072 --prefix-tokens 124518 \
  --prefix-cache-mode shared --prefix-read-mode auto \
  --redundant-experts-per-rank 1 --collective-startup-mode log2 \
  --matrix-compute-policy auto --homogeneous-graph-reuse auto \
  --output case.json

mindie-sim compose --model glm-5.3-flash --hardware Ascend950DT_9582 \
  --aic-count 16 --memory-gib 64 --hbm-peak-bytes-per-s 2e12 \
  --hbm-efficiency '{"axis":"aic_count","knots":[[8,0.5],[32,1.0]]}'
```

Scalar hardware flags and a `--hardware-derivation <JSON>` record must agree
when both specify the same field. `--memory-gib` is exactly `2**30` bytes.

## Verification gates

```sh
python3 -B scripts/check_no_magic_numbers.py --base <base> --root .
python3 .github/scripts/check_lf_eol.py
```

The numeric gate is differential: pre-existing literals in untouched statements
are grandfathered, while a new literal is reported with file/line/context.
