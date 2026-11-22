---
name: mindie-sim
description: Evaluate Ascend A2/A3/A5 LLM inference performance, audit vllm-ascend launch scripts, solve DP/TP/EP/DCP/PP & PD-disaggregated deployments, and export Excel comparison reports and MindStudio Insight traces via mindie-sim.
---

# MindIE-Sim Agent Workflow

Use `mindie-sim` before writing custom AscendC/Triton kernels, editing framework fusion passes, or tuning `vllm-ascend` parallelism so every engineering recommendation is grounded in causal physics, HBM capacity, and Lean-4 certified `L*` bounds.

## Canonical Workflows

### 1. Zero-Config Model & Cluster Evaluation (`mindie-sim report`)
Run `mindie-sim report` directly—never hand-author raw `Case` JSON files for routine evaluation:

```bash
# Built-in catalog model (auto-solves HBM-safe DP/TP/EP/DCP/PP, concurrency & auto_va quant)
mindie-sim report --model kimi-k3.1 --hardware 950dt --num-devices 32 --context 128k --out-dir ./out

# Uncataloged blind-box HuggingFace config.json or local model directory
mindie-sim report --model ./new_hf_model/config.json --hardware a3 --tpot-ms 20 --out-dir ./out
```

### 2. Audit & Auto-Tune a User Launch Script (`--from-script`)
Whenever the user shares a `serve.sh` / `vllm serve` command, run `--from-script` immediately (static pure-`shlex` parsing; never executes the script):

```bash
mindie-sim report --from-script ./serve.sh --hardware a3 --out-dir ./out
```

This models the user's exact `Before` configuration, generates `<prefix>_optimized_serve.sh`, and quantifies immediate script/quant gains vs. deep kernel/framework gains.

### 3. Reconcile Against Measured `msprof` Captures (`--kernel-csv`)
When real NPU profiling exists, pass `--kernel-csv ./kernel_details.csv` alongside pinned parallelism (`--dp`, `--tp`, `--ep`, `--concurrency`) to populate Sheet 5 (`Measured_msprof_Calibration`).

## Reading Outputs & Presenting Conclusions (Never Scrape `.xlsx` XML)
1. **Read `<prefix>_agent_brief.json`** (`< 8 KB`) — never write `zipfile`/XML parsers to scrape `.xlsx`.
2. **Present the 3-Track Action Plan**:
   - **Track 1 — Config & Env (0 code changes)**: exact `HCCL_OP_EXPANSION_MODE` (`AIV` on A2/A3, `CCU` on A5), `VLLM_ASCEND_ENABLE_MLAPO=1`, `VLLM_ASCEND_ENABLE_FUSED_MC2=1`, parallelism, and `vllm serve` flags (plus `<prefix>_optimized_serve.sh` if `--from-script` was used).
   - **Track 2 — Framework (`vllm-ascend` / `MindIE`)**: graph capture, MSO overlap, fusion passes, and saved ms.
   - **Track 3 — Kernel Co-Design (Sheet 4 `card_id`s)**: shape/dtype contracts, `before -> target` `L*` attainment, and optional **Pass-2** re-export via `--kernel-designs-json ./out/agent_designs.json` after inspecting target kernel source code.
3. **Link Artifacts**: point the user to `<prefix>_comparison.xlsx` and `<prefix>_trace_view.json` (opens directly in MindStudio Insight).

## Production Quantization Policy (`auto_va`, `QuaRot`-First)
- `mindie-sim report` defaults to chip-matched `vllm-ascend` production quantization (`auto_va`), because Ascend NPUs rarely serve native BF16 in production:
  - **Atlas A2 / A3 (`64G`)**: MoE -> `w4a8` + `c8` (`int8`) KV (`>= 32k`); Dense -> `w8a8` + `c8` KV (`>= 32k`).
  - **Ascend 950DT (`A5`, `96G/144G`)**: MoE -> `mxfp8_mxfp4` + `fp8` KV/Indexer; Dense -> `mxfp8` + `fp8` KV.
- **`QuaRot`-First Rule**: Always recommend `*-QuaRot` Eco-Tech checkpoints first when available (Hadamard outlier rotation yields higher accuracy than plain W8A8/W4A8). For uncataloged models, recommend `msmodelslim` with `QuaRot` + `--quantization ascend` while keeping `norm`, `router`, `embed_tokens`, and `lm_head` in BF16.
- Pass `--weight-quant checkpoint` or `--weight-quant bf16` only when explicitly evaluating unquantized baselines.

## Quick Discovery Commands
- `mindie-sim quickstart` — `< 45`-line cheat sheet (`--json` for structured output).
- `mindie-sim catalog` / `mindie-sim catalog --hardware` / `--presets` / `--quant` — list models, SKU aliases (`950dt`, `a5`, `a3`, `910c`, `a2`, `910b`), presets, and quant modes.
