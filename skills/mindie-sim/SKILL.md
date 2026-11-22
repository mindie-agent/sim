---
name: mindie-sim
description: Evaluate Ascend A2/A3/A5 LLM inference performance, distinguish strict Simulation vs target-ceiling Modeling, audit vllm-ascend launch scripts, solve Iso-SLA DP/TP/EP/DCP/PP & PD-disaggregated deployments, and export Excel comparison reports and MindStudio Insight traces via mindie-sim.
---

# MindIE-Sim Agent Workflow

Use `mindie-sim` before writing custom AscendC/Triton kernels, editing framework fusion passes, or tuning `vllm-ascend` parallelism so every engineering recommendation is grounded in causal physics, HBM capacity, and Lean-4 certified `L*` bounds.

## Core Distinction: Simulation (`仿真`) vs. Target Modeling (`目标性能建模`)

Never blur implemented-stack simulation with unimplemented target-ceiling projections:

| Mode | Entrypoints | Binding Contract | Use When |
| --- | --- | --- | --- |
| **Simulation (`仿真`)** | `simulate_operator`, `simulate_native_call`, `simulate_aclnn_call`, `simulate_graph`, `simulate_model`, `simulate_service`, `mindie-sim report --mode simulation`, `mindie-sim solve --mode simulation` | **Hard-locked `allow_unbound_formula=False`**. Evaluates ONLY Layer-2 bound implementations (`merged` / `unmerged_pr` / `bind_kernel_implementation`); fails fast with `CapabilityGap("unbound_operator_implementation")` on unbound ops. | Predicting real runtime latency, validating against `msprof`, or running hardware what-if studies (e.g., HBM BW `4.0 -> 1.4 TB/s` via `derive_hardware_sku` / `scale_case_hardware`). |
| **Modeling (`目标性能建模`)** | `model_target_ceiling`, `simulate_formula_kernel` (`FormulaKernelSpec`), `mindie-sim report --mode modeling` *(default)*, `mindie-sim solve --mode modeling` | **`allow_unbound_formula=True`**. Independently solves `Before [当前栈仿真基线]` `(Parallel₀, B₀)` vs `After [卡SLA目标建模终态]` `(Parallel₁*, B₁*)` under `TPOT <= SLA & HBM <= Cap`, stamping every projected op with `[未绑定实现 · 公式×硬件预演]`. | Establishing SLA-constrained Single-Die TPS ceilings (`us / (tok·die)`), sizing fused AscendC/Triton kernels, and producing Sheet 4 development backlogs. |

## Canonical Agent Workflows

### 1. Pass-1 -> Pass-2 Agent Co-Design Loop (`mindie-sim report`)
Run `mindie-sim report` directly—never hand-author raw `Case` JSON files or scrape binary `.xlsx` XML:

```bash
# Pass 1: Zero-config Iso-SLA Modeling report (auto-solves DP/TP/EP/DCP/PP, concurrency & auto_va quant)
mindie-sim report --model kimi-k3.1 --hardware 950dt --num-devices 32 --context 128k --out-dir ./out

# Uncataloged blind-box HuggingFace config.json or local model directory
mindie-sim report --model ./new_hf_model/config.json --hardware a3 --tpot-ms 20 --out-dir ./out

# Strict Simulation Mode (Layer-2 bound implementations only, zero unbound formula ops)
mindie-sim report --model qwen3-32b --hardware a3 --mode simulation --compare-track merged --target-track unmerged_pr --out-dir ./out
```

Then read `<prefix>_agent_brief.json` (`< 8 KB`):
1. Check **`semantic_audit_triggers`** (opaque/cross-boundary subgraphs or dynamic control flow) and **`macro_wall_trigger`** (`HBM_CAPACITY_WALL` vs `LATENCY_COMPUTE_WALL`).
2. Inspect the cited `vllm-ascend` / model source files directly to verify fusion semantics or architectural trade-offs.
3. **Pass 2**: Write a semantic rewrite plan (`plan.json`) and/or per-card kernel design notes (`designs.json`) and re-run:

```bash
mindie-sim report --model kimi-k3.1 --hardware 950dt --num-devices 32 --context 128k \
  --rewrite-plan ./out/plan.json --kernel-designs-json ./out/designs.json --out-dir ./out
```

### 2. Macro Wall Co-Design (`HBM_CAPACITY_WALL` vs `LATENCY_COMPUTE_WALL`)
Act directly on `agent_brief.json["macro_wall_trigger"]`:
- **`HBM_CAPACITY_WALL`** (`B_vram < B_sla`, SLA has slack but HBM is full): prioritize **compute-for-memory (`以算换存`)** — `auto_va` QuaRot quantization (`MXFP8+MXFP4` on A5, `W4A8+C8` on A2/A3), FP8/C8 KV cache, KV pooling/DRAM offload, Engram/PLE host offload, or higher `DP/DCP` before chasing micro-kernel latency.
- **`LATENCY_COMPUTE_WALL`** (`TPOT` hits SLA while HBM headroom remains): prioritize **memory-for-compute (`以存换算`)** — operator fusion (`--rewrite-plan`), `MLAPO`, `MC2/FlashComm1`, `MSO` shared-expert overlap, `ACLGraph`, or lower `TP` / `Attn-DP` to cut collective latency.

### 3. Audit & Auto-Tune a User Launch Script (`--from-script`)
Whenever the user shares a `serve.sh` / `vllm serve` command, run `--from-script` (pure static `shlex` parsing; never executes the script):

```bash
mindie-sim report --from-script ./serve.sh --hardware a3 --out-dir ./out
```

This models the exact `Before` configuration, generates `<prefix>_optimized_serve.sh`, and quantifies immediate 0-code config/quant gains vs. deep kernel/framework gains.

### 4. Reconcile Against Measured `msprof` Captures (`--kernel-csv`)
When real NPU profiling exists, pass `--kernel-csv ./kernel_details.csv` alongside pinned parallelism (`--dp`, `--tp`, `--ep`, `--concurrency`) to populate Sheet 5 (`Measured_msprof_Calibration`).

## Two-Layer Operator SDK (Python)
1. **Layer 1 — Formula-Driven Hardware Projection (Modeling Mode ONLY)**:
   Declare `FormulaKernelSpec` + `FormulaTensorSpec` and call `simulate_formula_kernel(spec, hardware)` to verify UB/L1/L0 residency and `L*` pipe bounds (`binding_state="unbound_formula_projection"`).
2. **Layer 2 — Real Implementation Binding (Simulation Mode)**:
   Promote the kernel via `ext = bind_kernel_implementation(spec, implementation=..., recipe=..., aclnn_api=...)`, load `ctx = load_extensions(extensions=[ext])`, and run `simulate_operator(case, op, extensions=ctx)` / `simulate_graph(...)` under strict Simulation Mode.

## Presenting Conclusions & Dual-Metric Conservation
1. **Always read `<prefix>_agent_brief.json`** (`< 8 KB`) — never scrape binary `.xlsx` XML.
2. **Report BOTH conserved metrics**: Single-Step Exposed Latency (`ms`) AND Amortized Cost (`us / (tok·die) = 10^6 / Single-Die TPS`), ranked by `Δ(us / tok·die)` so variable-`(Parallel, B)` Iso-SLA comparisons remain apples-to-apples.
3. **Present the 3-Track Action Plan**:
   - **Track 1 — Config & Env (0 code changes)**: `HCCL_OP_EXPANSION_MODE` (`AIV` on A2/A3, `CCU` on A5), `VLLM_ASCEND_ENABLE_MLAPO=1`, `VLLM_ASCEND_ENABLE_FUSED_MC2=1`, co-domain parallelism (`DP · TP (共域 DCP) · EP · SP`), and `<prefix>_optimized_serve.sh`.
   - **Track 2 — Framework (`vllm-ascend` / `MindIE`)**: graph capture, MSO overlap, fusion passes, and saved `us/(tok·die)`.
   - **Track 3 — Kernel Co-Design (Sheet 4 `card_id`s)**: shape/dtype contracts, `binding_state`, `before -> target` `L*` attainment, and Pass-2 `--rewrite-plan` / `--kernel-designs-json`.
4. **Link Artifacts**: point to `<prefix>_comparison.xlsx` and `<prefix>_trace_view.json` (MindStudio Insight).

## Production Quantization Policy (`auto_va`, `QuaRot`-First)
- **Atlas A2 / A3 (`64G`)**: MoE -> `w4a8` + `c8` (`int8`) KV (`>= 32k`); Dense -> `w8a8` + `c8` KV (`>= 32k`).
- **Ascend 950DT (`A5`, `96G/144G`)**: MoE -> `mxfp8_mxfp4` + `fp8` KV/Indexer; Dense -> `mxfp8` + `fp8` KV.
- **`QuaRot`-First Rule**: Always recommend `*-QuaRot` Eco-Tech checkpoints first; for uncataloged models recommend `msmodelslim` with `QuaRot` + `--quantization ascend` while keeping `norm`, `router`, `embed_tokens`, and `lm_head` in BF16. Pass `--weight-quant checkpoint` or `--weight-quant bf16` only when explicitly evaluating unquantized baselines.

## Quick Discovery Commands
- `mindie-sim quickstart` — `< 45`-line cheat sheet (`--json` for structured output).
- `mindie-sim catalog` / `mindie-sim catalog --hardware` / `--presets` / `--quant` — list models, SKU aliases (`950dt`, `a5`, `a3`, `910c`, `a2`, `910b`), presets, and quant modes.
