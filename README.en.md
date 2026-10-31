<p align="center">
  <img src="https://raw.githubusercontent.com/mindie-agent/mindie-agent/main/assets/brand/mindie-agent-logo.png" alt="MindIE Agent logo" width="128" height="128">
</p>

<h1 align="center">MindIE Agent Simulator (<code>mindie-sim</code>)</h1>

<p align="center">
  <strong>Give your AI coding agent a physics-grounded Ascend NPU performance compass — a unified causal engine across operators, subgraphs, full models, and distributed serving with 4D efficiency attribution, counterfactual Roofline target fitting, and SLA hardware scaling throughput solving</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Release-v0.8.1-blue" alt="Release v0.8.1">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/Interface-CLI%20%7C%20Python%20API%20%7C%20PrimitiveStore-00ADD8" alt="CLI | Python API | PrimitiveStore">
  <img src="https://img.shields.io/badge/Target-Ascend%20NPU%20(A2%20910B%20%7C%20A3%20910C%20%7C%20A5%20950)-FF6B00" alt="Ascend NPU A2/A3/A5">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="./README.md">中文</a> · <strong>English</strong> · <a href="#why-sim">Why sim?</a> · <a href="#quick-start">Quick Start</a> · <a href="#v081-highlights--full-fleet-convergence">v0.8.1 Highlights</a> · <a href="#cli-commands--python-api-reference">CLI & Python API</a> · <a href="#real-world-usage-examples">Examples</a> · <a href="#documentation">Documentation</a> · <a href="https://github.com/mindie-agent/mindie-agent">MindIE Agent Main Repo</a>
</p>

---

## Why `sim`?

When optimizing custom kernels or distributed LLM inference (`DP × TP × EP × PP × PCP` and disaggregated Prefill-Decode serving for DeepSeek, Qwen, GLM, or Kimi) on Ascend NPUs, the biggest waste of engineering time is **blind trial-and-error without a causal performance baseline**:

- **"Which hardware resource actually bottlenecks this operator or model?"** — Back-of-the-envelope `FLOPs / peak_tflops` estimates ignore on-chip `UB`/`L1` alignment, tile tails, multi-stream contention, and kernel invocation overheads (`startup_ms` / `minimum_ms`). `mindie-sim` decomposes every operator, subgraph, and network pass into **`Cube_MFU`, `Vec_MFU`, `MBU`, `Link_Util`, and `CV_Overlap`**, pinpointing whether work is `memory_bound`, `cube_bound`, `vector_bound`, `link_bound`, or `launch_bound`.
- **"If we fuse or tune these memory-bound kernels to reach `0.80x MBU`, how many milliseconds will end-to-end Decode drop?"** — Before writing hundreds of lines of AscendC or Triton code, counterfactual Roofline target fitting (`fit_efficiency_targets`) computes the exact latency reduction while preserving physical work conservation and kernel startup floors.
- **"Under a `<= 30 ms` Decode TPOT SLA, what is the maximum batch size and throughput — and what if we scale HBM bandwidth or compute by `2x`?"** — The hardware scaling & SLA solver (`solve_hardware_sla_throughput`) jointly solves the VRAM capacity ceiling (`B*_VRAM`) and SLA latency ceiling (`B*_SLA`) across hardware scaling scenarios (`2x HBM`, `2x Compute`, `1x HW + 0.80x MBU`).
- **"How do we evaluate isolated operators, subgraphs, callbacks, and full serving with consistent physics?"** — `mindie-sim` uses a **single unified causal engine (`prediction.engine.Engine`)** across isolated operator programs (`operators`), shared-resource target graphs (`graph`), callback protocols (`call`), candidate comparisons (`study`), and full-service scheduling (`predict` / `predict_pd`).

```mermaid
flowchart TB
    Input["📥 Input Specs & Empirical Evidence<br/>(Case JSON / Model Catalog / Hardware Specs / Profiler Structure & Observations)"]
    Frontend["🧱 Structural Lowering & Work Conservation (schema / catalog / frontend)<br/>(Attention / MLA / GDN / KDA / MoE Routing / Quantization & KV Cache Capacity)"]
    Program["⚙️ Finite Resource Programs & Pipelines (program / matrix / invocations)<br/>(M/N/K Tiling & Tails · Cube/Vector/MTE/HBM/Link Service · Startup & Completion Floors)"]
    Engine["🔗 Unified Causal Event Engine (prediction.engine.Engine)<br/>(operators Isolated Costs · graph Shared-Resource DAG · call Protocols · predict Serving)"]
    Analysis["🧭 4D Efficiency, Counterfactual Roofline & SLA Scaling Solver (efficiency / study)<br/>(Cube_MFU · Vec_MFU · MBU · Link_Util · CV_Overlap · 0.80x MBU Fit · B*_SLA Throughput)"]

    Input --> Frontend --> Program --> Engine --> Analysis
```

---

## Quick Start

### Option 1: Send One Sentence to Your AI Agent (Recommended)

Copy and paste the line below to Codex, Claude Code, Kimi Code, or your internal agent so it installs and verifies `mindie-sim` automatically:

```text
Please install and configure https://github.com/mindie-agent/sim for me
```

<details>
<summary>🤖 <b>AI Agent Auto-Install & Workflow Protocol (Instructions for AI Agents)</b></summary>

If you are an AI Agent receiving the prompt above, execute the following steps to install and verify `mindie-sim` (`v0.8.1`):

1. **Check Python Version**: Ensure Python `>= 3.11` is available (`mindie-sim` has zero `torch` or `torch_npu` runtime import dependencies and runs cleanly on macOS, Linux, and Windows).
2. **Install the CLI and Package**:
   - Install the pinned release wheel (recommended):
     ```bash
     uv pip install https://github.com/mindie-agent/sim/releases/download/v0.8.1/mindie_sim-0.8.1-py3-none-any.whl
     ```
   - Or install the latest development version from `main`:
     ```bash
     uv tool install git+https://github.com/mindie-agent/sim@main
     ```
3. **Verify CLI & Model Catalog**:
   ```bash
   mindie-sim catalog
   ```
4. **Agent Optimization Workflow**:
   - Before writing or modifying custom kernels, fusion patterns, or parallel layouts (`TP`/`DP`/`EP`/`PP`/`PCP`), run `mindie-sim operators`, `mindie-sim graph`, `mindie-sim predict`, or `mindie-sim study` to quantify stage latencies and resource bottlenecks.
   - To evaluate 4D efficiency (`Cube_MFU`, `Vec_MFU`, `MBU`, `Link_Util`, `CV_Overlap`), counterfactual `0.80x MBU` optimization headroom, or SLA-constrained maximum throughput, call `analyze_network_efficiency`, `fit_efficiency_targets`, and `solve_hardware_sla_throughput` from `mindie_sim.prediction.efficiency`.

</details>

### Option 2: Manual One-Line Install in Terminal (For Developers)

With Python `>= 3.11` and `uv` (or `pip`), install in one command:

```bash
# Install the v0.8.1 release wheel
uv pip install https://github.com/mindie-agent/sim/releases/download/v0.8.1/mindie_sim-0.8.1-py3-none-any.whl

# Or install as a global CLI tool from GitHub
uv tool install git+https://github.com/mindie-agent/sim@main
```

Clone the repository to run local examples and tests:

```bash
git clone https://github.com/mindie-agent/sim.git
cd sim
uv pip install -e ".[dev]"

# Run the built-in quickstart prediction
mindie-sim predict examples/quickstart.json

# Run the offline unit test suite
pytest -q
```

---

## `v0.8.1` Highlights & Full-Fleet Convergence

`mindie-sim v0.8.1` extends the unified causal engine with **4D efficiency attribution, counterfactual multi-pipe Roofline target fitting, and hardware scaling SLA throughput solving**, backed by full-fleet convergence across Ascend A3 and A5 long-context workloads (see [`docs/releases/sim-0.8.1.md`](./docs/releases/sim-0.8.1.md)):

### 1. 4D Efficiency Decomposition & Bottleneck Attribution (`analyze_operator_row` / `analyze_network_efficiency`)

Conserves physical FLOPs, vector OPs, HBM bytes, and interconnect bytes across individual operators, layer groups, structural subgraphs, and whole-network passes:

| Metric | Physical Definition | Diagnostic Value |
| --- | --- | --- |
| **`Cube_MFU`** | Matrix Cube utilization ($\text{Cube FLOPs} / (\text{Peak Cube FLOPS} \times T)$) | Measures AIC utilization on MatMul, GroupedMatMul, and MLA projections |
| **`Vec_MFU`** | Vector pipe utilization ($\text{Vector OPs} / (\text{Peak Vector OPS} \times T)$) | Identifies bottlenecks in Softmax, RoPE, RMSNorm, GDN/KDA recurrence, and quantization |
| **`MBU`** | HBM bandwidth utilization ($\text{HBM Bytes} / (\text{Peak HBM BW} \times T)$) | Pinpoints memory-bound Decode operators and redundant GM round-trips |
| **`Link_Util`** | Intra-/inter-package link utilization ($\text{Link Bytes} / (\text{Peak Link BW} \times T)$) | Quantifies collective communication overhead (AllReduce, ReduceScatter, AllGather, AllToAll) |
| **`CV_Overlap`** | Cube-Vector pipeline overlap ratio | Measures internal compute pipe overlap inside fused (`cv_fused`) kernels |

Every operator is classified by execution kind (`cube` / `vector` / `cv_fused` / `communication` / `memory_only`) and dominant bottleneck (`memory_bound` / `cube_bound` / `vector_bound` / `link_bound` / `launch_bound`).

### 2. Counterfactual Multi-Pipe Roofline Target Fitting (`evaluate_target_efficiency` / `fit_efficiency_targets`)

Want to know how much faster Decode becomes if memory-bound kernels are optimized to `0.80x MBU`? `fit_efficiency_targets` evaluates counterfactual efficiency caps ($t_{\text{body}} = \max_k W_k / (\eta_k \cdot P_k)$) while preserving kernel `startup_ms` and `minimum_ms`:
- Reproduces baseline simulator latency with **0.00% drift** at `1.0x` scaling;
- Computes per-subgraph and whole-network `baseline_ms`, `target_ms`, `saved_ms`, and `speedup_ratio` under configurable targets (e.g., `target_mbu=0.80`, `target_cube_mfu=0.75`, `target_vec_mfu=0.65`).

### 3. Hardware Scaling & SLA Throughput Solver (`scale_case_hardware` / `solve_hardware_sla_throughput`)

Supports non-mutating counterfactual hardware scaling (`hbm_bw_scale`, `cube_flops_scale`, `vector_ops_scale`, `link_bw_scale`, `hbm_capacity_scale`) and jointly solves maximum throughput under a target Decode latency SLA (e.g., `<= 30.0 ms`):
- **VRAM Capacity Ceiling (`B*_VRAM`)**: Maximum sequences per replica after accounting for model weights, runtime workspace, draft caches, and per-token KV cache growth;
- **SLA Latency Ceiling (`B*_SLA`)**: Monotonic binary search over batch sizes to find the largest concurrency meeting `latency_ms <= sla_ms`;
- **Feasible Optimum (`B* = min(B*_SLA, B*_VRAM)`)**: Evaluates `DEFAULT_SLA_SCENARIOS` (`baseline`, `2x_hbm`, `2x_compute`, `2x_cube_only`, `2x_hbm_2x_compute`, `target_0.8x_mbu`) and reports peak `tokens/s` and the active limiting factor (`sla_latency` / `vram_capacity` / `sla_unachievable_at_b1`).

### 4. Full-Fleet A3 / A5 Convergence Benchmarks

Validated jointly with [`mindie-profiler v0.5.3`](https://github.com/mindie-agent/profiler) across real hardware captures:

| Benchmark Cohort | Cases | Within `<= 15%` Error | Mean Relative Error | Coverage Highlights |
| --- | --- | --- | --- | --- |
| **Registered Decode Suite** | **280 cases** | **280 / 280 (100.0%)** | **2.15%** | A3 / A5, `TP`/`DP`/`EP`, 64K–128K long context, KDA/GDN, MXFP4 MoE, MLA/GQA |
| **Registered Prefill Suite** | **15 cases** | **15 / 15 (100.0%)** | **2.21%** | Max error `8.42%`, covering chunked prefill and long-context prefill |
| **Unregistered DSH Heldout Suite** | **561 cases** | **546 / 561 (97.33%)** | **3.66%** | Zero per-case tuning, generalized via `mindie-model-structure/1` |
| **Full Comparable Fleet Total** | **856 cases** | **841 / 856 (98.25%)** | **3.14%** | Bounded working set `< 50 MiB` peak RSS |

---

## CLI Commands & Python API Reference

### CLI Subcommands

| Subcommand | Purpose | Example |
| --- | --- | --- |
| `mindie-sim predict` | End-to-end serving or body-forward (`--scope serving\|body`) performance prediction | `mindie-sim predict examples/quickstart.json` |
| `mindie-sim inspect` | Parse a Case and emit normalized model structure, KV memory plan, and lowered operators | `mindie-sim inspect examples/quickstart.json` |
| `mindie-sim operators` | Evaluate isolated device resource costs for each operator in a Prefill/Decode pass | `mindie-sim operators examples/quickstart.json --phase decode --tokens 1` |
| `mindie-sim graph` | Compose and evaluate a target graph on shared device resources and streams | `mindie-sim graph examples/quickstart.json --phase decode --tokens 1 --sequences 8` |
| `mindie-sim call` | Evaluate a complete declared callback and repetition protocol | `mindie-sim call examples/quickstart.json examples/calls/rmsnorm.json` |
| `mindie-sim study` | Compare multiple candidate configurations across operator or graph probes | `mindie-sim study examples/studies/parallel_study.json` |
| `mindie-sim catalog` | List built-in standard model specifications or inspect a named model | `mindie-sim catalog` |
| `mindie-sim observations` | Stream-index observation directories or attach a `mindie-profiler` kernel catalogue | `mindie-sim observations /tmp/obs.db --scan ./data` |
| `mindie-sim calibrate` / `bind-profiler` | Import measured primitive samples or Profiler bindings into `PrimitiveStore` | `mindie-sim calibrate /tmp/primitives.db samples.jsonl` |
| `mindie-sim validate` / `accuracy` | Compare predictions against observations on an explicit timing boundary | `mindie-sim validate pred.json obs.json --boundary body` |
| `mindie-sim fit-rate` / `fit-service` | Fit a single-resource rate or primitive work-response service curve | `mindie-sim fit-service observations.json` |

### Core Python API

```python
from mindie_sim import (
    predict,                       # End-to-end serving or body-forward prediction
    predict_pd,                    # Disaggregated Prefill-Decode (PD) prediction
    operator_costs,                # Isolated per-operator resource costs and physical work
    graph_costs,                   # Shared-resource target graph execution costs
    call_costs,                    # Callback protocol composition costs
    study,                         # Multi-candidate comparative study
    analyze_operator_row,          # Single-operator 4D efficiency (Cube_MFU/Vec_MFU/MBU/Link/CV_Overlap)
    analyze_network_efficiency,    # Whole-network, layer-group & subgraph 4D efficiency breakdown
    evaluate_target_efficiency,    # Single-operator counterfactual Roofline target evaluation
    fit_efficiency_targets,        # Whole-network counterfactual Roofline fitting (e.g. 0.80x MBU)
    scale_case_hardware,           # Non-mutating counterfactual hardware bandwidth/compute/VRAM scaling
    solve_hardware_sla_throughput, # Joint SLA latency & VRAM capacity maximum throughput solver
)
```

---

## Real-World Usage Examples

### Example 1: Run Serving Prediction, Isolated Operator Costs & Target Graph Analysis from CLI

```bash
# 1. Run end-to-end serving prediction (TTFT, TPOT, throughput, and schedule trace)
mindie-sim predict examples/quickstart.json

# 2. Inspect isolated per-operator device costs for a Decode step (tokens=1, sequences=16)
mindie-sim operators examples/quickstart.json --phase decode --tokens 1 --sequences 16

# 3. Evaluate the full Decode target graph on shared device resources and streams
mindie-sim graph examples/quickstart.json --phase decode --tokens 1 --sequences 16 --scope body
```

### Example 2: 4D Efficiency Breakdown, `0.80x MBU` Counterfactual Fit & `30 ms` SLA Scaling Solver

```python
import json
from pathlib import Path
from mindie_sim import (
    analyze_network_efficiency,
    fit_efficiency_targets,
    solve_hardware_sla_throughput,
)

case = json.loads(Path("examples/quickstart.json").read_text())

# 1. Compute 4D efficiency and bottleneck distribution for Decode (batch=16, context=4096)
eff = analyze_network_efficiency(
    case, phase="decode", tokens=1, context=4096, sequences=16, scope="body"
)
print("Whole-network summary:", eff["whole_network"])
print("Dominant bound breakdown:", eff["bound_breakdown"])

# 2. Counterfactual fit: how much faster is Decode if memory-bound kernels reach 0.80x MBU?
target = fit_efficiency_targets(
    case,
    phase="decode",
    tokens=1,
    context=4096,
    sequences=16,
    target_mbu=0.80,
    target_cube_mfu=0.75,
    target_vec_mfu=0.65,
)
print(
    f"Baseline: {target['baseline_total_ms']:.2f} ms -> "
    f"Target (0.80x MBU): {target['target_total_ms']:.2f} ms "
    f"(Speedup: {target['speedup_ratio']:.2f}x)"
)

# 3. Solve maximum throughput under a 30ms Decode SLA across baseline and 2x scaled hardware
sla_report = solve_hardware_sla_throughput(
    case,
    sla_ms=30.0,
    context=4096,
    tokens=1,
    max_batch_search=256,
)
for row in sla_report["scenarios"]:
    print(
        f"[{row['label']}] B*_SLA={row['b_star_sla']}, B*_VRAM={row['b_star_vram']}, "
        f"B*={row['b_star']} -> Max Throughput={row['max_throughput_tokens_per_s']:.1f} tok/s "
        f"({row['limiting_factor']})"
    )
```

---

## Repository Structure

```text
sim/
├── mindie_sim/                      # Core Python package
│   ├── __init__.py / __main__.py    # Public Python API exports and mindie-sim CLI entrypoint
│   ├── schema/                      # Case, Hardware, ModelSpec, ParallelSpec, and SloSpec contracts
│   ├── catalog/ & model_catalog.py  # Hardware profile catalog and built-in standard model definitions
│   ├── backends/                    # Native model config and Ascend runtime parameter adapters
│   ├── prediction/                  # Unified causal engine, operator/graph/call evaluators, 4D efficiency & SLA solver
│   ├── observations/                # Streaming observation index, primitive calibration, validation & service fitting
│   ├── structure*.py                # Model structure contracts, subgraph mapping, and implementation selection
│   └── study.py                     # Multi-candidate comparative study runner
├── examples/                        # Ready-to-run Case, Call, Study, and component examples
├── docs/                            # Architecture, operator mechanisms, convergence reports & release notes
├── tests/                           # Unit and full-fleet regression test suite
└── pyproject.toml                   # Package build configuration and CLI script definition
```

---

## Documentation

| Topic | Link |
| --- | --- |
| **`v0.8.1` Release Notes (4D Efficiency, Roofline Fitting & SLA Scaling)** | [`docs/releases/sim-0.8.1.md`](./docs/releases/sim-0.8.1.md) |
| **Unified Causal Architecture & Resource Time Semantics** | [`docs/architecture.md`](./docs/architecture.md) |
| **Modeling Fidelity, Scope & Refinement Policy** | [`docs/modeling-fidelity.md`](./docs/modeling-fidelity.md) |
| **Input Case Contracts & Hardware/Parallel Specs** | [`docs/inputs.md`](./docs/inputs.md) |
| **Target Graph Evaluation (`graph`) & Runtime Diagnosis** | [`docs/graph-analysis.md`](./docs/graph-analysis.md) · [`docs/graph-runtime-diagnosis.md`](./docs/graph-runtime-diagnosis.md) |
| **Callback Analysis (`call`) & Invocation Cost Contracts** | [`docs/call-analysis.md`](./docs/call-analysis.md) · [`docs/invocation-costs.md`](./docs/invocation-costs.md) |
| **Observable Service Regimes & Resource Services** | [`docs/observable-service-regimes.md`](./docs/observable-service-regimes.md) · [`docs/resource-services.md`](./docs/resource-services.md) |
| **Comparative Studies (`study`) & Observation Calibration** | [`docs/studies.md`](./docs/studies.md) · [`docs/observations.md`](./docs/observations.md) |
| **Profiler Structure Mapping & Full-Fleet Model Convergence** | [`docs/profiler-structure.md`](./docs/profiler-structure.md) · [`docs/model-convergence.md`](./docs/model-convergence.md) |
| **Companion Profiler: `mindie-profiler` (In-Place Trace Analysis & Structure Attribution)** | [`mindie-agent/profiler`](https://github.com/mindie-agent/profiler) |

---

## License

Released under the [MIT License](./LICENSE).
