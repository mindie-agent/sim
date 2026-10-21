# `mindie-sim` (`mindie_sim`)

Standalone Ascend NPU 4-layer simulation and theoretical boundary solver for MindIE Agent (`v0.0.3`):

1. **Layer 1 (`hardware` / `sku_profile` / `comm`)**:
   - Multi-SKU `DeviceCapabilityProfile` (`cube_spec`, `vector_spec`, `memory_hierarchy_spec`, `clock_and_topology_spec`) across A2 (`Ascend910B1..B4`), A3 (`Ascend910_93*` dual-die), and A5 (`Ascend950DT`/`950PR`), with 3-tier resolution (`.ini` + SKU registry + external JSON -> live `npu-smi` & `lspci` probe override -> caller override) and safe degradation (`ambiguous_soc_sku_requires_exact_soc_or_ini_or_probe`).
   - Full-path 7-channel interconnect & host-device transfer solver (`d2d_intra_package`, `d2d_intra_node`, `d2d_inter_node`, `h2d`, `d2h`, `rh2d`, `rd2h`), including `pinned` vs `pageable` H2D/D2H, PCIe `LnkSta` downgrade detection, `direct_rdma` vs `host_staged` chunked pipeline for `RH2D`/`RD2H`, and 3-tier hierarchical collective bounds.
   - On-chip buffer (`L1`/`L0A`/`L0B`/`L0C`/`UB`) alignment/overlap validation and sustained thermal clock sensitivity matrices.
2. **Layer 2 (`pipeline`)**:
   - Multi-phase periodic microstage graph expansion (`MTE2`/`MTE1`/`M`/`FIX`/`V`/`MTE3`), Karp maximum cycle-mean solver (`minimum_period_ns`, `minimum_tile_ii_ns`), critical recurrence cycle extraction, and dimensionless topological `--self-check`.
3. **Layer 3 (`subgraph`)**:
   - Upfront Phase 0 producer-consumer subgraph fusion analyzer (`removable_gm_bytes`, hard feasibility gates for collective barriers, `L1`/`UB` residency limits, non-elidable `DCP` normalization, and concurrent critical-path overlap `max(P+C, Q+G)`).
4. **Layer 4 (`net`)**:
   - Multi-dimensional parallel step decomposition (`DP x TP x EP x PP x DCP`, `DSpark` acceptance length, `Prefix Cache` hit rate, `D2D`/`RH2D` transfer overlap) and read-only bridge to pinned `msmodeling` (`tensor_cast`).


## A3 predictive and normative dual model (v0.7)

The package also provides a runnable A3 dual-model path:

* `predict`/`mindie_sim.predict`: typed machine/model/workload/runtime input,
  Dense/GQA/MLA/MoE shape lowering, graph/KV/MoE recipes, request scheduler,
  event-driven Cube/Vector/MTE/HBM/D2D contention, TTFT/ITL/TPOT/E2E,
  throughput/SLO goodput and iteration/shape traces.
* `should`/`mindie_sim.should_cost`: a separate semantic normative path with
  optimistic lower bound `L`, enumerated legal plans and feasible target `U`
  plus a witness timeline. It is not a predictive result multiplied by an
  efficiency factor.
* `SQLiteCalibrationStore`: raw-sample inventory/coverage, deduplicated
  measurement requests, executable manifest, CPU closed loop and optional
  managed `torch_npu` runner. `predict` writes requests only with explicit
  opt-in. No A3 measurement is bundled or fabricated.
* `search`: full predictive simulation of every legal candidate, OOM/SLO
  filtering and an independently rechecked recommendation.

See `docs/a3-dual-model.md` and `examples/README.md`. Quick commands:

```bash
python -m mindie_sim predict --input examples/a3_dense_predict.json
python -m mindie_sim predict --input examples/a3_mla_moe_predict.json
python -m mindie_sim should  --input examples/a3_should_dense.json
python -m mindie_sim search  --input examples/a3_search_dense.json
```

## A2 predictive support and fitted physical calibration (v0.8, measured round)

The predictive path now targets **A2 (`Ascend910B4`) and A3 (`Ascend910_9362`)**
through the same machine-physics layer, with no per-SKU fudge factors:

* **A2 first-class**: profile-derived `MachineSpec` (clock precedence
  override > live `npu-smi` > CANN `.ini`), evidence-corrected AIC/L2 catalog,
  HCCS mesh + no-SIO link model, mesh vs switched semantics, and per-op fabric
  scope instead of `links[0]`.
* **Physical calibration** (`mindie_sim/a3_physical_calibration.py`): a versioned,
  provenance-bearing artifact fitted only on train measurements
  (`launch + max(compute, memory)` with fitted tiles, wave quantization,
  L2 blend and shape-tail terms). `predict --physical-calibration PATH
  --timing-scope SCOPE` uses it; unmodeled families fall back with the artifact's
  own reason. Timing scopes (batched amortized vs per-call) are never mixed.
* **Structural fixes**: additive launch (no max-composed 20 µs floor), AIC count
  from the machine, invented efficiency floors removed, correct PV contraction
  axis, exact `machine_key` calibration joins.

Measured on real A2/A3 hardware in this round (train-only fits; blinded test set 2
frozen before any outcome): A2 344 train conditions / 66 heldout, A3 212 train /
80 heldout. Fitted CV: **A2 GEMM 9.71 %**, A2 add/rmsnorm/swiglu/softmax 2.4–4.0 %,
**A3 GEMM 16.05 % (per-call) / 16.10 % (batched)**, A3 rmsnorm 6.13 %, softmax 6.98 %.
Frozen held-out test set 2: **A2 22.80 % WAPE (coverage 0.9545)**,
**A3 45.72 % WAPE (coverage 0.9125)**, combined 29.67 % over 136/146 conditions.
Both verdicts are **rejected** against the predeclared gates; the fitted layer is
strongest where it applies (A2 GEMM 4.78 %, A3 GEMM 16.87 %), and the remaining
error is concentrated in unmodeled batched families and protocol/declaration
mismatches. A5 is deliberately deferred.

Full method, tables, gates, limitations and evidence paths:
[`docs/a2-a3-measured-accuracy.md`](docs/a2-a3-measured-accuracy.md).

## Large-operator calibration and whole-network validation

The model now carries measured **fused attention** (GQA and absorbed MLA,
prefill/decode), **grouped matmul / MoE** and **MC2** (dispatch + combine +
grouped matmul) physics, fitted train-only with 5-fold CV, and it is validated
against a real serving benchmark on A3 (Qwen2.5-7B, TP1, vLLM-Ascend, eager and
ACLGraph):

* A3 fitted CV: attention **4.19 %**, `moe_gate_up` 10.98 %, `moe_down` 2.36 %,
  grouped matmul 1.65 %, dense GEMM 8.84 % (per-call) / 11.50 % (batched).
* Whole-network graph mode (after the attention-byte, weight-streaming and
  replayed-graph launch fixes): TPOT **14.70 / 15.14 / 16.52 ms** for
  128 / 1024 / 4096-token prompts vs measured **15.71 / 15.24 / 15.10 ms**;
  the batched scope brackets them at 14.64 / 14.68 / 14.83 ms.
* MC2 is a **sum**, not an overlap: chain = 1.0–1.2× the measured legs.
* The pre-fix long-prompt error was traced to materialising the attention score
  matrix to HBM (18.15 GB/layer → 0.305 GB/layer); prefill prediction dropped
  1147.6 → 225.3 ms analytic (323.1 ms fitted) for a measured 342.6 ms TTFT.

Method, contracts (128-expert GMM cap, A2 silent mask, absorbed-MLA support),
measured campaigns and residual gaps:
[`docs/large-operator-calibration.md`](docs/large-operator-calibration.md).
