# V4.1 DYNAMIC_BUFFER4 two-node sensitivity

This control runs the public `predict(..., scope="body")` path for the full 40-layer V4.1 source case. It uses eight ranks across two nodes (`ranks_per_node=4`), TP1/DP8/EP8, q1, context2048 and the explicitly declared symmetric cohort (one valid query row on each DP replica). That peer workload is a scenario, not a reconstruction from client c8 or graph capacity.

The fixed input is `examples/cases/high-concurrency-20261030/v41-dynamic-buffer4-conditional/a5_v41_fp8_tp1_dp8_ep8_tpot10ms_c8-dynamic-source.json`. Only node placement and the selected hardware axis change. The producer reuses `tools.deployment_generalization.scaled_case`; it retains the parsed rates, compute peaks, service modes, link definitions, source-prepared operator work and physical link service in `data/verification/v41-dynamic-two-node-sensitivity-20261030.json`.

The four axes are:

- Network: inter-node per-port bandwidth, 25/50/100 GB/s; link latency remains 5 µs and intra-node bandwidth remains 392 GB/s.
- HBM: 2/4/8 TB/s per device.
- AIC: all declared compute peaks, MTE1, MTE2_AIC and Fixpipe scale together by 0.5/1/2.
- AIV: Vector, MTE2_AIV, MTE3 and the declared register resources scale together by 0.5/1/2; dependent Vector service modes follow the same scale.

A single 1x baseline is shared across four pairs of 0.5x/2x runs, for nine fresh worker processes. Each process has a 60-second simulation limit and 10-second receipt/cleanup allowance. The campaign has a 600-second budget and does not launch a row without its complete 70-second allowance. Incomplete rows remain visible.

This is a hardware-parameter and execution-boundary check. It does not establish measured operator/body accuracy, PD handoff behavior, arbitrary topology support, or universal equality of the compact source program under heterogeneous contention. The latter remains limited as documented in `mxfp4-dynamic-buffer4.md`. A faster unused or noncritical resource may plateau; only actual nonzero link service demonstrates a network workload.


The nine runs completed in 311.48 seconds total. Runtime content stayed at package `90f5ed66354d75a33c0940a8e62124912ee3f5db95196f9e866df3428b7369e7` (equal to integrated `12e2073b`); no source, precision, peer-work or recipe changes occurred. All runs retained 1,468,488 Engine tasks and identical prepared operator counters, with zero live tasks/programs/buffers. Maximum per-row wall time was 41.79 seconds and peak RSS 409.45 MiB. Host submission was excluded in every body probe.

| Axis | 0.5× body ms | 1× body ms | 2× body ms | Classification |
|---|---:|---:|---:|---|
| network | 9.937264 | 9.888112 | 9.863536 | strict |
| hbm | 12.010850 | 9.888112 | 9.851933 | strict |
| aic | 13.512361 | 9.888112 | 8.758368 | strict |
| aiv | 12.086002 | 9.888112 | 9.195448 | strict |

All four axes are strictly monotone for this input; no plateau or reverse response occurred. Inter-node resource service is exactly **19,660,800 bytes** in every run, so the network result is supported by actual traffic. The network effect is small: the 0.5×–2× spread is 0.073728 ms. The HBM 2× result also has little further benefit, although halving HBM is costly. These outcomes identify the sensitivity of this declared graph, not a requirement that every hardware axis produce a large speedup.

`execution.work` records prepared operator work: 2,328,776,736,768 FLOPs, 1,111,485,760.1920805 Vector operations, 97,387,637,952 HBM bytes and 34,406,400 logical link bytes. Source-issued matrix tiles are included, so the FLOP total is not the valid-row arithmetic lower bound. Physical inter-node bytes are a subset of all logical communication and remain reported separately. No observations, error targets or multiplicative correction enter this scan.
