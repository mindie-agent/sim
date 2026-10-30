# V4.1 MC2 boundary audit and BF16-wire conditional control

The seven A5 dynamic-buffer4 plus partial-rotary inputs all currently use generic dispatch/combine. They declare TP1/DP8/EP8, hidden width 5120, 384 routed experts, top-k 6 and one **local** shared expert. No `native_moe_protocol` selector is installed. Those original inputs remain unchanged.

Retained complete tensor contracts identify a real representation mismatch with the existing ordinary `arch35_fullmesh` binding. For c1, DispatchV3 consumes BF16 `[1,5120]` plus INT32 route IDs `[1,6]`, and produces FP8 E4M3 `[48,5120]` with E8M0 `[48,160]` scales. The 48 output rows are **capacity**, not known active assignments. CombineV3 consumes BF16 expert outputs and FP32 routing weights, then publishes BF16 `[1,5120]`. Active mask values, peer routes and native fullmesh tiling key remain unknown.

The shared protocol source at `ops-transformer@a3925fd49cd256ad478d2faf983bf2961b8ae2c8` has an MX branch. `moe_distribute_dispatch_v3_apt.cpp` selects quantized fullmesh templates; the V2 A5 fullmesh header `TokenToExpertInQuant` (486–530) applies `QuantProcess` **before** token framing and remote-window publication. Therefore the current ordinary BF16 protocol plus a later `expert_up.quantize` cannot represent the captured atomic DispatchV3 boundary. Merely naming it DispatchV3 would be wrong.

The new input `examples/cases/high-concurrency-20261030/v41-mc2-conditional/a5_v41_c1-bf16-wire-fullmesh-conditional.json` is an explicit **alternative BF16-wire scenario**, not a promoted capture binding. It preserves all parent matrices, quantization, attention, hardware and local shared-expert work; it adds only the fullmesh selector and its evidence conditions. The finite peer window is explicitly 8, not inferred from timing. The demonstration uses symmetric `[1,1,1,1,1,1,1,1]` DP requests; a rank0 q1 observation does not establish that peer distribution.

The existing protocol source excludes dedicated shared-expert ranks and elastic remapping. One locally computed shared expert does not require a dedicated shared rank. Uniform peer traffic remains conditional; forward bytes and the reverse transposed matrix are preserved. Router weights and receiving-side expert activation quantization remain real dependencies. The new declaration neither removes that quantizer nor claims it was inside the native dispatch.

## One-layer results

The receipt pins all seven parent hashes, compact profiler ABI receipts, three source header hashes and the runtime commit. Only one complete layer is simulated; no new full-body runs or raw CSV scans are performed.

| Conditional program | One-layer ms | Dispatch/combine wire per direction |
|---|---:|---:|
| Existing generic collective | 0.181508 | 12.737 us |
| Direct-peer window8 only | 0.159908 | 1.937 us |
| Fullmesh BF16 phases + window8 | 0.160275 | Framed payload plus independent count/status channel |

Thus the roughly 25 us previously visible per layer is **two** generic directions, not one native kernel. All three one-layer Engine runs finish with zero live tasks/programs and released buffers after a 7 ms external-input readiness boundary. Source count-prefix/status work, framing, receive local copy, reduction and output publication are included in the MC2 phase graph. Their native group spans are recorded separately; phase elapsed sums must not be added as a body correction.

The retained c1 population has 6,080 calls per direction: Dispatch mean45.332 us/P50 33.641 us, Combine mean11.198 us/P50 11.141 us. Dispatch includes long outliers up to1175.402 us. These are whole native task boundaries, with peer/control effects; they cannot calibrate link latency or be assigned wholesale to AIV service. The conditional program does **not** close this measured gap.

For width5120, the shared row builder gives BF16 fullmesh frame11264 B versus MX frame6144 B (5120 data +160 scale bytes before triple/alignment/framing). The next narrow implementation is to expose the already-declared MX row format/quantization in the ordinary MC2 graph, remove or move the receiving-side activation quantizer only when its exact dtype/scale contract matches, and preserve reverse BF16 framing plus all count/status dependencies. Captured quant-mode/clip flags, active masks and routes still need explicit conditions; scalar/polling latency cannot be invented from the residual.

Reproduce the case and one-layer controls with:

```sh
python data/verification/v41-mc2-conditional-20261030.producer.py \
  -o data/verification/v41-mc2-conditional-20261030.json
```
