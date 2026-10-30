# Kimi finite peer window declaration

The 17 retained high-concurrency Kimi variant cases and
`examples/cases/kimi-k3-a5-sp-right-pad-conditional.json` now explicitly select
`direct_peer_collective` with `max_inflight_per_rank: 8` for both
`expert_dispatch` and `expert_combine`. This is the user's requested transport
scenario. Input hashes and prior bindings are retained in
`data/verification/kimi-direct-peer-window8-cases-20261030.json`.

Eight is the finite number of in-flight peer messages allowed per rank. Link
bandwidth, per-hop propagation, wire contention, directed payloads and inverse
return traffic retain their existing physical declarations. This does not
change A3/A5 rates, insert a measured floor, or identify the native binary's
peer window. The cases preserve their distinct checkpoint precision, topology,
workload, routing and source selections.

The older DP4/TP8/EP32 case retains its conditional mixed MXFP8/MXFP4 checkpoint
and explicit source SP padding. The newer TP8/EP8 BF16 cases retain their
16-expert, Top-8 declarations; they are not converted into the older
896-expert, Top-16 model. The A3 variant cases use this generic transport
scenario without claiming an arch35 implementation.

The boundary is peer transport only. Generic route packing, external activation
quantization, expert GEMMs and weighted output reduction remain in the caller.
Source-native MC2 count/status, row framing and local receive-copy phases are a
separate explicit selector under review and are not installed in these cases.
Consequently a transport span must not be presented as a full
`MoeDistributeDispatchV3` / `MoeDistributeCombineV3` kernel accuracy result.
