# Direct-peer MoE transport candidate

The Kimi EP32 diagnostic exposed a protocol mismatch: its generic
`ResourceEnvelope` charged each dispatch/return `31 * 5 us = 155 us` before
approximately `0.054 us` of link service. This is a serialized-round estimate.
It is not a property of all native MoE dispatch/combine implementations.

`direct_peer_collective` adds an explicit finite window of independent peer
messages to the existing Topology/Engine. It preserves the supplied directed
traffic matrix, source-port/shared-fabric contention, transaction framing,
per-hop propagation delay, and each participating rank's input readiness. A
rank finishes only after all its incoming and outgoing messages retire. Idle
senders still receive; zero cells have no wire or propagation work. No global
barrier is inserted between unrelated peers.

```json
{
  "runtime": {
    "kernels": {
      "recipes": {
        "expert_dispatch": {
          "kind": "direct_peer_collective",
          "max_inflight_per_rank": 32
        },
        "expert_combine": {
          "kind": "direct_peer_collective",
          "max_inflight_per_rank": 32
        }
      }
    }
  }
}
```

The window is mandatory and explicit. `32` is a transport scenario, **not a
measured vendor dispatch setting**. Each slot holds one coalesced peer message
through remote arrival; this is conservative about slot reuse relative to a
sender queue that releases at injection completion. It does not identify the
native per-token queue or AIV-owner assignment. `path_model="store_forward"`
allows explicit multi-hop paths; the default `direct` rejects them. The
existing generic, ring and pairwise choices remain available and unchanged.

## Source evidence and support boundary

Local retained repository: `refs/ops-transformer`, commit
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8`. The four files below are clean at
that revision. This is source-mechanism evidence, not a binary hash match to
the historical Kimi capture.

| File | SHA256 |
|---|---|
| `mc2/moe_distribute_dispatch_v3/op_kernel/arch35/moe_distribute_dispatch_v3_apt.cpp` | `85724d3f0d8271fbb1d467ab013e00aa8b385e72ab9e9e4f36dcae0fecd5bf3b` |
| `mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_v2_a5_full_mesh.h` | `76db25293e4d3b34784012ace8fd824a977e4055fa974a9e3370c99975467c1e` |
| `mc2/moe_distribute_combine_v3/op_kernel/arch35/moe_distribute_combine_v3_apt.cpp` | `2cc24e36c4c9c8cbb2028ca19d26425ad054e1f2c842b2d5391b81175498d207` |
| `mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_mte.h` | `133aef717fce163e06f942c82ffb162e5cc0fe8d8e9c56fcbaaccd13afd63429` |

The V3 arch35 entries reuse V2 A5 full-mesh dispatch and A5 MTE combine. In
the selected full-mesh dispatch branch:

- Lines 388–405 split AIVs into token-sending and expert-count/prefix groups.
- Lines 636–687 distribute actual routed assignments across sending AIVs;
  `expertId / experts_per_rank` selects each direct destination window.
- Lines 505–517 and 543–557 use finite MTE3 output buffers and hard-event
  dependencies. Lines 690–705 derive the buffer count from remaining UB and
  the finite event-flag domain. They do not impose a `rank-1` receive-forward
  round chain.
- Lines 833–893 send count/status records. Lines 1813–1824 run payload and
  count groups concurrently, then materialize received rows locally.

Combine lines 829–842 partition actual returned token rows across AIVs.
Lines 857–894 send each row to the destination encoded in `expandIdx`, and
lines 902 onward build a remote-window data-as-flag message. Lines 1620–1631
then perform local receive/combine. This also has no pairwise round barrier.

The retained implementation packs 480-byte data pieces into 512-byte blocks
with flags, alongside triple/scale metadata. These are **native protocol
bytes**, distinct from any separately declared link transaction framing.
This first candidate does not add them because the ordinary frontend supplies
only aggregate useful peer payloads, not actual integer routed token cells.
It likewise does not claim to price native count/status exchange, polling,
fused quantization or local window repacking. Existing caller operators remain
in the full graph; only the selected transport program changes. Avoid adding
native local work again until the corresponding caller work is replaced.

## Same-input Kimi control

Run `tools/verify_direct_peer_transport.py CASE -o REPORT` with the retained
Kimi current-checkpoint case. The committed output is
`data/verification/kimi-direct-peer-transport-20261030.json`.

The scenario is q1, B1 per DP replica, DP4/TP8/EP32, four 8-die hosts. It is
not proof that all four replicas were active in the original concurrency-one
capture. The tool obtains one actual ordinary dispatch/return pair via
`target_graphs(scope="body")`, then prices only that complete transport cohort.
All candidates retain the same bytes, 1.8 us intra-host delay and 5 us
inter-host delay. It does not run or relabel a full-body comparison.

| Protocol candidate | Dispatch + return graph ms |
|---|---:|
| Existing aggregate serial estimate | 0.310107520 |
| Causal pairwise rounds | 0.271725714 |
| Direct-peer window 1 | 0.265323520 |
| Direct-peer window 8 | 0.033723520 |
| Direct-peer window 32 | 0.010107520 |

The window-32 pair projected over 92 identical declared MoE layers is
0.92989184 ms of **transport-only service**. This is not comparable by itself
to the reported 3.6041 ms activity sum of complete native dispatch/combine
kernels. The difference identifies missing boundary terms and unknown native
scheduling, not a new latency floor to fit. No window is selected by minimizing
error against that measurement.

Operator reports show `setup_ms=0` for this program because propagation is
charged by each message's arrival event. The old aggregate estimate is retained
as `unused_serial_round_estimate_ms`, explicitly not charged a second time.

Independent tests cover finite-slot recurrences on asymmetric and transposed
traffic, shared/per-port wire contention, transaction tails, delayed rank
entry, idle senders, multi-hop framing, and ordinary frontend/predict use.
Every completion leaves zero live tasks and conserves source-attributed wire.
