# Full-duplex switch service and physical message accounting

`per_port` (including the existing `switched` input alias) now gives each rank
independent `tx:portN` and `rx:portN` capacity. Every routed transmission occupies
its source Tx and destination Rx simultaneously. This fixes receiver incast:
three senders targeting one receiver share the receiver rate; opposite-direction
transfers on a full-duplex pair can overlap. Causal peer, ring, HCCL AlltoAllV
and MC2 transport use the same coupled Engine service.

A link can additionally declare `switch_capacity_bytes_per_ms`, a shared switch
plane/uplink capacity. Per-hop propagation is
`latency_us + switch_hops * switch_hop_latency_us`. The first term is the
non-switch propagation declaration; a measured total that already includes ASIC
forwarding must not be charged again. These fields also survive Ascend
`transfer_channels` overrides. A hierarchical algorithm must still declare its
actual groups and inter-level dependencies; adding a switch is not a substitute
for an HCCL selector or algorithm.

`wire_bytes` remains physical traffic summed once per routed hop. Tx, Rx and
switch reservations appear separately as `capacity_byte_service` and resource
demand. Thus two endpoint reservations do not double the reported wire bytes.
Existing per-message alignment, transaction and protocol-header rounding happens
once before these simultaneous reservations, including each message tail.

The plan's proposed SIO/HCCS/UB flit sizes and RoCE overhead are not verified
universal protocol specifications. No such guessed values were inserted into
SKU defaults. Explicit framing declarations already execute and are tested;
measured effective bandwidth must be distinguished from physical wire bandwidth
before adding header costs. A zero-byte ordinary alltoall cell remains no send;
a separate explicit synchronization signal may carry propagation dependencies.

Engine controls verify incast versus disjoint destinations, opposite directions,
switch-plane oversubscription, 0.5/1/2 link rates, delayed input, framing once,
round/hop propagation and serialized hardware round trips. Existing causal
collective and source-mask controls retain their logical send/receive volumes.
