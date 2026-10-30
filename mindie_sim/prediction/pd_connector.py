"""Physical connector service for a complete P/D cache handoff.

[Role] Bind declared connector framing, propagation and capacity to topology.
[Invariants] Endpoints 0/1 denote prefill/decode pools, not inferred model ranks.
  One handoff is one logical message; wire bytes exclude duplicated capacity
  demands. HBM work is supplied separately by each pool's representative shard.
[Timeouts & Locks] Constant-sized connector graph and aggregate counters only.
[Side-Effect Policy] Submits tasks only to the caller's in-memory engine.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware, Link
from .communication import MessageFlow
from .topology import Topology


class PDConnector:
    def __init__(self, raw):
        explicit = "link" in raw
        declaration = dict(raw["link"] if explicit else raw)
        if not explicit:
            declaration["capacity_bytes_per_ms"] = declaration.pop("bytes_per_s") / 1000
        declaration.setdefault("name", "connector")
        declaration.setdefault("members", (0, 1))
        declaration["members"] = tuple(declaration["members"])
        self.link = Link(**declaration)
        if set(self.link.members) != {0, 1}:
            raise ModelingError("pd_connector_members_require_pool_endpoints_0_1")
        if self.link.semantics not in {
                "shared", "direct", "per_pair", "per_port", "fully_connected", "mesh"}:
            raise ModelingError(f"link_semantics_unknown:{self.link.semantics}")
        self.topology = Topology(Hardware("pd_connector", (), (self.link,)), None, "pd:")
        # Existing flat connector inputs keep their public resource identity.
        self.alias = ({f"pd:link:{self.link.name}": "pd_link"}
                      if not explicit else {})
        self.wire_bytes = 0.0
        self.transactions = 0 if self.link.transaction_bytes is not None else None

    def resources(self):
        return tuple(replace(resource, name=self.alias.get(resource.name, resource.name))
                     for resource in self.topology.resources())

    def submit(self, engine, name, volume, memory_work):
        hop, = self.topology.route_message(MessageFlow(0, 1, volume)).hops
        ready = engine.submit(name + ":latency", latency=hop.latency_ms)
        demand = {self.alias.get(key, key): value for key, value in hop.demand.items()}
        self.wire_bytes += hop.wire_bytes
        if self.transactions is not None:
            self.transactions += hop.transactions
        memory = engine.submit(name + ":memory", memory_work, after=(ready,))
        wire = engine.submit(name + ":wire", demand, after=(ready,), coupled=True)
        return engine.submit(name, after=(memory, wire), metadata={
            "boundary": "pd_pool_connector", "useful_bytes": volume,
            "wire_bytes": hop.wire_bytes, "transactions": hop.transactions,
        })

    def summary(self):
        return {"boundary": "pd_pool_connector", "endpoints": {"0": "prefill", "1": "decode"},
                "message_boundary": "one_complete_destination_sharded_handoff",
                "memory_boundary": "one_local_shard_per_pool_replica_stage",
                "link": self.link.to_dict(), "wire_bytes": self.wire_bytes,
                "transactions": self.transactions}
