"""Connector transport resource aliases.

[Role] Expose shared topology plans under the connector's resource names.
[Invariants] Aliases preserve flow, chunk and hop identities and physical work.
[Timeouts & Locks] Finite traversal of flows, chunks and hops. No locks.
[Side-Effect Policy] Returns an in-memory plan and read view of topology resources.
"""
from dataclasses import replace


class _AliasedTopology:
    """Read view of a connector topology with its historical public names."""

    def __init__(self, topology, alias):
        self._topology = topology
        self._alias = alias
        self._resources = tuple(
            replace(resource, name=alias.get(resource.name, resource.name))
            for resource in topology.resources())
        self._resource_rates = {resource.name: resource.rate
                                for resource in self._resources}

    def resources(self):
        return self._resources

    def __getattr__(self, name):
        return getattr(self._topology, name)


def _alias_plan(self, plan):
    """Keep the public resource identity while using the shared planner."""
    if not self.alias:
        return plan
    flows = []
    for path in plan.flows:
        chunks = []
        for chunk in path.chunks:
            hops = tuple(
                replace(hop,
                        resource=self.alias.get(hop.resource, hop.resource),
                        additional_resources=tuple(
                            self.alias.get(name, name)
                            for name in hop.additional_resources))
                for hop in chunk.hops)
            chunks.append(replace(chunk, hops=hops))
        flows.append(replace(path, chunks=tuple(chunks)))
    proxy = _AliasedTopology(self.topology, self.alias)
    return replace(plan, topology=proxy, flows=tuple(flows))
