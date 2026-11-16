"""Exposed-memory attribution and fixed-plan counterfactuals.

[Role] Attribute a physical graph's span to Cube, Vector, HBM and Link buckets,
  separating exposed memory stalls from fully overlapped service.
[Invariants] Coverage fractions are not MFU. Cube and Vector coverage cannot
  hide the same memory interval twice; the estimate is a lower bound on exposed
  memory unless stage service intervals prove a disjoint union. Buckets remain
  nonnegative and telescope to the reported graph span when the graph is total.
[Timeouts & Locks] Pure arithmetic over bounded node sets; no I/O or locks.
[Side-Effect Policy] Returns new mappings; does not mutate reports or plans.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from mindie_sim.schema.common import ModelingError
from .roofline_graph import PhysicalWorkGraph, PhysicalWorkNode

ZERO = 0
ZERO_FLOAT = 0.0
ONE_FLOAT = 1.0
DEFAULT_CUBE_COVER = 1.0
DEFAULT_VECTOR_COVER = 1.0


@dataclass(frozen=True)
class ExposedHbm:
    memory_ms: float
    cube_covered_ms: float
    vector_covered_ms: float
    exposed_ms: float
    basis: str

    def to_dict(self) -> dict:
        return {
            "memory_ms": self.memory_ms,
            "cube_covered_ms": self.cube_covered_ms,
            "vector_covered_ms": self.vector_covered_ms,
            "exposed_ms": self.exposed_ms,
            "basis": self.basis,
        }


def estimate_exposed_hbm(memory_ms: float, cube_ms: float, vector_ms: float, *,
                         cube_cover_fraction: float,
                         vector_cover_fraction: float) -> float:
    """Return ``max(0, mem - eta_C*cube - eta_V*vector)`` without clipping inputs."""
    values = (memory_ms, cube_ms, vector_ms)
    if any(not math.isfinite(value) or value < ZERO_FLOAT for value in values):
        raise ModelingError("exposed_hbm_requires_nonnegative_finite_service")
    for name, value in (("cube_cover_fraction", cube_cover_fraction),
                        ("vector_cover_fraction", vector_cover_fraction)):
        if not math.isfinite(value) or value < ZERO_FLOAT or value > ONE_FLOAT:
            raise ModelingError(f"exposed_hbm_cover_fraction_invalid:{name}")
    covered = (float(cube_cover_fraction) * float(cube_ms)
               + float(vector_cover_fraction) * float(vector_ms))
    return max(ZERO_FLOAT, float(memory_ms) - covered)


def attribute_resource_exposure(report: Mapping[str, object],
                                plan: PhysicalWorkGraph) -> Mapping[str, object]:
    """Attach four physical buckets and an exposed-memory estimate to a report."""
    if not isinstance(plan, PhysicalWorkGraph):
        raise ModelingError("resource_attribution_requires_physical_graph")
    graph_span = float(report.get("graph_duration_ms",
                                  report.get("duration_ms", ZERO_FLOAT)) or ZERO_FLOAT)
    cube = math.fsum(node.cube_ms for node in plan.nodes)
    vector = math.fsum(node.vector_ms for node in plan.nodes)
    memory = math.fsum(node.hbm_ms for node in plan.nodes)
    link = math.fsum(node.link_ms for node in plan.nodes)
    exposed = estimate_exposed_hbm(
        memory, cube, vector,
        cube_cover_fraction=DEFAULT_CUBE_COVER,
        vector_cover_fraction=DEFAULT_VECTOR_COVER)
    buckets = {
        "cube": {"service_ms": cube, "exposed_ms": max(ZERO_FLOAT, cube - memory)},
        "vector": {"service_ms": vector, "exposed_ms": max(ZERO_FLOAT, vector - memory)},
        "hbm": {
            "logical_bytes": float(report.get("logical_hbm_bytes",
                                              report.get("hbm_bytes",
                                                         ZERO_FLOAT)) or ZERO_FLOAT),
            "physical_hbm_bytes": float(report.get("hbm_bytes", ZERO_FLOAT) or ZERO_FLOAT),
            "l2_hit_bytes": float(report.get("l2_hit_bytes", ZERO_FLOAT) or ZERO_FLOAT),
            "memory_service_ms": memory,
            "exposed_hbm_estimate_ms": exposed,
            "exposure_basis": "pipelined_coverage_lower_bound",
        },
        "link": {"wire_ms": link, "exposed_ms": max(ZERO_FLOAT, link - memory)},
        "control": {"launch_or_idle_ms": max(
            ZERO_FLOAT, graph_span - max(cube, vector, memory, link))},
    }
    return {
        "buckets": buckets,
        "memory_service_ms": memory,
        "exposed_hbm_estimate_ms": exposed,
        "fixed_plan_counterfactual": fixed_plan_exposure(plan),
        "basis": "four_clock_service_buckets_not_whole_operator_assignment",
    }


def fixed_plan_exposure(plan: PhysicalWorkGraph) -> Mapping[str, float]:
    """Same-plan zero-HBM counterfactual under the graph's dependency order."""
    if not isinstance(plan, PhysicalWorkGraph):
        raise ModelingError("fixed_plan_exposure_requires_physical_graph")
    nodes = {node.node_id: node for node in plan.nodes}
    predecessors = plan.predecessors()
    baseline: dict[str, float] = {}
    zero_hbm: dict[str, float] = {}
    visiting: set[str] = set()

    def finish(node_id, field):
        if node_id in visiting:
            raise ModelingError("fixed_plan_exposure_cycle")
        visiting.add(node_id)
        value = ZERO_FLOAT
        for predecessor in predecessors.get(node_id, ()):
            value = max(value, finish(predecessor, field))
        node = nodes[node_id]
        if field == "baseline":
            own = node.ideal_ms
        else:
            own = max(node.launch_ms,
                      node.fixed_ms + max(node.cube_ms, node.vector_ms,
                                          node.link_ms))
        visiting.remove(node_id)
        return value + own

    for node_id in nodes:
        baseline[node_id] = finish(node_id, "baseline")
        zero_hbm[node_id] = finish(node_id, "zero_hbm")
    base_span = max(baseline.values(), default=ZERO_FLOAT)
    zero_span = max(zero_hbm.values(), default=ZERO_FLOAT)
    return {
        "baseline_ms": base_span,
        "zero_hbm_service_ms": zero_span,
        "delta_h_ms": max(ZERO_FLOAT, base_span - zero_span),
        "basis": "same_plan_same_dependencies_zero_hbm_service",
    }
