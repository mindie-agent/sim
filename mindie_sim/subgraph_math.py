"""Fusion-invariant subgraph DAG math and shared efficiency evaluation.

[Role]
  Derive one ``SubgraphMathSummary`` per exported semantic subgraph and one
  ``subgraph_math_by_family`` roll-up, then evaluate Cube/Vector MFU and
  boundary/realized MBU through the single function shared by ``mindie-sim``
  and ``mindie-profiler``.

[Invariants]
  * Fusion never changes algebraic compute. Algebraic counters are therefore
    populated only from an immutable pre-lowering ledger; until that producer
    exists they stay ``None`` with ``algebraic_work_unavailable`` and physical
    counters are never renamed into algebraic ones.
  * Boundary HBM bytes require an explicit access ledger (storage identity,
    direction and byte interval) plus a cold-HBM basis. Missing page/cache
    extents stay ``None`` with an explicit reason instead of an invented zero.
  * ``effective_physical_work`` charges every unfused semantic node once and
    every selected native fusion once; a partially selected fusion has unknown
    realized work rather than a proportional invention.
  * Overlapping semantic-family ownership is a hard error: family roll-ups use
    disjoint unions and are never advertised as whole-model totals.
  * Efficiency evaluation preserves a valid observed ``0`` and scales work by
    the number of complete compatible invocations.

[Timeouts & Locks]
  Pure arithmetic over one already-exported structure graph; no Engine run,
  device access, waits or locks.

[Side-Effect Policy]
  Pure functions returning new records; ``attach_subgraph_math`` mutates only
  the caller-owned payload it is explicitly given, before its content digest.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math
from dataclasses import dataclass
from typing import Any, TypedDict

SEMANTIC_FAMILIES: tuple[str, ...] = (
    "attention", "indexer", "moe_ffn", "dense_ffn", "mhc_residual", "norm_quant",
    "recurrent_gdn_kda",
)
ROLE_FAMILIES: dict[str, str] = {
    "attention_residual": "mhc_residual",
    "residual": "mhc_residual",
    "hc_pre": "mhc_residual",
    "hc_post": "mhc_residual",
    "compressor": "attention",
    "mla_prolog": "attention",
    "qkv_norm_rope": "attention",
    "key_pool": "indexer",
    "pool_key_indexer": "indexer",
    "lightning_indexer": "indexer",
    "quant_lightning_indexer": "indexer",
    "router_selection": "moe_ffn",
    "grouped_gate_up_situ_quant": "moe_ffn",
    "grouped_gate_up_activation_quant": "moe_ffn",
    "situ_quant": "norm_quant",
    "gdn_chunk": "recurrent_gdn_kda",
    "kda_recurrent": "recurrent_gdn_kda",
    "kda_chunk": "recurrent_gdn_kda",
}
DECLARATION_FAMILIES: dict[str, str] = {
    "mla": "attention",
    "dsa": "attention",
    "qkv_preprocess": "attention",
    "compressed_attention": "attention",
    "attn_res_prefix": "mhc_residual",
    "pooled_indexer": "indexer",
    "moe": "moe_ffn",
    "situ": "norm_quant",
    "mhc": "mhc_residual",
    "gdn": "recurrent_gdn_kda",
    "kda": "recurrent_gdn_kda",
}
WORK_FIELDS = ("hbm_bytes", "flops", "vector_ops", "link_bytes")
ALGEBRAIC_WORK_UNAVAILABLE = "algebraic_work_unavailable"
BOUNDARY_LEDGER_UNAVAILABLE = "boundary_access_ledger_unavailable"
ROLLUP_SCOPE = "declared_semantic_subgraph_union"
COMM_UNATTRIBUTED = "unattributed"


class SubgraphEfficiencyReport(TypedDict):
    """Shared efficiency record returned by ``evaluate_subgraph_efficiency``."""

    algebraic_cube_mfu: float | None
    algebraic_vector_mfu: float | None
    minimal_boundary_mbu: float | None
    realized_mbu: float | None
    fusion_headroom_bytes: float | None
    byte_accounting_consistent: bool | None


@dataclass(frozen=True)
class SubgraphMathSummary:
    """One exported subgraph instance under active parameterization."""

    subgraph_id: str
    subgraph_kind: str
    semantic_family: str | None
    phase: str
    layer_multiplicity: int
    parameterization: Mapping[str, Any]
    algebraic_cube_flops: float | None
    algebraic_cube_flops_by_dtype: Mapping[str, float] | None
    algebraic_vector_ops: float | None
    minimal_boundary_hbm_bytes: float | None
    realized_hbm_bytes: float | None
    unavailable_reasons: tuple[str, ...]
    parallel_comm_bytes: float | None = None
    parallel_comm_by_domain: Mapping[str, float] | None = None
    boundary_breakdown_bytes: Mapping[str, float] | None = None
    fusion_hbm_overhead_bytes: float | None = None
    fusion_byte_efficiency: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe record; unavailable metrics stay ``None``."""
        return {
            "subgraph_id": self.subgraph_id,
            "subgraph_kind": self.subgraph_kind,
            "semantic_family": self.semantic_family,
            "phase": self.phase,
            "layer_multiplicity": self.layer_multiplicity,
            "parameterization": _jsonable(dict(self.parameterization)),
            "algebraic_cube_flops": self.algebraic_cube_flops,
            "algebraic_cube_flops_by_dtype": (
                None if self.algebraic_cube_flops_by_dtype is None
                else _jsonable(dict(self.algebraic_cube_flops_by_dtype))),
            "algebraic_vector_ops": self.algebraic_vector_ops,
            "minimal_boundary_hbm_bytes": self.minimal_boundary_hbm_bytes,
            "realized_hbm_bytes": self.realized_hbm_bytes,
            "fusion_hbm_overhead_bytes": self.fusion_hbm_overhead_bytes,
            "fusion_byte_efficiency": self.fusion_byte_efficiency,
            "parallel_comm_bytes": self.parallel_comm_bytes,
            "parallel_comm_by_domain": (
                None if self.parallel_comm_by_domain is None
                else _jsonable(dict(self.parallel_comm_by_domain))),
            "boundary_breakdown_bytes": (
                None if self.boundary_breakdown_bytes is None
                else _jsonable(dict(self.boundary_breakdown_bytes))),
            "unavailable_reasons": list(self.unavailable_reasons),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def semantic_family_of_group(group: Mapping[str, Any]) -> str | None:
    """Explicit semantic-owner mapping; ``None`` reports an uncovered owner."""
    family = ROLE_FAMILIES.get(str(group.get("role") or ""))
    if family is not None:
        return family
    declaration = DECLARATION_FAMILIES.get(str(group.get("family") or ""))
    if declaration is not None:
        return declaration
    structure = group.get("structure") or {}
    if structure.get("attention_kind"):
        return "attention"
    ffn = structure.get("ffn_kind")
    if ffn == "moe":
        return "moe_ffn"
    if ffn:
        return "dense_ffn"
    return None


def boundary_bytes(accesses, *, complete, hbm_cold):
    """Union explicit read/write byte intervals; ``None`` without a complete ledger.

    ``accesses`` items are ``(physical_storage_key, "read" | "write", start,
    stop)``. Storage keys include placement and allocation identity. Boundary
    tensor bytes are not an unconditional minimum HBM transfer when data may
    already reside in L2, so ``hbm_cold`` is required.
    """
    if not complete or not hbm_cold:
        return None

    spans = defaultdict(list)
    for storage_key, direction, start, stop in accesses:
        if direction not in {"read", "write"} or start < 0 or stop < start:
            raise ValueError("invalid_boundary_access")
        spans[(storage_key, direction)].append((start, stop))

    total = 0
    for ranges in spans.values():
        end = 0
        for start, stop in sorted(ranges):
            total += max(0, stop - max(start, end))
            end = max(end, stop)
    return total


def effective_physical_work(report, member_ids):
    """Charge unfused members and selected fusions exactly once.

    ``resource_operators`` contains selected fused operators only, so summing it
    for an unfused graph is zero while adding it to every semantic node charges
    a selected fusion twice. This resolves each selected fusion once and returns
    ``(totals, reasons)`` where ``totals`` is ``None`` when work is incomplete.
    """
    nodes = {n["node_id"]: n for n in report["nodes"]}
    resources = {n["node_id"]: n for n in report["resource_operators"]}
    selected = frozenset(member_ids)
    if not selected <= nodes.keys():
        raise ValueError("unknown_subgraph_member")

    owners = {}
    covered = defaultdict(set)
    for node_id, node in nodes.items():
        boundary = node.get("resource_boundary") or {}
        if boundary.get("kind") != "internal_to_selected_native_fusion":
            continue
        resource_ids = tuple(boundary["operator_ids"])
        if not resource_ids or any(key not in resources for key in resource_ids):
            raise ValueError("dangling_resource_boundary")
        owners[node_id] = resource_ids
        for resource_id in resource_ids:
            covered[resource_id].add(node_id)

    atoms = {}
    for node_id in selected:
        if node_id not in owners:
            atoms[("node", node_id)] = nodes[node_id]
            continue
        for resource_id in owners[node_id]:
            if not covered[resource_id] <= selected:
                return None, ("partial_selected_fusion:" + resource_id,)
            atoms[("resource", resource_id)] = resources[resource_id]

    totals = dict.fromkeys(WORK_FIELDS, 0.0)
    for atom in atoms.values():
        work = atom.get("physical_work")
        if not isinstance(work, dict):
            return None, ("physical_work_unavailable:" + atom["node_id"],)
        for field in WORK_FIELDS:
            value = work.get(field)
            if (not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                return None, ("physical_work_incomplete:" + atom["node_id"],)
            totals[field] += value
    return totals, ()


def exclusive_family_members(groups, family_for):
    """Union members per family and reject unresolved cross-family ownership."""
    by_family = defaultdict(set)
    for group in groups:
        family = family_for(group)
        if family is not None:
            by_family[family].update(group["node_ids"])

    assigned = set()
    for family, members in sorted(by_family.items()):
        if assigned & members:
            raise ValueError("overlapping_semantic_family_ownership:" + family)
        assigned.update(members)
    return by_family


def _parameterization(report: Mapping[str, Any], group: Mapping[str, Any]) -> dict[str, Any]:
    """Declared parameterization provenance; unknown entries stay ``None``."""
    model = report.get("model") or {}
    execution = report.get("execution") or {}
    quant = model.get("quant") or {}
    software = execution.get("software") or {}
    parallel = software.get("parallel")
    selected = group.get("resource_boundary") or {}
    return {
        "kv_cache_format": quant.get("kv_dtype") or model.get("kv_dtype"),
        "weight_act_dtype": {
            "weight": quant.get("weight_format") or model.get("weight_dtype"),
            "activation": quant.get("activation_format") or model.get("activation_dtype"),
        },
        "parallel_strategy": dict(parallel) if isinstance(parallel, Mapping) else None,
        "fusion_boundary": (
            str(selected.get("kind")) if selected.get("kind")
            else "semantic_declared_graph"),
    }


def _comm_by_domain(report: Mapping[str, Any], member_ids) -> dict[str, float] | None:
    """Attribute declared physical link bytes by explicit parallel domain."""
    nodes = {n["node_id"]: n for n in report["nodes"]}
    resources = {n["node_id"]: n for n in report["resource_operators"]}
    selected = frozenset(member_ids)
    atoms: dict[tuple[str, str], Mapping[str, Any]] = {}
    for node_id in selected:
        boundary = nodes[node_id].get("resource_boundary") or {}
        if boundary.get("kind") == "internal_to_selected_native_fusion":
            for resource_id in boundary["operator_ids"]:
                atoms[("resource", resource_id)] = resources[resource_id]
        else:
            atoms[("node", node_id)] = nodes[node_id]
    totals: dict[str, float] = defaultdict(float)
    for atom in atoms.values():
        work = atom.get("physical_work") or {}
        link = work.get("link_bytes")
        if not isinstance(link, (int, float)) or link <= 0:
            continue
        meta = (atom.get("operator") or {}).get("metadata") or {}
        domain = meta.get("parallel_domain") or meta.get("collective_domain")
        totals[str(domain) if domain else COMM_UNATTRIBUTED] += float(link)
    return dict(sorted(totals.items())) if totals else None


def _summary(report: Mapping[str, Any], group: Mapping[str, Any]) -> SubgraphMathSummary:
    totals, reasons = effective_physical_work(report, group["node_ids"])
    reasons = (*reasons, ALGEBRAIC_WORK_UNAVAILABLE, BOUNDARY_LEDGER_UNAVAILABLE)
    realized = None if totals is None else totals["hbm_bytes"]
    return SubgraphMathSummary(
        subgraph_id=str(group.get("subgraph_id")),
        subgraph_kind=str(group.get("role") or group.get("family") or "unknown"),
        semantic_family=semantic_family_of_group(group),
        phase=str(group.get("phase") or (report.get("execution") or {}).get("phase") or ""),
        layer_multiplicity=1,
        parameterization=_parameterization(report, group),
        algebraic_cube_flops=None,
        algebraic_cube_flops_by_dtype=None,
        algebraic_vector_ops=None,
        minimal_boundary_hbm_bytes=None,
        realized_hbm_bytes=realized,
        unavailable_reasons=reasons,
        parallel_comm_bytes=None if totals is None else totals["link_bytes"],
        parallel_comm_by_domain=_comm_by_domain(report, group["node_ids"]),
    )


def compute_subgraph_math_summaries(report: Mapping[str, Any]) -> list[SubgraphMathSummary]:
    """Compute one summary per exported subgraph from the already-exported graph."""
    return [_summary(report, group) for group in report.get("subgraphs") or ()]


def _family_scopes(groups, family_for) -> dict[str, dict[tuple, set[str]]]:
    scopes: dict[str, dict[tuple, set[str]]] = defaultdict(lambda: defaultdict(set))
    for group in groups:
        family = family_for(group)
        if family is None:
            continue
        key = (group.get("replica"), group.get("stage"), group.get("phase"))
        scopes[family][key].update(group["node_ids"])
    return scopes


def build_subgraph_math_rollup(report: Mapping[str, Any]) -> dict[str, Any]:
    """Disjoint per-family union totals; never a whole-model total."""
    groups = list(report.get("subgraphs") or ())
    families = exclusive_family_members(groups, semantic_family_of_group)
    scopes = _family_scopes(groups, semantic_family_of_group)
    rollup: dict[str, Any] = {}
    for family in sorted(families):
        realized = 0.0
        link = 0.0
        complete = True
        reasons: list[str] = []
        for key, ids in sorted(scopes[family].items(), key=lambda item: str(item[0])):
            totals, why = effective_physical_work(report, ids)
            if totals is None:
                complete = False
                reasons.extend(why)
                break
            realized += totals["hbm_bytes"]
            link += totals["link_bytes"]
        occurrences = sum(1 for group in groups
                          if semantic_family_of_group(group) == family)
        rollup[family] = {
            "subgraph_occurrences": occurrences,
            "member_node_count": len(families[family]),
            "realized_hbm_bytes": realized if complete else None,
            "parallel_comm_bytes": link if complete else None,
            "algebraic_cube_flops": None,
            "algebraic_vector_ops": None,
            "minimal_boundary_hbm_bytes": None,
            "unavailable_reasons": list(dict.fromkeys(
                (*reasons, ALGEBRAIC_WORK_UNAVAILABLE, BOUNDARY_LEDGER_UNAVAILABLE))),
        }
    uncovered = sorted({
        node_id for group in groups if semantic_family_of_group(group) is None
        for node_id in group["node_ids"]
    })
    return {
        "schema": "mindie-subgraph-math-by-family/1",
        "scope": ROLLUP_SCOPE,
        "totals_additive": False,
        "note": ("Inclusive parent and child summaries coexist; per-family unions are "
                 "disjoint and are not whole-model totals."),
        "semantic_families": list(SEMANTIC_FAMILIES),
        "families": rollup,
        "uncovered_node_ids": uncovered,
    }


def attach_subgraph_math(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach summaries and the family roll-up before the content digest exists."""
    summaries = compute_subgraph_math_summaries(payload)
    groups = payload.get("subgraphs") or []
    for group, summary in zip(groups, summaries):
        group["math_summary"] = summary.to_dict()
    payload["subgraph_math_by_family"] = build_subgraph_math_rollup(payload)
    return payload


def evaluate_subgraph_efficiency(
    summary, *, duration_s, hardware_peaks,
    observed_hbm_bytes=None, invocations=1,
):
    """Shared MFU/MBU evaluation; observed bytes are already the cohort total."""
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s_must_be_positive_finite")
    if type(invocations) is not int or invocations <= 0:
        raise ValueError("complete_invocation_count_required")

    def total(value):
        return None if value is None else value * invocations

    def ratio(work, peak):
        if work is None:
            return None
        if not math.isfinite(work) or work < 0:
            raise ValueError("invalid_work")
        if peak is None or not math.isfinite(peak) or peak <= 0:
            return None
        return work / (duration_s * peak)

    cube = None
    if summary.algebraic_cube_flops_by_dtype is not None:
        peaks_by_dtype = hardware_peaks.get("compute_peaks") or {}
        terms = [
            ratio(total(work), peaks_by_dtype.get(dtype))
            for dtype, work in summary.algebraic_cube_flops_by_dtype.items()
        ]
        if all(term is not None for term in terms):
            cube = sum(terms)

    boundary = total(summary.minimal_boundary_hbm_bytes)
    realized = (
        total(summary.realized_hbm_bytes)
        if observed_hbm_bytes is None else observed_hbm_bytes
    )
    headroom = (
        None if boundary is None or realized is None
        else realized - boundary
    )
    return {
        "algebraic_cube_mfu": cube,
        "algebraic_vector_mfu": ratio(
            total(summary.algebraic_vector_ops),
            hardware_peaks.get("vector_peak_ops_per_s"),
        ),
        "minimal_boundary_mbu": ratio(
            boundary, hardware_peaks.get("hbm_peak_bytes_per_s"),
        ),
        "realized_mbu": ratio(
            realized, hardware_peaks.get("hbm_peak_bytes_per_s"),
        ),
        "fusion_headroom_bytes": headroom,
        "byte_accounting_consistent": (
            None if headroom is None else headroom >= 0
        ),
    }


__all__ = [
    "ALGEBRAIC_WORK_UNAVAILABLE", "BOUNDARY_LEDGER_UNAVAILABLE", "ROLLUP_SCOPE",
    "SEMANTIC_FAMILIES", "SubgraphEfficiencyReport", "SubgraphMathSummary",
    "attach_subgraph_math", "boundary_bytes", "build_subgraph_math_rollup",
    "compute_subgraph_math_summaries", "effective_physical_work",
    "evaluate_subgraph_efficiency", "exclusive_family_members",
    "semantic_family_of_group",
]
