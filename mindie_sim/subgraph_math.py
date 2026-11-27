"""Fusion-invariant subgraph DAG math and shared efficiency evaluation.

[Role]
  Derive one ``SubgraphMathSummary`` per exported semantic subgraph and one
  ``subgraph_math_by_family`` roll-up, then evaluate Cube/Vector MFU and
  boundary/realized MBU through the single function shared by ``mindie-sim``
  and ``mindie-profiler``.

[Invariants]
  * Fusion never changes algebraic compute. Algebraic counters come from the
    untouched pre-lowering semantic member nodes, so a fused resource boundary
    changes realized bytes but never the semantic Cube/Vector totals; physical
    counters are never renamed into algebraic ones.
  * Boundary HBM bytes are the unfused semantic member bytes minus the internal
    producer/consumer round-trips that a selected fusion keeps on chip. An
    incomplete semantic ledger stays ``None`` with an explicit reason instead
    of an invented zero.
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
from collections.abc import Mapping, Sequence
import math
from dataclasses import dataclass
from typing import Any, TypedDict

SEMANTIC_FAMILIES: tuple[str, ...] = (
    "attention", "indexer", "moe_ffn", "dense_ffn", "mhc_residual", "norm_quant",
    "recurrent_gdn_kda",
)
ROLE_FAMILIES: dict[str, str] = {
    "attention_residual": "mhc_residual",
    "attn_res_prefix_point": "mhc_residual",
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
    "grouped_gate_up_act": "moe_ffn",
    "situ_quant": "norm_quant",
    "gated_rmsnorm": "norm_quant",
    "gdn_chunk": "recurrent_gdn_kda",
    "gdn_recurrent": "recurrent_gdn_kda",
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
    "norm_gate": "norm_quant",
    "mhc": "mhc_residual",
    "gdn": "recurrent_gdn_kda",
    "kda": "recurrent_gdn_kda",
}
WORK_FIELDS = ("hbm_bytes", "flops", "vector_ops", "link_bytes")
SEMANTIC_WORK_FIELDS = ("flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes")
CANONICAL_DTYPE_FAMILIES = {
    "fp8_e4m3": "fp8",
    "fp8_e5m2": "fp8",
    "fp4_e2m1": "fp4",
    "mxfp4": "fp4",
    "int8_w8a8": "int8",
}
ALGEBRAIC_WORK_UNAVAILABLE = "algebraic_work_unavailable"
BOUNDARY_LEDGER_UNAVAILABLE = "boundary_access_ledger_unavailable"
REALIZED_BELOW_BOUNDARY = "realized_hbm_below_minimal_boundary"
ROLLUP_SCOPE = "declared_semantic_subgraph_union"
COMM_UNATTRIBUTED = "unattributed"
_PERSISTENT_STATE_WRITE_FAMILIES = frozenset({"cache_store", "kv_cache_write"})


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


def _composite_shared_node_ids(groups, nodes_by_id=None) -> set[str]:
    shared: set[str] = set()
    for group in groups:
        coverage = group.get("coverage") or {}
        if isinstance(coverage, Mapping) and (
            coverage.get("composite_shared") or coverage.get("kind") == "composite_shared"
        ):
            shared.update(group.get("node_ids") or ())
    if nodes_by_id:
        for node_id, node in nodes_by_id.items():
            meta = ((node.get("operator") or {}).get("metadata") or {})
            if meta.get("composite_shared_subgraphs"):
                shared.add(node_id)
    return shared


def exclusive_family_members(groups, family_for, *, nodes_by_id=None):
    """Union members per family and reject unresolved cross-family ownership."""
    composite_shared = _composite_shared_node_ids(groups, nodes_by_id)
    by_family = defaultdict(set)
    primary_owner: dict[str, str] = {}
    for group in groups:
        family = family_for(group)
        if family is None:
            continue
        for node_id in group["node_ids"]:
            if node_id in composite_shared:
                owner = primary_owner.setdefault(node_id, family)
                if owner == family:
                    by_family[family].add(node_id)
            else:
                by_family[family].add(node_id)

    assigned = set()
    for family, members in sorted(by_family.items()):
        if assigned & members:
            raise ValueError("overlapping_semantic_family_ownership:" + family)
        assigned.update(members)
    return by_family


def _finite_work(work: Mapping[str, Any], field: str) -> bool:
    value = work.get(field)
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _node_input_records(node: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    values = node.get("values")
    if isinstance(values, Mapping) and isinstance(values.get("inputs"), Sequence):
        return values["inputs"]
    raw_inputs = node.get("inputs")
    if isinstance(raw_inputs, Sequence) and not isinstance(raw_inputs, (str, bytes)):
        return raw_inputs
    return ()


def _producer_aliases(node_id: str, node: Mapping[str, Any] | None) -> set[str]:
    aliases = {node_id}
    if "/" in node_id:
        aliases.add(node_id.split("/", 1)[1])
    if isinstance(node, Mapping):
        op_id = node.get("op_id")
        if op_id:
            aliases.add(str(op_id))
        op = node.get("operator")
        if isinstance(op, Mapping) and op.get("op_id"):
            aliases.add(str(op["op_id"]))
    return aliases


def _typed_edge_bytes(
    consumer_node: Mapping[str, Any],
    producer_id: str,
    producer_node: Mapping[str, Any] | None = None,
) -> float | None:
    """Return declared consumer read bytes from ``producer_id`` when typed values resolve them."""
    records = _node_input_records(consumer_node)
    if not records:
        return None
    aliases = _producer_aliases(producer_id, producer_node)
    matched: list[Mapping[str, Any]] = []
    all_resolved = True
    for item in records:
        if not isinstance(item, Mapping):
            all_resolved = False
            continue
        prod = str(item.get("producer") or "")
        nbytes = item.get("bytes")
        if prod in aliases:
            if not isinstance(nbytes, (int, float)) or isinstance(nbytes, bool):
                return None
            if item.get("materialized") is False:
                continue
            matched.append(item)
        elif not prod or not isinstance(nbytes, (int, float)) or isinstance(nbytes, bool):
            all_resolved = False
    if matched:
        return sum(max(0.0, float(item["bytes"])) for item in matched)
    if all_resolved:
        return 0.0
    return None


def _writes_persistent_state(node: Mapping[str, Any]) -> bool:
    """True when a node writes persistent cross-step state (e.g. KV or index cache)."""
    op = node.get("operator") if isinstance(node.get("operator"), Mapping) else {}
    family = str(node.get("family") or op.get("family") or "").strip().lower()
    if family in _PERSISTENT_STATE_WRITE_FAMILIES:
        return True
    meta = op.get("metadata")
    if not isinstance(meta, Mapping):
        return False
    if bool(meta.get("cache_visible")):
        return True
    return str(meta.get("state_value_output") or "").strip().lower() == "persistent_cache"


def _shared_external_input_reuse(
    nodes: Mapping[str, Mapping[str, Any]],
    members: Sequence[str],
    member_set: set[str],
    eliminated_reads_by_member: Mapping[str, float],
) -> float:
    """Eliminate duplicate HBM reads of the same external operand across fused members."""
    member_aliases: set[str] = set()
    for mid in member_set:
        member_aliases.update(_producer_aliases(mid, nodes.get(mid)))
    seen_external: set[tuple[str, str]] = set()
    saved = 0.0
    for consumer in members:
        records = _node_input_records(nodes[consumer])
        if not records:
            continue
        consumer_mte2 = float(nodes[consumer]["physical_work"]["mte2_bytes"])
        remaining = max(0.0, consumer_mte2 - float(eliminated_reads_by_member.get(consumer, 0.0)))
        if remaining <= 0.0:
            continue
        consumer_keys: set[tuple[str, str]] = set()
        for item in records:
            if not isinstance(item, Mapping) or item.get("materialized") is False:
                continue
            prod = str(item.get("producer") or "")
            if not prod or prod in member_aliases:
                continue
            nbytes = item.get("bytes")
            if not isinstance(nbytes, (int, float)) or isinstance(nbytes, bool) or nbytes <= 0:
                continue
            out_name = str(item.get("output") or item.get("name") or "")
            key = (prod, out_name)
            if key in consumer_keys:
                continue
            consumer_keys.add(key)
            if key in seen_external:
                credit = min(remaining, float(nbytes))
                saved += credit
                remaining -= credit
            else:
                seen_external.add(key)
    return saved


def semantic_subgraph_ledger(
    report: Mapping[str, Any], member_ids, *, external_outputs=None,
    realized_hbm_bytes: float | None = None,
) -> dict[str, Any] | None:
    """Closed-form semantic Cube/Vector work and minimal boundary HBM bytes.

    The pre-lowering semantic members are the immutable program: a selected
    native fusion replaces their physical boundary, never their algebraic work.
    Internal producer writes and consumer reads disappear from the boundary
    ledger, capped by the consumers' declared MTE2 capacity. An incomplete
    member ledger returns ``None`` rather than a partial or invented total.
    """
    nodes = {node["node_id"]: node for node in report["nodes"]}
    members = tuple(member_ids)
    if not members:
        return None
    for member in members:
        node = nodes.get(member)
        if node is None or not isinstance(node.get("deps"), (list, tuple)):
            return None
        work = node.get("physical_work")
        if not isinstance(work, Mapping):
            return None
        if not all(_finite_work(work, field) for field in SEMANTIC_WORK_FIELDS):
            return None
        if not isinstance(work.get("matrix_flops"), Mapping):
            return None

    algebraic_cube_flops = sum(
        float(nodes[member]["physical_work"]["flops"]) for member in members)
    by_dtype: dict[str, float] = defaultdict(float)
    for member in members:
        work = nodes[member]["physical_work"]
        matrix_flops = work["matrix_flops"]
        positive_matrix = {
            str(dtype): float(value) for dtype, value in matrix_flops.items() if value > 0
        }
        if positive_matrix:
            for dtype, value in positive_matrix.items():
                by_dtype[dtype] += value
            continue
        flops = float(work["flops"])
        if flops > 0:
            operator = nodes[member].get("operator")
            dtype = (operator.get("dtype") if isinstance(operator, Mapping) else None)
            if dtype:
                by_dtype[str(dtype)] += flops

    member_set = set(members)
    internal_consumers = {
        member: [consumer for consumer in members
                 if member in nodes[consumer]["deps"]]
        for member in members
    }
    if external_outputs is not None:
        ext_out = set(external_outputs)
    else:
        ext_out = {
            member for member in members
            if any(member in (node.get("deps") or ())
                   for node_id, node in nodes.items() if node_id not in member_set)
            or not internal_consumers[member]
        }

    persistent_writers = {
        member for member in members
        if _writes_persistent_state(nodes[member])
    }

    unfused_hbm = sum(
        float(nodes[member]["physical_work"]["hbm_bytes"]) for member in members)
    write_eliminable = {
        member for member in members
        if member not in ext_out
        and member not in persistent_writers
        and internal_consumers[member]
    }
    read_eliminable = {
        member for member in members
        if internal_consumers[member]
    }
    eliminated_writes = sum(
        float(nodes[member]["physical_work"]["mte3_bytes"])
        for member in write_eliminable
    )
    eliminated_reads_by_member: dict[str, float] = {}
    for consumer in members:
        producers = [dep for dep in nodes[consumer]["deps"]
                     if dep in read_eliminable]
        if not producers:
            eliminated_reads_by_member[consumer] = 0.0
            continue
        consumer_mte2 = float(nodes[consumer]["physical_work"]["mte2_bytes"])
        edge_bound = 0.0
        for producer in producers:
            typed_edge = _typed_edge_bytes(nodes[consumer], producer, nodes.get(producer))
            producer_mte3 = float(nodes[producer]["physical_work"]["mte3_bytes"])
            if typed_edge is not None:
                edge_bound += min(typed_edge, producer_mte3, consumer_mte2)
            else:
                edge_bound += min(producer_mte3, consumer_mte2)
        eliminated_reads_by_member[consumer] = min(consumer_mte2, edge_bound)
    shared_ext_reads = _shared_external_input_reuse(
        nodes, members, member_set, eliminated_reads_by_member,
    )
    eliminated_reads = sum(eliminated_reads_by_member.values()) + shared_ext_reads
    eliminated_internal = eliminated_writes + eliminated_reads

    boundary_hbm = max(0.0, unfused_hbm - eliminated_internal)
    del realized_hbm_bytes
    return {
        "algebraic_cube_flops": algebraic_cube_flops,
        "algebraic_cube_flops_by_dtype": dict(sorted(by_dtype.items())),
        "algebraic_vector_ops": sum(
            float(nodes[member]["physical_work"]["vector_ops"]) for member in members),
        "boundary_hbm": boundary_hbm,
        "boundary_breakdown_bytes": {
            "unfused_member_hbm_bytes": unfused_hbm,
            "eliminated_internal_roundtrip_bytes": unfused_hbm - boundary_hbm,
            "minimal_boundary_hbm_bytes": boundary_hbm,
        },
    }


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
    realized = None if totals is None else totals["hbm_bytes"]
    ledger = semantic_subgraph_ledger(
        report, group["node_ids"],
        external_outputs=group.get("external_outputs"),
        realized_hbm_bytes=realized,
    )
    if ledger is None:
        reasons = (*reasons, ALGEBRAIC_WORK_UNAVAILABLE, BOUNDARY_LEDGER_UNAVAILABLE)
        cube = by_dtype = vector = boundary = breakdown = None
    else:
        cube = ledger["algebraic_cube_flops"]
        by_dtype = ledger["algebraic_cube_flops_by_dtype"]
        vector = ledger["algebraic_vector_ops"]
        boundary = ledger["boundary_hbm"]
        breakdown = ledger["boundary_breakdown_bytes"]
    overhead = efficiency = None
    if realized is not None and boundary is not None:
        if realized + 1e-9 < boundary:
            reasons = (*reasons, REALIZED_BELOW_BOUNDARY)
            overhead = realized - boundary
        else:
            overhead = max(0.0, realized - boundary)
            efficiency = boundary / realized if realized > 0 else 1.0
    return SubgraphMathSummary(
        subgraph_id=str(group.get("subgraph_id")),
        subgraph_kind=str(group.get("role") or group.get("family") or "unknown"),
        semantic_family=semantic_family_of_group(group),
        phase=str(group.get("phase") or (report.get("execution") or {}).get("phase") or ""),
        layer_multiplicity=1,
        parameterization=_parameterization(report, group),
        algebraic_cube_flops=cube,
        algebraic_cube_flops_by_dtype=by_dtype,
        algebraic_vector_ops=vector,
        minimal_boundary_hbm_bytes=boundary,
        realized_hbm_bytes=realized,
        unavailable_reasons=reasons,
        parallel_comm_bytes=None if totals is None else totals["link_bytes"],
        parallel_comm_by_domain=_comm_by_domain(report, group["node_ids"]),
        boundary_breakdown_bytes=breakdown,
        fusion_hbm_overhead_bytes=overhead,
        fusion_byte_efficiency=efficiency,
    )


def compute_subgraph_math_summaries(report: Mapping[str, Any]) -> list[SubgraphMathSummary]:
    """Compute one summary per exported subgraph from the already-exported graph."""
    return [_summary(report, group) for group in report.get("subgraphs") or ()]


def _family_scopes(groups, family_for, *, allowed_members=None) -> dict[str, dict[tuple, set[str]]]:
    scopes: dict[str, dict[tuple, set[str]]] = defaultdict(lambda: defaultdict(set))
    for group in groups:
        family = family_for(group)
        if family is None:
            continue
        key = (group.get("replica"), group.get("stage"), group.get("phase"))
        members = group["node_ids"]
        if allowed_members is not None:
            allowed = allowed_members.get(family, set())
            members = [nid for nid in members if nid in allowed]
        scopes[family][key].update(members)
    return scopes


def build_subgraph_math_rollup(report: Mapping[str, Any]) -> dict[str, Any]:
    """Disjoint per-family union totals; never a whole-model total."""
    groups = list(report.get("subgraphs") or ())
    nodes_by_id = {n["node_id"]: n for n in (report.get("nodes") or ())}
    families = exclusive_family_members(
        groups, semantic_family_of_group, nodes_by_id=nodes_by_id
    )
    scopes = _family_scopes(
        groups, semantic_family_of_group, allowed_members=families
    )
    rollup: dict[str, Any] = {}
    for family in sorted(families):
        realized = 0.0
        link = 0.0
        cube = 0.0
        vector = 0.0
        boundary = 0.0
        complete = True
        reasons: list[str] = []
        for key, ids in sorted(scopes[family].items(), key=lambda item: str(item[0])):
            ordered_ids = tuple(sorted(ids))
            totals, why = effective_physical_work(report, ordered_ids)
            ledger = semantic_subgraph_ledger(report, ordered_ids)
            if totals is None:
                complete = False
                reasons.extend(why)
                break
            if ledger is None:
                complete = False
                reasons.extend((ALGEBRAIC_WORK_UNAVAILABLE, BOUNDARY_LEDGER_UNAVAILABLE))
                break
            realized += totals["hbm_bytes"]
            link += totals["link_bytes"]
            cube += ledger["algebraic_cube_flops"]
            vector += ledger["algebraic_vector_ops"]
            boundary += ledger["boundary_hbm"]
        occurrences = sum(1 for group in groups
                          if semantic_family_of_group(group) == family)
        rollup[family] = {
            "subgraph_occurrences": occurrences,
            "member_node_count": len(families[family]),
            "realized_hbm_bytes": realized if complete else None,
            "parallel_comm_bytes": link if complete else None,
            "algebraic_cube_flops": cube if complete else None,
            "algebraic_vector_ops": vector if complete else None,
            "minimal_boundary_hbm_bytes": boundary if complete else None,
            "unavailable_reasons": list(dict.fromkeys(reasons)),
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


def _dtype_peak(peaks_by_dtype: Mapping[str, Any], dtype: str):
    """Exact declared peak first, then the canonical dtype-family peak."""
    peak = peaks_by_dtype.get(dtype)
    if peak is not None:
        return peak
    return peaks_by_dtype.get(CANONICAL_DTYPE_FAMILIES.get(dtype, ""))


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
    by_dtype = summary.algebraic_cube_flops_by_dtype
    if by_dtype is not None:
        peaks_by_dtype = hardware_peaks.get("compute_peaks") or {}
        cube_flops = summary.algebraic_cube_flops
        if not by_dtype:
            if cube_flops == 0.0:
                # A pure Vector subgraph issues no Cube work: its Cube MFU is a
                # real observed zero, not an unavailable ledger.
                cube = 0.0
        else:
            by_dtype_sum = sum(float(work) for work in by_dtype.values())
            if (by_dtype_sum > 0.0 and (
                cube_flops is None
                or math.isclose(by_dtype_sum, float(cube_flops), rel_tol=1e-6, abs_tol=1e-9)
            )):
                terms = [
                    ratio(total(work), _dtype_peak(peaks_by_dtype, str(dtype)))
                    for dtype, work in by_dtype.items()
                ]
                if terms and all(term is not None for term in terms):
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
    "ALGEBRAIC_WORK_UNAVAILABLE", "BOUNDARY_LEDGER_UNAVAILABLE",
    "REALIZED_BELOW_BOUNDARY", "ROLLUP_SCOPE",
    "SEMANTIC_FAMILIES", "SubgraphEfficiencyReport", "SubgraphMathSummary",
    "attach_subgraph_math", "boundary_bytes", "build_subgraph_math_rollup",
    "compute_subgraph_math_summaries", "effective_physical_work",
    "evaluate_subgraph_efficiency", "exclusive_family_members",
    "semantic_family_of_group", "semantic_subgraph_ledger",
]
