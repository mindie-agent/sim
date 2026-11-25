"""Communication local-domain resolution, recipe synthesis and binding.

[Role] Bind normalized collective observations into calibrated transport recipes.
[Invariants] Explicit provenance and positive physical rates.
[Timeouts & Locks] Pure in-memory binding; no locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

from typing import Mapping

from mindie_sim.schema.common import ModelingError
from .schema import A5_ARCH_FAMILY, A5_HARDWARE_KEY, A5_SKU
from .schema import (
    NATIVE_WINDOW_ADMITTED_REPEATS,
    NATIVE_WINDOW_WARMUP_CALLS,
)

_NATIVE_LOCAL_DIRECTION = {"all_gather": "read", "reduce_scatter": "send"}
_NATIVE_LOCAL_COLLECTIVES = tuple(_NATIVE_LOCAL_DIRECTION)
_NATIVE_LOCAL_COLLECTIVE_SET = frozenset(_NATIVE_LOCAL_COLLECTIVES)

_APPLICABILITY_KEYS = (
    ("platform", "family", "platform.family"),
    ("platform", "variant", "platform.variant"),
    ("platform", "sku", "platform.sku"),
    ("platform", "key", "platform.key"),
    ("platform", "arch_family", "platform.arch_family"),
    ("topology", "kind", "topology.kind"),
    ("topology", "scope", "topology.scope"),
    ("topology", "group_size", "topology.group_size"),
    ("topology", "host", "topology.host"),
    ("topology", "nodes", "topology.nodes"),
    ("topology", "nic", "topology.nic"),
    ("software", "stack", "software.stack"),
    ("software", "cann", "software.cann"),
    ("software", "execution_core", "software.execution_core"),
    ("software", "torch", "software.torch"),
    ("software", "torch_npu", "software.torch_npu"),
    ("software", "hccl_algorithm", "software.hccl_algorithm"),
    ("software", "hccl_protocol", "software.hccl_protocol"),
    ("transport_window", "kind", "transport_window.kind"),
    ("transport_window", "phase", "transport_window.phase"),
    ("transport_window", "warmup_calls", "transport_window.warmup_calls"),
    ("transport_window", "admitted_repeats", "transport_window.admitted_repeats"),
    ("transport_window", "concurrent_collectives",
     "transport_window.concurrent_collectives"),
    ("units", "work", "units.work"),
    ("units", "duration", "units.duration"),
    ("units", "rate", "units.rate"),
    ("units", "rate_scope", "units.rate_scope"),
    ("units", "active_units", "units.active_units"),
    ("units", "dtype", "units.dtype"),
    ("units", "service_work_unit", "units.service_work_unit"),
    ("shared_domain", "id", "shared_domain.id"),
    ("shared_domain", "kind", "shared_domain.kind"),
    ("shared_domain", "scope", "shared_domain.scope"),
    ("shared_domain", "members", "shared_domain.members"),
    ("shared_domain", "host", "shared_domain.host"),
    ("shared_domain", "inter_node", "shared_domain.inter_node"),
    ("shared_domain", "nic", "shared_domain.nic"),
)
_APPLICABILITY_KEY_SET = frozenset(key for _, _, key in _APPLICABILITY_KEYS) | {
    "direction", "hardware",
}
_LINEAGE_KEYS = (
    ("source", "summary_sha256", "source.summary_sha256"),
    ("source", "manifest_sha256", "source.manifest_sha256"),
    ("source", "versions_sha256", "source.versions_sha256"),
    ("source", "environment_image_id", "source.environment_image_id"),
)
_LINEAGE_KEY_SET = frozenset(key for _, _, key in _LINEAGE_KEYS)



def _flatten_sections(mapping, paths):
    flat = {}
    for outer, inner, flat_key in paths:
        section = mapping.get(outer)
        if isinstance(section, Mapping) and inner in section:
            flat[flat_key] = section[inner]
    return flat


def _applicability_sections(domain):
    if "conditions" in domain and isinstance(domain["conditions"], Mapping):
        return domain["conditions"]
    if "applicability" in domain and isinstance(domain["applicability"], Mapping):
        return domain["applicability"]
    if any(isinstance(domain.get(outer), Mapping)
           for outer, _, _ in _APPLICABILITY_KEYS):
        return domain
    return None


def _complete_applicability_identity(flat):
    """Add scalar hardware and reject a hardware/platform identity conflict."""
    hardware = flat.get("hardware")
    platform_key = flat.get("platform.key")
    platform_sku = flat.get("platform.sku")
    expected = platform_key or platform_sku
    if hardware is not None and expected is not None and str(hardware) != str(expected):
        raise ValueError("a5_local_service_hardware_identity_conflict")
    return flat


def a5_local_service_applicability_key(domain):
    """Canonical target-applicability key; source lineage is separate."""
    if not isinstance(domain, Mapping):
        raise ValueError("a5_local_service_domain_requires_mapping")
    if "applicability_key" in domain:
        return _complete_applicability_identity(
            {str(key): value for key, value in
             dict(domain["applicability_key"]).items()})
    conditions = _applicability_sections(domain)
    if conditions is not None:
        flat = _flatten_sections(conditions, _APPLICABILITY_KEYS)
        if "hardware" in conditions:
            flat["hardware"] = conditions["hardware"]
        if "direction" in conditions:
            flat["direction"] = conditions["direction"]
        return _complete_applicability_identity(flat)
    if not any(isinstance(value, Mapping) for value in domain.values()):
        flat = {str(key): value for key, value in domain.items()
                if key in _APPLICABILITY_KEY_SET}
        return _complete_applicability_identity(flat)
    return {}


def a5_local_service_lineage_key(domain):
    """Canonical calibration-source lineage, not target applicability."""
    if not isinstance(domain, Mapping):
        raise ValueError("a5_local_service_domain_requires_mapping")
    if "lineage_key" in domain:
        return {str(key): value for key, value in dict(domain["lineage_key"]).items()}
    source = domain.get("source")
    if isinstance(source, Mapping):
        flat = _flatten_sections({"source": source}, _LINEAGE_KEYS)
        if flat:
            return flat
        return {key: source.get(inner) for outer, inner, key in _LINEAGE_KEYS
                if outer == "source" and inner in source}
    if not any(isinstance(value, Mapping) for value in domain.values()):
        return {str(key): value for key, value in domain.items()
                if key in _LINEAGE_KEY_SET}
    return {}


def a5_local_service_domain_key(domain):
    """Backward-compatible merged key; require() checks the two parts apart."""
    return {**a5_local_service_applicability_key(domain),
            **a5_local_service_lineage_key(domain)}


def _declared_applicability(record):
    if not isinstance(record, Mapping):
        raise ValueError("a5_local_service_record_requires_mapping")
    declared_app = dict(record.get("applicability_key") or {})
    declared_lineage = dict(record.get("lineage_key") or {})
    if not declared_app and not declared_lineage:
        declared = dict(record.get("domain_key") or {})
        declared_app = {key: value for key, value in declared.items()
                        if key in _APPLICABILITY_KEY_SET}
        declared_lineage = {key: value for key, value in declared.items()
                            if key in _LINEAGE_KEY_SET}
    if not declared_app and not declared_lineage:
        raise ValueError("a5_local_service_record_requires_declared_domain")
    return declared_app, declared_lineage


def _presence_mismatches(declared, requested):
    missing = sorted(key for key in declared if key not in requested)
    changed = sorted(key for key in declared if key in requested
                     and requested[key] != declared[key])
    return missing, changed


def require_a5_local_service_applicability(record, domain):
    """Match the target environment only; calibration lineage is separate.

    This is the local counterpart of T06 ``service_conditions_match``:
    declared paths must be present and equal; extra requested paths and
    source lineage are not applicability.
    """
    declared_app, _declared_lineage = _declared_applicability(record)
    requested_app = a5_local_service_applicability_key(domain)
    missing, changed = _presence_mismatches(declared_app, requested_app)
    if missing or changed:
        raise ModelingError(
            "a5_local_service_domain_mismatch:applicability:"
            + ",".join(missing + changed))
    return declared_app


def require_a5_local_service_domain(record, domain):
    """Reject a reuse/supply respecification outside the measured domain.

    Applicability (target environment) and lineage (calibration source) are
    compared separately so a source hash cannot stand in for a changed runtime
    condition, and a shared source cannot hide a changed torch/CANN/algorithm.
    """
    declared_app, declared_lineage = _declared_applicability(record)
    require_a5_local_service_applicability(record, domain)
    requested_lineage = a5_local_service_lineage_key(domain)
    missing_lineage, changed_lineage = _presence_mismatches(
        declared_lineage, requested_lineage)
    if missing_lineage or changed_lineage:
        raise ModelingError(
            "a5_local_service_domain_mismatch:lineage:"
            + ",".join(missing_lineage + changed_lineage))
    return {**declared_app, **declared_lineage}


def _native_local_domain(source_identity, *, group_size, stack, cann,
                         direction, dtype):
    software_versions = source_identity.get("python_packages") or {}
    shared = {
        "id": f"ub_mesh:{source_identity['host']}:W{int(group_size)}",
        "kind": "ub_mesh", "scope": "intra_node",
        "members": int(group_size), "host": source_identity["host"],
        "inter_node": False, "nic": "na",
    }
    window = {
        "kind": "native_single_target_call", "phase": "aicpu_child",
        "warmup_calls": NATIVE_WINDOW_WARMUP_CALLS,
        "admitted_repeats": NATIVE_WINDOW_ADMITTED_REPEATS,
        "profiled_ranks": [0, int(group_size) - 1],
        "clock_domain": "device", "concurrent_collectives": False,
        "start": "unique hccl target-call marker/range",
        "end": "target-call completion",
    }
    conditions = {
        # ``hardware`` stays the T06/profile hardware key string; the
        # structured identity lives under ``platform`` so conditions can be
        # consumed by service_conditions_match without a key/type conflict.
        "hardware": A5_HARDWARE_KEY,
        "platform": {"family": A5_ARCH_FAMILY, "variant": "DT",
                     "sku": A5_SKU, "key": A5_HARDWARE_KEY,
                     "arch_family": A5_ARCH_FAMILY},
        "topology": {"kind": "ub_mesh", "scope": "intra_node",
                     "group_size": int(group_size),
                     "host": source_identity["host"], "nodes": 1, "nic": "na"},
        "software": {
            "stack": str(stack), "cann": str(cann),
            "execution_core": "aicpu",
            "torch": software_versions.get("torch"),
            "torch_npu": software_versions.get("torch_npu"),
            "hccl_algorithm": source_identity.get("hccl_algorithm"),
            "hccl_protocol": source_identity.get("hccl_protocol"),
        },
        "transport_window": window,
        "units": {"work": "bytes_per_rank_input", "duration": "ms",
                  "rate": "bytes_per_second", "rate_scope": "per_owner",
                  "active_units": int(group_size), "dtype": dtype,
                  "service_work_unit": "bytes"},
        "direction": direction,
        "shared_domain": shared,
    }
    measurement = {
        "kind": "native_target_call", "phase": "aicpu_child",
        "resource": "aicpu_child",
        "work_coordinate": "per_rank_input_bytes", "duration_unit": "ms",
        "source_kind": "native_ai_cpu_child",
        "service_semantics": "aicpu_child_collective_phase",
        "complete_collective_call": False, "complete_directed_call": False,
        "link_transit_identified": False, "hop_scaling": "not_derived",
        "algorithm_identified": False, "direction_observed": False,
    }
    domain = {
        **conditions,
        "source": source_identity,
        "measurement_boundary": measurement,
        "provenance": {
            "direction_basis": "hccl_source_template_native_direction",
            "direction_observed": False,
            "algorithm_identified": False,
            "shared_domain_basis": "declared_intrahost_ub_mesh_and_group_size",
            "transport_window_basis": "native_target_call_manifest",
        },
    }
    applicability_key = a5_local_service_applicability_key(conditions)
    lineage_key = a5_local_service_lineage_key({"source": source_identity})
    domain_key = {**applicability_key, **lineage_key}
    return domain, domain_key, applicability_key, lineage_key


def _local_service_evidence_gaps(collective, group_size):
    return [
        "link_transit_is_not_separated_from_the_aicpu_child_phase",
        "per_hop_link_price_is_not_identified",
        f"measured_only:{collective}:W{int(group_size)}:bf16:aicpu:intrahost:rack08:cann9.2.0",
        "cross_node_aiv_and_other_skus_are_unmeasured",
        "pr_variant_is_unmeasured",
        "hccl_algorithm_and_protocol_are_not_exposed_by_the_python_abi",
        "heldout_split_is_by_payload_not_by_environment",
    ]


from copy import deepcopy
from dataclasses import replace
from typing import Mapping

from mindie_sim.prediction.collectives.topology import Topology
from mindie_sim.prediction.service_regime_domains.services import ServiceCurve, service_identity
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from .schema import A5_HARDWARE_KEY
from .schema import (
    INTRA_NODE_MAX_RANKS,
    _CANONICAL_EXPANSION_MODES,
    _collective,
    _dtype,
    _explicit_pair,
    _explicit_scope,
    _mc2_callback_measurement,
    _require_a5,
)
from .schema import (
    NormalizedServiceStage,
    build_normalized_service_recipe,
    build_normalized_service_stage,
    normalized_service_duration_ms,
    service_curve_from_normalized,
)
from .ingest import (
    _fold_timings,
    _iter_json_events,
    _read_stream_value,
    _stream_json,
    _weighted_median,
)


_EXPANSION_MODES = frozenset({"aicpu", "aiv"})
_BYTE_RESOURCES = frozenset({"link", "hbm", "mte1", "mte2", "mte3", "fixpipe"})
_SUPPORTED_COLLECTIVES = frozenset({
    "all_reduce", "reduce_scatter", "all_gather", "all_to_all", "all_to_allv",
    "p2p_send_recv", "mc2_dispatch", "mc2_combine",
})
_COLLECTIVE_DIRECTIONS = {
    "all_gather": "read", "all_reduce": "send", "reduce_scatter": "send",
    "all_to_all": "send", "all_to_allv": "send", "p2p_send_recv": "directed",
    "mc2_dispatch": "dispatch", "mc2_combine": "combine",
}
_TOPOLOGY_GROUP_AXES = frozenset({
    "tp", "ep", "dp", "cp", "dcp", "pcp", "dcp_tp", "dcp_pcp", "dcp_dp",
    "dcp_a2a", "etp", "local_tp",
})
_GROUP_SIZE_META_KEYS = (
    "world_size", "comm_group_size", "group_size", "ranks", "group_ranks",
)
_PER_RANK_INPUT_COORDINATES = frozenset({
    "per_rank_input_bytes", "bytes_per_rank_input",
})
_DIRECT_LINK_COORDINATES = frozenset({
    "link_bytes", "wire_bytes",
})


def __getattr__(name: str):
    if name in {"_A5CommPlan", "build_a5_comm_recipe"}:
        import sys; _recipes = sys.modules[__name__]
        value = getattr(_recipes, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _curve_work_coordinate(curve, measurement=None) -> str:
    if isinstance(measurement, Mapping) and measurement.get("work_coordinate"):
        return str(measurement["work_coordinate"])
    if curve is None:
        return ""
    provenance = getattr(curve, "provenance", None) or {}
    identity = getattr(curve, "identity", None) or {}
    for container in (provenance, identity):
        boundary = container.get("measurement_boundary")
        if isinstance(boundary, Mapping) and boundary.get("work_coordinate"):
            return str(boundary["work_coordinate"])
    declared = str(provenance.get("work_scope") or identity.get("work_scope") or "")
    if declared:
        return declared
    if getattr(curve, "resource", None) == "aicpu_child":
        return "per_rank_input_bytes"
    return ""


def _wire_factor(collective, world_size, curve=None, *, measurement=None):
    """Captured payload per rank -> the declared collective's physical work.

    This is a coordinate conversion for the ring/tree recipe, not a hop price.
    A whole-collective completion envelope is never converted into a per-hop
    link rate by this factor; its provenance records
    ``link_transit_identified=False``.
    """
    rank_count = max(1, int(world_size or 1))
    name = _collective(collective)
    if rank_count <= 1:
        return 1.0
    coord = _curve_work_coordinate(curve, measurement)
    if coord in _DIRECT_LINK_COORDINATES:
        return 1.0
    if name == "all_gather":
        # Per-rank input shard (N/W) -> complete ring/mesh remote wire traffic (W - 1) * N/W.
        return float(rank_count - 1)
    if name == "reduce_scatter":
        # Native captures record full per-rank input bytes (N), so directional
        # remote traffic is (W - 1)/W * N. Legacy campaign rows record the local
        # output shard (N/W), whose remote traffic is (W - 1) * (N/W).
        if coord in _PER_RANK_INPUT_COORDINATES:
            return float(rank_count - 1) / rank_count
        return float(rank_count - 1)
    if name == "all_reduce":
        return 2.0 * (rank_count - 1) / rank_count
    if name == "all_to_all":
        return float(rank_count - 1) / rank_count
    if name == "all_to_allv":
        if coord in _PER_RANK_INPUT_COORDINATES:
            return float(rank_count - 1) / rank_count
        provenance = getattr(curve, "provenance", None) or {}
        identity = getattr(curve, "identity", None) or {}
        if (dict(provenance).get("derived_from") == "all_to_all"
                or dict(identity).get("derived_from") == "all_to_all"):
            return float(rank_count - 1) / rank_count
        return 1.0
    if name in {"mc2_dispatch", "mc2_combine"}:
        # MC2 comm-lane rows already record directional link_bytes.
        return 1.0
    return 1.0


def _fit_bundle(ingested=None):
    from .calibration import fit_a5_service_curves
    return fit_a5_service_curves(ingested)


def _resolve_bundle(bundle_or_case, case_or_op, op):
    if op is None:
        bundle, case, operation = _fit_bundle(), bundle_or_case, case_or_op
    else:
        bundle, case, operation = bundle_or_case, case_or_op, op
    if not hasattr(bundle, "lookup_curve"):
        bundle = _fit_bundle(bundle)
    return bundle, case, operation


def _validate_hardware(source, *, identity=False):
    source = dict(source or {})
    if identity:
        keys = (source.get("key"), source.get("sku"))
        architecture = source.get("architecture")
    else:
        keys = (source.get("hardware_key"), source.get("sku"))
        architecture = source.get("arch_family")
    for key in keys:
        if key or architecture:
            _require_a5(key or A5_HARDWARE_KEY, architecture)


def _require_case_a5(parsed):
    machine = parsed.machine
    _require_a5(machine.hardware_key or machine.sku, machine.arch_family)
    if machine.hardware_key:
        _require_a5(machine.hardware_key, machine.arch_family)
    if machine.sku:
        _require_a5(machine.sku, machine.arch_family)


def _group_size_from_metadata(metadata: Mapping) -> int | None:
    for key in _GROUP_SIZE_META_KEYS:
        value = metadata.get(key)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            if value:
                return len(value)
            continue
        size = int(value)
        if size > 0:
            return size
    return None


def _explicit_rank_list(metadata: Mapping, topo: Topology) -> tuple[int, ...] | None:
    for key in ("group_ranks", "ranks"):
        raw = metadata.get(key)
        if isinstance(raw, (list, tuple)) and raw:
            if all(isinstance(r, int) and not isinstance(r, bool) and r in topo._rank_node for r in raw):
                return tuple(int(r) for r in raw)
    return None


def _resolved_group_ranks(case, op, metadata: Mapping, *, device=None) -> tuple[Topology, tuple[int, ...]] | tuple[None, ()]:
    dies = tuple(getattr(case.machine, "dies", ()) or ())
    if not dies:
        return None, ()
    parallel = case.runtime.parallel
    stage = int(metadata.get("source_stage", getattr(op, "stage", 0) or 0))
    width = max(1, int(parallel.tp) * int(parallel.pcp))
    stage_span = max(1, int(parallel.pp) * width)
    selected = device if device is not None else dies[stage * width]
    die_idx = next(
        (i for i, die in enumerate(dies) if die is selected or die.rank == selected.rank),
        stage * width,
    )
    replica = die_idx // stage_span
    total_replicas = max(1, int(parallel.dp) * int(getattr(parallel, "external_dp", 1) or 1))
    max_replicas = min(total_replicas, len(dies) // stage_span)
    topo = Topology(case.machine, parallel)
    explicit_ranks = _explicit_rank_list(metadata, topo)
    if explicit_ranks is not None:
        return topo, explicit_ranks
    axis = metadata.get("group_axis", "tp")
    if axis == "pp" or str(metadata.get("collective", op.family)).strip().lower() == "pp_p2p":
        dest_stage = (
            int(metadata["destination_stage"])
            if "destination_stage" in metadata
            else stage + 1
        )
        if 0 <= stage < parallel.pp and 0 <= dest_stage < parallel.pp and dest_stage != stage:
            pairs = tuple(
                (src, dst)
                for src, dst in zip(topo.ranks(replica, stage), topo.ranks(replica, dest_stage))
                if src != dst
            )
            matched = [pair for pair in pairs if selected.rank in pair]
            if matched or pairs:
                return topo, matched[0] if matched else pairs[0]
        return topo, ()
    if 0 <= replica < max_replicas and 0 <= stage < parallel.pp and axis in _TOPOLOGY_GROUP_AXES:
        if axis != "local_tp" or metadata.get("local_tp_width"):
            groups = tuple(topo.groups(axis, replica, stage, metadata=metadata))
            if groups:
                matched = [grp for grp in groups if selected.rank in grp]
                chosen = matched[0] if matched else groups[0]
                return topo, tuple(int(rank) for rank in chosen)
    return topo, ()


def _world_size(case, op, *, device=None):
    metadata = dict(op.metadata)
    explicit = _group_size_from_metadata(metadata)
    if explicit is not None:
        return explicit
    if case is not None:
        _topo, ranks = _resolved_group_ranks(case, op, metadata, device=device)
        if ranks:
            return len(ranks)
        axis = metadata.get("group_axis")
        parallel = case.runtime.parallel
        if axis and hasattr(parallel, axis):
            return int(getattr(parallel, axis))
        return int(parallel.tp * parallel.cp or len(case.machine.dies) or 1)
    return INTRA_NODE_MAX_RANKS


def _placement_for(case, op, collective, world_size, pair, *, device=None):
    del collective
    metadata = dict(op.metadata)
    if "placement_scope" in metadata and metadata["placement_scope"] is not None:
        explicit = _explicit_scope(metadata["placement_scope"])
        if explicit is None:
            raise ModelingError(
                f"a5_placement_scope_unknown:{metadata['placement_scope']}")
        return explicit
    if pair is not None:
        norm_pair = _explicit_pair(pair) or str(pair).strip()
        if case is not None and getattr(case.machine, "dies", ()):
            topo = Topology(case.machine, case.runtime.parallel)
            parts = norm_pair.split("->", 1)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                src, dst = int(parts[0]), int(parts[1])
                if src in topo._rank_node and dst in topo._rank_node:
                    if topo.location(src)["node"] == topo.location(dst)["node"]:
                        return "intra_node_ub"
                    return "cross_node_urma_far" if norm_pair == "0->24" else "cross_node_urma"
        if norm_pair == "0->24":
            return "cross_node_urma_far"
        if norm_pair == "0->1":
            return "intra_node_ub"
        return "cross_node_urma" if int(world_size) > INTRA_NODE_MAX_RANKS else "intra_node_ub"
    if case is not None:
        topo, ranks = _resolved_group_ranks(case, op, metadata, device=device)
        if topo is not None and ranks and len(ranks) == int(world_size):
            nodes = {topo.location(rank)["node"] for rank in ranks}
            if len(nodes) > 1:
                return "cross_node_urma"
            linked_ranks = {int(rank) for link in case.machine.links for rank in link.members}
            for idx, src in enumerate(ranks):
                dst = ranks[(idx + 1) % len(ranks)]
                if src != dst and src in linked_ranks and dst in linked_ranks:
                    if any(topo.link_scope(link_name) == "cross_node"
                           for link_name, _, _ in topo.route(src, dst)):
                        return "cross_node_urma"
            return "intra_node_ub"
    if int(world_size) > INTRA_NODE_MAX_RANKS:
        return "cross_node_urma"
    return "intra_node_ub"


def _resource_for(op, metadata, collective):
    explicit = metadata.get("a5_service_resource") or metadata.get("service_resource")
    if explicit:
        return str(explicit)
    if op.link_bytes or collective in _SUPPORTED_COLLECTIVES:
        return "link"
    return "hbm" if op.hbm_bytes else "link"


def _measurement_boundary(curve, collective):
    """Preserve the measured phase; never relabel it as another boundary."""
    declared = curve.provenance.get("measurement_boundary")
    descriptor = dict(declared) if isinstance(declared, Mapping) else {}
    semantics = curve.provenance.get("service_semantics")
    if not descriptor:
        if collective in {"mc2_dispatch", "mc2_combine"}:
            direction = "dispatch" if collective == "mc2_dispatch" else "combine"
            descriptor = _mc2_callback_measurement(direction)
        elif semantics == "aicpu_child_collective_phase" or curve.resource == "aicpu_child":
            phase, complete = "aicpu_child", False
        elif curve.boundary == "resource_phase" or curve.resource == "collective_phase":
            phase, complete = "collective_phase", False
        elif collective == "p2p_send_recv":
            phase, complete = "directed_p2p_call", False
        else:
            phase, complete = "collective_call", True
        if not descriptor:
            descriptor = {
                "kind": "native_target_call" if phase == "aicpu_child" else "device_call",
                "phase": phase, "resource": curve.resource,
                "work_coordinate": ("per_rank_input_bytes" if phase == "aicpu_child"
                                    else "captured_per_rank_payload"),
                "duration_unit": "ms",
                "source_kind": ("native_ai_cpu_child" if phase == "aicpu_child"
                                else "campaign_device_call"),
                "service_semantics": semantics or (
                    "aicpu_child_collective_phase" if phase == "aicpu_child" else
                    "directed_p2p_call" if phase == "directed_p2p_call" else
                    "collective_phase_service" if phase == "collective_phase" else
                    "whole_collective_call_envelope"),
                "complete_collective_call": complete,
                "complete_directed_call": phase == "directed_p2p_call",
                "link_transit_identified": False, "hop_scaling": "not_derived",
                "algorithm_identified": False,
                "direction_observed": phase == "directed_p2p_call",
            }
    descriptor.setdefault("complete_collective_call", False)
    descriptor.setdefault("complete_directed_call", False)
    descriptor.setdefault("link_transit_identified", False)
    descriptor.setdefault("hop_scaling", "not_derived")
    descriptor.setdefault("direction_observed", False)
    descriptor.setdefault("algorithm_identified", False)
    descriptor.setdefault("work_coordinate", "captured_per_rank_payload")
    descriptor.setdefault("phase", "unknown")
    descriptor.setdefault("service_semantics", semantics)
    return descriptor


def _consumer_role(descriptor):
    phase = descriptor.get("phase")
    if phase == "aicpu_child":
        return "aicpu_child_completion_floor"
    if phase in {"mc2_dispatch", "mc2_combine"}:
        return f"{phase}_callback_envelope"
    if phase == "collective_phase":
        return "resource_phase_envelope"
    if phase == "directed_p2p_call":
        return "directed_p2p_call_envelope"
    if phase == "collective_call":
        return "collective_call_completion_envelope"
    return "measured_call_envelope"


def _require_complete_call(descriptor, required):
    if required and not descriptor.get("complete_collective_call"):
        raise ModelingError(
            "a5_comm_measurement_boundary_cannot_serve_complete_collective_call:"
            + str(descriptor.get("phase") or "unknown"))


def _bind_to_bundle(bundle, case, op, scope="none", *, device=None,
                    resource=None, boundary=None, active_units=None,
                    require_complete_call=False, production=False,
                    **lookup_kwargs):
    if production:
        raise ModelingError("measured_a5_call_binding_is_diagnostic_only")
    parsed = (case if hasattr(case, "machine") and not isinstance(case, Mapping)
              else parse_case(case))
    _require_case_a5(parsed)
    metadata = dict(op.metadata)
    if device is None:
        device = parsed.machine.dies[
            op.stage * parsed.runtime.parallel.tp * parsed.runtime.parallel.pcp]
    selectors = dict(lookup_kwargs)
    explicit = {key: selectors.get(key) for key in (
        "world_size", "expansion_mode", "placement_scope", "pair",
        "collective", "dtype", "stack", "cann", "algorithm",
        "protocol") if selectors.get(key) is not None}
    hardware_key = selectors.pop("hardware_key", None)
    sku = selectors.pop("sku", None)
    arch_family = selectors.pop("arch_family", None) or parsed.machine.arch_family
    _validate_hardware({"hardware_key": hardware_key, "sku": sku,
                        "arch_family": arch_family})
    hardware_key = (hardware_key or sku or parsed.machine.hardware_key
                    or parsed.machine.sku)
    placement = selectors.pop("placement_scope", None)
    world = selectors.pop("world_size", None) or _world_size(parsed, op, device=device)
    mode = selectors.pop("expansion_mode", None) or metadata.get(
        "expansion_mode", metadata.get("expansion_regime", "unspecified"))
    dtype = selectors.pop("dtype", None) or _dtype(
        metadata.get("comm_dtype") or op.dtype or parsed.model.compute_dtype)
    pair = selectors.pop("pair", None) or _explicit_pair(metadata.get("p2p_pair"))
    collective = _collective(
        selectors.pop("collective", None) or metadata.get("collective")
        or metadata.get("a5_collective") or op.family)
    if collective not in _SUPPORTED_COLLECTIVES:
        collective = _collective(metadata.get("kernel") or op.recipe)
    if collective not in _SUPPORTED_COLLECTIVES:
        raise ModelingError(f"a5_curve_unavailable:{collective}")
    placement = placement or _placement_for(
        parsed, op, collective, world, pair, device=device)
    boundary = (boundary or selectors.pop("boundary", None)
                or ("resource_phase" if metadata.get("device_kernel_group")
                    else "device_kernel"))
    for key in ("stack", "cann", "algorithm", "protocol"):
        if selectors.get(key) is None and metadata.get(key) is not None:
            selectors[key] = metadata[key]
    curve = bundle.lookup_curve(
        collective, world, placement_scope=placement, dtype=dtype,
        expansion_mode=mode, pair=pair, hardware_key=hardware_key,
        arch_family=arch_family, boundary=boundary, **selectors)
    bound_metadata = dict(metadata)
    if "world_size" in explicit:
        bound_metadata["world_size"] = int(world)
        for alias in ("comm_group_size", "group_size", "ranks", "group_ranks", "steps"):
            bound_metadata.pop(alias, None)
    if "expansion_mode" in explicit:
        bound_metadata["expansion_mode"] = mode
        if "expansion_regime" in bound_metadata:
            bound_metadata["expansion_regime"] = mode
    for k, v in (("placement_scope", placement), ("pair", pair),
                 ("collective", collective)):
        if k in explicit:
            bound_metadata["p2p_pair" if k == "pair" else k] = v
    if "collective" in explicit and "a5_collective" in bound_metadata:
        bound_metadata["a5_collective"] = collective
    for key in ("algorithm", "protocol", "stack", "cann"):
        if key in explicit:
            bound_metadata[key] = selectors[key]
    if "dtype" in explicit and "comm_dtype" in bound_metadata:
        bound_metadata["comm_dtype"] = dtype
    bound_op = op
    if bound_metadata != metadata:
        bound_op = replace(op, metadata=tuple(sorted(bound_metadata.items())))
    if "dtype" in explicit and dtype != bound_op.dtype:
        bound_op = replace(bound_op, dtype=dtype)
    identity = service_identity(parsed, bound_op, scope, device=device)
    resource = resource or _resource_for(op, metadata, collective)
    if resource not in _BYTE_RESOURCES:
        raise ModelingError(f"a5_comm_curve_requires_byte_resource:{resource}")
    unit = "bytes"
    units = active_units or device.units(resource)
    measurement = _measurement_boundary(curve, collective)
    _require_complete_call(measurement, require_complete_call)
    link_wire_factor = _wire_factor(collective, world, curve, measurement=measurement)
    wire_factor = link_wire_factor if resource == "link" else 1.0
    knots = curve.knots
    source_semantics = (curve.provenance.get("service_semantics")
                        or measurement.get("service_semantics")
                        or ("directed_p2p_call" if collective == "p2p_send_recv"
                            else "whole_collective_call_envelope"))
    provenance = {
        **curve.provenance, "bound_a5_curve": curve.model_id,
        "boundary": boundary, "bound_stack": curve.identity.get("stack"),
        "bound_metadata": dict(bound_metadata),
        "measurement_boundary": measurement,
        "measured_resource": measurement.get("resource"),
        "consumption_resource": resource,
        "source_service_semantics": source_semantics,
        "service_semantics": source_semantics,
        "consumer_role": _consumer_role(measurement),
        "complete_collective_call_response": bool(
            measurement.get("complete_collective_call")),
        "complete_directed_call_response": bool(
            measurement.get("complete_directed_call")),
        "direction": curve.provenance.get(
            "direction", _COLLECTIVE_DIRECTIONS.get(collective, "unspecified")),
        "direction_observed": bool(measurement.get("direction_observed", False)),
        "algorithm_identified": bool(measurement.get("algorithm_identified", False)),
        "shared_domain": curve.provenance.get("shared_domain"),
        "transport_window": curve.provenance.get("transport_window"),
        "work_scope": measurement.get(
            "work_coordinate", "captured_per_rank_payload"),
        # A measured phase/call is a completion envelope, not a per-hop transit
        # price and not a licence to compose an invented hop count.
        "link_transit_identified": bool(
            measurement.get("link_transit_identified", False)),
        "hop_scaling": measurement.get("hop_scaling", "not_derived"),
    }
    if link_wire_factor != 1.0:
        provenance["link_wire_factor"] = link_wire_factor
    if wire_factor != 1.0:
        knots = tuple((float(work) * wire_factor, float(duration))
                      for work, duration in curve.knots)
        provenance["raw_knots"] = curve.knots
        provenance["wire_factor"] = wire_factor
        provenance["wire_coordinate_semantics"] = (
            "captured per-rank payload converted to the declared collective's "
            "physical work coordinate; this is not a hop price")
    return ServiceCurve(resource, unit, identity, boundary, units,
                        curve.startup_ms, knots, provenance)


def bind_service_curve(bundle_or_case, case_or_op, op=None, scope="none",
                       production=False, **kwargs):
    """Bind one calibrated curve to a case/operator; the bundle argument is optional."""
    bundle, case, operation = _resolve_bundle(bundle_or_case, case_or_op, op)
    return _bind_to_bundle(bundle, case, operation, scope,
                           production=production, **kwargs)


def _regime_from_bundle(bundle, case, op, scope="none", *, regime_id=None,
                        coordinate="link", outside="analytic",
                        require_complete_call=False, production=False,
                        **lookup_kwargs):
    if production:
        raise ModelingError("to_service_regime_is_diagnostic_only")
    if coordinate not in _BYTE_RESOURCES:
        raise ModelingError(f"a5_comm_curve_requires_byte_resource:{coordinate}")
    if (lookup_kwargs.get("boundary") not in (None, "device_kernel")
            or dict(op.metadata).get("device_kernel_group")):
        raise ModelingError("service_regime_requires_device_kernel_boundary")
    curve = _bind_to_bundle(bundle, case, op, scope,
                            require_complete_call=require_complete_call,
                            **lookup_kwargs)
    if curve.boundary != "device_kernel":
        raise ModelingError("service_regime_requires_device_kernel_boundary")
    identity = curve.identity
    match = {"hardware": identity["hardware"]["key"], "family": op.family,
             **{key: identity[key] for key in ("runtime", "revision", "format",
                                               "dtype", "layout", "replay_scope",
                                               "implementation")}}
    if "transport" in identity:
        match["transport"] = deepcopy(identity["transport"])
    if op.recipe:
        match["recipe"] = op.recipe
    else:
        match["kernel"] = identity["kernel"]
    metadata = dict(curve.provenance.get("bound_metadata") or op.metadata)
    for key in ("collective", "a5_collective", "comm_dtype", "group_axis", "steps",
                "world_size", "comm_group_size", "group_size", "ranks",
                "group_ranks", "placement_scope", "p2p_pair", "expansion_mode",
                "expansion_regime", "algorithm", "protocol", "stack", "cann"):
        if key in metadata:
            match[f"metadata.{key}"] = metadata[key]
    raw_knots = curve.provenance.get("raw_knots") or curve.knots
    wire_factor = float(
        curve.provenance.get("wire_factor")
        or curve.provenance.get("link_wire_factor")
        or 1.0
    )
    knots = [[float(work), float(duration * 1000.0)]
             for work, duration in raw_knots]
    coord_spec = {"path": f"work.{coordinate}"}
    if coordinate == "link" and wire_factor != 1.0:
        coord_spec["scale"] = 1.0 / wire_factor
    return {
        "id": regime_id or (f"a5_comm:{op.family}:{op.recipe or identity['kernel']}:"
                            f"{curve.model_id[:12]}"),
        "match": match,
        "coordinates": {"bytes": coord_spec},
        "domain": {"bytes": {"min": knots[0][0], "max": knots[-1][0]}},
        "response": {"kind": "piecewise_linear", "coordinate": "bytes",
                     "knots": knots, "outside": outside},
        "provenance": {
            "boundary": ("measured A5 "
                         + str(curve.provenance.get("measurement_boundary", {}).get(
                             "phase", "unknown")) + " response"),
            # T20 C07: this rule is a source-bearing diagnostic projection of
            # one measured curve.  Ordinary production pricing uses physical
            # work + normalized services, not an injected old whole-call rule.
            "diagnostic_only": True,
            "source_bearing": True,
            "production_eligible": False,
            "model_id": curve.model_id,
            "stack": curve.provenance.get("bound_stack"),
            "measurement_boundary": curve.provenance.get("measurement_boundary"),
            "measured_resource": curve.provenance.get("measured_resource"),
            "consumption_resource": curve.provenance.get("consumption_resource"),
            "source_service_semantics": curve.provenance.get(
                "source_service_semantics"),
            "service_semantics": curve.provenance.get("service_semantics"),
            "consumer_role": curve.provenance.get("consumer_role"),
            "complete_collective_call_response": bool(curve.provenance.get(
                "complete_collective_call_response", False)),
            "complete_directed_call_response": bool(curve.provenance.get(
                "complete_directed_call_response", False)),
            "direction": curve.provenance.get("direction"),
            "direction_observed": bool(curve.provenance.get(
                "direction_observed", False)),
            "direction_basis": curve.provenance.get("direction_basis"),
            "algorithm_identified": bool(curve.provenance.get(
                "algorithm_identified", False)),
            "shared_domain": curve.provenance.get("shared_domain"),
            "transport_window": curve.provenance.get("transport_window"),
            "work_scope": curve.provenance.get("work_scope"),
            "link_transit_identified": bool(curve.provenance.get(
                "link_transit_identified", False)),
            "hop_scaling": curve.provenance.get("hop_scaling", "not_derived"),
            "coordinate_basis": ("captured_per_rank_payload_not_link_transit"
                                 if coordinate == "link" else
                                 f"captured_{coordinate}_work"),
        },
    }


def to_service_regime(bundle_or_case, case_or_op, op=None, scope="none",
                      require_complete_call=False, production=False, **kwargs):
    """Return a ``service_regimes`` rule bound to the calibrated response."""
    bundle, case, operation = _resolve_bundle(bundle_or_case, case_or_op, op)
    return _regime_from_bundle(
        bundle, case, operation, scope,
        require_complete_call=require_complete_call,
        production=production, **kwargs)


def _duration_for(curve, lookup_bytes):
    estimate = curve.estimate(lookup_bytes)
    if estimate is not None:
        return estimate.duration_ms
    if lookup_bytes < curve.knots[0][0]:
        x0, t0 = curve.knots[0]
        return curve.startup_ms + (t0 - curve.startup_ms) * max(
            0.0, lookup_bytes) / x0
    x0, t0 = curve.knots[-2]
    x1, t1 = curve.knots[-1]
    slope = max((t1 - t0) / (x1 - x0), (t1 - curve.startup_ms) / x1)
    return t1 + slope * (lookup_bytes - x1)


from dataclasses import dataclass, field, replace
from typing import Mapping

from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import KernelProgram, Stage
from mindie_sim.schema.common import ModelingError
from .schema import A5_HARDWARE_KEY
from .schema import (
    INTRA_NODE_MAX_RANKS,
    _collective,
    _dtype,
    _explicit_pair,
    _mode,
)

DEFAULT_ACTIVE_AIV_BLOCKS = 16
_DTYPE_ELEMENT_BYTES = {
    "int8": 1.0,
    "fp8-e4m3": 1.0,
    "fp32": 4.0,
    "int32": 4.0,
}
_DEFAULT_ELEMENT_BYTES = 2.0



@dataclass(frozen=True)
class _A5CommPlan(OperatorPlan):
    """A physical communication program with a measured completion floor.

    ``metadata`` remains the ordinary engine selector record.  The measured
    phase and consumer role are typed attributes so a consumer can require a
    complete collective call without parsing free text or a filename.
    """

    minimum_ms: float = 0.0
    measurement_boundary: Mapping = field(default_factory=dict)
    consumer_role: str = ""
    source_service_semantics: str = ""
    complete_collective_call_response: bool = False
    complete_directed_call_response: bool = False
    direction_observed: bool = False
    direction_basis: str = ""
    algorithm_identified: bool = False
    link_transit_identified: bool = False
    hop_scaling: str = "not_derived"
    diagnostic_only: bool = False
    whole_call_floor: bool = False
    production_eligible: bool = False

    def lower(self, context):
        if context.service_boundary != "device_kernel":
            raise ModelingError("service_curve_boundary_mismatch")
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("body", after=("setup",),
                  program=self.program(replace(context, setup_ms=0, minimum_ms=0))),
        ), minimum_ms=max(context.minimum_ms, self.minimum_ms))


def _a5_collective_program(context, *, hbm, mte2, mte3, vector, duration,
                           op_mode, active_aiv_blocks):
    _validate_hardware(
        context.service_identity.get("hardware", {}), identity=True)
    link_work = {res: val for res, val in context.work.items()
                 if res not in context.resource_names.values()}
    stages = [Stage("link", link_work) if link_work
              else Stage("link", latency=duration)]
    for res_name, amount in (("hbm", hbm), ("mte2", mte2), ("mte3", mte3)):
        if amount > 0:
            stages.append(Stage(
                res_name, context.bind(Work(resources={res_name: amount}))))
    if vector > 0:
        place = {}
        if op_mode == "aiv":
            count = context.resource_units.get("vector", context.core_count)
            units = tuple(range(min(max(1, int(active_aiv_blocks)), count)))
            place = {context.resource_names["vector"]: units}
        stages.append(Stage(
            "vector", context.bind(Work(resources={"vector": vector})),
            placement=place))
    return KernelProgram(tuple(stages))


def _recipe_from_bundle(bundle, collective=None, *, world_size=None,
                        case=None,
                        expansion_mode="unspecified",
                        active_aiv_blocks=DEFAULT_ACTIVE_AIV_BLOCKS,
                        allow_unspecified_fallback=False,
                        require_complete_call=False, production=False,
                        **lookup_kwargs):
    if production:
        raise ModelingError("measured_call_recipe_is_diagnostic_only")
    parsed_case = (
        None if case is None
        else case if hasattr(case, "machine") and not isinstance(case, Mapping)
        else parse_case(case)
    )
    if parsed_case is not None:
        _require_case_a5(parsed_case)
    mode = _mode(expansion_mode)
    if mode not in _CANONICAL_EXPANSION_MODES and mode != "unspecified":
        raise ModelingError(f"a5_expansion_mode_unknown:{expansion_mode}")
    spec = dict(lookup_kwargs)
    coordinate = spec.pop("coordinate", "link")
    if coordinate not in _BYTE_RESOURCES:
        raise ModelingError(f"a5_comm_curve_requires_byte_resource:{coordinate}")
    if spec.pop("boundary", None) not in (None, "device_kernel"):
        raise ModelingError("service_curve_boundary_mismatch")
    placement = spec.pop("placement_scope", None)
    hardware_key = spec.pop("hardware_key", None)
    sku = spec.pop("sku", None)
    arch_family = spec.pop("arch_family", None)
    _validate_hardware({"hardware_key": hardware_key, "sku": sku,
                        "arch_family": arch_family})
    hardware_key = hardware_key or sku

    def build(op, geometry):
        del geometry
        meta = dict(op.metadata)
        if meta.get("device_kernel_group"):
            raise ModelingError("service_curve_boundary_mismatch")
        _validate_hardware(meta)
        op_mode = mode
        if op_mode == "unspecified" and (
            meta.get("expansion_mode") or meta.get("expansion_regime")
        ):
            op_mode = _mode(
                meta.get("expansion_mode") or meta.get("expansion_regime"))
            if op_mode not in _CANONICAL_EXPANSION_MODES and op_mode != "unspecified":
                raise ModelingError(f"a5_expansion_mode_unknown:{op_mode}")
        name = _collective(
            collective or meta.get("collective")
            or meta.get("a5_collective") or op.family)
        world = world_size or _group_size_from_metadata(meta)
        if world is None and parsed_case is not None:
            world = _world_size(parsed_case, op)
        if world is None:
            if meta.get("group_axis"):
                raise ModelingError(
                    f"a5_comm_recipe_world_size_required:{meta['group_axis']}")
            world = INTRA_NODE_MAX_RANKS
        dtype = _dtype(
            spec.get("dtype") or meta.get("comm_dtype")
            or getattr(op, "dtype", None) or meta.get("dtype") or "bf16")
        pair = spec.get("pair") or _explicit_pair(meta.get("p2p_pair"))
        scope = placement or _placement_for(parsed_case, op, name, world, pair)
        software = {
            k: spec[k] if k in spec else meta.get(k)
            for k in ("stack", "cann", "algorithm", "protocol")
        }
        extra = {
            k: v for k, v in spec.items()
            if k not in {"dtype", "pair", "stack", "cann", "algorithm", "protocol"}
        }
        curve = bundle.lookup_curve(
            name, world, placement_scope=scope, dtype=dtype,
            expansion_mode=op_mode,
            allow_unspecified_fallback=allow_unspecified_fallback, pair=pair,
            hardware_key=hardware_key or A5_HARDWARE_KEY,
            arch_family=arch_family, boundary="device_kernel",
            **software, **extra)
        if curve.boundary != "device_kernel":
            raise ModelingError("service_curve_boundary_mismatch")
        measurement = _measurement_boundary(curve, name)
        _require_complete_call(measurement, require_complete_call)
        curve_mode = (curve.identity.get("expansion_regime")
                      or curve.identity.get("expansion_mode") or "unspecified")
        metadata = {
            "a5_collective": name, "world_size": int(world),
            "placement_scope": scope, "dtype": dtype,
            "requested_expansion_mode": op_mode,
            "curve_expansion_mode": curve_mode,
            "unspecified_fallback": bool(curve_mode != op_mode),
            "model_id": curve.model_id,
            **{k: software[k] if software[k] is not None else curve.identity.get(k)
               for k in ("stack", "cann", "algorithm", "protocol")},
        }
        wire_factor = (
            _wire_factor(name, world, curve, measurement=measurement)
            if coordinate == "link" else 1.0)
        raw_amount = float(
            getattr(op, coordinate + "_bytes", 0)
            or getattr(op, "link_bytes", 0)
            or getattr(op, "hbm_bytes", 0) or 1.0)
        link_bytes = float(
            getattr(op, "link_bytes", 0)
            or (getattr(op, coordinate + "_bytes", 0)
                if coordinate == "link" else 0) or 0.0)
        lookup_bytes = raw_amount / wire_factor if coordinate == "link" else raw_amount
        duration = float(_duration_for(curve, lookup_bytes) or 0.0)
        hbm, mte2, mte3 = (
            float(getattr(op, k, 0) or 0)
            for k in ("hbm_bytes", "mte2_bytes", "mte3_bytes"))
        elem_bytes = _DTYPE_ELEMENT_BYTES.get(dtype, _DEFAULT_ELEMENT_BYTES)
        vector = (
            (float(getattr(op, "vector_ops", 0) or 0)
             or max(1.0, raw_amount / elem_bytes))
            if op_mode == "aiv" else float(getattr(op, "vector_ops", 0) or 0))
        if op_mode != "aiv":
            hbm = max(1.0, hbm or raw_amount)
        resources = {
            "hbm": hbm,
            **{k: v for k, v in (
                ("vector", vector), ("link", link_bytes),
                ("mte2", mte2), ("mte3", mte3)) if v > 0},
        }
        work = Work(resources=resources)

        def program(context):
            return _a5_collective_program(
                context, hbm=hbm, mte2=mte2, mte3=mte3, vector=vector,
                duration=duration, op_mode=op_mode, active_aiv_blocks=active_aiv_blocks)

        return _A5CommPlan(
            work, program, minimum_ms=duration, metadata=metadata,
            measurement_boundary=measurement,
            consumer_role=_consumer_role(measurement),
            source_service_semantics=(
                curve.provenance.get("service_semantics")
                or measurement.get("service_semantics") or ""),
            complete_collective_call_response=bool(
                measurement.get("complete_collective_call")),
            complete_directed_call_response=bool(
                measurement.get("complete_directed_call")),
            direction_observed=bool(measurement.get("direction_observed", False)),
            direction_basis=str(curve.provenance.get("direction_basis") or ""),
            algorithm_identified=bool(
                measurement.get("algorithm_identified", False)),
            link_transit_identified=bool(
                measurement.get("link_transit_identified", False)),
            hop_scaling=measurement.get("hop_scaling", "not_derived"),
            diagnostic_only=True, whole_call_floor=True,
            production_eligible=False)

    return PlannedRecipe(build)


def build_a5_comm_recipe(bundle_or_collective=None, collective=None,
                         allow_unspecified_fallback=False,
                         require_complete_call=False, production=False,
                         **kwargs):
    """Build a PlannedRecipe from the calibrated curve; bundle is optional."""
    if hasattr(bundle_or_collective, "lookup_curve"):
        bundle, name = bundle_or_collective, collective
    else:
        bundle, name = _fit_bundle(), (collective or bundle_or_collective)
    return _recipe_from_bundle(
        bundle, name,
        allow_unspecified_fallback=allow_unspecified_fallback,
        require_complete_call=require_complete_call,
        production=production, **kwargs)
