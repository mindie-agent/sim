"""Hardware resize and candidate enumeration for topology sweeps.

[Role] Project an existing hardware declaration onto a different die count
  while preserving declared link semantics, rates and user topology overrides,
  and enumerate structurally valid parallel shardings.
[Invariants] No generated channel is copied from an old materialization.
  Derived SKUs keep their base profile and restore their own identity. Raw
  user declarations are sanitized against the same link semantics that the
  backend materializer validates. The generated machine has exactly the
  requested die count.
[Timeouts & Locks]
  Pure in-memory arithmetic; no I/O, devices, locks or background work.
[Side-Effect Policy]
  Inputs are never mutated; every returned hardware record is newly built.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from typing import Any

from mindie_sim.backends.ascend import (
    materialize as ascend_materialize,
    profiles as ascend_profiles,
)
from mindie_sim.backends.nvidia import (
    materialize as nvidia_materialize,
    profiles as nvidia_profiles,
)
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.parallel import ParallelSpec

GENERATED_CHANNELS = frozenset({"d2d_intra_4p", "d2d_intra_8p", "d2d_inter_pod"})
ACTIVE_GRAPH_CHANNELS = frozenset({
    "d2d_intra_package", "d2d_intra_node", "d2d_inter_node",
})
VALID_LINK_SEMANTICS = frozenset({
    "shared", "per_port", "per_pair", "direct", "fully_connected", "mesh",
    "switched",
})
PLACEHOLDER_SEMANTICS = frozenset({"none", "unknown", ""})
USER_CHANNEL_KEYS = frozenset({
    "effective_uni_bw_gbps", "calibrated_uni_bw_gbps", "derived_uni_bw_gbps",
    "uni_peak_bw_gbps", "peak_capacity_bytes_per_ms",
    "bw_utilization", "bw_ratio", "utilization",
    "latency_us", "alpha_us", "semantics",
    "transaction_bytes", "alignment_bytes", "protocol_overhead_bytes",
    "switch_hops", "switch_hop_latency_us", "switch_capacity_bytes_per_ms",
})
LINK_FRAMING_KEYS = (
    "transaction_bytes", "alignment_bytes", "protocol_overhead_bytes",
    "switch_hops", "switch_hop_latency_us", "switch_capacity_bytes_per_ms",
)
PRESERVED_TOPOLOGY_OVERRIDE_KEYS = (
    "intra_bw_4p_gbps", "intra_bw_8p_gbps", "superpod_size", "max_dies",
    "superpod_bw_gbps", "inter_pod_bw_gbps", "superpod_latency_us",
    "inter_pod_latency_us", "topology_preset",
)
TRANSFER_OVERRIDE_KEYS = (
    "transfer_channels", "inter_node_bw_gbps", "intra_node_bw_gbps",
    "inter_node_bw_bytes_per_s", "intra_node_bw_bytes_per_s",
    "intra_package_bw_gbps", "intra_package_bw_bytes_per_s",
    "inter_node_bw_util", "intra_node_bw_util", "intra_package_bw_util",
    "inter_node_latency_us", "intra_node_latency_us",
    "intra_package_latency_us",
)
BYTES_PER_MS_PER_GBPS = 1_000_000.0
DEFAULT_RANKS_PER_NODE = 8
A3_RANKS_PER_NODE = 16
PCP_ATTENTION_TYPES = frozenset({
    "gqa", "mla", "kda", "gdn", "dsa", "sfa", "paired_sfa",
    "projection_only",
})


def check_pos_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name}_must_be_int_ge_{minimum}:{value}")
    return int(value)


def check_pos_float(name: str, value: Any) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or float(value) <= 0.0):
        raise ValueError(f"{name}_must_be_positive_float:{value}")
    return float(value)


def _divisors(value: int) -> list[int]:
    return [item for item in range(1, value + 1) if value % item == 0]


def attention_types(model: ModelSpec) -> set[str]:
    plans = getattr(model, "layer_plans", ()) or ()
    if plans:
        kinds = {
            str(plan.attention).strip().lower()
            for plan in plans if str(plan.attention).strip()
        }
        if kinds:
            return kinds
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    declared = facts.get("attention_types")
    if isinstance(declared, Mapping):
        kinds = {str(kind).strip().lower() for kind, count in declared.items() if count}
        if kinds:
            return kinds
    return {str(getattr(model, "attention", "gqa") or "gqa").strip().lower()}


def ep_layout_valid(spec: ParallelSpec) -> bool:
    if spec.ep > spec.ep_domain or spec.ep_domain % spec.ep != 0:
        return False
    if not spec.ep_spans_pcp:
        return not (spec.ep > 1 and spec.ep <= spec.tp
                    and spec.tp % spec.ep != 0)
    local_domain = spec.tp * spec.pcp
    if spec.ep <= local_domain and local_domain % spec.ep != 0:
        return False
    if spec.ep > local_domain and spec.ep % local_domain != 0:
        return False
    return True


def extract_model_geometry(
    case_or_model: CaseSpec | ModelSpec | Mapping[str, Any] | str,
) -> dict[str, Any]:
    if isinstance(case_or_model, str):
        from mindie_sim.model_catalog import load_model
        return extract_model_geometry(load_model(case_or_model))
    if isinstance(case_or_model, CaseSpec):
        return extract_model_geometry(case_or_model.model)
    if isinstance(case_or_model, ModelSpec):
        meta = dict(case_or_model.metadata)
        plans = tuple(getattr(case_or_model, "layer_plans", ()) or ())
        gqa_kv_heads = tuple(sorted({
            int(p.num_kv_heads) for p in plans
            if str(p.attention).strip().lower() == "gqa" and int(p.num_kv_heads) > 0
        }))
        has_mla = any(str(p.attention).strip().lower() == "mla" for p in plans)
        from mindie_sim.catalog.context_parallel import is_v41_replicated_cache_model
        return {
            "num_heads": int(case_or_model.num_heads),
            "num_kv_heads": int(case_or_model.num_kv_heads
                               or case_or_model.num_heads),
            "gqa_kv_heads": gqa_kv_heads,
            "has_mla": has_mla,
            "replicated_cache_dcp_disallowed": is_v41_replicated_cache_model(case_or_model),
            "num_experts": int(case_or_model.num_experts),
            "attention": str(case_or_model.attention).strip().lower(),
            "attention_types": attention_types(case_or_model),
            "layers": int(case_or_model.layers),
            "mla_replicate": bool(meta.get("mla_replicate", False)),
        }
    if isinstance(case_or_model, Mapping):
        raw_model = (case_or_model["model"]
                     if ("model" in case_or_model
                         and isinstance(case_or_model["model"],
                                        (Mapping, ModelSpec)))
                     else case_or_model)
        if isinstance(raw_model, ModelSpec):
            return extract_model_geometry(raw_model)
        if not isinstance(raw_model, Mapping):
            raise TypeError(
                f"unsupported_model_spec_type:{type(raw_model).__name__}")
        raw_meta = (raw_model.get("metadata")
                    if isinstance(raw_model.get("metadata"), Mapping) else {})
        mla_replicate = bool(raw_model.get("mla_replicate", False)
                             or raw_meta.get("mla_replicate", False))
        info = extract_model_geometry(ModelSpec.from_mapping(raw_model))
        info["mla_replicate"] = info["mla_replicate"] or mla_replicate
        return info
    raise TypeError(
        f"unsupported_case_or_model_type:{type(case_or_model).__name__}")


def _valid_dcp_options(
    *, dp: int, tp: int, pcp: int, has_mla: bool, gqa_kv_heads: Sequence[int],
    allowed_dcps: set[int] | None,
    replicated_cache_dcp_disallowed: bool = False,
) -> list[int]:
    """Enumerate valid 3-axis DCP degrees for one ``(dp, tp, pcp)`` tuple."""
    from mindie_sim.schema.parallel import factor_dcp_axes

    if replicated_cache_dcp_disallowed:
        return [1] if (allowed_dcps is None or 1 in allowed_dcps) else []
    kv_reps = tuple(
        (tp // kv if (tp >= kv and tp % kv == 0) else 1)
        for kv in gqa_kv_heads
    )
    domain = dp * tp * pcp
    options: list[int] = []
    for dcp in _divisors(domain):
        if allowed_dcps is not None and dcp not in allowed_dcps:
            continue
        if dcp == 1:
            options.append(1)
            continue
        pcp_f, tp_g, dp_w = factor_dcp_axes(dcp, pcp=pcp, tp=tp, dp=dp)
        if (pcp % pcp_f != 0 or tp % tp_g != 0 or dp % dp_w != 0
                or pcp_f * tp_g * dp_w != dcp):
            continue
        if not has_mla:
            if not kv_reps:
                continue
            valid_gqa = True
            max_layer_tp = 1
            for kv_rep in kv_reps:
                if tp_g <= 1:
                    ltp = 1
                elif kv_rep > 1 and tp_g <= kv_rep and kv_rep % tp_g == 0:
                    ltp = tp_g
                elif kv_rep == 1:
                    ltp = 1
                else:
                    valid_gqa = False
                    break
                if ltp > max_layer_tp:
                    max_layer_tp = ltp
            if not valid_gqa or pcp_f * max_layer_tp * dp_w < dcp:
                continue
        options.append(dcp)
    return options


def enumerate_parallel_candidates(
    case_or_model: CaseSpec | ModelSpec | Mapping[str, Any] | str, *,
    num_devices: int,
    allow_dp: bool = True, allow_ep: bool = True,
    allow_pp: bool = False, allow_pcp: bool = False,
    allow_dcp: bool = False,
    candidate_tps: Sequence[int] | None = None,
    ep_candidates: Sequence[int] | None = None,
    pp_candidates: Sequence[int] | None = None,
    pcp_candidates: Sequence[int] | None = None,
    dcp_candidates: Sequence[int] | None = None,
) -> list[dict[str, int]]:
    """Deterministically generate valid parallel shardings."""
    devices = check_pos_int("num_devices", num_devices, minimum=1)
    geometry = extract_model_geometry(case_or_model)
    num_heads, kv_heads = geometry["num_heads"], geometry["num_kv_heads"]
    gqa_kv_heads = tuple(geometry.get("gqa_kv_heads") or ())
    has_mla = bool(geometry.get("has_mla") or geometry["attention"] == "mla")
    experts = geometry["num_experts"]
    kinds = set(geometry["attention_types"])
    layers, mla_replicate = geometry["layers"], geometry["mla_replicate"]
    include_dcp = bool(allow_dcp or dcp_candidates is not None)
    all_divs = _divisors(devices)
    allowed_tps = ({check_pos_int("candidate_tp", item, minimum=1)
                    for item in candidate_tps}
                   if candidate_tps is not None else None)
    tps = ([item for item in all_divs if item in allowed_tps]
           if allowed_tps is not None else all_divs)
    allowed_eps = ({check_pos_int("ep_candidate", item, minimum=1)
                    for item in ep_candidates}
                   if ep_candidates is not None else None)
    allowed_pps = ({check_pos_int("pp_candidate", item, minimum=1)
                    for item in pp_candidates}
                   if pp_candidates is not None else None)
    allowed_pcps = ({check_pos_int("pcp_candidate", item, minimum=1)
                     for item in pcp_candidates}
                    if pcp_candidates is not None else None)
    allowed_dcps = ({check_pos_int("dcp_candidate", item, minimum=1)
                     for item in dcp_candidates}
                    if dcp_candidates is not None else None)
    candidates: list[dict[str, int]] = []
    for tp in tps:
        if num_heads > 0 and not mla_replicate:
            if num_heads % tp != 0:
                continue
            kv_list = gqa_kv_heads or (() if has_mla else ((kv_heads,) if kv_heads > 0 else ()))
            if any(kv % tp != 0 and tp % kv != 0 for kv in kv_list):
                continue
        remainder_tp = devices // tp
        pps = ([item for item in _divisors(remainder_tp)
                if item in allowed_pps and (layers <= 0 or item <= layers)]
               if allowed_pps is not None
               else [item for item in _divisors(remainder_tp)
                     if item <= max(1, layers)] if allow_pp else [1])
        for pp in pps:
            remainder_pp = remainder_tp // pp
            pcps = ([item for item in _divisors(remainder_pp)
                     if item in allowed_pcps]
                    if allowed_pcps is not None
                    else _divisors(remainder_pp) if allow_pcp else [1])
            for pcp in pcps:
                if pcp > 1 and not kinds <= PCP_ATTENTION_TYPES:
                    continue
                dp = remainder_pp // pcp
                if not allow_dp and dp != 1:
                    continue
                ep_options = ([item for item in _divisors(dp * tp * pcp)
                               if ep_layout_valid(ParallelSpec(
                                   dp=dp, tp=tp, pp=pp, ep=item, cp=1,
                                   dcp=1, pcp=pcp))]
                              if (experts > 0 and allow_ep) else [1])
                if allowed_eps is not None:
                    ep_options = [item for item in ep_options
                                  if item in allowed_eps]
                dcp_options = (
                    _valid_dcp_options(
                        dp=dp, tp=tp, pcp=pcp, has_mla=has_mla,
                        gqa_kv_heads=gqa_kv_heads or ((kv_heads,) if (kv_heads > 0 and not has_mla) else ()),
                        allowed_dcps=allowed_dcps,
                        replicated_cache_dcp_disallowed=bool(geometry.get("replicated_cache_dcp_disallowed")),
                    )
                    if include_dcp else [1]
                )
                for ep in sorted(ep_options):
                    for dcp in dcp_options:
                        row = {"dp": dp, "tp": tp, "ep": ep, "pp": pp, "pcp": pcp}
                        if include_dcp:
                            row["dcp"] = dcp
                        candidates.append(row)
    candidates.sort(key=lambda item: (item["tp"], item["dp"], item["ep"],
                                      item["pp"], item["pcp"], item.get("dcp", 1)))
    return candidates


def resolve_ranks_per_node(parsed: CaseSpec, num_devices: int) -> int:
    original = int(parsed.machine.ranks_per_node)
    if original >= 1:
        return min(num_devices, original)
    from mindie_sim.backends.ascend import ASCEND_TOPOLOGY_PRESETS
    topology = dict(getattr(parsed.machine, "topology_options", ()))
    preset = topology.get("topology_preset")
    preset_ranks = int(ASCEND_TOPOLOGY_PRESETS.get(
        str(preset), {}).get("ranks_per_node", 0) or 0)
    if preset_ranks >= 1:
        return min(num_devices, preset_ranks)
    fallback = (A3_RANKS_PER_NODE if (
        parsed.machine.arch_family == "A3"
        or parsed.machine.dies_per_package >= 2) else DEFAULT_RANKS_PER_NODE)
    return min(num_devices, fallback)


def project_transfer_channel_input(
    raw_channel: Mapping[str, Any], link: Any | None = None, *,
    materialized: bool = True,
) -> dict[str, Any] | None:
    """Project one declaration channel onto the backend's validated input ABI."""
    record = {
        str(key): deepcopy(value) for key, value in dict(raw_channel).items()
        if key in USER_CHANNEL_KEYS
    }
    record.pop("channel_semantics", None)
    if link is not None:
        raw_semantics = str(raw_channel.get("semantics") or "").strip()
        record["semantics"] = (raw_semantics
                               if raw_semantics in VALID_LINK_SEMANTICS
                               else str(link.semantics))
    else:
        raw_semantics = (raw_channel.get("semantics")
                         or raw_channel.get("channel_semantics"))
        semantics = (str(raw_semantics).strip()
                     if raw_semantics is not None else "")
        if semantics in VALID_LINK_SEMANTICS:
            record["semantics"] = semantics
        elif (not materialized and semantics
              and semantics not in PLACEHOLDER_SEMANTICS):
            record["channel_semantics"] = semantics
        else:
            record.pop("semantics", None)
    if link is not None:
        utilization = float(getattr(link, "bw_utilization", 1.0) or 1.0)
        record["latency_us"] = float(link.latency_us)
        for key in LINK_FRAMING_KEYS:
            value = getattr(link, key, None)
            if value is not None:
                record[key] = value
        if not math.isclose(utilization, 1.0):
            for key in ("effective_uni_bw_gbps", "calibrated_uni_bw_gbps",
                        "derived_uni_bw_gbps"):
                record.pop(key, None)
            peak_bytes_ms = float(link.effective_peak_capacity_bytes_per_ms)
            record["uni_peak_bw_gbps"] = peak_bytes_ms / BYTES_PER_MS_PER_GBPS
            record["peak_capacity_bytes_per_ms"] = peak_bytes_ms
            record["bw_utilization"] = utilization
        else:
            for key in ("bw_utilization", "bw_ratio", "utilization",
                        "peak_capacity_bytes_per_ms", "calibrated_uni_bw_gbps",
                        "derived_uni_bw_gbps"):
                record.pop(key, None)
            record["effective_uni_bw_gbps"] = (
                float(link.capacity_bytes_per_ms) / BYTES_PER_MS_PER_GBPS)
    elif materialized:
        if record.get("peak_capacity_bytes_per_ms") is not None:
            peak_bytes_ms = float(record["peak_capacity_bytes_per_ms"])
            for key in ("effective_uni_bw_gbps", "calibrated_uni_bw_gbps",
                        "derived_uni_bw_gbps"):
                record.pop(key, None)
            record["uni_peak_bw_gbps"] = peak_bytes_ms / BYTES_PER_MS_PER_GBPS
        if all(record.get(key) is None for key in (
                "effective_uni_bw_gbps", "uni_peak_bw_gbps",
                "calibrated_uni_bw_gbps", "derived_uni_bw_gbps",
                "peak_capacity_bytes_per_ms")):
            return None
    return record or None


def _first_link_by_channel(machine: Hardware) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for link in machine.links:
        if link.channel in ACTIVE_GRAPH_CHANNELS and link.channel not in result:
            result[link.channel] = link
    return result


def materialize_machine_for_devices(
    parsed: CaseSpec, raw_case: CaseSpec | Mapping[str, Any], num_devices: int,
) -> Hardware:
    """Materialize a Hardware descriptor for exactly ``num_devices`` dies."""
    original_count = len(parsed.machine.dies)
    if original_count == num_devices:
        return parsed.machine
    base_device = deepcopy(parsed.machine.dies[0].to_dict())
    for key in ("rank", "node", "package"):
        base_device.pop(key, None)
    ranks_per_node = resolve_ranks_per_node(parsed, num_devices)
    sku = parsed.machine.sku
    assumptions = dict(getattr(parsed.machine, "assumptions", ()) or ())
    derived = sku not in ascend_profiles() and sku not in nvidia_profiles()
    base_sku = (str(assumptions.get("derivation.base") or "")
                if derived else sku)
    is_ascend = bool(base_sku and base_sku in ascend_profiles())
    is_nvidia = bool(base_sku and base_sku in nvidia_profiles())
    if derived and not (is_ascend or is_nvidia):
        raise ModelingError(f"resize_base_sku_unregistered:{sku}")
    if is_ascend or is_nvidia:
        raw_machine: dict[str, Any] = {
            "backend": "ascend" if is_ascend else "nvidia",
            "name": parsed.machine.name,
            "hardware_key": parsed.machine.hardware_key,
            "sku": base_sku,
            "num_devices": num_devices,
            "ranks_per_node": ranks_per_node,
            "dies_per_package": parsed.machine.dies_per_package,
            "device": base_device,
        }
        raw_machine.update({key: deepcopy(value) for key, value in
                            getattr(parsed.machine, "topology_options", ())})
        links = _first_link_by_channel(parsed.machine)
        channels: dict[str, dict[str, Any]] = {}
        for name, value in getattr(parsed.machine, "transfer_channels", ()):
            if name in GENERATED_CHANNELS:
                continue
            projected = project_transfer_channel_input(
                dict(value), links.get(name), materialized=True)
            if projected is not None:
                channels[name] = projected
        for name, link in links.items():
            if name in channels:
                continue
            projected = project_transfer_channel_input({}, link,
                                                       materialized=True)
            if projected is not None:
                channels[name] = projected
        if channels:
            raw_machine["transfer_channels"] = channels
        if getattr(parsed.machine, "collective_protocols", None):
            raw_machine["collective_protocols"] = dict(
                parsed.machine.collective_protocols)
        if isinstance(raw_case, Mapping) and isinstance(
                raw_case.get("machine"), Mapping):
            source_machine = raw_case["machine"]
            for key in PRESERVED_TOPOLOGY_OVERRIDE_KEYS:
                if key in source_machine and source_machine[key] is not None:
                    raw_machine[key] = deepcopy(source_machine[key])
            if "links" not in source_machine:
                for key in TRANSFER_OVERRIDE_KEYS:
                    if key in source_machine and source_machine[key] is not None:
                        raw_machine[key] = deepcopy(source_machine[key])
                source_channels = source_machine.get("transfer_channels")
                if isinstance(source_channels, Mapping):
                    for name, value in source_channels.items():
                        if name in GENERATED_CHANNELS or not isinstance(
                                value, Mapping):
                            continue
                        projected = project_transfer_channel_input(
                            value, links.get(name), materialized=False)
                        if projected is not None:
                            channels[name] = projected
                if channels:
                    raw_machine["transfer_channels"] = channels
        materializer = ascend_materialize if is_ascend else nvidia_materialize
        materialized = dict(materializer(raw_machine))
        if derived:
            materialized.update({
                "name": parsed.machine.name,
                "sku": sku,
                "hardware_key": parsed.machine.hardware_key,
                "assumptions": assumptions,
            })
        return Hardware.from_mapping(materialized)

    machine_dict = deepcopy(parsed.machine.to_dict())
    machine_dict["num_devices"] = num_devices
    machine_dict["ranks_per_node"] = ranks_per_node
    machine_dict["devices"] = [
        {**deepcopy(base_device), "rank": index}
        for index in range(num_devices)
    ]
    new_links = []
    for link in machine_dict.get("links", ()):
        copied = deepcopy(link)
        members = list(copied.get("members", ()))
        copied["members"] = (list(range(num_devices))
                             if len(members) == original_count
                             else [item for item in members
                                   if item < num_devices])
        if len(copied["members"]) > 1 or num_devices == 1:
            new_links.append(copied)
    if not new_links and num_devices > 1 and machine_dict.get("links"):
        fallback = deepcopy(machine_dict["links"][0])
        fallback["members"] = list(range(num_devices))
        new_links.append(fallback)
    machine_dict["links"] = new_links
    return Hardware.from_mapping(machine_dict)
