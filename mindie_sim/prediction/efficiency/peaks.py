"""Hardware peak rates and operator work bounds.

[Role] Resolve stage and link supply and derive nominal per-pipe service bounds
  from operator reports.
[Invariants] Rates retain their declared hardware and rank boundaries. Work
  and modeled invocation costs retain their original units and provenance.
[Timeouts & Locks] In-memory lookups and arithmetic with no device locks.
[Side-Effect Policy] Input cases and reports are read-only.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Device

from ..collectives.topology import Topology
from ..native_binding import bind_native_call
from ..structure import parse_case


DEFAULT_TOKENS = 1
DEFAULT_CONTEXT = 4096
DEFAULT_SEQUENCES = 1
DEFAULT_OUTPUT_BUDGET = 1024
DEFAULT_STAGE = 0
DEFAULT_REPLICA = 0

_MS_PER_S = 1000.0
_GBPS_TO_BYTES_PER_MS = 1_000_000.0
_CACHELINE_128B_BYTES = 128.0
_VECTOR_FP16_FMA_FLOPS_PER_ISSUE = 4.0
_LAUNCH_DIAGNOSTIC_MIN_THRESHOLD_MS = 0.002
_LAUNCH_DIAGNOSTIC_SHORT_DURATION_MS = 0.006
_LAUNCH_DIAGNOSTIC_DURATION_FACTOR = 2.5
_LAUNCH_DIAGNOSTIC_MAX_WORK_SHARE = 0.25

_HBM_ROLES = frozenset({"hbm", "memory", "gm_hbm_cacheline_128b"})
_CUBE_ROLES = frozenset({"cube", "aic"})
_VEC_ROLES = frozenset({"vector", "vbitsort32", "scalar_store_issue", "ub_copy", "aiv", "reg_issue_slots", "reg_alu_lanes", "reg_special_lanes", "reg_reduce_lanes", "reg_shuffle_lanes", "reg_data_writes", "reg_ub_accesses", "reg_ub_stores"})
_DTYPE_ALIASES = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "float8_e4m3": "fp8_e4m3",
    "float8_e5m2": "fp8_e5m2",
    "hif8": "hifloat8",
}
_FP8_VARIANTS = frozenset({"fp8_e4m3", "fp8_e5m2"})
_LINK_RESOURCE_PREFIXES = ("link:", "port:", "switch:", "sdma_flow:")
_NON_OTHER_SERVICED_PREFIXES = (
    "memory@", "gm_hbm", "matrix@", "vector@", "scheduler@",
    "native_call_service", "ub_inj:",
)
_BOUNDS = ("memory_bound", "cube_bound", "vector_bound", "link_bound", "launch_bound")
_KINDS = ("cube", "vector", "cv_fused", "communication", "memory_only")
_SCALE_KEYS = ("hbm_bw_scale", "cube_flops_scale", "vector_ops_scale", "link_bw_scale", "hbm_capacity_scale")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _check_pos_float(name: str, val: Any) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or float(val) <= 0.0:
        raise ValueError(f"{name}_must_be_positive:{val}")
    return float(val)


def _validate_geometry(phase: str, tokens: int, context: int, sequences: int, scope: str) -> None:
    if phase not in {"prefill", "decode"} or scope not in {"body", "model"}:
        raise ValueError(f"efficiency_invalid_phase_or_scope:{phase}:{scope}")
    for name, val, minimum in (("tokens", tokens, 1), ("sequences", sequences, 1), ("context", context, 0)):
        if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
            raise ValueError(f"efficiency_invalid_{name}:{val}")


def resolve_efficiency_scope(case: CaseSpec | Mapping[str, Any], scope: str | None) -> str:
    """Resolve the accounted program scope; ``None`` follows the declared stack.

    An explicit ``"body"``/``"model"`` is honoured.  An omitted scope selects the
    full ``"model"`` assembly (embedding, LM head and Draft/MTP/DSpark proposal
    programs) whenever the case declares speculative decoding, so callers that
    forget the argument never silently drop the draft stage from a step.
    """
    if scope is not None:
        return scope
    from .capacity_window import _resolve_speculative_tokens

    parsed = parse_case(case)
    query_tokens, _ = _resolve_speculative_tokens(
        parsed, output_budget=None, tokens_per_seq=None, emitted_tokens_per_seq=None)
    return "model" if query_tokens > 1 else "body"


def _resolve_link_bw_bytes_per_ms(case: CaseSpec, scope: str) -> float:
    """Effective machine-wide link bandwidth in bytes/ms (slowest configured link)."""
    links = parse_case(case).machine.links
    if not links:
        return 0.0
    return float(min(link.capacity_bytes_per_ms for link in links))


def _resolve_link_peak_bw_bytes_per_ms(case: CaseSpec, scope: str) -> float:
    """Peak machine-wide link bandwidth before configured utilization is applied."""
    links = parse_case(case).machine.links
    if not links:
        return 0.0
    return float(min(
        getattr(link, "effective_peak_capacity_bytes_per_ms", link.capacity_bytes_per_ms)
        for link in links
    ))


def _role_scale_factor(role: str, *, hbm_bw_scale: float, cube_flops_scale: float, vector_ops_scale: float, link_bw_scale: float) -> float:
    """Scale only the named physical supply; MTE/L2/copy stay independent unless explicitly coupled."""
    r = str(role or "").strip()
    if r in _HBM_ROLES or r.startswith(("hbm", "gm_hbm")):
        return hbm_bw_scale
    if r in _CUBE_ROLES or r.startswith("cube"):
        return cube_flops_scale
    if r in _VEC_ROLES or r.startswith(("vector", "vreg_", "sreg_", "reg_", "ub_", "aiv")):
        return vector_ops_scale
    return link_bw_scale if (r == "link" or r.startswith("link")) else 1.0


def _die_cube_peak(die: Device, dtype: str, default_dtype: str = "bf16") -> float:
    raw = str(dtype or default_dtype or "bf16").strip()
    lower = raw.lower()
    canonical = _DTYPE_ALIASES.get(lower, lower)
    if raw and raw != canonical:
        try:
            return float(die.peak(raw))
        except ModelingError:
            pass
    return float(die.peak(canonical))


def _resolve_cube_peak(hw: Mapping[str, Any], dtype: str, default_dtype: str = "bf16") -> float:
    raw = str(dtype or default_dtype or hw.get("default_dtype") or "bf16").strip()
    lower = raw.lower()
    canonical = _DTYPE_ALIASES.get(lower, lower)
    dies = hw.get("dies") or ((hw["die"],) if hw.get("die") is not None else ())
    if dies:
        return min(_die_cube_peak(d, raw, default_dtype) for d in dies)
    peaks = hw.get("compute_peaks", {})
    for key in (raw, canonical, lower):
        if not key:
            continue
        try:
            return float(peaks[key])
        except KeyError:
            continue
    if canonical in _FP8_VARIANTS:
        return float(peaks["fp8"])
    raise KeyError(raw)


def _link_effective_bps(lnk: Any, active_width: int) -> float:
    """Effective per-die link bandwidth (B/s) aligned with causal_tail link supply."""
    raw_bps = float(getattr(lnk, "capacity_bytes_per_ms", 0.0) or 0.0) * 1000.0
    if raw_bps <= 0.0:
        return 0.0
    sem = str(getattr(lnk, "semantics", "") or "").strip().lower()
    members = getattr(lnk, "members", ()) or ()
    width = max(1, int(active_width))
    span = min(width, len(members)) if members else width
    if sem == "per_pair":
        eff = raw_bps * float(max(1, span - 1))
        ep_rates = [
            float(r) * 1000.0
            for _, r in (getattr(lnk, "endpoint_rates", ()) or ())
            if float(r) > 0.0
        ]
        if ep_rates:
            eff = min(eff, max(ep_rates))
    elif sem == "shared":
        eff = raw_bps / float(max(1, span))
    else:
        eff = raw_bps
    sw_cap = getattr(lnk, "switch_capacity_bytes_per_ms", None)
    if sw_cap is not None and float(sw_cap) > 0.0:
        eff = min(eff, float(sw_cap) * 1000.0)
    return eff


def extract_hardware_peaks(case: CaseSpec | Mapping[str, Any], *, stage: int = DEFAULT_STAGE, replica: int = DEFAULT_REPLICA) -> dict[str, Any]:
    """Extract conservative stage-level Cube, Vector, HBM, MTE, and Link peak rates in SI units (/s)."""
    parsed = parse_case(case)
    p = parsed.runtime.parallel
    width, start = p.tp * p.pcp, (replica * p.pp + stage) * (p.tp * p.pcp)
    dies = tuple(parsed.machine.dies[start:start + width]) or (parsed.machine.dies[0],)
    common_dtypes = set(dict(dies[0].compute_peaks))
    for d in dies[1:]:
        common_dtypes &= set(dict(d.compute_peaks))
    compute_peaks, rates_list = {dt: min(float(d.peak(dt)) for d in dies) for dt in sorted(common_dtypes)}, [dict(d.rates) for d in dies]
    vec_peak, hbm_peak = min(float(r["vector"]) for r in rates_list), min(float(r["hbm"]) for r in rates_list)
    mte1_peak, fixpipe_peak = min(float(r.get("mte1", 0.0)) for r in rates_list), min(float(r.get("fixpipe", 0.0)) for r in rates_list)
    mte2_peak, mte3_peak = min(float(r.get("mte2_aic", r.get("mte2", r["hbm"]))) for r in rates_list), min(float(r.get("mte3", r["hbm"])) for r in rates_list)
    links = parsed.machine.links
    active_width = max(p.tp, p.ep, p.pcp, p.cp, p.dcp, p.etp, *p.local_tp_widths, 1)
    intra_links = [lnk for lnk in links
                   if "intra_package" in lnk.channel
                   or "intra_node" in lnk.channel
                   or len(lnk.members) <= parsed.machine.dies_per_package]
    inter_links = [lnk for lnk in links if lnk not in intra_links]
    intra_bw = max((_link_effective_bps(lnk, active_width) for lnk in intra_links), default=0.0)
    inter_bw = (
        min((bw for lnk in inter_links if (bw := _link_effective_bps(lnk, active_width)) > 0.0),
            default=0.0)
        or intra_bw
    )
    link_peak = (
        intra_bw if (active_width <= max(parsed.machine.dies_per_package,
                                         parsed.machine.ranks_per_node)
                     and intra_bw > 0.0) else inter_bw
    ) if links else 0.0
    is_lane_issue = "lane operation per cycle" in str(dict(parsed.machine.assumptions).get("vector_issue", ""))
    return {
        "die": dies[0], "dies": dies, "compute_peaks": compute_peaks, "default_dtype": parsed.model.compute_dtype or "bf16",
        "vector_peak_ops_per_s": vec_peak, "vector_fp32_fma_flops_per_s": (2.0 if is_lane_issue else 1.0) * vec_peak, "vector_fp16_fma_flops_per_s": (_VECTOR_FP16_FMA_FLOPS_PER_ISSUE if is_lane_issue else 1.0) * vec_peak,
        "hbm_peak_bytes_per_s": hbm_peak, "mte1_peak_bytes_per_s": mte1_peak, "mte2_peak_bytes_per_s": mte2_peak,
        "mte3_peak_bytes_per_s": mte3_peak, "fixpipe_peak_bytes_per_s": fixpipe_peak, "link_peak_bytes_per_s": float(link_peak),
        "intra_package_link_bytes_per_s": float(intra_bw or link_peak), "inter_package_link_bytes_per_s": float(inter_bw or link_peak), "dies_per_package": parsed.machine.dies_per_package,
    }


def _operator_invocation_bounds(op: Mapping[str, Any], row: Mapping[str, Any], parsed: CaseSpec, dies: Sequence[Device], duration_ms: float, flops: float, vector_ops: float, link_bytes: float) -> tuple[float, float]:
    """Return (fixed_startup_ms, inclusive_minimum_ms) without host launch_us."""
    basis, startup_ms, minimum_ms = row.get("basis") or {}, 0.0, 0.0
    inv_models = basis.get("invocation_models") or ()
    if inv_models:
        startup_ms, minimum_ms = max(float(i.get("startup_ms", 0.0) or 0.0) for i in inv_models), max(float(i.get("minimum_ms", 0.0) or 0.0) for i in inv_models)
    else:
        meta = dict(op.get("metadata") or {})
        explicit_cls = meta.get("invocation_class") or (meta.get("program_execution") or {}).get("execution_class")
        inv_rules, matched_rule = parsed.runtime.kernels.get("invocations", ()), False
        if inv_rules:
            ident = {"hardware": parsed.machine.hardware_key or parsed.machine.sku, "runtime": parsed.runtime.vllm_impl, "revision": parsed.runtime.vllm_version, "format": parsed.runtime.kernel_format, "kernel": meta.get("kernel", meta.get("kernel_name", op.get("recipe", ""))), "implementation": meta.get("implementation"), "dtype": op.get("dtype") or parsed.model.compute_dtype, "layout": op.get("layout", ""), "replay_scope": meta.get("graph_replay_scope", "none"), "recipe": op.get("recipe", ""), "family": op.get("family", ""), "phase": op.get("phase", "")}
            cands = [(len(r.get("match", {})), r) for r in inv_rules if all(ident.get(k) == v for k, v in (r.get("match") or {}).items())]
            if cands:
                top = [c[1] for c in cands if c[0] == max(x[0] for x in cands)]
                if len(top) == 1:
                    startup_ms, minimum_ms, matched_rule = float(top[0].get("startup_ms", 0.0) or 0.0), float(top[0].get("minimum_ms", 0.0) or 0.0), True
        if not matched_rule and not (link_bytes > 0 and not explicit_cls):
            classes = (explicit_cls,) if explicit_cls else tuple(r for r, amt in (("cube", flops), ("vector", vector_ops)) if amt > 0)
            for d in dies:
                inv_map = dict(d.kernel_invocations)
                for cls_name in (c for c in classes if c in inv_map):
                    startup_ms, minimum_ms = max(startup_ms, float(inv_map[cls_name].startup_ms)), max(minimum_ms, float(inv_map[cls_name].minimum_ms))
    return min(startup_ms, duration_ms), min(minimum_ms, duration_ms)


def diagnostic_legacy_invocation_bounds(basis: Mapping[str, Any]) -> dict[str, Any]:
    """Explicit diagnostic replay of archived service-regime floors.

    Never called by the ordinary analysis path.  The returned fixed/minimum
    values retain the source rule id and are marked ``production_prediction``
    false so legacy whole-call floors cannot silently re-enter timing.
    """
    fixed_ms, minimum_ms, sources = 0.0, 0.0, []
    for reg in basis.get("service_regimes", ()) or ():
        queries, resp = reg.get("queries", {}), reg.get("response", {})
        if (queries.get("observable_response") or queries.get("joint_native_service")) and float(resp.get("fixed_us", 0.0) or 0.0) > 0:
            sources.append(str(reg.get("rule_id") or reg.get("id") or "service_regime"))
            if resp.get("terms") or resp.get("services"):
                fixed_ms = max(fixed_ms, float(resp["fixed_us"]) / 1000.0)
            else:
                minimum_ms = max(minimum_ms, float(resp["fixed_us"]) / 1000.0)
    return {"fixed_ms": fixed_ms, "minimum_ms": minimum_ms,
            "source_rule_ids": sources, "production_prediction": False}


def legacy_service_regime_diagnostics(case: CaseSpec | Mapping[str, Any]) -> dict[str, Any] | None:
    """Expose archived service-regime rules as diagnostic-only, never as production price."""
    parsed = parse_case(case)
    metadata = dict(parsed.runtime.metadata)
    existing = metadata.get("legacy_service_regime_diagnostics")
    if isinstance(existing, Mapping):
        return deepcopy(dict(existing))
    rules = list(parsed.runtime.kernels.get("service_regimes", ()) or ())
    if not rules:
        return None
    return {"production_prediction": False, "scaled": False,
            "source": "runtime.kernels.service_regimes",
            "rules": deepcopy(rules)}


def _operator_comm_groups(
    topology: Topology,
    op: Mapping[str, Any],
    replica: int,
) -> tuple[tuple[int, ...], ...]:
    raw_meta = op.get("metadata") or ()
    metadata = dict(raw_meta) if isinstance(raw_meta, Mapping) else {k: v for k, v in raw_meta}
    axis = str(metadata.get("group_axis") or "tp")
    family = str(metadata.get("collective") or op.get("family") or "").strip().lower()
    stage = int(metadata.get("source_stage", op.get("stage", 0)) or 0)
    if family == "pp_p2p" or axis == "pp":
        dest = int(metadata.get("destination_stage", stage + 1))
        if 0 <= stage < topology.parallel.pp and 0 <= dest < topology.parallel.pp and dest != stage:
            return tuple(
                (src, dst)
                for src, dst in zip(topology.ranks(replica, stage), topology.ranks(replica, dest))
                if src != dst
            )
        return ()
    try:
        groups = tuple(topology.groups(axis, replica, stage, metadata=metadata))
    except ModelingError:
        return ()
    return tuple(group for group in groups if len(group) > 1)


def _link_peak_for_group(
    link: Any,
    group: tuple[int, ...],
    width: int,
    *,
    active_senders: int | None = None,
) -> float:
    if getattr(link, "bw_utilization", None) is not None:
        raw_bps = float(link.capacity_bytes_per_ms or link.peak_capacity_bytes_per_ms) * _MS_PER_S
    else:
        raw_bps = float(link.peak_capacity_bytes_per_ms or link.capacity_bytes_per_ms) * _MS_PER_S
    group_set = set(group)
    ep_rates = [
        float(rate) * _MS_PER_S
        for rank, rate in (getattr(link, "endpoint_rates", ()) or ())
        if float(rate) > 0.0 and (not group_set or int(rank) in group_set)
    ]
    if link.semantics == "per_pair":
        peak = raw_bps * max(1, width - 1)
        if ep_rates:
            peak = min(peak, min(ep_rates))
    elif link.semantics == "shared":
        senders = max(1, width if active_senders is None else int(active_senders))
        peak = raw_bps / senders
    else:
        peak = min(ep_rates) if ep_rates else raw_bps
    sw_cap = getattr(link, "switch_capacity_bytes_per_ms", None)
    if sw_cap is not None and float(sw_cap) > 0.0:
        peak = min(peak, float(sw_cap) * _MS_PER_S)
    return peak


def _routed_group_links(
    topology: Topology,
    group: tuple[int, ...],
    group_scope: str,
    *,
    is_pp: bool,
) -> list[Any]:
    pairs = ((group[0], group[1]),) if is_pp else tuple(
        (src, dst) for idx, src in enumerate(group) for dst in group[idx + 1:]
    )
    scoped_routed: list[Any] = []
    all_routed: list[Any] = []
    seen_scoped: set[str] = set()
    seen_all: set[str] = set()
    for src, dst in pairs:
        try:
            hops = topology.route(src, dst)
        except ValueError:
            continue
        pair_scope = topology.scope_for_group((src, dst))
        for link_name, _, _ in hops:
            lnk = topology.links.get(link_name)
            if lnk is None:
                continue
            if link_name not in seen_all:
                seen_all.add(link_name)
                all_routed.append(lnk)
            if (pair_scope == group_scope and topology.link_scope(link_name) == group_scope
                    and link_name not in seen_scoped):
                seen_scoped.add(link_name)
                scoped_routed.append(lnk)
    return scoped_routed or all_routed


def _resolve_operator_link_peak(
    parsed,
    op,
    hw,
    *,
    replica: int = DEFAULT_REPLICA,
) -> float:
    """Physical per-rank link peak for this operator's declared group axis."""
    links = parsed.machine.links
    if not links:
        return float(hw.get("link_peak_bytes_per_s", 0.0) or 0.0)
    raw_meta = op.get("metadata") or ()
    metadata = dict(raw_meta) if isinstance(raw_meta, Mapping) else {k: v for k, v in raw_meta}
    axis = str(metadata.get("group_axis") or "tp")
    family = str(metadata.get("collective") or op.get("family") or "").strip().lower()
    is_pp = family == "pp_p2p" or axis == "pp"
    topology = Topology(parsed.machine, parsed.runtime.parallel)
    groups = _operator_comm_groups(topology, op, replica)
    if groups:
        group_peaks: list[float] = []
        for group in groups:
            width = len(group)
            group_scope = topology.scope_for_group(group)
            selected = _routed_group_links(topology, group, group_scope, is_pp=is_pp)
            if not selected:
                group_set = set(group)
                selected = [
                    lnk for lnk in links
                    if group_set <= set(lnk.members) and topology.link_scope(lnk.name) == group_scope
                ] or list(links)
            if is_pp:
                peaks_for_group = [
                    _link_peak_for_group(
                        lnk, group, width,
                        active_senders=max(1, len({src for src, _ in groups if src in lnk.members})),
                    )
                    for lnk in selected
                ]
            else:
                peaks_for_group = [_link_peak_for_group(lnk, group, width) for lnk in selected]
            group_peaks.append(min(peaks_for_group))
        return min(group_peaks)
    width = max(1, int(getattr(parsed.runtime.parallel, axis, 1) or 1))
    candidates = [lnk for lnk in links if lnk.members and len(lnk.members) >= width] or list(links)
    within_node = width <= max(1, parsed.machine.ranks_per_node)
    selected = [
        lnk for lnk in candidates
        if (topology.link_scope(lnk.name) in {"package", "node"}) == within_node
    ] or candidates
    return max(_link_peak_for_group(lnk, (), width) for lnk in selected)


def _split_link_namespace(name: str) -> tuple[str, str] | None:
    text = str(name or "")
    for prefix in _LINK_RESOURCE_PREFIXES:
        if text.startswith(prefix):
            return "", text
        marker = f":{prefix}"
        pos = text.find(marker)
        if pos > 0:
            return text[:pos + 1], text[pos + 1:]
    if text.endswith(":switch_plane"):
        return "", text
    return None


def _physical_link_service_with_owner(
    comm_info: Mapping[str, Any],
) -> tuple[float, float, str, dict[str, float]] | None:
    resources = dict(comm_info.get("resources") or {})
    wire: dict[str, float] = {}
    capacity: dict[str, float] = {}
    for name, item in resources.items():
        if not isinstance(item, Mapping):
            continue
        amount = float(item.get("wire_bytes", 0.0) or 0.0)
        peak = float(item.get("peak_bytes_per_ms", 0.0) or 0.0)
        if amount > 0.0:
            wire[str(name)] = amount
        if peak > 0.0:
            capacity[str(name)] = peak
    if not wire or not set(wire) <= set(capacity):
        return None
    best_name = ""
    best_ms = -1.0
    service_by_res: dict[str, float] = {}
    for name in sorted(wire):
        service_ms = wire[name] / capacity[name]
        service_by_res[name] = service_ms
        if service_ms > best_ms:
            best_ms = service_ms
            best_name = name
    return best_ms, math.fsum(wire.values()), best_name, service_by_res


def physical_link_service(comm_info: Mapping[str, Any]) -> tuple[float, float] | None:
    """Exact per-physical-link byte service from the communication ledger.

    Returns ``(max(wire_bytes / peak_bytes_per_ms), sum(wire_bytes))`` only
    when every published resource carries its own matching capacity.  The
    numerator and denominator then cover the same rank boundary, so utilization
    is never repaired by clipping or by averaging uneven traffic away.
    """
    resolved = _physical_link_service_with_owner(comm_info)
    return None if resolved is None else (resolved[0], resolved[1])


def _serviced_link_service(
    serviced: Mapping[str, Any],
    comm_info: Mapping[str, Any],
    parsed: CaseSpec,
    *,
    resource_rates: Mapping[str, float] | None = None,
) -> tuple[float, float, str, dict[str, float]] | None:
    """Exact physical link service from executed Engine ``serviced`` entries."""
    entries: list[tuple[str, float, str, str, bool]] = []
    for raw_name, raw_val in serviced.items():
        amount = float(raw_val or 0.0)
        if amount <= 0.0:
            continue
        name = str(raw_name)
        split = _split_link_namespace(name)
        if split is not None:
            entries.append((name, amount, split[0], split[1], False))
        elif name.startswith("ub_inj:") or ":ub_inj:" in name:
            entries.append((name, amount, "", name, True))
    if not entries:
        return None
    comm_res = dict(comm_info.get("resources") or {})
    pair_rates: dict[str, float] = {}
    for res_name, item in comm_res.items():
        if not isinstance(item, Mapping):
            continue
        peak = float(item.get("peak_bytes_per_ms", 0.0) or 0.0)
        if peak > 0.0 and ":pair" in str(res_name):
            pair_rates.setdefault(str(res_name).split(":pair", 1)[0], peak)
    topo_rates_by_ns: dict[str, dict[str, float]] = {}
    best_ms, best_name, wire_total = -1.0, "", 0.0
    service_by_res: dict[str, float] = {}
    for name, amount, ns, bare, is_ub_inj in sorted(entries, key=lambda item: item[0]):
        if bare.startswith("sdma_flow:"):
            service_ms = amount
        else:
            cap = float((resource_rates or {}).get(name, 0.0) or 0.0)
            if cap <= 0.0:
                item = comm_res.get(name)
                cap = float(item.get("peak_bytes_per_ms", 0.0) or 0.0) if isinstance(item, Mapping) else 0.0
            if cap <= 0.0 and not is_ub_inj and ":pair" in name:
                cap = pair_rates.get(name.split(":pair", 1)[0], 0.0)
            if cap <= 0.0 and not is_ub_inj:
                rates = topo_rates_by_ns.get(ns)
                if rates is None:
                    topo = Topology(parsed.machine, parsed.runtime.parallel, namespace=ns)
                    rates = {res.name: float(res.rate) for res in topo.resources()}
                    topo_rates_by_ns[ns] = rates
                cap = rates.get(name, 0.0)
            if cap <= 0.0:
                if is_ub_inj:
                    continue
                return None
            service_ms = amount / cap
            if not is_ub_inj:
                wire_total += amount
        service_by_res[name] = service_ms
        if service_ms > best_ms:
            best_ms = service_ms
            best_name = name
    if not service_by_res:
        return None
    return best_ms, wire_total, best_name, service_by_res


def _serviced_other_non_hbm_ms(
    serviced: Mapping[str, Any],
    dies: Sequence[Device],
    parsed: CaseSpec,
    *,
    resource_rates: Mapping[str, float] | None = None,
) -> float:
    """Max executed service (ms) over non-HBM, non-cube, non-vector, non-link device resources."""
    rates_list = [dict(d.rates) for d in dies]
    best = 0.0
    for raw_name, raw_val in serviced.items():
        amount = float(raw_val or 0.0)
        if amount <= 0.0:
            continue
        name = str(raw_name)
        if name.startswith(_NON_OTHER_SERVICED_PREFIXES) or _split_link_namespace(name) is not None:
            continue
        if name.startswith("ub_inj:") or ":ub_inj:" in name:
            continue
        rate_per_ms = float((resource_rates or {}).get(name, 0.0) or 0.0)
        if rate_per_ms <= 0.0:
            if name.startswith("host@") or ":host@" in name:
                rate_per_ms = 1.0
            elif name.startswith("host_solo:") and hasattr(parsed.machine, "host_solo_bw_gbps"):
                parts = name.split(":")
                if len(parts) >= 2:
                    gbps = parsed.machine.host_solo_bw_gbps(parts[1])
                    if gbps is not None and float(gbps) > 0.0:
                        rate_per_ms = float(gbps) * _GBPS_TO_BYTES_PER_MS
            elif name.startswith("host_rc:") and hasattr(parsed.machine, "host_root_complex_ceiling_gbps"):
                parts = name.split(":")
                if parts:
                    gbps = parsed.machine.host_root_complex_ceiling_gbps(parts[-1])
                    if gbps is not None and float(gbps) > 0.0:
                        rate_per_ms = float(gbps) * _GBPS_TO_BYTES_PER_MS
            elif "@" in name:
                role = name.split("@", 1)[0]
                matching = [float(r[role]) for r in rates_list if role in r and float(r[role]) > 0.0]
                if matching:
                    rate_per_ms = min(matching) / _MS_PER_S
        if rate_per_ms > 0.0:
            best = max(best, amount / rate_per_ms)
    return best


def _serviced_hbm_exec_ms(
    serviced: Mapping[str, Any],
    dies: Sequence[Device],
    hbm_peak_bytes_per_s: float,
    *,
    resource_rates: Mapping[str, float] | None = None,
) -> float:
    rates_list = [dict(d.rates) for d in dies]
    best = 0.0
    for raw_name, raw_val in serviced.items():
        amount = float(raw_val or 0.0)
        if amount <= 0.0:
            continue
        name = str(raw_name)
        if not name.startswith(("memory@", "gm_hbm")):
            continue
        rate_per_ms = float((resource_rates or {}).get(name, 0.0) or 0.0)
        if rate_per_ms <= 0.0:
            role = name.split("@", 1)[0] if "@" in name else name
            if role == "gm_hbm_cacheline_128b":
                matching = [float(r[role]) for r in rates_list if role in r and float(r[role]) > 0.0]
                if matching:
                    rate_per_ms = min(matching) / _MS_PER_S
                elif hbm_peak_bytes_per_s > 0.0:
                    rate_per_ms = (hbm_peak_bytes_per_s / _CACHELINE_128B_BYTES) / _MS_PER_S
            elif hbm_peak_bytes_per_s > 0.0:
                rate_per_ms = hbm_peak_bytes_per_s / _MS_PER_S
        if rate_per_ms > 0.0:
            best = max(best, amount / rate_per_ms)
    return best


def _op_native_call_id(op: Mapping[str, Any]) -> str:
    raw_meta = op.get("metadata") or ()
    metadata = tuple(raw_meta.items()) if isinstance(raw_meta, Mapping) else tuple(raw_meta)
    raw_mf = op.get("matrix_flops") or ()
    mf_items = raw_mf.items() if isinstance(raw_mf, Mapping) else raw_mf
    matrix_flops = tuple(sorted((str(k), float(v)) for k, v in mf_items if v))
    ns = SimpleNamespace(
        op_id=str(op.get("op_id") or ""),
        family=str(op.get("family") or ""),
        recipe=str(op.get("recipe") or ""),
        source_boundary=str(op.get("source_boundary") or ""),
        dtype=str(op.get("dtype") or ""),
        layout=str(op.get("layout") or "contiguous"),
        local_shape=tuple(op.get("local_shape") or ()),
        padded_shape=tuple(op.get("padded_shape") or ()),
        launch_shape=tuple(op.get("launch_shape") or ()),
        metadata=metadata,
        matrix_flops=matrix_flops,
    )
    binding = bind_native_call(ns)
    return str(binding.call_id) if binding is not None else ""


def _is_cohort_priced_collective(
    row: Mapping[str, Any],
    comm_info: Mapping[str, Any],
    parsed: CaseSpec,
) -> bool:
    """True when this operator row's link service already pools >= 2 replicas' traffic."""
    p = parsed.runtime.parallel
    if p.dp <= 1:
        return False
    if row.get("isolation_boundary") == "complete_collective_cohort":
        return True
    useful_ranks = comm_info.get("useful_bytes_by_rank") or {}
    if not isinstance(useful_ranks, Mapping) or len(useful_ranks) <= 1:
        return False
    topology = Topology(parsed.machine, p)
    rank_to_replica = {
        int(rank): rep
        for rep in range(p.dp)
        for st in range(p.pp)
        for rank in topology.ranks(rep, st)
    }
    reps = {rank_to_replica[int(rank)] for rank in useful_ranks if int(rank) in rank_to_replica}
    return len(reps) > 1


def analyze_operator_row(
    row: Mapping[str, Any],
    case: CaseSpec | Mapping[str, Any],
    *,
    peaks: Mapping[str, Any] | None = None,
    observed_duration_ms: float | None = None,
    pmu: Mapping[str, float] | None = None,
    resource_rates: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compute 4D efficiency (Cube_MFU, Vec_MFU, MBU, Link_Util) & CV-overlap for one operator."""
    parsed, op, count = parse_case(case), row["operator"], int(row.get("occurrences", 1))
    if isinstance(row.get("occurrences", 1), bool) or count < 1:
        raise ValueError(f"operator_occurrences_must_be_positive:{row.get('occurrences')}")
    raw_dur = observed_duration_ms if observed_duration_ms is not None else row["duration_ms"]
    if isinstance(raw_dur, bool) or not isinstance(raw_dur, (int, float)) or not math.isfinite(raw_dur) or float(raw_dur) < 0.0:
        raise ValueError(f"operator_duration_must_be_nonnegative_finite:{raw_dur}")
    duration_ms, stage, replica = float(raw_dur), int(op.get("stage", 0)), int(row.get("replica", 0))
    hw = dict(peaks) if peaks is not None else extract_hardware_peaks(parsed, stage=stage, replica=replica)
    dies, ref_work, has_jn = hw.get("dies") or (hw["die"],), dict(row.get("analytical_reference_work") or {}), bool(row.get("joint_native_service"))
    raw_meta = op.get("metadata") or ()
    metadata = dict(raw_meta) if isinstance(raw_meta, Mapping) else {k: v for k, v in raw_meta}
    res_work, serviced = dict(metadata.get("resource_work") or {}), dict(row.get("serviced") or {})

    serv_hbm, serv_vec = (sum(float(v) for k, v in serviced.items() if k.startswith("memory@")) if not has_jn else 0.0), (sum(float(v) for k, v in serviced.items() if k.startswith("vector@")) if not has_jn else 0.0)
    serv_mte1, serv_mte2, serv_mte3 = (sum(float(v) for k, v in serviced.items() if k.startswith("mte1@")) if not has_jn else 0.0), (sum(float(v) for k, v in serviced.items() if k.startswith("mte2")) if not has_jn else 0.0), (sum(float(v) for k, v in serviced.items() if k.startswith("mte3@")) if not has_jn else 0.0)
    comm_info = dict(row.get("communication") or {})
    comm_wire = comm_info.get("wire_by_resource") or {}
    comm_res = dict(comm_info.get("resources") or {})
    _non_lnk = ("memory@", "vector@", "matrix@", "mte", "fixpipe@", "l2@", "ub", "reg_", "sreg_", "vreg_", "scalar_", "vbitsort", "host", "scheduler@", "native_call_service", "sdma_flow:")
    if has_jn:
        serv_link = 0.0
    elif comm_wire:
        pair_sources = {str(key).split(":pair", 1)[0]
                        for key in comm_wire if ":pair" in str(key)}
        rank_div = max(1, len(comm_info.get("useful_bytes_by_rank") or ())
                       or len(pair_sources))
        serv_link = sum(float(v) for v in comm_wire.values()) / rank_div
    elif comm_info.get("wire_bytes") is not None:
        is_per_port = any(":tx:port" in k or ":rx:port" in k for k in comm_res)
        rank_div = (max(1, len(comm_info.get("useful_bytes_by_rank") or ()) or len(dies)) if is_per_port else 1)
        serv_link = float(comm_info["wire_bytes"]) / rank_div
    else:
        lnk_items = {
            k: float(v) for k, v in serviced.items()
            if not any(k.startswith(p) for p in _non_lnk)
            and not k.endswith(":switch_plane")
            and ":sdma_flow:" not in k
        }
        tx_items = [v for k, v in lnk_items.items() if ":tx:port" in k]
        serv_link = (sum(tx_items) / max(1, len(dies))) if tx_items else sum(lnk_items.values())

    m_raw = op.get("matrix_flops") or ()
    matrix_flops = {str(k): float(v) for k, v in (m_raw.items() if isinstance(m_raw, Mapping) else m_raw) if v}
    flops, dtype = max(float(op.get("flops", 0.0) or 0.0) + float(res_work.get("cube", 0.0) or 0.0), sum(matrix_flops.values()), float(ref_work.get("cube", 0.0) or 0.0)), str(op.get("dtype") or hw["default_dtype"] or "bf16")
    if matrix_flops:
        if dies:
            t_cube_ideal_ms = max(
                sum((fl / _die_cube_peak(d, dt, dtype)) * _MS_PER_S for dt, fl in matrix_flops.items())
                for d in dies
            )
        else:
            t_cube_ideal_ms = sum((fl / _resolve_cube_peak(hw, dt, dtype)) * _MS_PER_S for dt, fl in matrix_flops.items())
    else:
        t_cube_ideal_ms = (flops / _resolve_cube_peak(hw, dtype, hw["default_dtype"])) * _MS_PER_S if flops > 0 else 0.0
    vector_ops = max(float(op.get("vector_ops", 0.0) or 0.0) + float(res_work.get("vector", 0.0) or 0.0), serv_vec, float(ref_work.get("vector", 0.0) or 0.0))
    t_vec_ideal_ms = (vector_ops / hw["vector_peak_ops_per_s"]) * _MS_PER_S if vector_ops > 0 else 0.0
    hbm_bytes = max(float(op.get("hbm_bytes", 0.0) or 0.0) + float(res_work.get("hbm", 0.0) or 0.0), serv_hbm, float(ref_work.get("hbm", 0.0) or 0.0))
    mte1_bytes = max(float(metadata.get("mte1_bytes", 0.0) or 0.0) + float(res_work.get("mte1", 0.0) or 0.0), serv_mte1, float(ref_work.get("mte1", 0.0) or 0.0))
    mte2_bytes = max(float(op.get("mte2_bytes", 0.0) or 0.0) + float(res_work.get("mte2", 0.0) or 0.0), serv_mte2, float(ref_work.get("mte2", 0.0) or 0.0))
    mte3_bytes = max(float(op.get("mte3_bytes", 0.0) or 0.0) + float(res_work.get("mte3", 0.0) or 0.0), serv_mte3, float(ref_work.get("mte3", 0.0) or 0.0))
    t_hbm_ideal_ms = (hbm_bytes / hw["hbm_peak_bytes_per_s"]) * _MS_PER_S if hbm_bytes > 0 else 0.0
    logical_link_bytes = max(float(op.get("link_bytes", 0.0) or 0.0) + float(res_work.get("link", 0.0) or 0.0), float(ref_work.get("link", 0.0) or 0.0))
    serviced_link = _serviced_link_service(serviced, comm_info, parsed, resource_rates=resource_rates) if not has_jn else None
    physical_link = _physical_link_service_with_owner(comm_info)
    link_bytes = max(logical_link_bytes, serv_link)
    dominant_link_resource: str | None = None
    link_service_by_resource: dict[str, float] = {}
    if serviced_link is not None:
        t_link_ideal_ms, _physical_link_bytes, dominant_link_resource, link_service_by_resource = serviced_link
    elif physical_link is not None:
        t_link_ideal_ms, _physical_link_bytes, dominant_link_resource, link_service_by_resource = physical_link
    else:
        op_link_peak = _resolve_operator_link_peak(parsed, op, hw, replica=replica)
        t_link_ideal_ms = ((link_bytes / op_link_peak) * _MS_PER_S
                           if (link_bytes > 0.0 and op_link_peak > 0.0)
                           else 0.0)

    has_device_serviced = bool(serviced) and not has_jn
    t_other_non_hbm_ms = _serviced_other_non_hbm_ms(serviced, dies, parsed, resource_rates=resource_rates) if has_device_serviced else 0.0
    t_hbm_exec_ms = (
        _serviced_hbm_exec_ms(serviced, dies, hw["hbm_peak_bytes_per_s"], resource_rates=resource_rates)
        if has_device_serviced else None
    )
    matrix_vals = [float(v) for k, v in serviced.items() if k.startswith("matrix@") and float(v) > 0.0] if has_device_serviced else []
    t_cube_exec_ms = max(matrix_vals) if matrix_vals else None
    vec_vals = [float(v) for k, v in serviced.items() if k.startswith("vector@") and float(v) > 0.0] if has_device_serviced else []
    t_vec_exec_ms = (max(vec_vals) / (hw["vector_peak_ops_per_s"] / _MS_PER_S)) if (vec_vals and hw["vector_peak_ops_per_s"] > 0.0) else None
    t_link_exec_ms = serviced_link[0] if serviced_link is not None else None

    total_ms, safe_t = duration_ms * count, max(duration_ms, 1e-12)
    operator_kind = "cv_fused" if (flops > 0 and vector_ops > 0) else "cube" if flops > 0 else "vector" if vector_ops > 0 else "communication" if link_bytes > 0 else "memory_only"
    fixed_ms, minimum_ms = _operator_invocation_bounds(op, row, parsed, dies, duration_ms, flops, vector_ops, link_bytes)
    roofline_ideal_ms, launch_thresh = max(t_hbm_ideal_ms, t_cube_ideal_ms, t_vec_ideal_ms, t_link_ideal_ms), max(fixed_ms, minimum_ms, _LAUNCH_DIAGNOSTIC_MIN_THRESHOLD_MS)
    pipe_ideals = {"memory_bound": t_hbm_ideal_ms, "cube_bound": t_cube_ideal_ms, "vector_bound": t_vec_ideal_ms, "link_bound": t_link_ideal_ms}
    dominant_bound = "launch_bound" if (roofline_ideal_ms == 0.0 or (roofline_ideal_ms < launch_thresh and duration_ms <= max(_LAUNCH_DIAGNOSTIC_SHORT_DURATION_MS, _LAUNCH_DIAGNOSTIC_DURATION_FACTOR * launch_thresh) and roofline_ideal_ms / safe_t < _LAUNCH_DIAGNOSTIC_MAX_WORK_SHARE)) else max(pipe_ideals, key=pipe_ideals.__getitem__)
    cv_dominant_bound = cv_roofline_efficiency = cv_compute_share_ratio = cv_overlap_ratio = None
    roofline_efficiency = roofline_ideal_ms / max(duration_ms - fixed_ms, 1e-9) if roofline_ideal_ms > 0 else 0.0
    if operator_kind == "cv_fused":
        cv_dominant_bound, cv_roofline_efficiency, denom_comp = f"cv_{dominant_bound}", roofline_efficiency, t_cube_ideal_ms + t_vec_ideal_ms
        cv_compute_share_ratio = (t_vec_ideal_ms / denom_comp) if denom_comp > 0 else 0.5
        pmu_data = pmu or row.get("pmu")
        if isinstance(pmu_data, Mapping) and ("aic_mac_time_us" in pmu_data or "aiv_vec_time_us" in pmu_data or "compute_span_us" in pmu_data):
            t_c, t_v = float(pmu_data.get("aic_mac_time_us", t_cube_ideal_ms * _MS_PER_S)) / _MS_PER_S, float(pmu_data.get("aiv_vec_time_us", t_vec_ideal_ms * _MS_PER_S)) / _MS_PER_S
            t_span = float(pmu_data["compute_span_us"]) / _MS_PER_S if "compute_span_us" in pmu_data else (max(t_c + float(pmu_data.get("aic_wait_time_us", 0.0)) / _MS_PER_S, t_v + float(pmu_data.get("aiv_wait_time_us", 0.0)) / _MS_PER_S) if ("aic_wait_time_us" in pmu_data or "aiv_wait_time_us" in pmu_data) else max(duration_ms - fixed_ms, max(t_c, t_v)))
            cv_overlap_ratio = _clamp((t_c + t_v - t_span) / min(t_c, t_v), 0.0, 1.0) if min(t_c, t_v) > 0 else 0.0
        elif denom_comp > max(duration_ms - fixed_ms, 0.0) and min(t_cube_ideal_ms, t_vec_ideal_ms) > 0:
            cv_overlap_ratio = _clamp((denom_comp - max(duration_ms - fixed_ms, max(t_cube_ideal_ms, t_vec_ideal_ms))) / min(t_cube_ideal_ms, t_vec_ideal_ms), 0.0, 1.0)
    return {
        "physical_id": row.get("physical_id", ""), "op_id": op.get("op_id", ""), "family": op.get("family", ""), "recipe": op.get("recipe", ""),
        "replica": replica, "native_call_id": _op_native_call_id(op),
        "dominant_link_resource": dominant_link_resource,
        "link_service_by_resource": link_service_by_resource,
        "is_cross_replica_collective": _is_cohort_priced_collective(row, comm_info, parsed),
        "structure_block": str(dict(op.get("metadata") or ()).get("structure_block", "")),
        "program": row.get("program", ""), "phase": op.get("phase", ""), "stage": stage, "layer": int(op.get("layer", 0)), "dtype": dtype,
        "local_shape": tuple(op.get("local_shape") or ()), "occurrences": count, "duration_ms": duration_ms, "total_ms": total_ms,
        "fixed_ms": fixed_ms, "minimum_ms": minimum_ms, "work_basis": "analytical_reference" if has_jn else "executed_service",
        "analytical_reference_work": ref_work, "flops": flops, "matrix_flops": matrix_flops, "vector_ops": vector_ops, "hbm_bytes": hbm_bytes,
        "mte1_bytes": mte1_bytes, "mte2_bytes": mte2_bytes, "mte3_bytes": mte3_bytes, "link_bytes": link_bytes, "logical_link_bytes": logical_link_bytes,
        "t_cube_ideal_ms": t_cube_ideal_ms, "t_vec_ideal_ms": t_vec_ideal_ms, "t_hbm_ideal_ms": t_hbm_ideal_ms, "t_link_ideal_ms": t_link_ideal_ms,
        "t_other_non_hbm_ms": t_other_non_hbm_ms, "t_cube_exec_ms": t_cube_exec_ms, "t_vec_exec_ms": t_vec_exec_ms, "t_hbm_exec_ms": t_hbm_exec_ms, "t_link_exec_ms": t_link_exec_ms,
        "roofline_ideal_ms": roofline_ideal_ms, "cube_mfu": (t_cube_ideal_ms / safe_t) if flops > 0 else None, "vec_mfu": (t_vec_ideal_ms / safe_t) if vector_ops > 0 else None,
        "mbu": (t_hbm_ideal_ms / safe_t) if hbm_bytes > 0 else None, "link_util": (t_link_ideal_ms / safe_t) if link_bytes > 0 else None,
        "roofline_efficiency": roofline_efficiency, "operator_kind": operator_kind, "dominant_bound": dominant_bound,
        "cv_dominant_bound": cv_dominant_bound, "cv_roofline_efficiency": cv_roofline_efficiency, "cv_compute_share_ratio": cv_compute_share_ratio, "cv_overlap_ratio": cv_overlap_ratio,
    }
