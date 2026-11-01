"""Ascend A5 communication expansion and contention calibration.

[Role] Expose strictly keyed A5 UB-mesh/URMA service curves, an expansion
  regime resolver, MC2 chain overlap summaries and engine-binding helpers.
[Invariants] A curve is keyed by (collective, world_size, placement_scope,
  dtype, expansion regime, pair, stack, CANN, algorithm, protocol).  A requested
  AICPU/AIV mode is never promoted without observed or profiler evidence.
  Incompatible software stacks are never pooled.  Non-A5 hardware is rejected.
  Single-launch anchors never enter a fit.
[Timeouts & Locks] No timers, locks or subprocesses.  Ingestion of an explicit
  path is a bounded synchronous or ijson-streamed JSON read.
[Side-Effect Policy] Pure computation.  Never launches a device, writes
  calibration or mutates the primitive store.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from statistics import median
from typing import Mapping

from mindie_sim.prediction.services import ServiceCurve
from mindie_sim.schema.common import ModelingError
from .a5_comm_binding import (
    _SUPPORTED_COLLECTIVES, _fold_timings, _weighted_median, bind_service_curve,
    build_a5_comm_recipe, to_service_regime,
)
from .a5_comm_golden import (
    A5_ARCH_FAMILY, A5_HARDWARE_KEY, A5_SKU,
    EMBEDDED_A5_CAMPAIGNS, EMBEDDED_A5_MC2_CAMPAIGNS,
)
from .a5_comm_ingest import (
    _coerce_ingested, _collective, _dtype, _explicit_pair, _explicit_scope,
    _is_a5, _mode, _protocol_key, _require_a5,
    ingest_a5_campaign_data as _ingest_a5_campaign_data,
    resolve_expansion_regime as _resolve_expansion_regime,
)
from .service_fitting import fit_service_curve

__all__ = [
    "A5_ARCH_FAMILY", "A5_HARDWARE_KEY", "A5_SKU", "A5CommCalibrationBundle",
    "EMBEDDED_A5_CAMPAIGNS", "EMBEDDED_A5_MC2_CAMPAIGNS",
    "bind_service_curve", "build_a5_comm_recipe", "build_a5_followup_manifest",
    "compute_a5_contention_slowdown", "evaluate_a5_mc2_chains",
    "fit_a5_service_curves", "ingest_a5_campaign_data", "resolve_expansion_regime",
    "to_service_regime",
]

_EXPANSION_MODES = frozenset({"aicpu", "aiv"})
_PAIR_ORDER = {"0->1": 0, "0->4": 1, "0->8": 2, "0->24": 3}
_PRIMARY_STACKS = ("rack06", "rack06-07", "rack06-09")
_AMBIGUOUS = "a5_incompatible_software_stacks_require_selector"
def resolve_expansion_regime(requested_expansion_mode, observed_expansion_mode, *,
                             profiler_evidence=None):
    """Resolve one expansion regime without promoting an unconfirmed request."""
    return _resolve_expansion_regime(requested_expansion_mode, observed_expansion_mode,
                                     profiler_evidence=profiler_evidence)
def ingest_a5_campaign_data(source=None):
    """Normalize A5 campaigns into steady-state conditions and launch anchors."""
    return _ingest_a5_campaign_data(source)


def _data_for(ingested):
    return ingest_a5_campaign_data() if ingested is None else _coerce_ingested(ingested)


def _software_key(row):
    return (row.get("stack"), row.get("cann"), row.get("algorithm"),
            _protocol_key(row.get("protocol")))


def _sample_split(row, holdout_payloads):
    split = row.get("split")
    if split is None:
        return "heldout" if row["payload_bytes"] in holdout_payloads else "train"
    if split not in {"train", "heldout"}:
        raise ValueError("service_sample_split_requires_train_or_heldout")
    return split


def _curve_identity(collective, world_size, scope, dtype, regime, pair, evidence,
                    software):
    return {
        "hardware": {"key": A5_HARDWARE_KEY, "sku": A5_SKU,
                     "architecture": A5_ARCH_FAMILY},
        "transport": "ub_mesh" if scope.startswith("intra") else "urma",
        "collective": collective, "world_size": int(world_size),
        "placement_scope": scope, "dtype": dtype,
        "expansion_regime": regime, "p2p_pair": pair, "evidence": evidence,
        "stack": software[0], "cann": software[1], "algorithm": software[2],
        "protocol": software[3],
    }


def _fit_group(rows, holdout_payloads, key, evidence, extra_identity=None):
    collective, world_size, scope, dtype, regime, pair = key[:6]
    software = key[6:10] if len(key) > 6 else key[6:]
    boundary = (key[10] if len(key) > 10 else
                "resource_phase" if any(row.get("boundary") == "resource_phase"
                                        for row in rows) else "device_kernel")
    by_run, samples = defaultdict(list), []
    for row in rows:
        by_run[(row["payload_bytes"], row.get("run_id") or row["campaign"],
                _sample_split(row, holdout_payloads))].append(row)
    for (work, run_id, split), grp in by_run.items():
        if any("_weights" in r for r in grp):
            vals = []
            for r in grp:
                for val, wt in (r.get("_weights") or ((r["per_op_ms"], 1),)):
                    _fold_timings(vals, val, wt)
            r0 = min(grp, key=lambda r: int(r.get("repeat_index") or 0))
            samples.append({"work": work, "duration_ms": _weighted_median(vals),
                            "run_id": run_id, "repeat_index": int(r0.get("repeat_index") or 0),
                            "source": r0.get("source") or f"{r0['campaign']}:{collective}:W{world_size}:{work}",
                            "split": split})
        else:
            samples.extend({"work": work, "duration_ms": r["per_op_ms"], "run_id": run_id,
                            "repeat_index": int(r.get("repeat_index") or 0),
                            "source": r.get("source") or f"{r['campaign']}:{collective}:W{world_size}:{work}",
                            "split": split} for r in grp)
    train = {sample["work"] for sample in samples if sample["split"] == "train"}
    if len(train) < 2:
        return None, None, "a5_curve_requires_two_training_payloads"
    identity = _curve_identity(collective, world_size, scope, dtype, regime, pair,
                               evidence, software)
    identity["protocol"] = next((row.get("protocol") for row in rows
                                 if row.get("protocol") is not None), software[3])
    if extra_identity:
        identity.update(extra_identity)
    provenance = {
        "hardware_key": A5_HARDWARE_KEY, "sku": A5_SKU, "arch_family": A5_ARCH_FAMILY,
        "campaigns": sorted({row["campaign"] for row in rows}),
        "evidence": evidence, "holdout_payloads": sorted(holdout_payloads),
        "expansion_regime": regime,
        "expansion_unconfirmed": any(row["expansion_unconfirmed"] for row in rows),
        "p2p_pair": pair, "stack": software[0], "cann": software[1],
        "algorithm": software[2],
        "protocol": next((row.get("protocol") for row in rows if row.get("protocol") is not None), None)}
    if extra_identity and extra_identity.get("derived_from"):
        provenance["derived_from"] = extra_identity["derived_from"]
    provenance["boundary"] = boundary
    packet = {"resource": "link", "work_unit": "bytes", "identity": identity,
              "boundary": boundary, "active_units": max(1, int(world_size)),
              "samples": samples, "provenance": provenance}
    fitted = fit_service_curve(packet)
    return ServiceCurve.from_mapping(fitted["model"]), fitted, None


def _selector_match(curve, selectors):
    for name, wanted in selectors.items():
        if wanted is None:
            continue
        if name == "protocol":
            if _protocol_key(curve.identity.get(name)) != _protocol_key(wanted):
                return False
        elif name == "pair":
            if curve.identity.get("p2p_pair") != wanted:
                return False
        elif name == "boundary":
            if curve.boundary != wanted:
                return False
        elif curve.identity.get(name) != wanted:
            return False
    return True


def _select_software(cell, options):
    """Pick the primary 9.1.0 rack06 stack, or reject an ambiguous pool."""
    primary = [option for option in options if option[0][0] in _PRIMARY_STACKS
               and option[0][1] in {None, "9.1.0"}]
    if primary:
        primary.sort(key=lambda option: (option[0][1] != "9.1.0",
                                         _PRIMARY_STACKS.index(option[0][0]), str(option[0])))
        return primary[0][1]
    if len(options) == 1:
        return options[0][1]
    raise ModelingError(_AMBIGUOUS)


@dataclass(frozen=True)
class A5CommCalibrationBundle:
    """Strictly keyed A5 service curves plus their independent fit evidence."""

    curves: Mapping
    fits: tuple = ()
    holdout: tuple = ()
    single_op_anchors: tuple = ()
    campaigns: tuple = ()
    unfitted: tuple = ()
    provenance: Mapping = field(default_factory=dict)
    hardware_key: str = A5_HARDWARE_KEY
    arch_family: str = A5_ARCH_FAMILY
    stack_curves: Mapping = field(default_factory=dict)

    @property
    def cells(self):
        return tuple(sorted(self.curves))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) >= 6 and key in self.stack_curves:
                return self.stack_curves[key]
            if len(key) == 5:
                return self.curves[key]
        if key not in self.__dataclass_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def _candidates(self, collective, world_size, *, placement_scope=None,
                    dtype="bf16", expansion_mode="unspecified",
                    allow_unspecified_fallback=False):
        name = _collective(collective)
        if placement_scope is None:
            scope = ("cross_node_urma" if int(world_size) > 8
                     else "intra_node_ub")
        else:
            scope = _explicit_scope(placement_scope)
            if scope is None:
                raise ModelingError(
                    f"a5_placement_scope_unknown:{placement_scope}")
        base = (name, int(world_size), scope, _dtype(dtype))
        mode = _mode(expansion_mode)
        if mode in _EXPANSION_MODES:
            keys = [base + (mode,)]
            if allow_unspecified_fallback:
                keys.append(base + ("unspecified",))
            return keys, mode
        if mode != "unspecified":
            raise ModelingError(f"a5_expansion_mode_unknown:{expansion_mode}")
        return [base + ("unspecified",)], mode

    def _pick(self, candidates, selectors, mode):
        requested = any(value is not None for value in selectors.values())
        for candidate in candidates:
            direct = {curve.model_id: curve for key, curve in self.curves.items()
                      if key == candidate and _selector_match(curve, selectors)}
            if not requested and direct:
                return next(iter(direct.values()))
            matches = {curve.model_id: curve for key, curve
                       in self.stack_curves.items() if key[:5] == candidate
                       and _selector_match(curve, selectors)}
            matches.update(direct)
            if not matches:
                continue
            if len(matches) > 1 and len(direct) == 1:
                return next(iter(direct.values()))
            if len(matches) > 1 and selectors.get("boundary") is None:
                device = {curve.model_id: curve for curve in matches.values()
                          if curve.boundary == "device_kernel"}
                if len(device) == 1:
                    return next(iter(device.values()))
            if len(matches) > 1:
                raise ModelingError(_AMBIGUOUS)
            return next(iter(matches.values()))
        if mode in _EXPANSION_MODES:
            raise ModelingError(f"a5_expansion_regime_unconfirmed:{mode}")
        base = candidates[0]
        raise ModelingError(
            f"a5_curve_unavailable:{base[0]}:W{base[1]}:{base[2]}:{base[3]}")

    def lookup_curve(self, collective, world_size=None, *,
                     placement_scope=None, dtype="bf16",
                     expansion_mode="unspecified", hardware_key=A5_HARDWARE_KEY,
                     arch_family=None, allow_unspecified_fallback=False,
                     stack=None, cann=None, algorithm=None, protocol=None,
                     pair=None, boundary=None):
        _require_a5(hardware_key, arch_family)
        selectors = {"stack": stack, "cann": cann, "algorithm": algorithm,
                     "protocol": protocol}
        if boundary is not None:
            selectors["boundary"] = boundary
        if pair is not None:
            pair = _explicit_pair(pair) or str(pair).strip()
            selectors["pair"] = pair
        if isinstance(collective, (tuple, list)):
            key = tuple(collective)
            if len(key) == 6:
                selectors["stack"] = selectors["stack"] or key[5]
                key = key[:5]
            if len(key) != 5:
                raise ModelingError("a5_curve_key_requires_five_or_six_fields")
            if key in self.curves and not any(value is not None
                                              for value in selectors.values()):
                return self.curves[key]
            candidates, mode = self._candidates(
                key[0], key[1], placement_scope=key[2], dtype=key[3],
                expansion_mode=key[4], allow_unspecified_fallback=allow_unspecified_fallback)
            return self._pick(candidates, selectors, mode)
        candidates, mode = self._candidates(
            collective, world_size, placement_scope=placement_scope, dtype=dtype,
            expansion_mode=expansion_mode,
            allow_unspecified_fallback=allow_unspecified_fallback)
        if pair and candidates:
            expanded = []
            inferred = {"0->24": "cross_node_urma_far",
                        "0->1": "intra_node_ub"}.get(pair)
            for base in candidates:
                suffix = f"{base[2]}_{pair.replace('->', '_')}"
                if inferred and (placement_scope is None or (
                        pair == "0->24" and base[2] == "cross_node_urma")):
                    expanded.append(base[:2] + (inferred,) + base[3:])
                expanded.append(base[:2] + (suffix,) + base[3:])
                expanded.append(base)
            candidates = expanded
        return self._pick(candidates, selectors, mode)

    def to_dict(self):
        return {
            "schema": "a5-comm-calibration-bundle/1",
            "hardware_key": self.hardware_key, "arch_family": self.arch_family,
            "curves": {":".join((key[0], f"W{key[1]}", key[2], key[3], key[4])): curve.to_dict()
                       for key, curve in sorted(self.curves.items(), key=lambda item: str(item[0]))},
            "stack_curves": {":".join(str(part) for part in
                                      (key[0], f"W{key[1]}", key[2], key[3], key[4], *key[5:])):
                             curve.to_dict() for key, curve in
                             sorted(self.stack_curves.items(), key=lambda item: str(item[0]))},
            "fits": list(self.fits), "holdout": list(self.holdout),
            "single_op_anchors": list(self.single_op_anchors),
            "campaigns": list(self.campaigns), "unfitted": list(self.unfitted),
            "provenance": dict(self.provenance)}

    def bind_service_curve(self, case, op, scope="none", **lookup_kwargs):
        return bind_service_curve(self, case, op, scope, **lookup_kwargs)

    def to_service_regime(self, case, op, scope="none", **lookup_kwargs):
        return to_service_regime(self, case, op, scope, **lookup_kwargs)

    def build_a5_comm_recipe(self, collective=None, *,
                             allow_unspecified_fallback=False, **lookup_kwargs):
        return build_a5_comm_recipe(
            self, collective,
            allow_unspecified_fallback=allow_unspecified_fallback,
            **lookup_kwargs)


def fit_a5_service_curves(ingested=None, *, holdout_payloads=(65536, 16777216)):
    """Fit one ServiceCurve per strict A5 communication software partition."""
    data = _data_for(ingested)
    _require_a5(data.get("hardware_key") or data.get("sku") or A5_HARDWARE_KEY, data.get("arch_family"))
    holdout = frozenset(int(payload) for payload in holdout_payloads)
    groups = defaultdict(list)
    for row in data.get("conditions") or ():
        if row["single"] or row["collective"] not in _SUPPORTED_COLLECTIVES:
            continue
        key = (row["collective"], int(row["world_size"]), row["placement_scope"],
               row["dtype"], row["regime"],
               row["pair"] if row["collective"] == "p2p_send_recv" else None,
               *_software_key(row), row.get("boundary") or "device_kernel")
        groups[key].append(row)
    derived_keys = set()
    for key, rows in list(groups.items()):
        if key[0] != "all_to_all":
            continue
        target = ("all_to_allv",) + key[1:2] + key[2:]
        if target not in groups:
            groups[target] = rows
            derived_keys.add(target)
    curves, fits, holdout_rows, unfitted = {}, [], [], []
    software_maps = defaultdict(lambda: defaultdict(list))
    for key in sorted(groups, key=str):
        rows = groups[key]
        derived = key in derived_keys
        evidence = "derived_all_to_all_equivalent" if derived else "measured"
        extra = {"derived_from": "all_to_all"} if derived else None
        curve, fitted, error = _fit_group(rows, holdout, key, evidence, extra)
        if error:
            unfitted.append({"key": key, "reason": error})
            continue
        cell = key[:5]
        software_maps[key[6:10]][cell].append((key[5], curve, fitted, key))
        fits.append({**fitted, "cell": cell, **_cell_fields(cell),
                     "p2p_pair": key[5], "evidence": evidence,
                     "stack": key[6], "cann": key[7], "algorithm": key[8],
                     "boundary": key[10]})
        holdout_rows.extend({**row, "cell": cell, **_cell_fields(cell),
                             "p2p_pair": key[5], "split": "heldout"}
                            for row in fitted["heldout"])
    stack_curves = {}
    target_options = defaultdict(lambda: defaultdict(list))
    for software in sorted(software_maps, key=str):
        for cell, variants in software_maps[software].items():
            by_boundary = defaultdict(list)
            for entry in variants:
                by_boundary[entry[3][10]].append(entry)
            for boundary in sorted(by_boundary):
                boundary_variants = by_boundary[boundary]
                boundary_variants.sort(key=lambda value: (
                    _PAIR_ORDER.get(value[0], 99), str(value[3])))
                targets = defaultdict(list)
                for index, entry in enumerate(boundary_variants):
                    pair = entry[0]
                    if index == 0:
                        targets[cell].append(entry)
                        continue
                    suffix = pair.replace("->", "_")
                    targets[(cell[0], cell[1], f"{cell[2]}_{suffix}", cell[3],
                             cell[4])].append(entry)
                    if pair == "0->24":
                        targets[(cell[0], 24, cell[2], cell[3], cell[4])].append(entry)
                        targets[(cell[0], cell[1], "cross_node_urma_far", cell[3],
                                 cell[4])].append(entry)
                for target, entries in targets.items():
                    curve = entries[0][1]
                    stack_curves.setdefault(target + (software[0],), curve)
                    if boundary == "resource_phase":
                        stack_curves.setdefault(
                            target + (software[0], boundary), curve)
                    full_key = target + software
                    if boundary == "device_kernel" or full_key not in stack_curves:
                        stack_curves[full_key] = curve
                    else:
                        stack_curves[full_key + (boundary,)] = curve
                    target_options[target][boundary].append((software, curve))
    for cell, options_by_boundary in target_options.items():
        options = (options_by_boundary.get("device_kernel")
                   or options_by_boundary.get("resource_phase") or ())
        if options:
            curves[cell] = _select_software(cell, options)
    provenance = {
        "hardware_key": A5_HARDWARE_KEY, "sku": A5_SKU,
        "arch_family": A5_ARCH_FAMILY,
        "source_campaigns": list(data.get("campaign_labels") or ()),
        "holdout_payloads": sorted(holdout),
        "curve_count": len(curves), "stack_curve_count": len(stack_curves),
        "fit_count": len(fits), "unfitted_count": len(unfitted),
    }
    return A5CommCalibrationBundle(
        curves=curves, fits=tuple(fits), holdout=tuple(holdout_rows),
        single_op_anchors=tuple(data.get("single_op_anchors") or ()),
        campaigns=tuple(data.get("campaigns") or ()),
        unfitted=tuple(unfitted), provenance=provenance,
        stack_curves=stack_curves)


def _cell_fields(key):
    return dict(zip(("collective", "world_size", "placement_scope", "dtype",
                     "expansion_mode"), key[:5]))


def compute_a5_contention_slowdown(expansion_mode, *, concurrent_cube_util=0.0,
                                   concurrent_vector_util=0.0,
                                   concurrent_mte_util=0.0, active_aiv_blocks=16,
                                   total_aiv_units=64):
    """Shared-vector / MTE / HBM contention for an A5 execution mode."""
    mode = _mode(expansion_mode)
    if mode not in _EXPANSION_MODES and mode != "unspecified":
        raise ModelingError(f"a5_expansion_mode_unknown:{expansion_mode}")
    for name, value in (("concurrent_cube_util", concurrent_cube_util),
                        ("concurrent_vector_util", concurrent_vector_util),
                        ("concurrent_mte_util", concurrent_mte_util)):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"a5_contention_util_invalid:{name}")
    if int(total_aiv_units) < 1:
        raise ValueError("a5_contention_requires_positive_total_aiv_units")
    blocks = max(0, int(active_aiv_blocks))
    units = max(1, int(total_aiv_units))
    occupancy = min(1.0, blocks / units)
    cube = min(4.0, float(concurrent_cube_util))
    vector_demand = min(4.0, float(concurrent_vector_util))
    mte = min(4.0, float(concurrent_mte_util))
    mte_slowdown = 1.0 + 0.55 * mte + 0.10 * vector_demand + 0.05 * cube
    hbm_slowdown = 1.0 + 0.45 * mte + 0.15 * vector_demand + 0.20 * cube
    cube_slowdown = 1.0 + 0.35 * cube + 0.05 * mte
    vector_slowdown = 1.0
    if mode == "aiv":
        # AIV owns the vector pipe; more resident blocks and more concurrent
        # vector demand extend the same ALU service time.
        vector_slowdown = (1.0 + vector_demand * (0.55 + 0.45 * occupancy)
                           + 0.10 * cube)
    total = max(vector_slowdown, mte_slowdown, hbm_slowdown, cube_slowdown)
    transfer = 0.012 + 0.05 * mte + 0.01 * vector_demand
    sync = 0.008 + 0.02 * max(mte, vector_demand, cube) + 0.0005 * blocks
    return {
        "expansion_mode": mode,
        "vector_slowdown": vector_slowdown,
        "mte_slowdown": mte_slowdown,
        "hbm_slowdown": hbm_slowdown,
        "cube_slowdown": cube_slowdown,
        "contention_slowdown": total,
        "total_slowdown": total,
        "overall_slowdown": total,
        "vector_alu_slowdown": vector_slowdown,
        "transfer_latency_ms": transfer, "sync_latency_ms": sync,
        "transfer_ms": transfer, "sync_ms": sync,
        "active_aiv_blocks": blocks,
        "total_aiv_units": units,
        "aiv_occupancy": occupancy,
    }


def evaluate_a5_mc2_chains(ingested=None):
    """Summarize decomposed MC2 chain overlap and resolved expansion regime."""
    data = _data_for(ingested)
    rows = []
    for chain in data.get("mc2_chains") or ():
        dtype = _dtype(chain["dtype"])
        legs = chain.get("legs_ms") or {}
        chain_ms = float(chain["chain_p50_ms"])
        sum_ms = float(chain.get("sum_leg_p50_ms") or sum(legs.values()))
        ratio = chain_ms / sum_ms
        concurrent = max(0.01, min(1.0, 1.0 - min(1.0, ratio)))
        if any(key in chain for key in ("observed_expansion_mode",
                                        "requested_expansion_mode",
                                        "profiler_evidence")):
            mode = resolve_expansion_regime(
                chain.get("requested_expansion_mode"),
                chain.get("observed_expansion_mode"),
                profiler_evidence=chain.get("profiler_evidence"))["regime"]
        else:
            mode = _mode(chain.get("regime"))
        contention = compute_a5_contention_slowdown(
            mode, concurrent_cube_util=0.40 * concurrent,
            concurrent_vector_util=0.75 * concurrent,
            concurrent_mte_util=concurrent)
        rows.append({
            **chain, "dtype": dtype, "chain_p50_ms": chain_ms,
            "sum_leg_p50_ms": sum_ms, "overlap_ratio": ratio,
            "chain_over_sum": ratio,
            "expansion_regime": mode,
            "contention_degradation_multiplier": contention["contention_slowdown"],
            "vector_slowdown": contention["vector_slowdown"],
            "mte_slowdown": contention["mte_slowdown"],
            "hbm_slowdown": contention["hbm_slowdown"],
            "contended_chain_ms": chain_ms * contention["contention_slowdown"],
        })
    by_dtype = {}
    for dtype in sorted({row["dtype"] for row in rows}):
        subset = [row for row in rows if row["dtype"] == dtype]
        ratios = sorted(row["overlap_ratio"] for row in subset)
        multipliers = sorted(row["contention_degradation_multiplier"] for row in subset)
        by_dtype[dtype] = {
            "count": len(subset),
            "world_sizes": sorted({row["world_size"] for row in subset}),
            "campaigns": sorted({row["campaign"] for row in subset}),
            "overlap_ratio": {"min": ratios[0], "max": ratios[-1],
                              "median": median(ratios),
                              "mean": sum(ratios) / len(ratios)},
            "contention_degradation_multiplier": {
                "min": multipliers[0], "max": multipliers[-1],
                "median": median(multipliers)},
        }
    quantized = [row for row in rows
                 if row["dtype"] in {"int8", "fp8-e4m3"}]
    ratios = sorted(row["overlap_ratio"] for row in quantized)
    all_ratios = sorted(row["overlap_ratio"] for row in rows)
    multipliers = sorted(row["contention_degradation_multiplier"] for row in quantized)
    summary = {
        "schema": "a5-mc2-chain-evaluation/1", "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY, "chains": rows, "rows": rows,
        "dtypes": sorted(by_dtype), "by_dtype": by_dtype,
        "overlap_ratio": ({"min": ratios[0], "max": ratios[-1],
                           "median": median(ratios),
                           "mean": sum(ratios) / len(ratios)} if ratios else {}),
        "all_overlap_ratio": (
            {"min": all_ratios[0], "max": all_ratios[-1],
             "median": median(all_ratios),
             "mean": sum(all_ratios) / len(all_ratios)} if all_ratios else {}),
        "contention_degradation": (
            {"min": multipliers[0], "max": multipliers[-1],
             "median": median(multipliers)} if multipliers else {}),
    }
    return summary


def build_a5_followup_manifest(ingested=None):
    """Return a deterministic verified/pending cell manifest."""
    data = _data_for(ingested)
    verified = {}
    for row in data.get("conditions") or ():
        if row["single"]:
            continue
        key = (row["collective"], int(row["world_size"]), row["placement_scope"],
               row["dtype"], row["regime"], row.get("pair"))
        entry = verified.setdefault(key, {"campaigns": set(), "payloads": set(),
                                          "unconfirmed": False})
        entry["campaigns"].add(row["campaign"])
        entry["payloads"].add(row["payload_bytes"])
        entry["unconfirmed"] = entry["unconfirmed"] or row["expansion_unconfirmed"]
    verified_cells = [_ManifestCell({
        "collective": key[0], "world_size": key[1], "placement_scope": key[2],
        "dtype": key[3], "expansion_mode": key[4], "p2p_pair": key[5],
        "campaigns": sorted(value["campaigns"]),
        "payload_count": len(value["payloads"]),
        "expansion_unconfirmed": value["unconfirmed"], "evidence": "measured",
    }) for key, value in sorted(verified.items(), key=lambda item: str(item[0]))]
    has_confirmed = {"aicpu", "aiv"} <= {
        cell["expansion_mode"] for cell in verified_cells}
    has_four_hop = any(
        cell["collective"] == "p2p_send_recv"
        and cell["placement_scope"] == "intra_node_ub"
        and cell.get("p2p_pair") == "0->4" for cell in verified_cells)
    has_w64 = any(cell["world_size"] >= 64 for cell in verified_cells)
    wide_all_to_allv = {cell["world_size"] for cell in verified_cells
                        if cell["collective"] == "all_to_allv"}
    has_wide_all_to_allv = {16, 32} <= wide_all_to_allv
    pending_cells = []
    if not has_confirmed:
        pending_cells.append(_ManifestCell({
            "cell": "expansion_regime:aicpu/aiv",
            "requires": "msprof task-trace confirmation",
            "reason": "no measured row observed an AICPU or AIV expansion engine"}))
    if not has_w64:
        pending_cells.append(_ManifestCell({
            "cell": "collective:W64-8node",
            "requires": "W64 8-node expansion",
            "reason": "W32 is the widest measured mesh; W64 extrapolation is unverified"}))
    if not has_four_hop:
        pending_cells.append(_ManifestCell({
            "cell": "p2p_send_recv:0->4:intra_node_ub",
            "requires": "0->4 intra-node P2P pair",
            "reason": "only the 0->1 intra-node pair is measured"}))
    pending_cells.append(_ManifestCell({
        "cell": "contention:live_sweeps",
        "requires": "live concurrent contention sweeps",
        "reason": "contention multipliers are analytic, not a measured sweep"}))
    if not has_wide_all_to_allv:
        pending_cells.append(_ManifestCell({
            "cell": "all_to_allv:W16-W32",
            "requires": "measured all_to_allv lane",
            "reason": "all_to_allv curves outside W2-W8 are derived from all_to_all"}))
    pending_cells.sort(key=lambda item: item["cell"])
    return {
        "schema": "a5-followup-manifest/1",
        "hardware_key": A5_HARDWARE_KEY, "arch_family": A5_ARCH_FAMILY,
        "source_campaigns": sorted(data.get("campaign_labels") or ()),
        "verified_cells": verified_cells,
        "pending_cells": pending_cells,
        "verified_cell_ids": [f"{cell['collective']}:W{cell['world_size']}:"
                              f"{cell['placement_scope']}:{cell['dtype']}:"
                              f"{cell['expansion_mode']}" for cell in verified_cells],
        "pending_cell_ids": [cell["cell"] for cell in pending_cells],
        "summary": {
            "verified_cells": len(verified_cells),
            "pending_cells": len(pending_cells),
            "verified_world_sizes": sorted({cell["world_size"]
                                            for cell in verified_cells}),
            "unconfirmed_conditions": sum(1 for row in data.get("conditions") or ()
                                          if row["single"] is False
                                          and row["expansion_unconfirmed"]),
        },
    }


class _ManifestCell(dict):
    """Structured cell whose membership also matches its readable text."""

    def _text(self):
        return json.dumps(self, sort_keys=True)

    def __contains__(self, item):
        if super().__contains__(item):
            return True
        return str(item).lower() in self._text().lower()

    def __eq__(self, other):
        if isinstance(other, str):
            return other in {self.get("cell"), self._text()}
        return dict.__eq__(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def lower(self): return self._text().lower()
    def upper(self): return self._text().upper()
    def startswith(self, prefix): return self._text().startswith(prefix)
    def endswith(self, suffix): return self._text().endswith(suffix)
