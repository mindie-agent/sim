"""Communication service-curve calibration bundle and strict software partitioning.

[Role] Fit strictly keyed UB-mesh/URMA communication ``ServiceCurve`` objects
  from normalized campaign records and expose ``A5CommCalibrationBundle`` lookup,
  binding, regime, and recipe delegation.
[Invariants] Curves are keyed by (collective, world_size, placement_scope,
  dtype, expansion regime, pair, stack, CANN, algorithm, protocol). Unconfirmed
  requests never promote; incompatible software stacks never pool; single-launch
  anchors never enter fits; heldout payloads never train.
[Timeouts & Locks] Pure in-memory fitting; no timers, locks, or subprocesses.
[Side-Effect Policy] Pure computation; never launches a device, writes
  calibration files, or mutates primitive stores.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping

from mindie_sim.prediction.services import ServiceCurve
from mindie_sim.schema.common import ModelingError
from ..service_fitting import fit_service_curve
from .binding import (
    _SUPPORTED_COLLECTIVES,
    bind_service_curve,
    to_service_regime,
)
from .golden import A5_ARCH_FAMILY, A5_HARDWARE_KEY, A5_SKU
from .ingest import _coerce_ingested, ingest_a5_campaign_data
from .normalization import (
    INTRA_NODE_MAX_RANKS,
    _collective,
    _dtype,
    _explicit_pair,
    _explicit_scope,
    _fallback_measurement,
    _fallback_transport_window,
    _mode,
    _protocol_key,
    _require_a5,
    _validate_measurement_window,
    resolve_expansion_regime,
)
from .recipes import build_a5_comm_recipe
from .streaming import _fold_timings, _weighted_median

DEFAULT_HOLDOUT_PAYLOADS = (65536, 16_777_216)
_EXPANSION_MODES = frozenset({"aicpu", "aiv"})
_PAIR_ORDER = {"0->1": 0, "0->4": 1, "0->8": 2, "0->24": 3}
_PAIR_FALLBACK_ORDER = 99
_FAR_PAIR_WORLD_ALIAS = 24
# Source direction of the data movement each collective performs.  AllGather
# channels read peers; AllReduce/ReduceScatter steps and AllToAll exchanges
# write/send.  This is the HCCL contract direction, not a measured link sample.
_COLLECTIVE_DIRECTION = {
    "all_gather": "read", "all_reduce": "send", "reduce_scatter": "send",
    "all_to_all": "send", "all_to_allv": "send", "p2p_send_recv": "directed",
}
_PRIMARY_STACKS = ("rack06", "rack06-07", "rack06-09")
_AMBIGUOUS = "a5_incompatible_software_stacks_require_selector"

__all__ = [
    "A5CommCalibrationBundle",
    "DEFAULT_HOLDOUT_PAYLOADS",
    "bind_service_curve",
    "build_a5_comm_recipe",
    "fit_a5_service_curves",
    "ingest_a5_campaign_data",
    "resolve_expansion_regime",
    "to_service_regime",
]


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
                     "variant": "DT", "architecture": A5_ARCH_FAMILY},
        "transport": "ub_mesh" if scope.startswith("intra") else "urma",
        "collective": collective, "world_size": int(world_size),
        "placement_scope": scope, "dtype": dtype,
        "expansion_regime": regime, "p2p_pair": pair, "evidence": evidence,
        "stack": software[0], "cann": software[1], "algorithm": software[2],
        "protocol": software[3],
    }


def _shared_domain(scope, world_size, pair=None):
    """One physical sharing domain, not a per-call duration or hop price.

    A collective's members share the same intra-node UB mesh or the same
    cross-node URMA fabric.  A direct pair names its exact directed pair.
    """
    intra = str(scope).startswith("intra")
    domain = {
        "id": (f"{'ub_mesh' if intra else 'urma'}:{scope}:W{int(world_size)}"
               + (f":{pair}" if pair else "")),
        "kind": "ub_mesh" if intra else "urma",
        "scope": str(scope),
        "members": int(world_size),
        "inter_node": not intra,
    }
    if pair:
        domain["pair"] = str(pair)
    return domain


def _measurement_from_rows(rows, collective, boundary):
    """Canonicalize every row's measured phase before comparing/aggregating.

    Native extraction attaches ``measurement`` to every condition.  Legacy
    campaign rows without one get a labeled campaign fallback **per row**, so
    a mixed explicit/fallback group cannot silently inherit the explicit row's
    phase.  No row is classified as native by a path suffix or campaign name.
    """
    canonical = []
    for row in rows:
        declared = row.get("measurement")
        if isinstance(declared, Mapping):
            measurement = dict(declared)
            measurement.setdefault("explicit", True)
        else:
            measurement = _fallback_measurement(
                collective, boundary, row.get("algorithm") is not None)
        canonical.append(measurement)
    first = canonical[0]
    if any(item != first for item in canonical):
        raise ValueError("a5_measurement_boundary_mixed_sources")
    return first


def _transport_window(rows, collective, boundary):
    """Canonicalize every row's window; a missing row gets its own fallback."""
    canonical = []
    for row in rows:
        declared = row.get("transport_window")
        if isinstance(declared, Mapping):
            window = dict(declared)
        else:
            measurement = row.get("measurement")
            if not isinstance(measurement, Mapping):
                measurement = _fallback_measurement(
                    collective, boundary, row.get("algorithm") is not None)
            window = _fallback_transport_window(dict(measurement))
        canonical.append(window)
    first = canonical[0]
    if any(item != first for item in canonical):
        raise ValueError("a5_transport_window_mixed_sources")
    measurement = rows[0].get("measurement")
    if not isinstance(measurement, Mapping):
        measurement = _fallback_measurement(
            collective, boundary, rows[0].get("algorithm") is not None)
    _validate_measurement_window(dict(measurement), first)
    return first


def _measurement_boundary(collective, boundary, native=False,
                          algorithm_identified=False):
    """Describe the measured interval, not the consumer's later role.

    A native ``HcclLaunchAicpuKernel`` AI_CPU child is not a complete
    collective call.  A campaign per-op collective is a complete measured call
    envelope.  A directed p2p call is a complete directed call.  An explicit
    resource-phase row is a phase, not a device call.  Binding and export must
    preserve this descriptor instead of relabelling every collective.
    """
    if native:
        return {
            "kind": "native_target_call", "phase": "aicpu_child",
            "resource": "aicpu_child",
            "work_coordinate": "per_rank_input_bytes",
            "duration_unit": "ms", "source_kind": "native_ai_cpu_child",
            "service_semantics": "aicpu_child_collective_phase",
            "complete_collective_call": False,
            "complete_directed_call": False,
            "link_transit_identified": False,
            "hop_scaling": "not_derived",
            "algorithm_identified": bool(algorithm_identified),
            "direction_observed": False,
        }
    if boundary == "resource_phase":
        return {
            "kind": "resource_phase", "phase": "collective_phase",
            "resource": "collective_phase",
            "work_coordinate": "captured_per_rank_payload",
            "duration_unit": "ms", "source_kind": "campaign_resource_phase",
            "service_semantics": "collective_phase_service",
            "complete_collective_call": False,
            "complete_directed_call": False,
            "link_transit_identified": False,
            "hop_scaling": "not_derived",
            "algorithm_identified": bool(algorithm_identified),
            "direction_observed": False,
        }
    if collective == "p2p_send_recv":
        return {
            "kind": "device_call", "phase": "directed_p2p_call",
            "resource": "directed_p2p_call",
            "work_coordinate": "captured_per_rank_payload",
            "duration_unit": "ms", "source_kind": "campaign_device_call",
            "service_semantics": "directed_p2p_call",
            "complete_collective_call": False,
            "complete_directed_call": True,
            "link_transit_identified": False,
            "hop_scaling": "not_derived",
            "algorithm_identified": bool(algorithm_identified),
            "direction_observed": True,
        }
    return {
        "kind": "device_call", "phase": "collective_call",
        "resource": "collective_call",
        "work_coordinate": "captured_per_rank_payload",
        "duration_unit": "ms", "source_kind": "campaign_device_call",
        "service_semantics": "whole_collective_call_envelope",
        "complete_collective_call": True,
        "complete_directed_call": False,
        "link_transit_identified": False,
        "hop_scaling": "not_derived",
        "algorithm_identified": bool(algorithm_identified),
        "direction_observed": False,
    }


def _service_direction(collective, pair):
    if collective == "p2p_send_recv":
        return str(pair) if pair else "directed"
    return _COLLECTIVE_DIRECTION.get(collective, "unspecified")


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
            samples.append({
                "work": work, "duration_ms": _weighted_median(vals),
                "run_id": run_id, "repeat_index": int(r0.get("repeat_index") or 0),
                "source": (r0.get("source")
                           or f"{r0['campaign']}:{collective}:W{world_size}:{work}"),
                "split": split,
            })
        else:
            samples.extend({
                "work": work, "duration_ms": r["per_op_ms"], "run_id": run_id,
                "repeat_index": int(r.get("repeat_index") or 0),
                "source": (r.get("source")
                           or f"{r['campaign']}:{collective}:W{world_size}:{work}"),
                "split": split,
            } for r in grp)
    train = {sample["work"] for sample in samples if sample["split"] == "train"}
    if len(train) < 2:
        return None, None, "a5_curve_requires_two_training_payloads"
    identity = _curve_identity(collective, world_size, scope, dtype, regime, pair,
                               evidence, software)
    identity["protocol"] = next((row.get("protocol") for row in rows
                                 if row.get("protocol") is not None), software[3])
    direction = _service_direction(collective, pair)
    shared = _shared_domain(scope, world_size, pair)
    measurement = _measurement_from_rows(rows, collective, boundary)
    window = _transport_window(rows, collective, boundary)
    semantics = measurement["service_semantics"]
    identity.update({
        "direction": direction, "shared_domain": shared,
        "transport_window": window, "measurement_boundary": measurement,
        "service_semantics": semantics,
        "work_scope": "captured_per_rank_payload",
    })
    if extra_identity:
        identity.update(extra_identity)
    provenance = {
        "hardware_key": A5_HARDWARE_KEY, "sku": A5_SKU,
        "arch_family": A5_ARCH_FAMILY, "hardware_variant": "DT",
        "campaigns": sorted({row["campaign"] for row in rows}),
        "evidence": evidence, "holdout_payloads": sorted(holdout_payloads),
        "expansion_regime": regime,
        "expansion_unconfirmed": any(row["expansion_unconfirmed"] for row in rows),
        "p2p_pair": pair, "stack": software[0], "cann": software[1],
        "algorithm": software[2],
        "protocol": next((row.get("protocol") for row in rows
                          if row.get("protocol") is not None), None),
        "topology": {"placement_scope": scope, "world_size": int(world_size),
                     "transport": identity["transport"]},
        "software": {"stack": software[0], "cann": software[1],
                     "algorithm": software[2], "protocol": software[3]},
        "direction": direction, "shared_domain": shared,
        "transport_window": window, "service_semantics": semantics,
        "measurement_boundary": measurement,
        "work_scope": "captured_per_rank_payload",
        # A measured call/phase is a completion envelope, not a per-hop link
        # price.  Whether a link transit rate is identifiable is recorded
        # explicitly instead of deriving one by multiplying hops.
        "link_transit_identified": measurement["link_transit_identified"],
        "hop_scaling": measurement["hop_scaling"],
        "direction_basis": ("declared_directed_pair" if collective == "p2p_send_recv"
                            else "hccl_source_template_native_direction"),
        "direction_observed": measurement["direction_observed"],
        "algorithm_identified": measurement["algorithm_identified"],
        "shared_domain_basis": "declared_placement_scope_and_group",
        "transport_window_basis": "source_artifact_kind",
    }
    if extra_identity and extra_identity.get("derived_from"):
        provenance["derived_from"] = extra_identity["derived_from"]
    provenance["boundary"] = boundary
    packet = {
        "resource": measurement["resource"], "work_unit": "bytes",
        "identity": identity, "boundary": boundary,
        "active_units": max(1, int(world_size)),
        "samples": samples, "provenance": provenance,
    }
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
    del cell
    primary = [option for option in options if option[0][0] in _PRIMARY_STACKS
               and option[0][1] in {None, "9.1.0"}]
    if primary:
        primary.sort(key=lambda option: (
            option[0][1] != "9.1.0",
            _PRIMARY_STACKS.index(option[0][0]), str(option[0])))
        return primary[0][1]
    if len(options) == 1:
        return options[0][1]
    raise ModelingError(_AMBIGUOUS)


def _cell_fields(key):
    return dict(zip(("collective", "world_size", "placement_scope", "dtype",
                     "expansion_mode"), key[:5]))


@dataclass(frozen=True)
class A5CommCalibrationBundle:
    """Strictly keyed communication service curves plus their independent fit evidence."""

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
            scope = ("cross_node_urma" if int(world_size) > INTRA_NODE_MAX_RANKS
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
                expansion_mode=key[4],
                allow_unspecified_fallback=allow_unspecified_fallback)
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
            "curves": {
                ":".join((key[0], f"W{key[1]}", key[2], key[3], key[4])): curve.to_dict()
                for key, curve in sorted(self.curves.items(), key=lambda item: str(item[0]))
            },
            "stack_curves": {
                ":".join(str(part) for part in
                         (key[0], f"W{key[1]}", key[2], key[3], key[4], *key[5:])):
                curve.to_dict()
                for key, curve in sorted(self.stack_curves.items(), key=lambda item: str(item[0]))
            },
            "fits": list(self.fits), "holdout": list(self.holdout),
            "single_op_anchors": list(self.single_op_anchors),
            "campaigns": list(self.campaigns), "unfitted": list(self.unfitted),
            "provenance": dict(self.provenance),
        }

    def bind_service_curve(self, case, op, scope="none", *,
                           require_complete_call=False, production=False,
                           **lookup_kwargs):
        return bind_service_curve(
            self, case, op, scope,
            require_complete_call=require_complete_call,
            production=production, **lookup_kwargs)

    def to_service_regime(self, case, op, scope="none", *,
                          require_complete_call=False, production=False,
                          **lookup_kwargs):
        return to_service_regime(
            self, case, op, scope,
            require_complete_call=require_complete_call,
            production=production, **lookup_kwargs)

    def build_a5_comm_recipe(self, collective=None, *,
                             allow_unspecified_fallback=False,
                             require_complete_call=False, production=False,
                             **lookup_kwargs):
        return build_a5_comm_recipe(
            self, collective,
            allow_unspecified_fallback=allow_unspecified_fallback,
            require_complete_call=require_complete_call,
            production=production, **lookup_kwargs)


def fit_a5_service_curves(ingested=None, *,
                          holdout_payloads=DEFAULT_HOLDOUT_PAYLOADS):
    """Fit one ServiceCurve per strict communication software partition."""
    data = _data_for(ingested)
    _require_a5(data.get("hardware_key") or data.get("sku") or A5_HARDWARE_KEY,
                data.get("arch_family"))
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
        fits.append({
            **fitted, "cell": cell, **_cell_fields(cell),
            "p2p_pair": key[5], "evidence": evidence,
            "stack": key[6], "cann": key[7], "algorithm": key[8],
            "boundary": key[10],
        })
        holdout_rows.extend({
            **row, "cell": cell, **_cell_fields(cell),
            "p2p_pair": key[5], "split": "heldout",
        } for row in fitted["heldout"])
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
                    _PAIR_ORDER.get(value[0], _PAIR_FALLBACK_ORDER),
                    str(value[3])))
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
                        targets[(cell[0], _FAR_PAIR_WORLD_ALIAS, cell[2],
                                 cell[3], cell[4])].append(entry)
                        targets[(cell[0], cell[1], "cross_node_urma_far",
                                 cell[3], cell[4])].append(entry)
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
