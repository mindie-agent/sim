"""Diagnostic contention analysis, MC2 chain summaries, and follow-up manifests.

[Role] Summarize shared-pipe contention sensitivity, MC2 chain stage overlap,
  and deterministic verified/pending calibration cell manifests.
[Invariants] Analytic contention estimates remain diagnostic observations and
  never alter ordinary production pricing. AIV shares vector ALU pipes while
  AICPU/unspecified leaves vector slowdown at 1.0. Single-launch anchors never
  appear in verified follow-up cells.
[Timeouts & Locks] Pure in-memory analysis; no timers, locks, or I/O.
[Side-Effect Policy] Pure functions; never mutates input collections or launches
  device work.
"""
from __future__ import annotations

import json
import math
from statistics import median

from mindie_sim.schema.common import ModelingError
from .calibration import _data_for
from .golden import A5_ARCH_FAMILY, A5_HARDWARE_KEY
from .normalization import _EXPANSION_MODES, _dtype, _mode, resolve_expansion_regime
from .recipes import DEFAULT_ACTIVE_AIV_BLOCKS

DEFAULT_TOTAL_AIV_UNITS = 64
_CONTENTION_UTIL_CAP = 4.0
_MTE_SELF_CONTENTION = 0.55
_MTE_VECTOR_CONTENTION = 0.10
_MTE_CUBE_CONTENTION = 0.05
_HBM_MTE_CONTENTION = 0.45
_HBM_VECTOR_CONTENTION = 0.15
_HBM_CUBE_CONTENTION = 0.20
_CUBE_SELF_CONTENTION = 0.35
_CUBE_MTE_CONTENTION = 0.05
_AIV_VECTOR_BASE_CONTENTION = 0.55
_AIV_VECTOR_OCCUPANCY_CONTENTION = 0.45
_AIV_CUBE_CONTENTION = 0.10
_TRANSFER_BASE_MS = 0.012
_TRANSFER_MTE_MS = 0.05
_TRANSFER_VECTOR_MS = 0.01
_SYNC_BASE_MS = 0.008
_SYNC_UTIL_MS = 0.02
_SYNC_PER_AIV_BLOCK_MS = 0.0005

_MIN_MC2_CONCURRENT_UTIL = 0.01
_MC2_CUBE_UTIL_WEIGHT = 0.40
_MC2_VECTOR_UTIL_WEIGHT = 0.75

_TARGET_WIDE_MESH_RANKS = 64
_WIDE_ALL_TO_ALLV_RANKS = frozenset({16, 32})

__all__ = [
    "DEFAULT_TOTAL_AIV_UNITS",
    "build_a5_followup_manifest",
    "compute_a5_contention_slowdown",
    "evaluate_a5_mc2_chains",
]


def compute_a5_contention_slowdown(expansion_mode, *, concurrent_cube_util=0.0,
                                   concurrent_vector_util=0.0,
                                   concurrent_mte_util=0.0,
                                   active_aiv_blocks=DEFAULT_ACTIVE_AIV_BLOCKS,
                                   total_aiv_units=DEFAULT_TOTAL_AIV_UNITS):
    """Shared-vector / MTE / HBM contention for an expansion execution mode."""
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
    cube = min(_CONTENTION_UTIL_CAP, float(concurrent_cube_util))
    vector_demand = min(_CONTENTION_UTIL_CAP, float(concurrent_vector_util))
    mte = min(_CONTENTION_UTIL_CAP, float(concurrent_mte_util))
    mte_slowdown = (
        1.0 + _MTE_SELF_CONTENTION * mte
        + _MTE_VECTOR_CONTENTION * vector_demand
        + _MTE_CUBE_CONTENTION * cube
    )
    hbm_slowdown = (
        1.0 + _HBM_MTE_CONTENTION * mte
        + _HBM_VECTOR_CONTENTION * vector_demand
        + _HBM_CUBE_CONTENTION * cube
    )
    cube_slowdown = (
        1.0 + _CUBE_SELF_CONTENTION * cube
        + _CUBE_MTE_CONTENTION * mte
    )
    vector_slowdown = 1.0
    if mode == "aiv":
        # AIV owns the vector pipe; more resident blocks and more concurrent
        # vector demand extend the same ALU service time.
        vector_slowdown = (
            1.0
            + vector_demand * (
                _AIV_VECTOR_BASE_CONTENTION
                + _AIV_VECTOR_OCCUPANCY_CONTENTION * occupancy
            )
            + _AIV_CUBE_CONTENTION * cube
        )
    total = max(vector_slowdown, mte_slowdown, hbm_slowdown, cube_slowdown)
    transfer = (
        _TRANSFER_BASE_MS
        + _TRANSFER_MTE_MS * mte
        + _TRANSFER_VECTOR_MS * vector_demand
    )
    sync = (
        _SYNC_BASE_MS
        + _SYNC_UTIL_MS * max(mte, vector_demand, cube)
        + _SYNC_PER_AIV_BLOCK_MS * blocks
    )
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
        concurrent = max(_MIN_MC2_CONCURRENT_UTIL, min(1.0, 1.0 - min(1.0, ratio)))
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
            mode,
            concurrent_cube_util=_MC2_CUBE_UTIL_WEIGHT * concurrent,
            concurrent_vector_util=_MC2_VECTOR_UTIL_WEIGHT * concurrent,
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
        multipliers = sorted(
            row["contention_degradation_multiplier"] for row in subset)
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
    quantized = [row for row in rows if row["dtype"] in {"int8", "fp8-e4m3"}]
    ratios = sorted(row["overlap_ratio"] for row in quantized)
    all_ratios = sorted(row["overlap_ratio"] for row in rows)
    multipliers = sorted(
        row["contention_degradation_multiplier"] for row in quantized)
    return {
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

    def lower(self):
        return self._text().lower()

    def upper(self):
        return self._text().upper()

    def startswith(self, prefix):
        return self._text().startswith(prefix)

    def endswith(self, suffix):
        return self._text().endswith(suffix)


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
    has_w64 = any(
        cell["world_size"] >= _TARGET_WIDE_MESH_RANKS for cell in verified_cells)
    wide_all_to_allv = {cell["world_size"] for cell in verified_cells
                        if cell["collective"] == "all_to_allv"}
    has_wide_all_to_allv = _WIDE_ALL_TO_ALLV_RANKS <= wide_all_to_allv
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
        "verified_cell_ids": [
            f"{cell['collective']}:W{cell['world_size']}:"
            f"{cell['placement_scope']}:{cell['dtype']}:{cell['expansion_mode']}"
            for cell in verified_cells
        ],
        "pending_cell_ids": [cell["cell"] for cell in pending_cells],
        "summary": {
            "verified_cells": len(verified_cells),
            "pending_cells": len(pending_cells),
            "verified_world_sizes": sorted({cell["world_size"]
                                            for cell in verified_cells}),
            "unconfirmed_conditions": sum(
                1 for row in data.get("conditions") or ()
                if row["single"] is False and row["expansion_unconfirmed"]),
        },
    }
