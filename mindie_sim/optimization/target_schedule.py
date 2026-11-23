"""Hardware-grounded Die + Topology contention Engine scheduler & exposure attribution.

[Role] Schedule target-state operator graphs on a unified Die + Topology
  ``Engine`` containing per-stream ``units=1`` FIFO hardware queues
  (``stream_0/1/2``), die ``aic`` / ``aiv`` / ``hbm`` / ``mte2`` / ``mte3`` /
  ``fixpipe`` pools, cross-die ``Topology`` links, persistent whole-die
  admission reservations, and explicit ``StreamCoreBudget(isolation="disjoint")``
  core-split re-pricing, paired with additive Engine service-grant exposure
  attribution.
[Invariants] Same-stream events on one physical die have identically ``0.0 us``
  intersection. Persistent whole-die kernels occupy all core execution slots
  from admission through final completion. Concurrent active AIC/AIV grants
  never exceed physical die counts. Additive exposure conserves step wall time.
[Timeouts & Locks] Single-pass bounded Engine replay per candidate; no I/O or locks.
[Side-Effect Policy] Pure functional scheduling over immutable node records.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
import re
from typing import Any

from mindie_sim.prediction.collectives import Topology
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.prediction.streams import (
    MAIN_STREAM,
    StreamCoreBudget,
    StreamLane,
    StreamOwners,
    StreamPlan,
    bind_stream_owners,
)
from mindie_sim.primitives.loader import resolve_sim_rate_overlay
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import DatapathCapabilitySpec, datapath_for_die, ub_peer_edges_for_rank
from mindie_sim.schema.hardware import Device

from .causal_tail import (
    PERSISTENT_MOE_FAMILIES,
    is_persistent_whole_die_node,
)

US_PER_MS = 1000.0
MS_PER_SECOND = 1000.0
BYTES_PER_GB = 1_000_000_000.0
PERCENT_SCALE = 100.0
TIME_EPS_US = 1e-6
EXPOSED_EPS_US = 1e-3
SECONDARY_MASKER_SHARE = 0.25
MAX_MASKED_BY_LEN = 48

#: Admission capacity for concurrent non-persistent streams on one physical die.
DIE_EXEC_CONCURRENCY_SLOTS = 16
#: Minimum relative makespan improvement (0.5%) to adopt a disjoint core split.
DISJOINT_SPLIT_MIN_GAIN_RATIO = 0.005
#: Candidate main-lane AIC split fractions (numerator, denominator) evaluated for disjoint budgets.
DISJOINT_SPLIT_RATIOS: tuple[tuple[int, int], ...] = (
    (3, 4),
    (5, 8),
    (1, 2),
)
#: Default cube tile width used when estimating discrete wave counts from GEMM shape.
DEFAULT_CUBE_TILE_M = 128
DEFAULT_CUBE_TILE_N = 128
MATMUL_3D_RANK = 3
#: Default vector tile elements per AIV core for wave quantization under disjoint splits.
DEFAULT_VEC_TILE_ELEMS = 4096
#: HBM read/write asymmetry threshold matching FusedPipelineRecipe DRAM mode selection.
HBM_RW_ASYMMETRY_RATIO = 1.5
#: Pure-dispatch/combine families executing on AIV + Link with zero AIC usage.
ZERO_AIC_COMM_FAMILIES = frozenset({"dispatch", "combine", "alltoall", "alltoallv", "mc2_transport", "mc2_protocol"})
_COLLECTIVE_TASK_IDX_RE = re.compile(r"^c:(\d+):")


@dataclass(frozen=True, slots=True)
class DieScheduleResult:
    """Result of executing a target graph on the real Die + Topology Engine."""

    nodes: tuple[Any, ...]
    archetype_spans_us: Mapping[str, float]
    step_start_us: float
    step_end_us: float
    step_span_us: float
    stream_core_budget: Mapping[tuple[int, int, int], Mapping[str, Any]]
    stream_core_budget_summary: str
    max_same_stream_overlap_us: float


class ServiceGrantObserver:
    """Lightweight read-only ``WaveRecorder`` observer capturing per-op Engine grants."""

    __slots__ = ("current_node_idx", "task_to_node", "admit_ms_by_node", "_grant_acc", "_slot_gen")

    def __init__(self) -> None:
        self.current_node_idx: int | None = None
        self.task_to_node: dict[int, int] = {}
        self.admit_ms_by_node: dict[int, float] = {}
        self._grant_acc: dict[tuple[float, float, str, int], float] = {}
        self._slot_gen: dict[tuple[str, int, str], int] = {}

    @property
    def grants(self) -> list[tuple[float, float, str, int, float]]:
        return [(t0, t1, rname, idx, deliv) for (t0, t1, rname, idx), deliv in self._grant_acc.items()]

    def register_pipe(self, _name: str) -> int:
        return 0

    def on_submit(self, task_id: int, name: str = "", *_args: Any, **_kwargs: Any) -> None:
        idx = self.current_node_idx
        if idx is None and name:
            match = _COLLECTIVE_TASK_IDX_RE.match(str(name))
            if match is not None:
                idx = int(match.group(1))
        if idx is not None:
            self.task_to_node[int(task_id)] = idx

    def on_admit(self, task_id: int, now_ms: float, *_args: Any, **_kwargs: Any) -> None:
        idx = self.task_to_node.get(int(task_id))
        if idx is not None:
            prev = self.admit_ms_by_node.get(idx)
            if prev is None or now_ms < prev:
                self.admit_ms_by_node[idx] = float(now_ms)

    def on_service(
        self, task_id: int, physical_row: str, _owner: int,
        t0_ms: float, t1_ms: float, delivered: float, **_kwargs: Any,
    ) -> None:
        if delivered <= 0.0 or t1_ms <= t0_ms:
            return
        idx = self.task_to_node.get(int(task_id))
        if idx is not None:
            key = (float(t0_ms), float(t1_ms), str(physical_row), idx)
            self._grant_acc[key] = self._grant_acc.get(key, 0.0) + float(delivered)

    def allocate_slot_generation(self, pool: str, unit: int, name: str) -> tuple[int, int]:
        key = (str(pool), int(unit), str(name))
        gen = self._slot_gen.get(key, 0) + 1
        self._slot_gen[key] = gen
        return 0, gen

    def release_slot_generation(self, pool: str, unit: int, name: str) -> tuple[int, int]:
        return 0, self._slot_gen.get((str(pool), int(unit), str(name)), 0)

    def on_allocation(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def extract_cycle_pattern(self, *_args: Any, **_kwargs: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        return (), ()

    def on_period(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _short_node_label(node: Any) -> str:
    fam = str(getattr(node, "family", "") or "").lower()
    rec = str(getattr(node, "recipe", "") or "").lower()
    if fam in PERSISTENT_MOE_FAMILIES or "megamoe" in rec or "dispatch_ffn_combine" in rec:
        return "MoE Routed GMM / MegaMoE"
    if fam in {"allgather", "all_gather"}:
        return "Collective AllGather"
    if fam in {"indexer", "index_pool"}:
        return "Sparse Indexer"
    role = str(getattr(node, "logical_role", "") or getattr(node, "ascend_kernel", "") or getattr(node, "op_id", ""))
    return role[:MAX_MASKED_BY_LEN]


def _pick_witness_for_slice(
    active_idxs: Sequence[int],
    nodes: Sequence[Any],
    intervals_us: Sequence[tuple[float, float]],
    witness_set: frozenset[int],
) -> int:
    main_active = [i for i in active_idxs if int(nodes[i].stream_index) == 0]
    candidates = main_active if main_active else list(active_idxs)
    in_witness = [i for i in candidates if i in witness_set]
    pool = in_witness if in_witness else candidates
    return min(
        pool,
        key=lambda i: (
            0 if int(nodes[i].stream_index) == 0 else (1 if int(nodes[i].stream_index) == 2 else 2),
            -float(intervals_us[i][1]),
            -max(0.0, float(intervals_us[i][1]) - float(intervals_us[i][0])),
            int(nodes[i].ordinal),
            i,
        ),
    )


def _trace_completion_witness_set(
    nodes: Sequence[Any],
    intervals_us: Sequence[tuple[float, float]],
    causal_preds: Mapping[int, tuple[int, ...]],
) -> frozenset[int]:
    active_all = [i for i, (s, e) in enumerate(intervals_us) if e > s + TIME_EPS_US]
    if not active_all:
        return frozenset()
    cur = max(active_all, key=lambda i: (intervals_us[i][1], -int(nodes[i].stream_index), -i))
    witness: set[int] = set()
    while cur is not None and cur not in witness:
        witness.add(cur)
        cur_start = intervals_us[cur][0]
        cand_preds = [
            p for p in causal_preds.get(cur, ())
            if intervals_us[p][1] <= cur_start + EXPOSED_EPS_US and intervals_us[p][1] > intervals_us[p][0] + TIME_EPS_US
        ]
        if not cand_preds:
            break
        cur = max(cand_preds, key=lambda p: (intervals_us[p][1], -int(nodes[p].stream_index), -p))
    return frozenset(witness)


def _format_masked_labels(
    n: int,
    intervals_us: Sequence[tuple[float, float]],
    exposed: Sequence[float],
    mask_overlap: Sequence[Mapping[str, float]],
) -> list[str]:
    masked_by: list[str] = []
    for idx in range(n):
        dur_us = max(0.0, intervals_us[idx][1] - intervals_us[idx][0])
        if dur_us - exposed[idx] <= EXPOSED_EPS_US or not mask_overlap[idx]:
            masked_by.append("")
            continue
        ordered = sorted(mask_overlap[idx].items(), key=lambda kv: -kv[1])
        top_lbl, top_ov = ordered[0]
        if len(ordered) > 1 and ordered[1][1] > top_ov * SECONDARY_MASKER_SHARE:
            masked_by.append(f"{top_lbl} + {ordered[1][0]}"[:MAX_MASKED_BY_LEN])
        else:
            masked_by.append(top_lbl)
    return masked_by


def attribute_engine_exposure(
    nodes: Sequence[Any],
    intervals_us: Sequence[tuple[float, float]],
    observer: ServiceGrantObserver,
    causal_preds: Mapping[int, tuple[int, ...]],
    step_span_us: float,
) -> tuple[list[float], list[str]]:
    """Attribute Engine wall time additively across occurrences using physical service grants."""
    n = len(nodes)
    exposed = [0.0] * n
    mask_overlap: list[dict[str, float]] = [defaultdict(float) for _ in range(n)]
    if n == 0 or step_span_us <= TIME_EPS_US:
        return exposed, [""] * n

    witness_set = _trace_completion_witness_set(nodes, intervals_us, causal_preds)
    cuts_set: set[float] = {0.0, float(step_span_us)}
    for s_us, e_us in intervals_us:
        if e_us > s_us + TIME_EPS_US:
            cuts_set.add(max(0.0, min(step_span_us, s_us)))
            cuts_set.add(max(0.0, min(step_span_us, e_us)))
    grant_us: list[tuple[float, float, str, int, float]] = []
    for t0_ms, t1_ms, rname, idx, deliv in observer.grants:
        s_u = max(0.0, min(step_span_us, t0_ms * US_PER_MS))
        e_u = max(0.0, min(step_span_us, t1_ms * US_PER_MS))
        if e_u > s_u + TIME_EPS_US:
            grant_us.append((s_u, e_u, rname, idx, deliv))
            cuts_set.add(s_u)
            cuts_set.add(e_u)
    cuts = sorted(cuts_set)
    node_events = sorted(
        ((intervals_us[i][0], intervals_us[i][1], i) for i in range(n) if intervals_us[i][1] > intervals_us[i][0] + TIME_EPS_US),
        key=lambda item: item[0],
    )
    grant_us.sort(key=lambda item: item[0])
    n_ptr = g_ptr = 0
    live_nodes: list[tuple[float, int]] = []
    live_grants: list[tuple[float, float, str, int, float]] = []

    for lo, hi in zip(cuts, cuts[1:]):
        dt = hi - lo
        if dt <= TIME_EPS_US:
            continue
        mid = 0.5 * (lo + hi)
        while n_ptr < len(node_events) and node_events[n_ptr][0] <= mid:
            _, end_u, idx = node_events[n_ptr]
            if end_u > mid:
                live_nodes.append((end_u, idx))
            n_ptr += 1
        live_nodes = [(end_u, idx) for end_u, idx in live_nodes if end_u > mid]

        while g_ptr < len(grant_us) and grant_us[g_ptr][0] <= mid:
            if grant_us[g_ptr][1] > mid:
                live_grants.append(grant_us[g_ptr])
            g_ptr += 1
        live_grants = [g for g in live_grants if g[1] > mid]

        if not live_nodes:
            continue
        active_idxs = [idx for _, idx in live_nodes]
        if len(active_idxs) == 1:
            exposed[active_idxs[0]] += dt
            continue

        witness = _pick_witness_for_slice(active_idxs, nodes, intervals_us, witness_set)
        grants_by_res: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for g_lo, g_hi, rname, idx, deliv in live_grants:
            span = g_hi - g_lo
            if span > TIME_EPS_US:
                grants_by_res[rname][idx] += deliv * (dt / span)
        contested_res = [r for r, cl in grants_by_res.items() if witness in cl and len(cl) >= 2]
        witness_lbl = _short_node_label(nodes[witness])
        if contested_res:
            share_per_res = dt / float(len(contested_res))
            credited: set[int] = set()
            for rname in contested_res:
                claimants = grants_by_res[rname]
                tot = sum(claimants.values())
                if tot > 0.0:
                    for idx, amt in claimants.items():
                        exposed[idx] += share_per_res * (amt / tot)
                        credited.add(idx)
            for idx in active_idxs:
                if idx not in credited and int(nodes[idx].stream_index) != int(nodes[witness].stream_index):
                    mask_overlap[idx][witness_lbl] += dt
        else:
            exposed[witness] += dt
            for idx in active_idxs:
                if idx != witness and int(nodes[idx].stream_index) != int(nodes[witness].stream_index):
                    mask_overlap[idx][witness_lbl] += dt

    return exposed, _format_masked_labels(n, intervals_us, exposed, mask_overlap)


def _canonical_stream_lane_name(stream: str, stream_index: int) -> str:
    idx = int(stream_index)
    if idx == 0:
        return MAIN_STREAM
    raw = str(stream or "").strip()
    if raw and raw != MAIN_STREAM and raw.isascii() and not any(ch.isspace() for ch in raw):
        return raw
    return f"stream_{idx}"


def format_stream_core_budget_receipt(owners_by_lane: Mapping[str, StreamOwners]) -> str:
    """Format resolved integer core-budget receipt (e.g. ``disjoint (main: 24 AIC / 48 AIV, aux: 8 AIC / 16 AIV)``)."""
    if not owners_by_lane or not any(o.isolation == "disjoint" for o in owners_by_lane.values()):
        return ""
    parts = [f"{lid}: {int(o.aic_limit)} AIC / {int(o.aiv_limit)} AIV" for lid, o in owners_by_lane.items()]
    return f"disjoint ({', '.join(parts)})"


def _effective_die_supply(
    die: Device, links: Sequence[Any], channels: Sequence[Any], primitive_table: Any = None,
) -> tuple[DatapathCapabilitySpec, dict[str, float], dict[str, float], dict[str, tuple[str, float]], float]:
    ub_edges = ub_peer_edges_for_rank(die.rank, links, channels)
    datapath = datapath_for_die(die, ub_peer_edges=ub_edges)
    rates = dict(die.rates)
    peaks = dict(die.compute_peaks)
    service_modes = dict(die.service_modes)
    if primitive_table is not None:
        overlay = resolve_sim_rate_overlay(primitive_table, datapath, sku=str(getattr(die, "sku", "") or ""))
        for role, val in overlay.rates:
            if val > 0.0:
                rates[role] = min(val, rates[role]) if rates.get(role, 0.0) > 0.0 else val
        for dt, pk in overlay.compute_peaks:
            if pk > 0.0:
                peaks[dt] = pk
        if overlay.hbm_read_bytes_per_s is not None:
            service_modes["mte2_aiv_dram"] = ("mte2_aiv" if "mte2_aiv" in rates else "mte2", overlay.hbm_read_bytes_per_s)
        if overlay.hbm_write_bytes_per_s is not None:
            service_modes["hbm_write_dram"] = ("hbm", overlay.hbm_write_bytes_per_s)
        if overlay.hbm_rw11_bytes_per_s is not None:
            service_modes["hbm_rw11_dram"] = ("hbm", overlay.hbm_rw11_bytes_per_s)
        if overlay.hbm_rw21_bytes_per_s is not None:
            service_modes["hbm_rw21_dram"] = ("hbm", overlay.hbm_rw21_bytes_per_s)
    link_bps = float(rates.get("link") or 0.0)
    if links:
        def _lbps(lnk: Any) -> float:
            b_ms = float(getattr(lnk, "capacity_bytes_per_ms", 0.0) or 0.0)
            return b_ms * MS_PER_SECOND if b_ms > 0.0 else float(getattr(lnk, "capacity_gbps", 0.0) or 0.0) * BYTES_PER_GB

        best_bps = max((_lbps(lnk) for lnk in links), default=0.0)
        if best_bps > 0.0:
            link_bps = best_bps
    return datapath, rates, peaks, service_modes, link_bps


def _cube_peak_for_dtype(peaks: Mapping[str, float], dtype: str) -> float:
    tok = str(dtype or "bf16").strip().lower()
    aliases = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "fp8_e4m3": "fp8", "fp8_e5m2": "fp8"}
    for key in (tok, aliases.get(tok, tok), "bf16", "fp16"):
        val = float(peaks.get(key, 0.0) or 0.0)
        if val > 0.0:
            return val
    return max((float(v) for v in peaks.values() if float(v) > 0.0), default=0.0)


def _node_cube_peak_ms(node: Any, peaks: Mapping[str, float]) -> float:
    spec = getattr(node, "spec", None)
    entries = getattr(spec, "matrix_flops", ()) if spec is not None else ()
    if entries:
        total = sum(
            MS_PER_SECOND * float(fl) / pk
            for dt, fl in entries
            if float(fl) > 0.0 and (pk := _cube_peak_for_dtype(peaks, str(dt))) > 0.0
        )
        if total > 0.0:
            return total
    flops = float(getattr(node, "flops", 0.0) or 0.0)
    if flops <= 0.0:
        return 0.0
    pk = _cube_peak_for_dtype(peaks, str(getattr(node, "dtype", "bf16") or "bf16"))
    return (MS_PER_SECOND * flops / pk) if pk > 0.0 else 0.0


def _effective_hbm_rate_bps(
    hbm_read: float, hbm_write: float, hbm_peak: float, service_modes: Mapping[str, tuple[str, float]],
) -> float:
    def _mode_rate(name: str) -> float | None:
        entry = service_modes.get(name)
        return float(entry[1]) if (entry is not None and float(entry[1]) > 0.0) else None

    if hbm_read <= 0.0 or hbm_write > HBM_RW_ASYMMETRY_RATIO * hbm_read:
        rate = _mode_rate("hbm_write_dram") or _mode_rate("mte2_aiv_dram")
    elif hbm_write <= 0.0:
        rate = _mode_rate("mte2_aiv_dram")
    elif hbm_read > HBM_RW_ASYMMETRY_RATIO * hbm_write:
        rate = _mode_rate("hbm_rw21_dram") or _mode_rate("mte2_aiv_dram")
    else:
        rate = _mode_rate("hbm_rw11_dram") or _mode_rate("hbm_rw21_dram") or _mode_rate("mte2_aiv_dram")
    if rate is None or rate <= 0.0:
        return hbm_peak
    return min(rate, hbm_peak) if hbm_peak > 0.0 else rate


def _estimate_core_tiles(node: Any, *, kind: str, full_cores: int) -> int:
    spec = getattr(node, "spec", None)
    meta = dict(getattr(spec, "metadata", ()) or ()) if spec is not None else {}
    explicit = meta.get("matrix_tiles" if kind == "aic" else "vector_tiles") or meta.get("tiles")
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    shape = tuple(int(d) for d in (getattr(node, "shape", ()) or ()) if int(d) > 0)
    if kind == "aic":
        if len(shape) >= MATMUL_3D_RANK:
            m_t = max(1, math.ceil(shape[-MATMUL_3D_RANK] / DEFAULT_CUBE_TILE_M))
            n_t = max(1, math.ceil(shape[-2] / DEFAULT_CUBE_TILE_N))
            batch = math.prod(shape[:-MATMUL_3D_RANK]) if len(shape) > MATMUL_3D_RANK else 1
            return max(1, int(batch * m_t * n_t))
        if len(shape) == 2:
            return max(1, math.ceil(shape[0] / DEFAULT_CUBE_TILE_M) * math.ceil(shape[1] / DEFAULT_CUBE_TILE_N))
        return max(1, int(full_cores))
    elems = math.prod(shape) if shape else 0
    return max(1, math.ceil(elems / DEFAULT_VEC_TILE_ELEMS)) if elems > 0 else max(1, int(full_cores))


def _wave_scale_factor(tiles: int, full_cores: int, eff_cores: int) -> float:
    if eff_cores <= 0 or full_cores <= 0 or eff_cores >= full_cores:
        return 1.0
    return float(max(1, math.ceil(tiles / eff_cores))) / float(max(1, math.ceil(tiles / full_cores)))


def _node_physical_demand_and_placement(
    node: Any, *, die_tag: str, lane_id: str, datapath: DatapathCapabilitySpec,
    rates: Mapping[str, float], peaks: Mapping[str, float],
    service_modes: Mapping[str, tuple[str, float]], link_bps: float,
    owners: StreamOwners | None, has_topology_plan: bool, launch_floor_us: float,
) -> tuple[dict[str, float], dict[str, tuple[int, ...]], float, bool]:
    aic_full, aiv_full = max(1, int(datapath.aic_count)), max(1, int(datapath.aiv_count))
    is_disjoint = owners is not None and owners.isolation == "disjoint"
    aic_eff = int(owners.aic_limit) if is_disjoint else aic_full
    aiv_eff = int(owners.aiv_limit) if is_disjoint else aiv_full
    sfx = f":{lane_id}" if is_disjoint else ""

    cube_ms = _node_cube_peak_ms(node, peaks)
    vec_ops = max(0.0, float(getattr(node, "vector_ops", 0.0) or 0.0))
    hbm_r = max(0.0, float(getattr(node, "hbm_read_bytes", 0.0) or 0.0))
    hbm_w = max(0.0, float(getattr(node, "hbm_write_bytes", 0.0) or 0.0))
    hbm_tot = max(0.0, float(getattr(node, "hbm_total_bytes", 0.0) or 0.0))
    mte2_b = max(0.0, float(getattr(node, "mte2_bytes", 0.0) or 0.0))
    mte3_b = max(0.0, float(getattr(node, "mte3_bytes", 0.0) or 0.0))
    fixpipe_b = max(0.0, float(getattr(node, "fixpipe_bytes", 0.0) or 0.0))
    link_b = max(0.0, float(getattr(node, "link_bytes", 0.0) or 0.0))
    if str(getattr(node, "family", "") or "").strip().lower() in ZERO_AIC_COMM_FAMILIES:
        cube_ms = 0.0
    if cube_ms > 0.0 and aic_eff <= 0:
        raise ModelingError(f"stream_zero_aic_budget:{node.op_id}")
    if vec_ops > 0.0 and aiv_eff <= 0:
        raise ModelingError(f"stream_zero_aiv_budget:{node.op_id}")

    demand: dict[str, float] = {}
    solo_ms: dict[str, float] = {}
    aic_res, aiv_res = [], []
    pure_cube = cube_ms > 0.0 and vec_ops <= 0.0

    if cube_ms > 0.0:
        rname = f"matrix@{die_tag}{sfx}"
        demand[rname], solo_ms[rname] = cube_ms, cube_ms
        aic_res.append(rname)
    vec_rate_s = float(rates.get("vector") or 0.0)
    if vec_ops > 0.0 and vec_rate_s > 0.0:
        rname = f"vector@{die_tag}{sfx}"
        demand[rname], solo_ms[rname] = vec_ops, (vec_ops / vec_rate_s) * MS_PER_SECOND
        aiv_res.append(rname)
    hbm_peak = float(rates.get("hbm") or max(datapath.hbm_read_bytes_per_s, datapath.hbm_write_bytes_per_s) or 0.0)
    if hbm_tot > 0.0 and hbm_peak > 0.0:
        rname = f"memory@{die_tag}"
        eff_hbm = _effective_hbm_rate_bps(hbm_r, hbm_w, hbm_peak, service_modes)
        scaled = hbm_tot * (hbm_peak / eff_hbm) if eff_hbm > 0.0 else hbm_tot
        demand[rname], solo_ms[rname] = scaled, (scaled / hbm_peak) * MS_PER_SECOND
    if mte2_b > 0.0:
        use_aic_m2 = pure_cube or (is_disjoint and aiv_eff <= 0 and aic_eff > 0)
        m2_key = "mte2_aic" if (use_aic_m2 and "mte2_aic" in rates) else ("mte2_aiv" if (not use_aic_m2 and "mte2_aiv" in rates) else "mte2")
        m2_r = float(rates.get(m2_key) or hbm_peak or 0.0)
        if m2_r > 0.0:
            m2_sfx = sfx if (is_disjoint and (aic_eff > 0 if use_aic_m2 else aiv_eff > 0)) else ""
            rname = f"{m2_key}@{die_tag}{m2_sfx}"
            demand[rname], solo_ms[rname] = mte2_b, (mte2_b / m2_r) * MS_PER_SECOND
            (aic_res if use_aic_m2 else aiv_res).append(rname)
    if mte3_b > 0.0:
        m3_r = float(rates.get("mte3") or hbm_peak or 0.0)
        if m3_r > 0.0:
            use_aic_m3 = pure_cube or (is_disjoint and aiv_eff <= 0 and aic_eff > 0)
            m3_key = "mte3_aic" if (is_disjoint and use_aic_m3 and aic_eff > 0) else "mte3"
            m3_sfx = sfx if (is_disjoint and (aic_eff > 0 if use_aic_m3 else aiv_eff > 0)) else ""
            rname = f"{m3_key}@{die_tag}{m3_sfx}"
            demand[rname], solo_ms[rname] = mte3_b, (mte3_b / m3_r) * MS_PER_SECOND
            (aic_res if use_aic_m3 else aiv_res).append(rname)
    if fixpipe_b > 0.0:
        fp_r = float(rates.get("fixpipe") or hbm_peak or 0.0)
        if fp_r > 0.0:
            fp_sfx = sfx if (is_disjoint and aic_eff > 0) else ""
            rname = f"fixpipe@{die_tag}{fp_sfx}"
            demand[rname], solo_ms[rname] = fixpipe_b, (fixpipe_b / fp_r) * MS_PER_SECOND
            aic_res.append(rname)
    if link_b > 0.0 and not has_topology_plan and link_bps > 0.0:
        rname = f"link@{die_tag}"
        demand[rname], solo_ms[rname] = link_b, (link_b / link_bps) * MS_PER_SECOND

    base_dur_ms = max(0.0, float(getattr(node, "dur_us", 0.0) or 0.0)) / US_PER_MS
    floor_ms = max(0.0, float(launch_floor_us)) / US_PER_MS
    max_pipe_ms = max(solo_ms.values(), default=0.0)
    dom_bound = str(getattr(node, "dominant_bound", "") or "").lower()
    if (
        not has_topology_plan
        and base_dur_ms > max_pipe_ms + (TIME_EPS_US / US_PER_MS)
        and base_dur_ms > floor_ms + (TIME_EPS_US / US_PER_MS)
        and dom_bound != "launch_bound"
        and solo_ms
    ):
        pref = {
            "cube_bound": (f"matrix@{die_tag}{sfx}",),
            "vector_bound": (f"vector@{die_tag}{sfx}",),
            "memory_bound": (f"memory@{die_tag}", f"mte2_aiv@{die_tag}{sfx}", f"mte2_aic@{die_tag}{sfx}", f"mte2@{die_tag}{sfx}"),
            "link_bound": (f"link@{die_tag}",),
        }.get(dom_bound, ())
        dom_res = next((r for r in pref if r in solo_ms and solo_ms[r] > 0.0), None) or max(solo_ms.items(), key=lambda kv: kv[1])[0]
        if solo_ms[dom_res] > 0.0:
            demand[dom_res] *= base_dur_ms / solo_ms[dom_res]
            solo_ms[dom_res] = base_dur_ms

    if is_disjoint:
        for res_list, kind, full_n, eff_n, core_pref in (
            (aic_res, "aic", aic_full, aic_eff, "matrix@"),
            (aiv_res, "aiv", aiv_full, aiv_eff, "vector@"),
        ):
            if eff_n > 0 and res_list:
                wave = _wave_scale_factor(_estimate_core_tiles(node, kind=kind, full_cores=full_n), full_n, eff_n)
                frac = float(eff_n) / float(full_n)
                for rname in res_list:
                    if rname.startswith(core_pref):
                        demand[rname] *= wave * frac
                        solo_ms[rname] *= wave
                    elif sfx and rname.endswith(sfx):
                        solo_ms[rname] /= max(frac, TIME_EPS_US)

    max_pipe_ms = max(solo_ms.values(), default=0.0)
    latency_ms = 0.0 if has_topology_plan else (
        (max(floor_ms, max_pipe_ms) if max_pipe_ms > 0.0 else base_dur_ms) if is_disjoint else max(base_dur_ms, max_pipe_ms)
    )
    return demand, {}, latency_ms, bool(cube_ms > 0.0 or vec_ops > 0.0 or is_persistent_whole_die_node(node))


def _die_for_occurrence(case: Case, occ: OpOccurrence) -> Device:
    dies = case.machine.dies
    if not dies:
        raise ModelingError("target_schedule_requires_at_least_one_die")
    if occ.rank is not None:
        for die in dies:
            if int(die.rank) == int(occ.rank):
                return die
    width = max(1, int(case.runtime.parallel.tp) * int(case.runtime.parallel.pcp))
    idx = int(occ.stage) * width
    return dies[idx] if 0 <= idx < len(dies) else dies[0]


def _build_die_resources_and_pools(
    die_coords: Sequence[tuple[int, int, int]],
    supply_by_coord: Mapping[tuple[int, int, int], tuple[DatapathCapabilitySpec, dict[str, float], dict[str, float], dict[str, tuple[str, float]], float]],
    stream_indices_by_coord: Mapping[tuple[int, int, int], set[int]],
    topology: Topology | None,
    collective_ranks: set[int],
    *,
    owners_by_coord: Mapping[tuple[int, int, int], Mapping[str, StreamOwners]] | None = None,
) -> tuple[list[Resource], dict[str, float], dict[str, int]]:
    resources: list[Resource] = []
    pools: dict[str, float] = {}
    pool_units: dict[str, int] = {}
    seen_res: set[str] = set()

    def _add(res: Resource) -> None:
        if res.name not in seen_res:
            seen_res.add(res.name)
            resources.append(res)

    for coord in die_coords:
        die_tag = f"r{coord[0]}:s{coord[1]}:d{coord[2]}"
        datapath, rates, _peaks, _smodes, link_bps = supply_by_coord[coord]
        aic_full, aiv_full = max(1, int(datapath.aic_count)), max(1, int(datapath.aiv_count))
        for s_idx in sorted(stream_indices_by_coord[coord] | {0, 1, 2}):
            _add(Resource(f"stream_{s_idx}@{die_tag}", 1.0, "ms", "fifo", 1))
        _add(Resource(f"matrix@{die_tag}", 1.0, "peak_ms", "fair", 1))
        vec_r = float(rates.get("vector") or 0.0)
        if vec_r > 0.0:
            _add(Resource(f"vector@{die_tag}", vec_r / MS_PER_SECOND, "ops", "fair", 1))
        hbm_r = float(rates.get("hbm") or max(datapath.hbm_read_bytes_per_s, datapath.hbm_write_bytes_per_s) or 0.0)
        if hbm_r > 0.0:
            _add(Resource(f"memory@{die_tag}", hbm_r / MS_PER_SECOND, "bytes", "fair", 1))
        m_rates: dict[str, float] = {}
        for mkey in ("mte2_aic", "mte2_aiv", "mte2", "mte3", "fixpipe"):
            mrate = float(rates.get(mkey) or (hbm_r if mkey in {"mte2", "mte3", "fixpipe"} else 0.0))
            if mrate > 0.0:
                m_rates[mkey] = mrate
                _add(Resource(f"{mkey}@{die_tag}", mrate / MS_PER_SECOND, "bytes", "fair", 1))
        if link_bps > 0.0:
            _add(Resource(f"link@{die_tag}", link_bps / MS_PER_SECOND, "bytes", "fair", 1))
        for lid, own in (owners_by_coord or {}).get(coord, {}).items():
            if own.isolation != "disjoint":
                continue
            aic_f = float(own.aic_limit) / float(aic_full)
            aiv_f = float(own.aiv_limit) / float(aiv_full)
            if aic_f > 0.0:
                _add(Resource(f"matrix@{die_tag}:{lid}", aic_f, "peak_ms", "fair", 1))
                for mk in ("mte2_aic", "fixpipe", "mte2"):
                    if mk in m_rates:
                        _add(Resource(f"{mk}@{die_tag}:{lid}", (m_rates[mk] * aic_f) / MS_PER_SECOND, "bytes", "fair", 1))
                if "mte3" in m_rates:
                    _add(Resource(f"mte3_aic@{die_tag}:{lid}", (m_rates["mte3"] * aic_f) / MS_PER_SECOND, "bytes", "fair", 1))
            if aiv_f > 0.0:
                if vec_r > 0.0:
                    _add(Resource(f"vector@{die_tag}:{lid}", (vec_r * aiv_f) / MS_PER_SECOND, "ops", "fair", 1))
                for mk in ("mte2_aiv", "mte2", "mte3"):
                    if mk in m_rates:
                        _add(Resource(f"{mk}@{die_tag}:{lid}", (m_rates[mk] * aiv_f) / MS_PER_SECOND, "bytes", "fair", 1))
        pools[f"die_exec@{die_tag}"] = float(DIE_EXEC_CONCURRENCY_SLOTS)
        pool_units[f"die_exec@{die_tag}"] = 1

    if topology is not None and collective_ranks:
        for res in topology.resources_for_ranks(tuple(sorted(collective_ranks))):
            _add(res)
    return resources, pools, pool_units


def _max_same_stream_overlap(
    nodes: Sequence[Any], case: Case, intervals_us: Sequence[tuple[float, float]],
) -> float:
    by_ds: dict[tuple[int, int, int, int], list[tuple[float, float]]] = defaultdict(list)
    for idx, node in enumerate(nodes):
        s_us, e_us = intervals_us[idx]
        if e_us > s_us + TIME_EPS_US:
            occ = node.occurrence
            die = _die_for_occurrence(case, occ)
            by_ds[(int(occ.replica), int(occ.stage), int(die.rank), int(node.stream_index))].append((s_us, e_us))
    best = 0.0
    for ivs in by_ds.values():
        ivs.sort()
        for (_, pe), (cs, _) in zip(ivs, ivs[1:]):
            if pe > cs:
                best = max(best, pe - cs)
    return best


def _bind_stream_plan_by_coord(
    stream_plan: StreamPlan | None,
    die_coords: Sequence[tuple[int, int, int]],
    supply_by_coord: Mapping[tuple[int, int, int], Any],
) -> tuple[dict[tuple[int, int, int], dict[str, StreamOwners]], dict[tuple[int, int, int], dict[str, Any]], str]:
    owners_by_coord: dict[tuple[int, int, int], dict[str, StreamOwners]] = {}
    receipt_by_coord: dict[tuple[int, int, int], dict[str, Any]] = {}
    summary = ""
    if stream_plan is None:
        return owners_by_coord, receipt_by_coord, summary
    for coord in die_coords:
        bound = bind_stream_owners(stream_plan, supply_by_coord[coord][0])
        owners_by_coord[coord] = bound
        receipt_by_coord[coord] = {
            lid: {
                "stream_id": lid, "isolation": own.isolation,
                "aic_limit": int(own.aic_limit), "aiv_limit": int(own.aiv_limit),
                "aic_owners": list(own.aic), "aiv_owners": list(own.aiv),
            }
            for lid, own in bound.items()
        }
        if not summary:
            summary = format_stream_core_budget_receipt(bound)
    return owners_by_coord, receipt_by_coord, summary


def _materialize_scheduled_result(
    nodes: Sequence[Any], case: Case, arch_order: Sequence[str], intervals_us: Sequence[tuple[float, float]],
    exposed_list: Sequence[float], masked_list: Sequence[str], step_span_us: float,
    receipt_by_coord: Mapping[tuple[int, int, int], dict[str, Any]], budget_summary: str,
) -> DieScheduleResult:
    arch_spans: dict[str, float] = {aid: 0.0 for aid in arch_order}
    out_nodes: list[Any] = []
    for idx, node in enumerate(nodes):
        s_us, e_us = intervals_us[idx]
        dur_us = max(0.0, e_us - s_us)
        exp_us = min(dur_us, max(0.0, exposed_list[idx]))
        fully_hidden = dur_us > EXPOSED_EPS_US and exp_us <= EXPOSED_EPS_US
        eff_exp = 0.0 if fully_hidden else exp_us
        masker = masked_list[idx] if (dur_us - eff_exp > EXPOSED_EPS_US and masked_list[idx]) else ""
        l_star = float(node.l_star_lean_us)
        roof_eff = min(1.0, l_star / dur_us) if (dur_us > 0.0 and l_star > 0.0) else None
        upd: dict[str, Any] = dict(
            start_us=s_us, end_us=e_us, dur_us=dur_us, exposed_dur_us=eff_exp,
            is_overlapped=fully_hidden, masked_by=masker,
            gap_us=0.0 if fully_hidden else max(0.0, dur_us - l_star),
            gap_ratio=None if fully_hidden else ((dur_us / l_star) if l_star > 0.0 else None),
            roofline_efficiency=roof_eff,
            opt_attainment_pct=round(roof_eff * PERCENT_SCALE, 1) if roof_eff is not None else node.opt_attainment_pct,
            bound_display=f"Overlapped by {masker} (0 exposed)" if (fully_hidden and masker) else node.bound_display,
        )
        if hasattr(node, "stream_core_budget") and budget_summary:
            upd["stream_core_budget"] = budget_summary
        out_nodes.append(replace(node, **upd))
        arch_spans[str(node.archetype_id)] = arch_spans.get(str(node.archetype_id), 0.0) + eff_exp
    return DieScheduleResult(
        nodes=tuple(out_nodes), archetype_spans_us=arch_spans,
        step_start_us=0.0, step_end_us=step_span_us, step_span_us=step_span_us,
        stream_core_budget=dict(receipt_by_coord), stream_core_budget_summary=budget_summary,
        max_same_stream_overlap_us=_max_same_stream_overlap(nodes, case, intervals_us),
    )


def schedule_target_graph_on_engine(
    nodes: Sequence[Any],
    case: Case,
    *,
    collective_plans: Mapping[OpOccurrence, tuple[Any, float]] | None = None,
    topology: Topology | None = None,
    primitive_table: Any = None,
    stream_plan: StreamPlan | None = None,
    launch_floor_us: float = 0.0,
) -> DieScheduleResult:
    """Schedule ``nodes`` on one real Die + Topology ``Engine`` with FIFO, core, HBM & Link contention."""
    if not nodes:
        return DieScheduleResult((), {}, 0.0, 0.0, 0.0, {}, "", 0.0)

    col_plans = dict(collective_plans or {})
    by_arch: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    arch_order: list[str] = []
    die_by_coord: dict[tuple[int, int, int], Device] = {}
    stream_indices_by_coord: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for idx, node in enumerate(nodes):
        aid = str(node.archetype_id)
        if aid not in by_arch:
            arch_order.append(aid)
        by_arch[aid].append((idx, node))
        die = _die_for_occurrence(case, node.occurrence)
        coord = (int(node.occurrence.replica), int(node.occurrence.stage), int(die.rank))
        die_by_coord[coord] = die
        stream_indices_by_coord[coord].add(int(node.stream_index))

    die_coords = sorted(die_by_coord)
    supply_by_coord = {
        c: _effective_die_supply(die_by_coord[c], case.machine.links, case.machine.transfer_channels, primitive_table)
        for c in die_coords
    }
    owners_by_coord, receipt_by_coord, budget_summary = _bind_stream_plan_by_coord(
        stream_plan, die_coords, supply_by_coord,
    )
    col_ranks = {int(r) for plan, _ in col_plans.values() for r in getattr(plan, "ranks", ())}
    resources, pools, pool_units = _build_die_resources_and_pools(
        die_coords, supply_by_coord, stream_indices_by_coord, topology, col_ranks,
        owners_by_coord=owners_by_coord,
    )
    engine = Engine(resources, pools=pools, pool_units=pool_units, backend="python")
    observer = ServiceGrantObserver()
    engine.attach_wave_recorder(observer)
    root_ev = engine.submit("step_root")

    occ_to_idx = {node.occurrence: idx for idx, node in enumerate(nodes)}
    end_by_occ: dict[OpOccurrence, Any] = {}
    queue_tail: dict[tuple[tuple[int, int, int], int], tuple[Any, int | None]] = {}
    arch_barrier, arch_barrier_idxs = root_ev, []
    start_events: dict[int, Any] = {}
    end_events: dict[int, Any] = {}
    causal_preds: dict[int, tuple[int, ...]] = {}

    for arch_idx, aid in enumerate(arch_order):
        ordered = sorted(by_arch[aid], key=lambda item: (item[1].ordinal, item[1].start_us))
        arch_done_evs: list[Any] = []
        arch_done_idxs: list[int] = []
        for idx, node in ordered:
            occ = node.occurrence
            die = _die_for_occurrence(case, occ)
            coord = (int(occ.replica), int(occ.stage), int(die.rank))
            die_tag = f"r{coord[0]}:s{coord[1]}:d{coord[2]}"
            s_idx = int(node.stream_index)
            lane_id = _canonical_stream_lane_name(str(node.stream), s_idx)
            owners = owners_by_coord.get(coord, {}).get(lane_id)

            deps_ev: list[Any] = [arch_barrier]
            c_preds: list[int] = list(arch_barrier_idxs)
            for pred in node.predecessors:
                if pred in end_by_occ:
                    deps_ev.append(end_by_occ[pred])
                    if pred in occ_to_idx:
                        c_preds.append(occ_to_idx[pred])
            qkey = (coord, s_idx)
            if qkey in queue_tail:
                q_ev, q_idx = queue_tail[qkey]
                deps_ev.append(q_ev)
                if q_idx is not None:
                    c_preds.append(q_idx)

            causal_preds[idx] = tuple(dict.fromkeys(c_preds))
            ready_ev = engine.submit(f"r:{idx}", after=tuple({id(e): e for e in deps_ev}.values()))
            col_entry = col_plans.get(occ)
            datapath, rates, peaks, service_modes, link_bps = supply_by_coord[coord]
            demand, placement, latency_ms, uses_cores = _node_physical_demand_and_placement(
                node, die_tag=die_tag, lane_id=lane_id, datapath=datapath, rates=rates, peaks=peaks,
                service_modes=service_modes, link_bps=link_bps, owners=owners,
                has_topology_plan=(col_entry is not None), launch_floor_us=launch_floor_us,
            )
            is_persist = is_persistent_whole_die_node(node)
            if is_persist and owners is not None and owners.isolation == "disjoint":
                if int(owners.aic_limit) < int(datapath.aic_count) or int(owners.aiv_limit) < int(datapath.aiv_count):
                    raise ModelingError(f"persistent_kernel_requires_full_die:{node.op_id}")
            alloc = Allocation(f"exec:{idx}", f"die_exec@{die_tag}", float(DIE_EXEC_CONCURRENCY_SLOTS) if is_persist else 1.0)
            acq, rel = ((alloc,), (alloc,)) if uses_cores else ((), ())

            observer.current_node_idx = idx
            if col_entry is not None:
                plan, min_us = col_entry
                admit_ev = engine.submit(f"a:{idx}", after=(ready_ev,), acquire=acq) if acq else ready_ev
                done_deps: list[Any] = [plan.start(engine, f"c:{idx}:{node.op_id}", admit_ev)]
                if min_us > 0.0:
                    done_deps.append(engine.submit(f"f:{idx}", after=(admit_ev,), latency=min_us / US_PER_MS))
                if demand:
                    done_deps.append(engine.submit(f"d:{idx}", work=demand, after=(admit_ev,), placement=placement or None))
                start_events[idx] = admit_ev
                end_ev = engine.submit(f"e:{idx}", after=tuple(done_deps), free=rel)
            elif demand or latency_ms > 0.0 or acq:
                end_ev = engine.submit(
                    f"op:{idx}", work=demand or None, after=(ready_ev,),
                    latency=latency_ms, placement=placement or None, acquire=acq, free=rel,
                )
                start_events[idx] = end_ev
            else:
                start_events[idx] = end_ev = ready_ev
            observer.current_node_idx = None

            end_events[idx] = end_ev
            end_by_occ[occ] = end_ev
            queue_tail[qkey] = (end_ev, idx)
            arch_done_evs.append(end_ev)
            arch_done_idxs.append(idx)
        if arch_done_evs:
            arch_barrier = engine.submit(f"arch:{arch_idx}", after=tuple(arch_done_evs))
            arch_barrier_idxs = arch_done_idxs

    step_done = engine.submit("step_done", after=tuple(end_events.values()))
    engine.advance(until=(step_done,))
    engine.detach_wave_recorder()

    step_start_us = float(root_ev.time or 0.0) * US_PER_MS
    step_span_us = max(0.0, float(step_done.time or 0.0) * US_PER_MS - step_start_us)
    intervals_us = [
        (
            s_u := max(0.0, float(
                observer.admit_ms_by_node.get(i)
                if observer.admit_ms_by_node.get(i) is not None
                else (start_events[i].started if start_events[i].started is not None else (start_events[i].time or 0.0))
            ) * US_PER_MS - step_start_us),
            max(s_u, float(end_events[i].time or 0.0) * US_PER_MS - step_start_us),
        )
        for i in range(len(nodes))
    ]
    exposed_list, masked_list = attribute_engine_exposure(nodes, intervals_us, observer, causal_preds, step_span_us)
    return _materialize_scheduled_result(
        nodes, case, arch_order, intervals_us, exposed_list, masked_list,
        step_span_us, receipt_by_coord, budget_summary,
    )


def weighted_step_span_us(nodes: Sequence[Any]) -> float:
    """Return repeat-weighted exposed step duration in microseconds."""
    return sum(
        max(0.0, float(getattr(n, "exposed_dur_us", 0.0))) * max(1, int(getattr(n, "repeat_count", 1) or 1))
        for n in nodes
    )


def evaluate_disjoint_core_budgets(
    nodes: Sequence[Any],
    case: Case,
    *,
    shared_result: DieScheduleResult,
    collective_plans: Mapping[OpOccurrence, tuple[Any, float]] | None = None,
    topology: Topology | None = None,
    primitive_table: Any = None,
    launch_floor_us: float = 0.0,
) -> DieScheduleResult:
    """Test explicit ``StreamCoreBudget(isolation='disjoint')`` splits; adopt only if makespan drops."""
    stream_lanes: dict[str, tuple[bool, bool]] = {}
    for node in nodes:
        if float(node.dur_us) <= TIME_EPS_US:
            continue
        lname = _canonical_stream_lane_name(str(node.stream), int(node.stream_index))
        has_c = float(node.flops) > 0.0 and str(node.family).lower() not in ZERO_AIC_COMM_FAMILIES
        has_v = float(node.vector_ops) > 0.0
        prev_c, prev_v = stream_lanes.get(lname, (False, False))
        stream_lanes[lname] = (prev_c or has_c, prev_v or has_v)

    aux_lanes = [lid for lid, (hc, hv) in stream_lanes.items() if lid != MAIN_STREAM and (hc or hv)]
    if MAIN_STREAM not in stream_lanes or len(aux_lanes) != 1 or any(is_persistent_whole_die_node(n) for n in nodes):
        return shared_result
    if sum(float(n.dur_us) for n in shared_result.nodes) <= shared_result.step_span_us + EXPOSED_EPS_US:
        return shared_result

    die = case.machine.dies[0]
    datapath = datapath_for_die(die)
    aic_tot, aiv_tot = int(datapath.aic_count), int(datapath.aiv_count)
    if aic_tot < 2 or aiv_tot < 2:
        return shared_result

    aux_id = aux_lanes[0]
    aux_needs_aic, aux_needs_aiv = stream_lanes[aux_id]
    main_needs_aic, main_needs_aiv = stream_lanes[MAIN_STREAM]
    best = shared_result
    best_weighted = weighted_step_span_us(shared_result.nodes)
    threshold = best_weighted * (1.0 - DISJOINT_SPLIT_MIN_GAIN_RATIO)

    for num, den in DISJOINT_SPLIT_RATIOS:
        main_aic, aux_aic = ((aic_tot, 0) if not aux_needs_aic else ((aic_tot * num) // den, aic_tot - (aic_tot * num) // den))
        if aux_needs_aic and main_needs_aic and (main_aic <= 0 or aux_aic <= 0):
            continue
        main_aiv, aux_aiv = ((aiv_tot, 0) if not aux_needs_aiv else ((aiv_tot * num) // den, aiv_tot - (aiv_tot * num) // den))
        if aux_needs_aiv and main_needs_aiv and (main_aiv <= 0 or aux_aiv <= 0):
            continue
        extra_lanes = tuple(
            StreamLane(lid, StreamCoreBudget(aic=0, aiv=0, isolation="disjoint"))
            for lid in stream_lanes if lid not in {MAIN_STREAM, aux_id}
        )
        plan = StreamPlan(
            rank=int(die.rank),
            lanes=(
                StreamLane(MAIN_STREAM, StreamCoreBudget(aic=main_aic, aiv=main_aiv, isolation="disjoint")),
                StreamLane(aux_id, StreamCoreBudget(aic=aux_aic, aiv=aux_aiv, isolation="disjoint")),
                *extra_lanes,
            ),
        )
        try:
            cand = schedule_target_graph_on_engine(
                nodes, case, collective_plans=collective_plans, topology=topology,
                primitive_table=primitive_table, stream_plan=plan, launch_floor_us=launch_floor_us,
            )
        except ModelingError:
            continue
        cand_weighted = weighted_step_span_us(cand.nodes)
        if cand_weighted < best_weighted and cand_weighted <= threshold:
            best = cand
            best_weighted = cand_weighted
    return best


__all__ = [
    "DIE_EXEC_CONCURRENCY_SLOTS",
    "DieScheduleResult",
    "ServiceGrantObserver",
    "attribute_engine_exposure",
    "evaluate_disjoint_core_budgets",
    "format_stream_core_budget_receipt",
    "schedule_target_graph_on_engine",
    "weighted_step_span_us",
]
