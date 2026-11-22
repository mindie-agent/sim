"""Physical conservation and causal execution of the L0 primitives.

[Role] Exercise sealed declarations and lower each primitive into the CPU Engine.
[Invariants] Native work, certified bounds, live cache admission, and causal
  delays preserve their respective units and hardware boundaries.
[Timeouts & Locks] Small, finite Tier-0 programs with no waits on external work.
[Side-Effect Policy] Read packaged profiles and use tmp_path for tampered copies.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import replace

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from _support.hardware import machine, sku_case
from mindie_sim.prediction.collectives.ports import RootComplexKey
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.prediction.runtime.l2_residency import AdmissionMemoryView, bind_memory_access
from mindie_sim.prediction.runtime.memory_physics import (
    _L2_CACHE_DISABLE, AddressRun, AddressRunPlan, HbmBusSpec, MemoryAccessPattern, ReuseKey, VisibilityContract,
)
from mindie_sim.prediction.runtime.tile_solver import LiveBufferSpec
from mindie_sim.prediction.streams import StreamCoreBudget, StreamLane, StreamPlan
from mindie_sim.primitives import (
    UncertaintyBand, default_primitives_dir, frame_wire_bytes, load_manifest, load_primitives,
    lower_primitive, parse_primitive_table, resolve_primitive_rate, submit_lowered_primitive,
)
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.device_input import StarsSpec
from mindie_sim.schema.hardware_links import UNCALIBRATED_SKU_WARNING

pytestmark = pytest.mark.tier0

PRIMITIVES_DIR = default_primitives_dir()
HOST_DMA_SKU, CALIBRATED_REGISTER_SKU, UNMEASURED_REGISTER_SKU = "Ascend910_9362", "Ascend950DT_9582", "Ascend950PR_9579"
PRIVATE_SOURCE = re.compile(r'(?<![\d.])(?:\d{1,3}[._-]){3}\d{1,3}(?![\d.])'
                            r'|@|/(?:Users|home|root)/|\$\{?HOME\b|\b(?:username|user_name)\b')


def _datapath(sku): return datapath_for_die(machine(sku).dies[0])


@pytest.fixture(scope="module")
def register_device():
    dp = _datapath(CALIBRATED_REGISTER_SKU)
    return load_primitives(CALIBRATED_REGISTER_SKU, datapath=dp), dp


@pytest.fixture(scope="module")
def memory_device():
    sku = REPRESENTATIVE_SKUS["a2"]
    dp = _datapath(sku)
    return load_primitives(sku, datapath=dp), dp


def _digest(raw):
    body = {k: v for k, v in raw.items() if k != "envelope_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _bands(tree):
    for value in tree.values():
        if isinstance(value, UncertaintyBand): yield value
        elif isinstance(value, Mapping): yield from _bands(value)


def _with_latencies(table, pid, **delays):
    cell = table.cell(pid)
    bands = {m: UncertaintyBand(ns, bound_kind="latency_lower") for m, ns in delays.items()}
    return replace(table, primitives={**table.primitives, pid: replace(cell, metrics={**cell.metrics, **bands})})


def _stages(stages):
    for stage in stages:
        yield stage
        if stage.program is not None: yield from _stages(stage.program.stages)


def _engine(*results, rates=None, units=None, pools=None):
    capacities = {}
    for result in results:
        for stage in _stages(result.stages):
            for key, value in stage.metadata.items():
                assert type(key) is str and key != "occurrence"
                assert type(value) in (bool, int, float, str, bytes, type(None)) or (
                    type(value) is tuple and all(type(item) is int for item in value))
            for row in stage.work:
                capacities[row] = max(capacities.get(row, 0), stage.caps.get(row, 1))
        if result.bound_access is not None:
            capacities.update(result.bound_access.row_rates)
    capacities.update(rates or {})
    return Engine([Resource(row, rate, units=(units or {}).get(row, 1)) for row, rate in capacities.items()],
                  backend="python", trace_limit=256, pools=pools)


def _run(result, **resources):
    engine = _engine(result, **resources)
    done = submit_lowered_primitive(engine, result)
    engine.advance(until=(done,))
    assert done.time == pytest.approx(result.analytical_ms, abs=1e-12)
    return engine


def _trace(engine, stage): return next(r for r in engine.trace if r["task"].endswith(f":{stage}"))


def _read(key, *runs, policy="normal"):
    return MemoryAccessPattern(key.backing_allocation, 0, 0, 0, "read", "mte2_aiv",
                               sum(r.row_bytes * r.rows for r in runs), AddressRunPlan(runs), key, cache_policy=policy)


def _lower(pid, params, table, **kwargs):
    kwargs.setdefault("allow_uncalibrated_declaration", True)
    return lower_primitive(pid, params, table, **kwargs)


@pytest.mark.parametrize("pool,owner,wrong", [
    ("l1", "aic", "aiv"), ("l0a", "aic", "aiv"), ("l0b", "aic", "aiv"), ("l0c", "aic", "aiv"), ("ub", "aiv", "aic"),
])
def test_live_buffer_requires_physical_pool_owner(pool, owner, wrong):
    buffer = LiveBufferSpec("tile", pool, 32, 0, 1, owner_kind=owner)
    with pytest.raises(ModelingError, match="^live_buffer_pool_owner_mismatch:"):
        replace(buffer, owner_kind=wrong)
    for kind in ("aic", "aiv", "die"):
        assert replace(buffer, pool="l2", owner_kind=kind).bytes_per_owner == 32


def test_cache_policy_bypass_preserves_residency_and_last_use_retires_it():
    state = AdmissionMemoryView({0: 256})
    read = _read(ReuseKey(0, "resident", 0), AddressRun(0, 1, 192, 192))
    with pytest.raises(ModelingError, match="^memory_access_cache_policy_invalid:"):
        replace(read, cache_policy="invalid")
    assert state.admit_access(read, 192) == (0, 192)
    bypass = replace(read, cache_policy=_L2_CACHE_DISABLE)
    assert state.preview_access(bypass, 192) == state.admit_access(bypass, 192) == (0, 192)
    fresh = _read(ReuseKey(0, "fresh", 0, "aux"), *read.runs.runs, policy="disable")
    assert state.admit_access(fresh, 192) == (0, 192) and not state.is_resident(fresh.reuse_key) and state.active_stream_count(0) == 1
    assert state.admit_access(replace(read, cache_policy="last_use"), 192) == (192, 0) and not state.is_resident(read.reuse_key)
    vis = VisibilityContract(private_workspace=True, live_out=False)
    priv = MemoryAccessPattern("p", 0, 0, 0, "write", "mte3", 32, AddressRunPlan((AddressRun(0, 1, 32, 32),)),
                               ReuseKey(0, "p", 0), visibility=vis, cache_policy="last_use")
    assert state.preview_access(priv, 32) == state.admit_access(priv, 32) == (0, 32)
    cap_state = AdmissionMemoryView({0: 512})
    k1, k2 = ReuseKey(0, "a", 0, "main"), ReuseKey(0, "b", 0, "main")
    r1, r2 = _read(k1, AddressRun(0, 1, 192, 192)), _read(k2, AddressRun(0, 1, 192, 192))
    cap_state.admit_access(r1, 192); cap_state.admit_access(r2, 192)
    cap_state.exit_stream(0, "main"); cap_state.enter_stream(0, "aux")
    assert cap_state.preview_access(r1, 192) == cap_state.admit_access(r1, 192) == (0, 192) and cap_state.is_resident(k1) and not cap_state.is_resident(k2)
    wt_state, kw = AdmissionMemoryView({0: 512}), ReuseKey(0, "w", 0, "main")
    assert wt_state.admit_access(MemoryAccessPattern("w", 0, 0, 0, "write", "mte3", 32, AddressRunPlan((AddressRun(0, 1, 32, 32),)), kw, visibility=vis), 32) == (32, 0)
    assert wt_state.admit_access(MemoryAccessPattern("w", 0, 0, 0, "write", "mte3", 32, AddressRunPlan((AddressRun(32, 1, 32, 32),)), kw, visibility=VisibilityContract(private_workspace=False, live_out=True)), 32) == (0, 32)
    assert [wb.bytes for wb in wt_state.drain_writebacks(0)] == [32]


def test_periodic_sector_coverage_excludes_stride_gaps_and_hits_identical_reads():
    key, amount = ReuseKey(0, "strided", 0), 5000 * 32
    c0, c1 = (AddressRunPlan((AddressRun(b, 5000, 1, 64),)).sector_coverage() for b in (0, 1))
    assert c0.overlap_bytes(c1) == c0.union(c1).measure_bytes() == amount
    assert AddressRunPlan((AddressRun(0, 5000, 64, 96),)).sector_coverage().overlap_bytes(
        AddressRunPlan((AddressRun(160, 5000, 64, 96),)).sector_coverage()) == 159936
    assert AddressRunPlan((AddressRun(0, 5000, 1, 64), AddressRun(1, 5000, 1, 64))).dram_sector_and_service() == (amount, float(amount))
    read, gap = _read(key, AddressRun(0, 5000, 32, 64)), _read(key, AddressRun(32, 1, 32, 32))
    assert read.runs.sector_intervals() != ((0, 320000),) and read.runs.sector_coverage().measure_bytes() == amount
    state = AdmissionMemoryView({0: 256 * 1024})
    state.admit_access(_read(key, AddressRun(0, 1, 32, 32)), 32)
    bound = bind_memory_access(_read(key, AddressRun(0, 5000, 1, 64)), state, hbm_bus=HbmBusSpec(0, 1e9, 1e9, 0.0),
                               copy_hit_bytes_per_ms=1e9, copy_miss_bytes_per_ms=1e9, l2_bytes_per_ms=1e9,
                               pipe_row_id="mte2_aiv", l2_row_id="l2", commit=True)
    assert (bound.ledger.l2_hit_bytes, bound.ledger.dram_miss_copy_bytes, bound.ledger.dram_sector_bytes) == (32, 159968, 159968)
    assert state.preview_access(gap, 32) == (0, 32) and state.admit_access(read, amount) == (amount, 0)
    dirty_state, vis = AdmissionMemoryView({0: 200000}), VisibilityContract(private_workspace=True, live_out=False)
    for run, amt in ((AddressRun(0, 5000, 1, 64), amount), (AddressRun(32, 1, 32, 32), 32)):
        assert dirty_state.admit_access(MemoryAccessPattern("w", 0, 0, 0, "write", "mte3", amt, AddressRunPlan((run,)), ReuseKey(0, "w", 0), visibility=vis), amt) == (amt, 0)
    dirty_state.enter_stream(0, "aux")
    wbs = dirty_state.drain_writebacks(0)
    assert len(wbs) == 1 and wbs[0].bytes == amount + 32


def test_sealed_envelopes_match_manifest_and_preserve_calibrated_rate_bounds():
    manifest = load_manifest()
    entries = manifest["tables"]
    assert len(entries) == 5 and not PRIVATE_SOURCE.search(json.dumps(manifest))
    assert {e["file"] for e in entries.values()} == {p.name for p in PRIMITIVES_DIR.glob("*.json") if p.name != "manifest.json"}
    for sku, entry in entries.items():
        path = PRIMITIVES_DIR / entry["file"]
        cap = PRIMITIVES_DIR / "captures" / f"{path.stem}.capture.json"
        assert entry["compatible_skus"] == [sku] and cap.is_file() and not PRIVATE_SOURCE.search(cap.read_text(encoding="utf-8"))
        assert entry["capture_uri"].endswith("/" + hashlib.sha256(cap.read_bytes()).hexdigest())
        encoded = path.read_text(encoding="utf-8")
        assert not PRIVATE_SOURCE.search(encoded), sku
        raw, table = json.loads(encoded), load_primitives(sku, datapath=_datapath(sku))
        assert _digest(raw) == table.envelope_sha256 == entry["envelope_sha256"] and table.to_envelope_dict() == raw
        assert (table.sku, table.cann_version, table.provenance.capture_uri, table.provenance.calibrated) == (
            sku, entry["cann_version"], entry["capture_uri"], entry["calibrated"])
        assert table.warnings == (() if entry["calibrated"] else (UNCALIBRATED_SKU_WARNING,))
        assert len(table.primitives) == entry["total_primitives"] == 20
        assert sum(c.status == "calibrated" for c in table.primitives.values()) == entry["calibrated_primitive_count"]
        for cell in table.primitives.values():
            for band in _bands({"metrics": cell.metrics, "modes": cell.modes}):
                if band.bound_kind == "rate_upper" and band.p50 is not None:
                    assert 0 < band.p50 <= band.p99 <= band.certified_upper_capacity and band.cv_pct <= 5.0
    with pytest.raises(ModelingError, match="^primitive_table_sku_unknown_or_ambiguous:"):
        load_primitives(REPRESENTATIVE_SKUS["a5"], datapath=_datapath(REPRESENTATIVE_SKUS["a5"]))


@pytest.mark.parametrize("field", ("sku", "cann_version", "calibrated", "metric"))
def test_full_envelope_digest_detects_mutation_and_manifest_rejects_resealing(tmp_path, field, register_device):
    table, dp = register_device
    raw = table.to_envelope_dict()
    if field == "calibrated": raw["provenance"][field] = not raw["provenance"][field]
    elif field == "metric": raw["primitives"]["P01_datacopy"]["metrics"]["hbm_read_bytes_per_s"]["certified_upper_capacity"] *= 2
    else: raw[field] += "-modified"
    mismatch = "^primitive_table_sha256_mismatch$"
    with pytest.raises(ModelingError, match=mismatch): parse_primitive_table(raw)
    path = tmp_path / load_manifest()["tables"][table.sku]["file"]
    for digest in (raw["envelope_sha256"], _digest(raw)):
        raw["envelope_sha256"] = digest
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ModelingError, match=mismatch): load_primitives(path, datapath=dp)


def test_primitive_declarations_require_matching_datapath_and_explicit_unmeasured_rates(memory_device, register_device):
    memory, old_dp = memory_device
    register, _ = register_device
    uncal_dp = _datapath(UNMEASURED_REGISTER_SKU)
    uncal = load_primitives(UNMEASURED_REGISTER_SKU, datapath=uncal_dp)
    with pytest.raises(ModelingError, match="^primitive_table_arch_mismatch$"): load_primitives(register.sku, datapath=old_dp)
    with pytest.raises(ModelingError, match="^primitive_table_uncalibrated$"): load_primitives(UNMEASURED_REGISTER_SKU, datapath=uncal_dp, require_calibrated=True)
    for table in (memory, register, uncal):
        pid, metric = "P19_direct_peer_window", "peer_bytes_per_s"
        rate, warnings = resolve_primitive_rate(table, pid, metric, allow_uncalibrated_declaration=True)
        assert rate == table.cell(pid).band(metric).certified_upper_capacity > 0 and warnings == (UNCALIBRATED_SKU_WARNING,)
        with pytest.raises(ModelingError, match="^primitive_unmeasured_requires_declaration:"): resolve_primitive_rate(table, pid, metric, allow_uncalibrated_declaration=False)


def test_uncertainty_bounds_enclose_the_required_quantile():
    with pytest.raises(ModelingError, match="^uncertainty_band_upper_below_p99$"):
        UncertaintyBand(9, p05=7, p50=8, p95=9, p99=10, cv_pct=1.0, n_samples=4, bound_kind="rate_upper")
    with pytest.raises(ModelingError, match="^uncertainty_band_lower_above_p05$"):
        UncertaintyBand(6, p05=5, p50=8, p95=9, p99=10, cv_pct=1.0, n_samples=4, bound_kind="latency_lower")
    for bad in (dict(certified_upper_capacity=1, p50=100, n_samples=0),
                dict(certified_upper_capacity=100, p05=40, p50=50, p95=60, p99=70, cv_pct=1.0, n_samples=0)):
        with pytest.raises(ModelingError):
            UncertaintyBand(**bad)
    with pytest.raises(ModelingError, match="^primitive_unmeasured_requires_declaration$"):
        _ = UncertaintyBand(100.0).sim_rate_p50


@pytest.mark.parametrize("source", ["192.0.2.7", "node-192-0-2-7", "file://" + "/home" + "/fixture/capture", "user:token@lab"])
def test_provenance_rejects_raw_endpoint_and_credential_sources(register_device, source):
    table, _ = register_device
    for field in ("endpoint_alias", "capture_uri"):
        with pytest.raises(ModelingError, match="^primitive_provenance_secret_leak$"):
            replace(table.provenance, **{field: source})


def test_datacopy_strided_sector_union_and_live_l2_admission(register_device):
    table, dp = register_device
    key = ReuseKey(0, "tiles", 0)
    state = AdmissionMemoryView({0: 256})
    params = {"runs": (AddressRun(0, 3, 17, 64), AddressRun(8, 1, 16, 16)), "reuse_key": key}
    cold = _lower("P01_datacopy", params, table, datapath=dp, memory_state=state)
    ledger = cold.bound_access.ledger
    assert (ledger.logical_tensor_bytes, ledger.aligned_transfer_bytes, ledger.dram_sector_bytes) == (67, 128, 96)
    assert ledger.l2_hit_bytes == 0 and not state.is_resident(key)
    l2_rate = table.sim_rate_p50("P05_l2_residency_reuse", "l2_die_bytes_per_s", allow_uncalibrated_declaration=True) / 1000
    engine = _engine(cold, rates={"l2": l2_rate})
    first = submit_lowered_primitive(engine, cold, name_prefix="cold")
    second = submit_lowered_primitive(engine, cold, after=(first,), name_prefix="warm")
    engine.advance(until=(second,))
    assert (engine.delivered["hbm_read@d0"], engine.delivered["mte2_aiv"], engine.delivered["l2"]) == pytest.approx((96, 256, 128))
    bypass = _lower("P01_datacopy", {**params, "cache_policy": "disable"}, table, datapath=dp, memory_state=state)
    assert _run(bypass).delivered["hbm_read@d0"] == pytest.approx(96) and state.is_resident(key)


def test_load_data_separate_fractal_operands_follow_serial_setup(register_device):
    table, dp = register_device
    pid, uncal_dp = "P02_load_data", _datapath(UNMEASURED_REGISTER_SKU)
    with pytest.raises(ModelingError, match="^primitive_unmeasured_requires_declaration:"):
        lower_primitive(pid, {"m": 17, "n": 33, "k": 31}, load_primitives(UNMEASURED_REGISTER_SKU, datapath=uncal_dp), datapath=uncal_dp)
    table = _with_latencies(table, pid, setup_ns=750)
    result = _lower(pid, {"m": 17, "n": 33, "k": 31}, table, datapath=dp)
    engine = _run(result)
    elapsed = 0
    for operand, amount in (("a", 2 * 2 * 512), ("b", 3 * 2 * 512)):
        setup, transfer = (_trace(engine, f"load_{operand}{suffix}") for suffix in (":setup", ""))
        assert setup["start_ms"] == pytest.approx(elapsed)
        elapsed += 750 * 1e-6
        assert setup["end_ms"] == transfer["start_ms"] == pytest.approx(elapsed)
        assert not setup["serviced"] and transfer["serviced"]["mte1"] == pytest.approx(amount)
        rate = table.sim_rate_p50(pid, f"l1_to_l0{operand}_bytes_per_s_per_aic", allow_uncalibrated_declaration=True)
        elapsed += amount * 1000 / rate
        assert transfer["end_ms"] == pytest.approx(elapsed)


def test_fixpipe_destination_controls_width_and_requires_direct_ub_route(memory_device, register_device):
    old_table, old_dp = memory_device
    table, dp = register_device
    pid, dims = "P03_fixpipe", {"m": 2, "n": 33}
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_unsupported:l0c->ub"):
        _lower(pid, {**dims, "dst": "ub"}, old_table, datapath=old_dp)
    for dst, split, width in (("gm", "NO_SPLIT", 33), ("ub", "NO_SPLIT", 48), ("ub", "SPLIT_N", 64)):
        result = _lower(pid, {**dims, "dst": dst, "split_mode": split}, table, datapath=dp)
        meta = dict(result.metadata)
        assert (meta["n_eff"], meta["input_l0c_bytes"], meta["output_bytes"]) == (width, 2 * 33 * 4, 2 * width * 2)
        engine = _run(result)
        if dst == "gm": assert engine.delivered["hbm_write@d0"] == pytest.approx(160)
        else: assert engine.serviced["fixpipe"] == pytest.approx(2 * width * 2) and not engine.delivered


def test_mte3_direct_l1_route_and_one_read_modify_write_charge(memory_device, register_device):
    old_table, old_dp = memory_device
    table, dp = register_device
    pid, params = "P04_mte3_store", {"dst": "l1", "bytes": 33}
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_unsupported:ub->l1"): _lower(pid, params, old_table, datapath=old_dp)
    engine = _run(_lower(pid, params, table, datapath=dp))
    assert engine.serviced["mte3"] == pytest.approx(64) and not engine.delivered
    atomic = _lower(pid, {"bytes": 33, "atomic": True}, table, datapath=dp)
    assert atomic.bound_access.ledger.dram_sector_bytes == 2 * 64
    engine = _run(atomic)
    assert all(engine.delivered[row] == pytest.approx(64) for row in ("mte3", "hbm_read@d0", "hbm_write@d0"))


def test_l2_knee_counts_distinct_streams_and_bypass_removes_l2_work(register_device):
    table, dp = register_device
    state = AdmissionMemoryView({0: 512})
    reads = [_read(ReuseKey(0, n, 0, s), AddressRun(0, 1, 192, 192)) for n, s in (("weights", "main"), ("states", "main"), ("auxiliary", "aux"))]
    base = {"working_set_bytes": 192, "pattern": reads[0]}
    for index, (pattern, streams, knee) in enumerate(zip(reads, (1, 1, 2), (512, 512, 256))):
        state.admit_access(pattern, 192)
        assert state.active_stream_count(0) == streams and state.effective_l2_bytes(0) == knee
        for policy in ("normal", "disable"):
            result = _lower("P05_l2_residency_reuse", {**base, "cache_policy": policy}, table, datapath=dp, memory_state=state)
            assert dict(result.metadata)["effective_l2_bytes"] == knee
            assert _run(result).serviced.get("l2", 0) == pytest.approx(192 if index < 2 and policy == "normal" else 0)


@pytest.mark.parametrize("m,n", [(49, 63), (81, 33)])
def test_mmad_issued_flops_and_pipe_barrier_threshold(register_device, m, n):
    table, dp = register_device
    pid, k = "P06_mmad", 33
    params = dict(m=m, n=n, k=k, m_iss=16, n_iss=32, k_iss=32, mode="bf16")
    issued = 2 * ((m + 15) // 16 * 16) * ((n + 31) // 32 * 32) * 64
    band = table.cell(pid).band("cube_only_rate", mode="bf16")
    table = _with_latencies(table, pid, pipe_m_barrier_ns=125)
    for role, rate in (("sim", band.p50), ("dual", band.certified_upper_capacity)):
        result = _lower(pid, params, table, datapath=dp, role=role)
        has_barrier = (m // 16) * (n // 16) < 10
        assert any(s.name == "PipeBarrier<PIPE_M>" for s in result.stages) is has_barrier
        assert dict(result.metadata)["issued_flops"] == issued
        engine = _run(result)
        assert engine.serviced["cube"] == pytest.approx(issued)
        assert engine.now == pytest.approx(issued * 1000 / rate + has_barrier * 125e-6)
    assert band.p50 < band.certified_upper_capacity


def test_production_stage_profile_engine_preserves_units_and_native_counters(register_device):
    table, _ = register_device
    prof = EvaluationContext.create(sku_case(table.sku)).bindings.stage_profiles[0]
    ctx = RecipeContext(None, {}, prof.names, {prof.names[k]: r for k, (r, _) in prof.rates.items()},
                        prof.datapath.aic_count, pool_names=prof.stage_pools,
                        resource_units=dict(prof.units), compute_peaks=prof.compute_peaks, datapath=prof.datapath)
    mm = _lower("P06_mmad", {"m": 1, "n": 16, "k": 16, "mode": "bf16"}, table, datapath=prof.datapath, context=ctx)
    eng = Engine(prof.resources, backend="python", pools=prof.pools)
    done = submit_lowered_primitive(eng, mm)
    eng.advance(until=(done,))
    assert done.time == pytest.approx(mm.analytical_ms, abs=1e-12) and done.time < 1e-5 and eng.delivered["cube_flops"] == pytest.approx(8192)
    cases = (("P07_vec_alu", {"elements": 512, "clock_hz": 1.65e9}), ("P08_vec_sfu", {"elements": 512, "clock_hz": 1.65e9}),
             ("P09_vec_reduce_cast", {"rows": 2, "reduction_dim": 64, "clock_hz": 1.65e9}), ("P10_sort_topk", {"n": 256}),
             ("P11_simt", {"frontend": "tla_dsl", "threads": 64, "warp_instructions": 2}),
             ("P15_rts_stars", {"descriptors": [{"work": {prof.names["cube"]: 1.0}}]}))
    for pid, params in cases:
        r = _lower(pid, params, table, datapath=prof.datapath, context=ctx)
        e = Engine(prof.resources, backend="python", pools=prof.pools)
        ev = submit_lowered_primitive(e, r)
        e.advance(until=(ev,))
        assert ev.time == pytest.approx(r.analytical_ms, abs=1e-12) and ev.time > 0


def test_mmad_weight_conversion_prologues_finish_before_cube(register_device):
    table, dp = register_device
    dec = dp.compute_arch.declared
    no_mx = replace(dp, compute_arch=replace(dp.compute_arch, declared=replace(
        dec, native_modes=tuple(m for m in dec.native_modes if "mxfp8" not in m.operands))))
    with pytest.raises(UnmodeledCapabilityError, match="native_matrix_mode_required:mxfp8"):
        _lower("P06_mmad", dict(m=16, n=5, k=13, mode="a8w4_mx"), table, datapath=no_mx)
    for mode, end, expected in (("w4a8", "unpack:workspace_write", {"mte2_aiv": 33, "vec_alu": 130, "mte3": 65}),
                                ("a8w4_mx", "mx:decompress_b", {"mte1": 75})):
        native = {"native_mode": "fp8_e4m3"} if mode == "a8w4_mx" else {}
        result = _lower("P06_mmad", dict(m=16, n=5, k=13, mode=mode, mx_scale_bytes=10, **native), table, datapath=dp)
        engine = _run(result)
        assert {row: engine.serviced[row] for row in expected} == pytest.approx(expected)
        assert _trace(engine, "mmad")["start_ms"] == pytest.approx(_trace(engine, end)["end_ms"])


@pytest.mark.parametrize("pid,pipe", [
    ("P07_vec_alu", "vec_alu"), ("P08_vec_sfu", "vec_sfu"), ("P09_vec_reduce_cast", "vec_reduce"),
])
def test_vector_consumers_wait_for_raw_results_on_native_subpipes(register_device, pid, pipe):
    table, dp = register_device
    groups = (dict(group_id="producer", pipe=pipe, count=3, result_latency_cycles=7, last_issue_mark=False),
              dict(group_id="consumer", pipe=pipe, count=2, dependencies=("producer",)))
    engine = _run(_lower(pid, dict(groups=groups, clock_hz=1e9), table, datapath=dp))
    assert engine.serviced == pytest.approx({"vec_issue": 5, pipe: 5})
    assert _trace(engine, "consumer")["start_ms"] == pytest.approx(_trace(engine, "producer")["end_ms"] + 7e-6, abs=1e-12)


def test_mx_dynamic_quant_serializes_scale_reduction_before_cast(register_device):
    table, dp = register_device
    engine = _run(_lower("P09_vec_reduce_cast", dict(rows=2, reduction_dim=65, lanes=64,
                                                     mx_dynamic_quant=True, clock_hz=1e9), table, datapath=dp))
    end = 0
    for name in ("vabs", "vreduce_max", "e8m0_scale", "vmul", "vcast_pack"):
        stage = _trace(engine, name)
        assert stage["start_ms"] == pytest.approx(end, abs=1e-12)
        end = stage["end_ms"]
    assert engine.serviced == pytest.approx({"vec_issue": 19, "vec_reduce": 6, "vec_alu": 13})


def test_sortall_merge_fan_in_and_masked_histogram_trunks(memory_device, register_device):
    old_table, old_dp = memory_device
    table, dp = register_device
    pid = "P10_sort_topk"
    for n, merges in ((64, 1), (128, 1), (256, 3), (512, 5), (1024, 11), (2048, 21)):
        engine = _run(_lower(pid, {"n": n}, old_table, datapath=old_dp))
        assert (engine.serviced["vbitsort32"], engine.serviced["vmrgsort4"]) == pytest.approx((n // 32, merges))
    for n in (32, 4096):
        with pytest.raises(ModelingError, match="^primitive_sortall_length_unsupported$"):
            _lower(pid, {"n": n}, old_table, datapath=old_dp)
    trunk, splits, k = 16384, 3, 7
    n = 2 * trunk + 1
    engine = _run(_lower(pid, dict(n=n, splits=splits, k=k, algorithm="masked_hist256"), table, datapath=dp))
    for offset, width in ((0, trunk), (trunk, trunk), (2 * trunk, 1)):
        assert _trace(engine, f"masked_low_histogram:{offset}")["serviced"]["vhist256"] == pytest.approx(width)
    assert engine.serviced["vhist256"] == pytest.approx(3 * n + 3 * 256 + splits * k)
    assert engine.serviced["hbm"] == pytest.approx(splits * k * 2 * 4)
    assert _trace(engine, "split_merge")["start_ms"] == _trace(engine, "split_merge:workspace_read")["end_ms"]


@pytest.mark.parametrize("frontend,limit", [("tla_dsl", 2048), ("catlass", 1024)])
def test_simt_requires_ub_carveout_and_enforces_frontend_thread_limit(memory_device, register_device, frontend, limit):
    _, old_dp = memory_device
    table, dp = register_device
    pid = "P11_simt"
    assert old_dp.simt_ub_carveout_bytes == 0 < dp.simt_ub_carveout_bytes
    with pytest.raises(UnmodeledCapabilityError, match="simt_ub_carveout_required"): _lower(pid, {"frontend": frontend}, table, datapath=old_dp)
    with pytest.raises(ModelingError, match="^primitive_simt_thread_limit$"): _lower(pid, dict(frontend=frontend, threads=limit + 1), table, datapath=dp)
    assert _run(_lower(pid, dict(frontend=frontend, threads=limit, warp_instructions=3), table, datapath=dp)).serviced["simt"] == pytest.approx(limit // 32 * 3)


def test_unit_flag_serializes_delay_and_requires_single_l0c_slot(register_device):
    table, dp = register_device
    pid = "P12_intracore_flag"
    table = _with_latencies(table, pid, unitflag_ns=300)
    with pytest.raises(ModelingError, match="^primitive_unit_flag_requires_single_l0c_stage$"): _lower(pid, dict(enable_unit_flag=True, l0c_stages=2), table, datapath=dp)
    result = _lower(pid, dict(enable_unit_flag=True, l0c_stages=1), table, datapath=dp)
    engine = _engine(result)
    done = submit_lowered_primitive(engine, result, after=(engine.submit("producer", latency=500e-6),))
    engine.advance(until=(done,))
    assert done.time == pytest.approx((500 + 300) * 1e-6) and not engine.serviced


def test_crosscore_scopes_and_reverse_credit_gate_the_next_window(register_device):
    table, dp = register_device
    pid = "P13_crosscore_sync"
    table = _with_latencies(table, pid, mode0_aic_barrier_ns=90, mode1_subblock_ns=100, mode2_paired_ns=70, reverse_ns=250)
    for mode, event, ns in ((0, 9, 90), (1, 10, 100), (2, 0, 70)):
        engine = _run(_lower(pid, {"mode": mode}, table, datapath=dp))
        assert _trace(engine, "crosscore:0")["metadata"]["event_id"] == event and engine.now == pytest.approx(ns * 1e-6)
    engine = _run(_lower(pid, dict(mode=2, with_reverse=True, notifications=16), table, datapath=dp))
    credit = _trace(engine, "reverse_credit:14")
    assert _trace(engine, "crosscore:14")["end_ms"] == pytest.approx(15 * 70e-6) and credit["end_ms"] - credit["start_ms"] == pytest.approx(250e-6)
    assert _trace(engine, "crosscore:15")["start_ms"] == pytest.approx(credit["end_ms"])


def test_mutex_ring_ids_and_l1_reuse_require_matching_address_and_coordinate(register_device):
    table, dp = register_device
    pid = "P14_mutex_block_mmad"
    with pytest.raises(ModelingError, match="^primitive_mutex_id_limit$"): _lower(pid, {"l0b_stages": 25}, table, datapath=dp)
    steps = [{"l1a": dict(addr=a, coord=c, work={"mte2_aic": 32}, duration_ms=1)} for a, c in ((0, 0), (0, 0), (0, 1), (32, 1))]
    result = _lower(pid, dict(l0b_stages=24, steps=steps, l1a_slot_bytes=32), table, datapath=dp)
    assert dict(result.metadata) == {"mutex_ids": 28, "l1_reuse_hits": 1} and len(result.ring_buffers) == 5
    engine = _run(result, rates={"mte2_aic": 32}, pools={"l1": 32})
    assert engine.serviced["mte2_aic"] == pytest.approx(3 * 32) and engine.peak_allocated["l1"] == 32 and not any(engine.allocated.values())


def test_stars_descriptor_queue_enforces_first_issue_and_credit_depth(register_device):
    table, dp = register_device
    table = _with_latencies(table, "P15_rts_stars", host_submit_ns=0, aic_first_issue_ns=500, issue_interval_ns=200)
    result = _lower("P15_rts_stars", dict(queue_depth=1, lanes=1, descriptors=2), table, datapath=dp)
    spec = result.stars_spec
    assert isinstance(spec, StarsSpec) and (spec.queue_depth, spec.lanes) == (1, 1)
    assert spec.first_issue_us == pytest.approx(0.5) and spec.issue_interval_us == pytest.approx(0.2) and _run(result).now == pytest.approx(900e-6)


def test_stream_budgets_bind_disjoint_owners_and_derive_nested_span(register_device):
    table, dp = register_device
    pid, cap = "P16_multistream_core_control", 2
    rates, units = {"vector": dp.aiv_count}, {"vector": dp.aiv_count}
    context = RecipeContext(None, {}, {"vector": "vector"}, rates, dp.aic_count, resource_units=units)
    program = KernelProgram((Stage("vector_work", {"vector": 4}),))
    derived = _lower(pid, {"program": KernelProgram((Stage("a", {"r": 4.0}, caps={"r": 1.0}), Stage("b", {"r": 4.0}, caps={"r": 1.0})))}, table, datapath=dp)
    ev_ms = table.sim_rate_p50(pid, "stream_event_ns", allow_uncalibrated_declaration=True) * 1e-6
    assert derived.analytical_ms == pytest.approx(8.0 + ev_ms)
    for mode in ("disjoint", "shared_cap"):
        plan = StreamPlan(0, tuple(StreamLane(s, StreamCoreBudget(1, cap, mode)) for s in ("main", "aux")))
        results = [_lower(pid, dict(mode=mode, stream_plan=plan, stream_id=s, program=program),
                          table, datapath=dp, context=context) for s in ("main", "aux")]
        stages = [result.stages[-1].program.stages[0] for result in results]
        if mode == "disjoint":
            owners = [set(s.placement["vector"]) for s in stages]
            assert all(len(placement) == cap for placement in owners) and owners[0].isdisjoint(owners[1])
        else:
            assert all(not s.placement and s.caps["vector"] == cap for s in stages)
        for result in results:
            assert _run(result, rates=rates, units=units).serviced["vector"] == pytest.approx(4)


def test_aclgraph_cache_omits_eager_readback_after_metadata_publication(register_device):
    table, dp = register_device
    pid = "P17_aicpu_metadata_launch"
    table = _with_latencies(table, pid, host_select_ns=100, aicpu_base_ns=200, gm_pub_ns=300, eager_d2h_sync_ns=400)
    for cached, elapsed_ns in ((False, 1000), (True, 600)):
        engine = _run(_lower(pid, {"aclgraph_cached": cached}, table, datapath=dp))
        assert (any(row["task"].endswith(":eager_readback") for row in engine.trace) is not cached) and engine.now == pytest.approx(elapsed_ns * 1e-6)


@pytest.mark.parametrize("policy,wire", [("exact_plus_header", 33), ("pad_full_frame", 512)])
def test_collective_short_frame_charges_the_declared_wire_policy(register_device, policy, wire):
    table, dp = register_device
    assert frame_wire_bytes(1, 480, 512) == (33, 1) and frame_wire_bytes(1, 480, 512, policy) == (wire, 1)
    result = _lower("P18_hccl_collective_band", dict(payload_bytes=1, payload_bytes_per_frame=480,
                    wire_bytes_per_frame=512, short_final_frame_policy=policy), table, datapath=dp)
    assert dict(result.metadata)["frames"] == 1 and _run(result).serviced["link"] == pytest.approx(wire)


def test_peer_tokens_require_declared_edge_and_status_follows_payload(register_device):
    table, dp = register_device
    for params in ({"peer": 99, "token_payload_bytes": (1,)}, {"peer": 99, "peer_edges": ((0, 99),), "token_payload_bytes": (1,)}):
        with pytest.raises(UnmodeledCapabilityError, match="direct_peer_edge_required"):
            _lower("P19_direct_peer_window", params, table, datapath=dp)
    dp2 = datapath_for_die(machine(table.sku).dies[0], ub_peer_edges=((0, 1),))
    engine = _run(_lower("P19_direct_peer_window", {"token_payload_bytes": (1, 481), "status_bytes": 32}, table, datapath=dp2))
    data, status = _trace(engine, "transfer"), _trace(engine, "status")
    assert status["start_ms"] == pytest.approx(data["end_ms"])
    for row in ("peer:0->1", "peer_egress:d0"):
        assert data["serviced"][row] == pytest.approx(3 * 512) and status["serviced"][row] == pytest.approx(32)


def test_dma_pinned_startup_and_coupled_per_die_and_root_complex_service():
    dp = _datapath(HOST_DMA_SKU)
    table = load_primitives(HOST_DMA_SKU, datapath=dp)
    pid, amount = "P20_host_dma_sync", 4 * 1024 * 1024
    root = RootComplexKey(2, "shared", "d2h").resource_name("host:")
    rate = table.sim_rate_p50(pid, "host_d2h_bytes_per_s") / 1000
    cell = table.cell(pid)
    table = replace(table, primitives={**table.primitives, pid: replace(cell, metrics={
        **cell.metrics, "host_rc_d2h_bytes_per_s": UncertaintyBand(rate * 1000),
        "pinned_dma_startup_ns": UncertaintyBand(15000.0, bound_kind="latency_lower"),
        "pageable_dma_startup_ns": UncertaintyBand(35000.0, bound_kind="latency_lower")})})
    results, startup = [], []
    for die, kind in enumerate(("pinned", "pageable")):
        r = _lower(pid, dict(payload_bytes=amount, node=2, root_complex_id="shared", namespace="host:", die=die, memory_kind=kind), table, datapath=dp)
        results.append(r)
        dma = _trace(_run(r), "dma")
        assert dma["coupled"] and dma["serviced"] == pytest.approx({root: amount, f"host:host_solo:d2h:d{die}": amount})
        startup.append(dma["start_ms"])
        assert startup[-1] == pytest.approx(table.sim_rate_p50(pid, f"{kind}_dma_startup_ns", allow_uncalibrated_declaration=True) * 1e-6)
    assert startup[0] < startup[1]
    engine = _engine(*results)
    engine.advance(until=[submit_lowered_primitive(engine, r, name_prefix=f"copy:{i}") for i, r in enumerate(results)], first=False)
    assert engine.now == pytest.approx(startup[0] + 2 * amount / rate)


def test_bench_idle_npu_parser_rejects_prefix_match_and_die_mismatch(register_device):
    from tools.bench_l0_primitives import build_primitive_table_payload, canonical_capture_bytes, parse_npu_smi_idle_die
    info = "No running processes found in NPU 10\nNo running processes found in NPU 0\n"
    mapping = "NPU ID Chip ID Chip Logic ID Chip Phy-ID Chip Name\n0 0 0 0 Ascend910\n"
    common = "Chip ID : 0\nAicore Usage Rate(%) : 0\nHBM Usage Rate(%) : 4\nAicore Count : 20\nAicore Freq(MHZ) : 1650\n"
    with pytest.raises(RuntimeError, match="bench_npu_not_idle"): parse_npu_smi_idle_die(info, mapping, common, npu_id=1, chip_id=0, device_id=1)
    with pytest.raises(ValueError, match="bench_npu_die_mismatch"): parse_npu_smi_idle_die(info, mapping, common, npu_id=0, chip_id=0, device_id=1)
    with pytest.raises(ValueError, match="bench_npu_mapping_missing"): parse_npu_smi_idle_die(info, "", common, npu_id=0, chip_id=0, device_id=0)
    with pytest.raises(RuntimeError, match="bench_npu_telemetry_missing"): parse_npu_smi_idle_die(info, mapping, "", npu_id=0, chip_id=0, device_id=0)
    with pytest.raises(RuntimeError, match="bench_npu_clock_missing"):
        parse_npu_smi_idle_die(info, mapping, common.replace("1650", "0"), npu_id=0, chip_id=0, device_id=0)
    assert parse_npu_smi_idle_die(info, mapping, common, npu_id=0, chip_id=0, device_id=0)["observed_aic_count"] == 20
    assert not PRIVATE_SOURCE.search(canonical_capture_bytes(CALIBRATED_REGISTER_SKU, {"meta": "192.0.2.7 /home" + "/u user@h"}).decode())
    with pytest.raises(ValueError, match="bench_sample_sku_mismatch"):
        build_primitive_table_payload(sku=CALIBRATED_REGISTER_SKU, cann_version="9", endpoint_alias="lab", campaign_id="c", raw_measurements={"device_name": REPRESENTATIVE_SKUS["a5"]})
    assert register_device[0].cell("P20_host_dma_sync").band("host_h2d_bytes_per_s").certified_upper_capacity == pytest.approx(64.0e9) and not load_primitives(UNMEASURED_REGISTER_SKU, datapath=_datapath(UNMEASURED_REGISTER_SKU)).provenance.calibrated
