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
from _support.hardware import machine
from mindie_sim.prediction.collectives.ports import RootComplexKey
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.prediction.runtime.l2_residency import AdmissionMemoryView
from mindie_sim.prediction.runtime.memory_physics import (
    _L2_CACHE_DISABLE, AddressRun, AddressRunPlan, MemoryAccessPattern, ReuseKey,
)
from mindie_sim.prediction.runtime.ring_buffer import RingBuffer
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
HOST_DMA_SKU = "Ascend910_9382"
PRIVATE_SOURCE = re.compile(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])'
                            r'|@|/(?:Users|home|root)/|\$\{?HOME\b|\b(?:username|user_name)\b')


def _datapath(sku):
    return datapath_for_die(machine(sku).dies[0])


@pytest.fixture(scope="module")
def register_device():
    sku = REPRESENTATIVE_SKUS["a5"]
    dp = _datapath(sku)
    return load_primitives(sku, datapath=dp), dp


@pytest.fixture(scope="module")
def memory_device():
    sku = REPRESENTATIVE_SKUS["a2"]
    dp = _datapath(sku)
    return load_primitives(sku, datapath=dp, allow_uncalibrated_declaration=True), dp


def _digest(raw):
    body = {key: value for key, value in raw.items() if key != "envelope_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _bands(tree):
    for value in tree.values():
        if isinstance(value, UncertaintyBand):
            yield value
        elif isinstance(value, Mapping):
            yield from _bands(value)


def _with_latencies(table, pid, **delays):
    cell = table.cell(pid)
    bands = {metric: UncertaintyBand(ns, bound_kind="latency_lower") for metric, ns in delays.items()}
    return replace(table, primitives={**table.primitives, pid: replace(cell, metrics={**cell.metrics, **bands})})


def _stages(stages):
    for stage in stages:
        yield stage
        if stage.program is not None:
            yield from _stages(stage.program.stages)


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


def _trace(engine, stage):
    return next(row for row in engine.trace if row["task"].endswith(f":{stage}"))


def _read(key, *runs, policy="normal"):
    logical = sum(run.row_bytes * run.rows for run in runs)
    return MemoryAccessPattern(key.backing_allocation, 0, 0, 0, "read", "mte2_aiv",
                               logical, AddressRunPlan(runs), key, cache_policy=policy)


@pytest.mark.parametrize("pool,owner,wrong", [
    ("l1", "aic", "aiv"), ("l0a", "aic", "aiv"), ("l0b", "aic", "aiv"),
    ("l0c", "aic", "aiv"), ("ub", "aiv", "aic"),
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
    assert state.admit_access(fresh, 192) == (0, 192)
    assert not state.is_resident(fresh.reuse_key)
    assert state.active_stream_count(0) == 1
    assert state.admit_access(replace(read, cache_policy="last_use"), 192) == (192, 0)
    assert not state.is_resident(read.reuse_key)


def test_periodic_sector_coverage_excludes_stride_gaps_and_hits_identical_reads():
    key = ReuseKey(0, "strided", 0)
    read = _read(key, AddressRun(0, 5000, 32, 64))
    gap = _read(key, AddressRun(32, 1, 32, 32))
    amount = 5000 * 32
    assert read.runs.sector_intervals() != ((0, 320000),)
    assert read.runs.sector_coverage().measure_bytes() == amount
    state = AdmissionMemoryView({0: amount})
    assert state.admit_access(read, amount) == (0, amount)
    assert state.preview_access(gap, 32) == (0, 32)
    assert state.admit_access(read, amount) == (amount, 0)


def test_sealed_envelopes_match_manifest_and_preserve_calibrated_rate_bounds():
    manifest = load_manifest()
    entries = manifest["tables"]
    assert len(entries) == 5
    assert {entry["file"] for entry in entries.values()} == {
        path.name for path in PRIMITIVES_DIR.glob("*.json") if path.name != "manifest.json"}
    assert not PRIVATE_SOURCE.search(json.dumps(manifest))
    for sku, entry in entries.items():
        path = PRIMITIVES_DIR / entry["file"]
        encoded = path.read_text(encoding="utf-8")
        assert not PRIVATE_SOURCE.search(encoded), sku
        raw = json.loads(encoded)
        table = load_primitives(sku, datapath=_datapath(sku), allow_uncalibrated_declaration=True)
        assert _digest(raw) == table.envelope_sha256 == entry["envelope_sha256"]
        assert table.to_envelope_dict() == raw
        assert table.sku == sku and table.cann_version == entry["cann_version"]
        assert table.provenance.capture_uri == entry["capture_uri"]
        assert table.provenance.calibrated is entry["calibrated"]
        assert table.warnings == (() if entry["calibrated"] else (UNCALIBRATED_SKU_WARNING,))
        assert len(table.primitives) == entry["total_primitives"] == 20
        assert sum(cell.status == "calibrated" for cell in table.primitives.values()) == entry[
            "calibrated_primitive_count"]
        for cell in table.primitives.values():
            for band in _bands({"metrics": cell.metrics, "modes": cell.modes}):
                if band.bound_kind == "rate_upper":
                    assert band.sim_rate_p50 <= band.certified_upper_capacity, (sku, cell.primitive_id)
                    if band.p50 is not None:
                        assert band.p99 <= band.certified_upper_capacity
                        assert band.cv_pct <= 5.0
    alias = "Ascend950DT_9581"
    alias_table = load_primitives(alias, datapath=_datapath(alias))
    assert alias_table.envelope_sha256 == entries["Ascend950DT_9582"]["envelope_sha256"]


@pytest.mark.parametrize("field", ("sku", "cann_version", "calibrated", "metric"))
def test_full_envelope_digest_detects_mutation_and_manifest_rejects_resealing(tmp_path, field, register_device):
    table, dp = register_device
    raw = table.to_envelope_dict()
    if field == "calibrated":
        raw["provenance"][field] = not raw["provenance"][field]
    elif field == "metric":
        raw["primitives"]["P01_datacopy"]["metrics"]["hbm_read_bytes_per_s"]["certified_upper_capacity"] *= 2
    else:
        raw[field] += "-modified"
    mismatch = "^primitive_table_sha256_mismatch$"
    with pytest.raises(ModelingError, match=mismatch):
        parse_primitive_table(raw)
    path = tmp_path / load_manifest()["tables"][table.sku]["file"]
    for digest in (raw["envelope_sha256"], _digest(raw)):
        raw["envelope_sha256"] = digest
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ModelingError, match=mismatch):
            load_primitives(path, datapath=dp)


def test_primitive_declarations_require_matching_datapath_and_explicit_unmeasured_rates(
    memory_device, register_device,
):
    memory, old_dp = memory_device
    register, _ = register_device
    with pytest.raises(ModelingError, match="^primitive_table_arch_mismatch$"):
        load_primitives(register.sku, datapath=old_dp)
    with pytest.raises(ModelingError, match="^primitive_table_uncalibrated$"):
        load_primitives(memory.sku, datapath=old_dp, require_calibrated=True)
    for table in (memory, register):
        pid, metric = "P02_load_data", "l1_to_l0a_bytes_per_s_per_aic"
        rate, warnings = resolve_primitive_rate(table, pid, metric, allow_uncalibrated_declaration=True)
        assert rate == table.cell(pid).band(metric).certified_upper_capacity > 0
        assert warnings == (UNCALIBRATED_SKU_WARNING,)
        with pytest.raises(ModelingError, match="^primitive_unmeasured_requires_declaration:"):
            resolve_primitive_rate(table, pid, metric, allow_uncalibrated_declaration=False)


def test_uncertainty_bounds_enclose_the_required_quantile():
    with pytest.raises(ModelingError, match="^uncertainty_band_upper_below_p99$"):
        UncertaintyBand(9, p50=8, p99=10, bound_kind="rate_upper")
    with pytest.raises(ModelingError, match="^uncertainty_band_lower_above_p05$"):
        UncertaintyBand(6, p05=5, p50=8, bound_kind="latency_lower")


@pytest.mark.parametrize("source", ["192.0.2.7", "file://" + "/home" + "/fixture/capture", "user:token@lab"])
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
    cold = lower_primitive("P01_datacopy", params, table, datapath=dp, memory_state=state)
    ledger = cold.bound_access.ledger
    assert (ledger.logical_tensor_bytes, ledger.aligned_transfer_bytes, ledger.dram_sector_bytes) == (67, 128, 96)
    assert ledger.l2_hit_bytes == 0 and not state.is_resident(key)
    l2_rate = table.sim_rate_p50("P05_l2_residency_reuse", "l2_die_bytes_per_s") / 1000
    engine = _engine(cold, rates={"l2": l2_rate})
    first = submit_lowered_primitive(engine, cold, name_prefix="cold")
    second = submit_lowered_primitive(engine, cold, after=(first,), name_prefix="warm")
    engine.advance(until=(second,))
    assert engine.delivered["hbm_read@d0"] == pytest.approx(96)
    assert engine.delivered["mte2_aiv"] == pytest.approx(256)
    assert engine.delivered["l2"] == pytest.approx(128)
    bypass = lower_primitive("P01_datacopy", {**params, "cache_policy": "disable"}, table,
                             datapath=dp, memory_state=state)
    assert _run(bypass).delivered["hbm_read@d0"] == pytest.approx(96)
    assert state.is_resident(key)


def test_load_data_separate_fractal_operands_follow_serial_setup(register_device):
    table, dp = register_device
    pid = "P02_load_data"
    table = _with_latencies(table, pid, setup_ns=750)
    result = lower_primitive(pid, {"m": 17, "n": 33, "k": 31}, table, datapath=dp)
    engine = _run(result)
    elapsed = 0
    for operand, amount in (("a", 2 * 2 * 512), ("b", 3 * 2 * 512)):
        setup, transfer = (_trace(engine, f"load_{operand}{suffix}") for suffix in (":setup", ""))
        assert setup["start_ms"] == pytest.approx(elapsed)
        elapsed += 750 * 1e-6
        assert setup["end_ms"] == transfer["start_ms"] == pytest.approx(elapsed)
        assert not setup["serviced"] and transfer["serviced"]["mte1"] == pytest.approx(amount)
        elapsed += amount * 1000 / table.sim_rate_p50(pid, f"l1_to_l0{operand}_bytes_per_s_per_aic")
        assert transfer["end_ms"] == pytest.approx(elapsed)


def test_fixpipe_destination_controls_width_and_requires_direct_ub_route(memory_device, register_device):
    old_table, old_dp = memory_device
    table, dp = register_device
    pid, dims = "P03_fixpipe", {"m": 2, "n": 33}
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_unsupported:l0c->ub"):
        lower_primitive(pid, {**dims, "dst": "ub"}, old_table, datapath=old_dp)
    for dst, split, width in (("gm", "NO_SPLIT", 33), ("ub", "NO_SPLIT", 48), ("ub", "SPLIT_N", 64)):
        result = lower_primitive(pid, {**dims, "dst": dst, "split_mode": split}, table, datapath=dp)
        meta = dict(result.metadata)
        assert (meta["n_eff"], meta["input_l0c_bytes"], meta["output_bytes"]) == (width, 2 * 33 * 4, 2 * width * 2)
        engine = _run(result)
        if dst == "gm":
            assert engine.delivered["hbm_write@d0"] == pytest.approx(160)
        else:
            assert engine.serviced["fixpipe"] == pytest.approx(2 * width * 2)
            assert not engine.delivered


def test_mte3_direct_l1_route_and_one_read_modify_write_charge(memory_device, register_device):
    old_table, old_dp = memory_device
    table, dp = register_device
    pid, params = "P04_mte3_store", {"dst": "l1", "bytes": 33}
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_unsupported:ub->l1"):
        lower_primitive(pid, params, old_table, datapath=old_dp)
    engine = _run(lower_primitive(pid, params, table, datapath=dp))
    assert engine.serviced["mte3"] == pytest.approx(64) and not engine.delivered
    atomic = lower_primitive(pid, {"bytes": 33, "atomic": True}, table, datapath=dp)
    assert atomic.bound_access.ledger.dram_sector_bytes == 2 * 64
    engine = _run(atomic)
    for row in ("mte3", "hbm_read@d0", "hbm_write@d0"):
        assert engine.delivered[row] == pytest.approx(64)


def test_l2_knee_counts_distinct_streams_and_bypass_removes_l2_work(register_device):
    table, dp = register_device
    state = AdmissionMemoryView({0: 512})
    reads = [_read(ReuseKey(0, name, 0, stream), AddressRun(0, 1, 384, 384))
             for name, stream in (("weights", "main"), ("states", "main"), ("auxiliary", "aux"))]
    base = {"working_set_bytes": 384, "pattern": reads[0]}
    for index, (pattern, streams, knee) in enumerate(zip(reads, (1, 1, 2), (512, 512, 256))):
        state.admit_access(pattern, 384)
        assert state.active_stream_count(0) == streams and state.effective_l2_bytes(0) == knee
        for policy in ("normal", "disable"):
            hits = 384 if index < 2 and policy == "normal" else 0
            result = lower_primitive("P05_l2_residency_reuse", {**base, "cache_policy": policy},
                                     table, datapath=dp, memory_state=state)
            assert dict(result.metadata)["effective_l2_bytes"] == knee
            assert _run(result).serviced.get("l2", 0) == pytest.approx(hits)


@pytest.mark.parametrize("m,n", [(49, 63), (81, 33)])
def test_mmad_issued_flops_and_pipe_barrier_threshold(register_device, m, n):
    table, dp = register_device
    pid, k = "P06_mmad", 33
    params = dict(m=m, n=n, k=k, m_iss=16, n_iss=32, k_iss=32, mode="bf16")
    issued = 2 * ((m + 15) // 16 * 16) * ((n + 31) // 32 * 32) * 64
    band = table.cell(pid).band("cube_only_rate", mode="bf16")
    table = _with_latencies(table, pid, pipe_m_barrier_ns=125)
    for role, rate in (("sim", band.p50), ("dual", band.certified_upper_capacity)):
        result = lower_primitive(pid, params, table, datapath=dp, role=role)
        has_barrier = (m // 16) * (n // 16) < 10
        assert any(s.name == "PipeBarrier<PIPE_M>" for s in result.stages) is has_barrier
        assert dict(result.metadata)["issued_flops"] == issued
        engine = _run(result)
        assert engine.serviced["cube"] == pytest.approx(issued)
        assert engine.now == pytest.approx(issued * 1000 / rate + has_barrier * 125e-6)
    assert band.p50 < band.certified_upper_capacity


def test_mmad_weight_conversion_prologues_finish_before_cube(register_device):
    table, dp = register_device
    with pytest.raises(UnmodeledCapabilityError, match="native_matrix_mode_required:mxfp8"):
        lower_primitive("P06_mmad", dict(m=16, n=5, k=13, mode="a8w4_mx"), table, datapath=dp)
    for mode, end, expected in (
        ("w4a8", "unpack:workspace_write", {"mte2_aiv": 33, "vec_alu": 130, "mte3": 65}),
        ("a8w4_mx", "mx:decompress_b", {"mte1": 75}),
    ):
        native = {"native_mode": "fp8_e4m3"} if mode == "a8w4_mx" else {}
        result = lower_primitive("P06_mmad", dict(m=16, n=5, k=13, mode=mode, mx_scale_bytes=10, **native),
                                 table, datapath=dp)
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
    result = lower_primitive(pid, dict(groups=groups, clock_hz=1e9), table, datapath=dp)
    engine = _run(result)
    assert engine.serviced == pytest.approx({"vec_issue": 5, pipe: 5})
    ready = _trace(engine, "producer")["end_ms"] + 7e-6
    assert _trace(engine, "consumer")["start_ms"] == pytest.approx(ready, abs=1e-12)


def test_mx_dynamic_quant_serializes_scale_reduction_before_cast(register_device):
    table, dp = register_device
    result = lower_primitive("P09_vec_reduce_cast", dict(rows=2, reduction_dim=65, lanes=64,
                                                        mx_dynamic_quant=True, clock_hz=1e9), table, datapath=dp)
    engine = _run(result)
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
        engine = _run(lower_primitive(pid, {"n": n}, old_table, datapath=old_dp))
        assert engine.serviced["vbitsort32"] == pytest.approx(n // 32)
        assert engine.serviced["vmrgsort4"] == pytest.approx(merges)
    for n in (32, 4096):
        with pytest.raises(ModelingError, match="^primitive_sortall_length_unsupported$"):
            lower_primitive(pid, {"n": n}, old_table, datapath=old_dp)
    trunk, splits, k = 16384, 3, 7
    n = 2 * trunk + 1
    engine = _run(lower_primitive(pid, dict(n=n, splits=splits, k=k), table, datapath=dp))
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
    with pytest.raises(UnmodeledCapabilityError, match="simt_ub_carveout_required"):
        lower_primitive(pid, {"frontend": frontend}, table, datapath=old_dp)
    with pytest.raises(ModelingError, match="^primitive_simt_thread_limit$"):
        lower_primitive(pid, dict(frontend=frontend, threads=limit + 1), table, datapath=dp)
    result = lower_primitive(pid, dict(frontend=frontend, threads=limit, warp_instructions=3), table, datapath=dp)
    assert _run(result).serviced["simt"] == pytest.approx(limit // 32 * 3)


def test_unit_flag_serializes_delay_and_requires_single_l0c_slot(register_device):
    table, dp = register_device
    pid = "P12_intracore_flag"
    table = _with_latencies(table, pid, unitflag_ns=300)
    with pytest.raises(ModelingError, match="^primitive_unit_flag_requires_single_l0c_stage$"):
        lower_primitive(pid, dict(enable_unit_flag=True, l0c_stages=2), table, datapath=dp)
    result = lower_primitive(pid, dict(enable_unit_flag=True, l0c_stages=1), table, datapath=dp)
    engine = _engine(result)
    ready = engine.submit("producer", latency=500e-6)
    done = submit_lowered_primitive(engine, result, after=(ready,))
    engine.advance(until=(done,))
    assert done.time == pytest.approx((500 + 300) * 1e-6)
    assert not engine.serviced


def test_crosscore_scopes_and_reverse_credit_gate_the_next_window(register_device):
    table, dp = register_device
    pid = "P13_crosscore_sync"
    table = _with_latencies(table, pid, mode0_aic_barrier_ns=90, mode1_subblock_ns=100,
                            mode2_paired_ns=70, reverse_ns=250)
    for mode, event, ns in ((0, 9, 90), (1, 10, 100), (2, 0, 70)):
        result = lower_primitive(pid, {"mode": mode}, table, datapath=dp)
        engine = _run(result)
        assert _trace(engine, "crosscore:0")["metadata"]["event_id"] == event
        assert engine.now == pytest.approx(ns * 1e-6)
    result = lower_primitive(pid, dict(mode=2, with_reverse=True, notifications=16), table, datapath=dp)
    engine = _run(result)
    assert _trace(engine, "crosscore:14")["end_ms"] == pytest.approx(15 * 70e-6)
    credit = _trace(engine, "reverse_credit:14")
    assert credit["end_ms"] - credit["start_ms"] == pytest.approx(250e-6)
    assert _trace(engine, "crosscore:15")["start_ms"] == pytest.approx(credit["end_ms"])


def test_mutex_ring_ids_and_l1_reuse_require_matching_address_and_coordinate(register_device):
    table, dp = register_device
    pid = "P14_mutex_block_mmad"
    with pytest.raises(ModelingError, match="^primitive_mutex_id_limit$"):
        lower_primitive(pid, {"l0b_stages": 25}, table, datapath=dp)
    steps = [{"l1a": dict(addr=addr, coord=coord, work={"mte2_aic": 32}, duration_ms=1)}
             for addr, coord in ((0, 0), (0, 0), (0, 1), (32, 1))]
    result = lower_primitive(pid, dict(l0b_stages=24, steps=steps, l1a_slot_bytes=32), table, datapath=dp)
    assert dict(result.metadata) == {"mutex_ids": 28, "l1_reuse_hits": 1}
    assert len(result.ring_buffers) == 5 and all(isinstance(ring, RingBuffer) for ring in result.ring_buffers)
    engine = _run(result, rates={"mte2_aic": 32}, pools={"l1": 32})
    assert engine.serviced["mte2_aic"] == pytest.approx(3 * 32)
    assert engine.peak_allocated["l1"] == 32 and not any(engine.allocated.values())


@pytest.mark.parametrize("host_ns,issue_ns", [(1200, 800), (800, 1200)])
def test_stars_descriptor_throughput_is_limited_by_the_slower_issuer(register_device, host_ns, issue_ns):
    table, dp = register_device
    pid = "P15_rts_stars"
    table = _with_latencies(table, pid, host_submit_ns=host_ns, issue_interval_ns=issue_ns)
    result = lower_primitive(pid, dict(queue_depth=8, lanes=2), table, datapath=dp)
    spec = result.stars_spec
    assert isinstance(spec, StarsSpec) and (spec.queue_depth, spec.lanes) == (8, 2)
    assert spec.issue_interval_us == pytest.approx(issue_ns / 1000)
    assert _run(result).now == pytest.approx(max(host_ns, issue_ns) * 1e-6)


def test_stream_budgets_bind_disjoint_owners_or_cap_shared_service(register_device):
    table, dp = register_device
    pid = "P16_multistream_core_control"
    cap = 2
    rates, units = {"vector": dp.aiv_count}, {"vector": dp.aiv_count}
    context = RecipeContext(None, {}, {"vector": "vector"}, rates, dp.aic_count, resource_units=units)
    program = KernelProgram((Stage("vector_work", {"vector": 4}),))
    for mode in ("disjoint", "shared_cap"):
        plan = StreamPlan(0, tuple(StreamLane(s, StreamCoreBudget(1, cap, mode)) for s in ("main", "aux")))
        results = [lower_primitive(pid, dict(mode=mode, stream_plan=plan, stream_id=s, program=program,
                                             work_ms=2), table, datapath=dp, context=context) for s in ("main", "aux")]
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
    table = _with_latencies(table, pid, host_select_ns=100, aicpu_base_ns=200, gm_pub_ns=300,
                            eager_d2h_sync_ns=400)
    for cached, elapsed_ns in ((False, 1000), (True, 600)):
        result = lower_primitive(pid, {"aclgraph_cached": cached}, table, datapath=dp)
        engine = _run(result)
        assert any(row["task"].endswith(":eager_readback") for row in engine.trace) is not cached
        assert engine.now == pytest.approx(elapsed_ns * 1e-6)


@pytest.mark.parametrize("policy,wire", [("pad_full_frame", 512), ("exact_plus_header", 33)])
def test_collective_short_frame_charges_the_declared_wire_policy(register_device, policy, wire):
    table, dp = register_device
    assert frame_wire_bytes(1, 480, 512, policy) == (wire, 1)
    result = lower_primitive("P18_hccl_collective_band", dict(payload_bytes=1, payload_bytes_per_frame=480,
                             wire_bytes_per_frame=512, short_final_frame_policy=policy), table, datapath=dp)
    assert dict(result.metadata)["frames"] == 1
    assert _run(result).serviced["link"] == pytest.approx(wire)


def test_peer_tokens_frame_individually_and_status_follows_payload(register_device):
    table, dp = register_device
    result = lower_primitive("P19_direct_peer_window", {"token_payload_bytes": (1, 481), "status_bytes": 32},
                             table, datapath=dp)
    engine = _run(result)
    data, status = _trace(engine, "transfer"), _trace(engine, "status")
    assert status["start_ms"] == pytest.approx(data["end_ms"])
    for row in ("peer:0->1", "peer_egress:d0"):
        assert data["serviced"][row] == pytest.approx(3 * 512)
        assert status["serviced"][row] == pytest.approx(32)


def test_dma_pinned_startup_and_coupled_per_die_and_root_complex_service():
    dp = _datapath(HOST_DMA_SKU)
    table = load_primitives(HOST_DMA_SKU, datapath=dp)
    pid, amount = "P20_host_dma_sync", 32 * 1024
    root = RootComplexKey(2, "shared", "d2h").resource_name("host:")
    rate = table.sim_rate_p50(pid, "host_d2h_bytes_per_s") / 1000
    results, startup = [], []
    for die, kind in enumerate(("pinned", "pageable")):
        result = lower_primitive(pid, dict(payload_bytes=amount, node=2, root_complex_id="shared",
                                 namespace="host:", die=die, memory_kind=kind), table, datapath=dp)
        results.append(result)
        dma = _trace(_run(result), "dma")
        solo = f"host:host_solo:d2h:d{die}"
        assert dma["coupled"] and dma["serviced"] == pytest.approx({root: amount, solo: amount})
        startup.append(dma["start_ms"])
        assert startup[-1] == pytest.approx(table.sim_rate_p50(pid, f"{kind}_dma_startup_ns") * 1e-6)
    assert startup[0] < startup[1]
    engine = _engine(*results)
    done = [submit_lowered_primitive(engine, result, name_prefix=f"copy:{die}") for die, result in enumerate(results)]
    engine.advance(until=done, first=False)
    assert engine.now == pytest.approx(startup[0] + 2 * amount / rate)
