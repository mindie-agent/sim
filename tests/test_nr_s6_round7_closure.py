"""Round-7 closure: signed tails, communicators, cleanup status and bytes.

[Role]
  Re-run Astra's five remaining Round-7 counterexamples: signed terminal
  replacement and unreported observed totals, registered ``send_recv``
  communicators, a cleanup budget inside the wall deadline, cleanup-failure
  qualification through direct and CLI fitting, raw-byte artifact integrity
  and full CAS-slot immutability, plus a bounded diff-trace MCP envelope.

[Invariants]
  * Every assertion executes product code; no source grep substitutes for a
    timing, communicator, parser, storage or token-budget observation.
  * Probe tests use two real Python threads and the per-group rendezvous
    broker; the supervisor test uses a real SIGTERM-ignoring subprocess.
  * The one-second two-size probe plan is built by the planner and is never
    patched after admission.
  * Storage tests write only fresh ASCII roots under ``/tmp`` and restore them.
  * The envelope sweep covers every inspected declared version, every ready
    receipt/resolution and a broad invalid-handle grid at <=250 tokens.

[Timeouts & Locks]
  Broker threads and watchers are joined with bounded timeouts; the supervisor
  child is terminated/killed with bounded waits; no NPU or HCCL call is made.

[Side-Effect Policy]
  Writes only pytest temporary files, fresh short reproducer roots under
  ``/tmp`` and caller-owned abort files used by probe tests.
"""

from __future__ import annotations

import json
from pathlib import Path
__import__("pytest").importorskip("mindie_profiler")
from mindie_profiler import repro_atomic_io as checked_io
import shutil
import subprocess
import sys
import threading
import time

import pytest

from mindie_profiler.fabric_probe import (
    build_fabric_probe_plan, emit_fabric_probe_script, main as fabric_main,
    supervise_probe_processes,
)
from mindie_profiler.fabric_probe_fit import (
    fit_calibrated_fabric_profile, parse_fabric_probe_samples,
)
from mindie_profiler.op_api_registry import (
    call_mcp_tool as registry_mcp, load_catalog,
)
from mindie_profiler.repro_op import (
    call_mcp_tool as repro_mcp, generate_single_op_reproducer,
)
from mindie_profiler.repro_storage import resolve_repro_artifact, short_root
from mindie_profiler.token_budget import estimate_tokens
from mindie_sim.diff_trace import (
    attribute_diff, call_mcp_tool as diff_mcp, resolve_diff_artifact,
)

from nr_s6_round4_probe import BrokerDist, World, run_ranks
from nr_s6_support import paged_binding
from temp_cleanup import tempdir
from test_nr_s6_round2_closure import _ready_bindings


OPS = [
    {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": 1},
    {"name": "B", "stream": 0, "start_ms": 1, "duration_ms": 1},
]
SAMPLE_SCHEMA = "mindie-fabric-probe-sample/2"
TOPOLOGY = {"tiers": [{"tier": "intra_node_hccs", "ranks": [0, 1],
                       "protocol": "HCCS"}]}


def _trace(tail):
    return {"total_ms": 2 + tail, "ops": OPS,
            "gaps": [{"stream": 0, "start_ms": 2, "duration_ms": tail}]}


def _diff(predicted, captured):
    return attribute_diff(predicted, captured, launch_overhead_us=0)


def _codes(results):
    return sorted(code for code, _rows in results.values())


def _rows(results):
    return [item for _rank, (_code, rows) in results.items() for item in rows]


def _samples(results):
    return [item for item in _rows(results) if item.get("schema") == SAMPLE_SCHEMA]


def _fresh_root():
    return tempdir("r7cas-", directory="/tmp")


# ----------------------------------------------------------------------- F1
def test_f1_signed_terminal_replacement_is_bidirectional():
    shrink = _diff(_trace(4), _trace(2))
    assert shrink["stages"] == {"R0": 6.0, "R1": 4.0, "R2": 4.0,
                                "R3": 4.0, "R4_model": 4.0, "O": 4.0}
    assert shrink["attribution"]["sched_delta_ms"] == pytest.approx(-2.0)
    assert shrink["attribution"]["residual_ms"] == pytest.approx(0.0)
    assert shrink["reconciliation"]["reconciled"] is True
    assert shrink["status"] == "reconciled"

    grow = _diff(_trace(2), _trace(4))
    assert grow["attribution"]["sched_delta_ms"] == pytest.approx(2.0)
    assert grow["attribution"]["residual_ms"] == pytest.approx(0.0)
    assert grow["reconciliation"]["reconciled"] is True

    identity = _diff(_trace(4), _trace(4))
    assert all(value == 0.0 for value in identity["buckets"].values())
    assert identity["reconciliation"]["reconciled"] is True


def test_f1_missing_captured_total_never_reconciles():
    captured = {"ops": OPS,
                "gaps": [{"start_ms": 2, "duration_ms": 2}]}
    result = _diff(_trace(0), captured)
    assert result["stages"] == {"R0": 2.0, "R1": 4.0, "R2": 4.0,
                                "R3": 4.0, "R4_model": 4.0, "O": 4.0}
    assert result["attribution"]["residual_ms"] == pytest.approx(0.0)
    assert result["reconciliation"]["captured_total_explicit"] is False
    assert result["reconciliation"]["captured_terminal_evidence"] is True
    assert result["reconciliation"]["reconciled"] is False
    assert result["status"] == "partial"
    assert [issue["code"] for issue in result["issues"]] == [
        "captured_total_unreported"]


def test_f1_explicit_total_without_terminal_evidence_keeps_residual():
    for total, expected in ((4, -2.0), (2, -4.0)):
        result = _diff(_trace(4), {"total_ms": total, "ops": OPS})
        assert result["reconciliation"]["captured_total_explicit"] is True
        assert result["reconciliation"]["captured_terminal_evidence"] is False
        assert result["attribution"]["residual_ms"] == pytest.approx(expected)
        assert result["reconciliation"]["reconciled"] is False
        assert result["status"] == "partial"


def test_f1_zero_duration_terminal_gap_is_explicit_evidence():
    captured = {"total_ms": 2, "ops": OPS,
                "gaps": [{"stream": 0, "start_ms": 2, "duration_ms": 0}]}
    result = _diff(_trace(4), captured)
    assert result["stages"] == {"R0": 6.0, "R1": 2.0, "R2": 2.0,
                                "R3": 2.0, "R4_model": 2.0, "O": 2.0}
    assert result["attribution"]["sched_delta_ms"] == pytest.approx(-4.0)
    assert result["attribution"]["residual_ms"] == pytest.approx(0.0)
    assert result["reconciliation"]["captured_terminal_evidence"] is True
    assert result["reconciliation"]["reconciled"] is True


# ----------------------------------------------------------------------- F6
def _probe_namespace():
    plan = build_fabric_probe_plan(
        world_size=2, ranks_per_node=2, sizes=[4096], ops=["send_recv"],
        iterations=1, warmup=0, duration_s=1.0, topology=TOPOLOGY)
    script = emit_fabric_probe_script(plan)["script"]
    namespace = {"__name__": "round7_probe"}
    exec(compile(script, "<probe7>", "exec"), namespace)
    return plan, namespace


def test_f6_send_recv_blocks_on_registered_tier_communicator(tmp_path):
    plan, namespace = _probe_namespace()
    world = World(plan)
    dist = BrokerDist(world, 1)
    abort_file = tmp_path / "abort.json"
    namespace["ABORT_FILE"] = str(abort_file)
    groups = namespace["_tier_groups"](dist, 1)
    tier = groups["intra_node_hccs"][0]
    world_group = world.broker.world_group
    assert world_group is not tier
    assert world_group.abort_event is not tier.abort_event
    outcome = {}
    try:
        def blocked_recv():
            try:
                namespace["_call_once"](None, dist, "send_recv", object(), None,
                                        tier, [0, 1], 1, 1000)
                outcome["error"] = None
            except TimeoutError:
                raise
            except Exception as exc:  # the bounded peer wake must be the error
                outcome["error"] = type(exc).__name__

        thread = threading.Thread(target=blocked_recv, daemon=True)
        thread.start()
        cutoff = time.monotonic() + 0.5
        while (world.broker.last_send_recv_group is not tier
               and time.monotonic() < cutoff):
            time.sleep(0.005)
        assert world.broker.last_send_recv_group is tier
        world_group.abort()
        time.sleep(0.03)
        assert thread.is_alive()
        assert tier.abort_event.is_set() is False
        abort_file.write_text(json.dumps(
            {"kind": "abort", "rank": 0, "step": "round7",
             "error": "peer failed"}), encoding="utf-8")
        thread.join(0.3)
        peer_woke_before_timeout = not thread.is_alive()
        assert peer_woke_before_timeout is True
        assert outcome["error"] == "PeerAbortedError"
        assert tier.aborted is True
        assert world.broker.last_send_recv_group is tier
    finally:
        namespace["_stop_abort_watcher"]()


def test_f6_supervisor_cleanup_is_inside_the_wall_budget():
    code = ("import signal,time;"
            "signal.signal(signal.SIGTERM, lambda *args: None);"
            "print('ready', flush=True);"
            "time.sleep(30)")
    proc = subprocess.Popen([sys.executable, "-c", code],
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "ready"
        started = time.monotonic()
        result = supervise_probe_processes(
            [proc], wall_budget_s=0.12, poll_interval_s=0.005)
        elapsed = time.monotonic() - started
        assert result["status"] == "deadline_exceeded"
        assert result["exit_code"] == 2
        assert result["terminated_ranks"] == [0]
        assert proc.poll() is not None
        assert proc.returncode == -9
        assert elapsed <= 0.15
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        proc.stdout.close()


# ----------------------------------------------------------------------- F7
def _two_size_plan():
    return build_fabric_probe_plan(
        world_size=2, ranks_per_node=2, sizes=[4096, 8192],
        ops=["all_reduce"], iterations=1, warmup=0, duration_s=1.0,
        topology=TOPOLOGY)


def _latency():
    return {0: {4: ("advance", 0.0001), 8: ("advance", 0.0002)},
            1: {4: ("advance", 0.0001), 8: ("advance", 0.0002)}}


def _assert_cleanup_blocked(plan, results):
    samples = _samples(results)
    assert len(samples) == 4
    failed = [item for item in samples if item["rank"] == 1]
    assert failed and all(item["validated"] is False for item in failed)
    assert {item["status"] for item in failed} == {"cleanup_failed"}
    assert any(item.get("kind") == "cleanup_failed" for item in _rows(results))
    profile = fit_calibrated_fabric_profile(plan, _rows(results))
    assert profile["status"] == "partial"
    assert profile["run_evidence_blocked"] is True
    assert profile["usable_for_link_override"] is False
    assert profile["control_record_count"] >= 1


def test_f7_two_size_plan_covers_every_completion_edge():
    plan = _two_size_plan()
    assert plan["status"] == "admitted"
    assert plan["budget_s"] == 1.0
    assert plan["deadline"]["measurement_budget_s"] == 0.9
    baseline, healthy = run_ranks(plan, sync_actions=_latency())
    assert baseline.errors == {}
    assert _codes(healthy) == [0, 0]
    base_samples = _samples(healthy)
    assert len(base_samples) == 4
    assert all(item["validated"] is True for item in base_samples)
    profile = fit_calibrated_fabric_profile(plan, _rows(healthy))
    assert profile["status"] == "calibrated"
    assert profile["usable_for_link_override"] is True
    final_gate = len(baseline.broker.decision_logs[1])
    assert final_gate > 0

    late_min, late_min_results = run_ranks(
        plan, sync_actions=_latency(), decision_delay=(1, final_gate, 2.0))
    assert late_min.errors == {}
    late_min_codes = _codes(late_min_results)
    assert late_min_codes[1] == 2
    assert all(code in (0, 2) for code in late_min_codes)
    assert all(item["validated"] is False
               for item in _samples(late_min_results) if item["rank"] == 1)
    late_min_profile = fit_calibrated_fabric_profile(
        plan, _rows(late_min_results))
    assert late_min_profile["status"] == "partial"
    assert late_min_profile["run_evidence_blocked"] is True
    assert late_min_profile["usable_for_link_override"] is False

    late_destroy, late_destroy_results = run_ranks(
        plan, sync_actions=_latency(), destroy_delay={1: 2.0})
    assert late_destroy.errors == {}
    late_destroy_codes = _codes(late_destroy_results)
    assert late_destroy_codes[1] == 2
    assert all(code in (0, 2) for code in late_destroy_codes)
    assert all(item.get("status") == "deadline_exceeded"
               for item in _samples(late_destroy_results) if item["rank"] == 1)
    late_destroy_profile = fit_calibrated_fabric_profile(
        plan, _rows(late_destroy_results))
    assert late_destroy_profile["status"] == "partial"
    assert late_destroy_profile["run_evidence_blocked"] is True
    assert late_destroy_profile["usable_for_link_override"] is False

    failed, failed_results = run_ranks(
        plan, sync_actions=_latency(), fail_destroy={1: True})
    assert failed.errors == {}
    failed_codes = _codes(failed_results)
    assert failed_codes[1] == 2 and all(code in (0, 2)
                                       for code in failed_codes)
    _assert_cleanup_blocked(plan, failed_results)


def test_f7_cleanup_control_survives_direct_and_cli_fit(tmp_path):
    plan = _two_size_plan()
    world, results = run_ranks(
        plan, sync_actions=_latency(), fail_destroy={1: True})
    assert world.errors == {}
    records = _rows(results)
    parsed = parse_fabric_probe_samples(records)
    controls = [item for item in parsed if item.get("_control_record") is True]
    assert any(item.get("kind") == "cleanup_failed" for item in controls)
    profile = fit_calibrated_fabric_profile(plan, parsed)
    assert profile["status"] == "partial"
    assert profile["run_evidence_blocked"] is True
    assert profile["usable_for_link_override"] is False

    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(
        "\n".join(json.dumps(item) for item in records), encoding="utf-8")
    topology_path = tmp_path / "topology.json"
    topology_path.write_text(json.dumps(TOPOLOGY), encoding="utf-8")
    output_path = tmp_path / "profile.json"
    code = fabric_main([
        "--world-size", "2", "--ranks-per-node", "2", "--duration-s", "1",
        "--iterations", "1", "--warmup", "0", "--topology",
        str(topology_path), "--samples", str(samples_path), "--output",
        str(output_path)])
    assert code in (0, 2)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "partial"
    assert payload["run_evidence_blocked"] is True
    assert payload["usable_for_link_override"] is False

    healthy, healthy_results = run_ranks(
        plan, sync_actions=_latency())
    assert _codes(healthy_results) == [0, 0]
    clean = parse_fabric_probe_samples(_rows(healthy_results))
    assert all(item.get("validated") is True for item in clean)
    synthetic = clean + [{"kind": "cleanup_failed", "rank": 1,
                          "cleanup_failed": True, "status": "cleanup_failed"}]
    blocked = fit_calibrated_fabric_profile(plan, synthetic)
    assert blocked["status"] == "partial"
    assert blocked["run_evidence_blocked"] is True
    assert blocked["usable_for_link_override"] is False


# ---------------------------------------------------------------------- F10
def test_f10_crlf_tamper_fails_direct_and_mcp_resolution(monkeypatch):
    root = _fresh_root()
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
    first = repro_mcp("repro_op", {"occurrence": paged_binding(), "iters": 5})
    try:
        assert first["status"] == "ready"
        script = Path(first["artifact_path"])
        original = script.read_bytes()
        assert resolve_repro_artifact(first["artifact_id"])["resolved"] is True
        script.write_bytes(original.replace(b"\n", b"\r\n"))
        direct = resolve_repro_artifact(first["artifact_id"])
        via_tool = repro_mcp("repro_op",
                             {"resolve_artifact_id": first["artifact_id"]})
        assert direct["resolved"] is False
        assert direct["retrievable"] is False
        assert direct["error"] == "artifact_integrity_mismatch"
        assert via_tool["resolved"] is False
        assert via_tool["retrievable"] is False
        assert via_tool["error"] == "artifact_integrity_mismatch"
        script.write_bytes(original)
        assert resolve_repro_artifact(first["artifact_id"])["resolved"] is True
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_f10_alternate_names_cannot_touch_registered_slots(monkeypatch):
    root = _fresh_root()
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
    first = repro_mcp("repro_op", {"occurrence": paged_binding(), "iters": 5})
    try:
        assert first["status"] == "ready"
        script = Path(first["artifact_path"])
        manifest = Path(first["manifest_path"])
        script_bytes = script.read_bytes()
        manifest_bytes = manifest.read_bytes()
        attempts = (
            {"output_path": str(script.parent / "other.py")},
            {"output_path": str(script.parent / "nested" / "other.py")},
            {"output_path": str(manifest)},
            {"output_dir": str(script.parent / "nested")},
            {"output_dir": str(short_root() / "fresh-slot")},
        )
        for attempt in attempts:
            result = generate_single_op_reproducer(
                paged_binding(), iters=7, **attempt)
            assert result["status"] == "delivery_blocked"
            assert result["error"] == (
                "output_dir_conflicts_with_registered_artifact")
        link = root.parent / (root.name + "-link.py")
        link.symlink_to(script)
        try:
            result = generate_single_op_reproducer(
                paged_binding(), iters=7, output_path=link)
            assert result["status"] == "delivery_blocked"
            assert result["error"] == (
                "output_dir_conflicts_with_registered_artifact")
        finally:
            link.unlink()
        assert script.read_bytes() == script_bytes
        assert manifest.read_bytes() == manifest_bytes
        assert not checked_io.path_exists(script.parent / "other.py")
        assert not checked_io.path_exists(script.parent / "nested")
        assert not checked_io.path_exists(short_root() / "fresh-slot")
        assert resolve_repro_artifact(first["artifact_id"])["resolved"] is True
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------- MCP envelope sweep
def test_mcp_envelopes_are_bounded_on_every_route(monkeypatch):
    root = _fresh_root()
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
    document = load_catalog()
    envelopes = []
    for name, entry in document["operators"].items():
        for version in entry.get("cann_versions") or ():
            envelopes.append(registry_mcp(
                "inspect_op_abi",
                {"op_name": name, "cann_version": str(version)}))
        envelopes.append(registry_mcp("inspect_op_abi", {"op_name": name}))
    assert len(envelopes) == 132
    receipts = []
    for binding in _ready_bindings().values():
        receipt = repro_mcp("repro_op", {"occurrence": binding})
        receipts.append(receipt)
        envelopes.append(receipt)
    assert len(receipts) == 13
    assert all(receipt["status"] == "ready" for receipt in receipts)
    for receipt in receipts:
        envelopes.append(repro_mcp(
            "repro_op", {"resolve_artifact_id": receipt["artifact_id"]}))
    handles = ["", " ", "sha256:", "sha256:abc", "sha256:" + "z" * 64,
               "sha256:" + "A" * 64, "sha256:" + "0" * 63,
               "sha256:" + "0" * 65, "sha256:" + "0" * 64 + "\n",
               "sha256:" + "0" * 128, "sha256:" + "界" * 40,
               "sha256:" + "a" * 2000]
    handles += [None, 0, 1.5, True, [], {},
                ["sha256:" + "0" * 64]]
    while len(handles) < 106:
        handles.append("sha256:" + format(len(handles), "064x"))
    for handle in handles[:106]:
        envelopes.append(repro_mcp(
            "repro_op", {"resolve_artifact_id": handle}))
    assert len(envelopes) == 264

    identity = {"total_ms": 2, "ops": [
        {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": 2}]}
    multi = {"total_ms": 6, "ops": OPS,
             "gaps": [{"start_ms": 2, "duration_ms": 4}]}
    padding = {"aclnnGroupedMatmulV4": {
        "bucket_tokens": 64, "bound_args": {"group_list": [16, 32, 48, 64]}}}
    diff_envelopes = [
        diff_mcp({"predicted": identity, "captured": identity}),
        diff_mcp({"predicted": _trace(2), "captured": multi}),
        diff_mcp({"predicted": identity, "captured": identity,
                  "padding": padding, "cann_version": "9.0"}),
        diff_mcp({}),
    ]
    diff_envelopes.append(resolve_diff_artifact(
        diff_envelopes[0]["artifact_id"]))
    diff_envelopes.append(resolve_diff_artifact("sha256:" + "0" * 64))
    diff_envelopes.append(resolve_diff_artifact(""))
    diff_envelopes.append(resolve_diff_artifact(123))
    artifact_path = Path(diff_envelopes[0]["artifact_path"])
    artifact_path.write_bytes(artifact_path.read_bytes() + b"\n")
    diff_envelopes.append(resolve_diff_artifact(
        diff_envelopes[0]["artifact_id"]))
    try:
        assert diff_envelopes[-1]["error"] == "artifact_integrity_mismatch"
        assert len(envelopes) + len(diff_envelopes) == 273
        for envelope in envelopes + diff_envelopes:
            assert estimate_tokens(envelope) <= 250
            assert envelope["receipt"]["within_budget"] is True
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("route", ["cas", "caller"])
@pytest.mark.parametrize("leaf", ["repro.py", "manifest.json"])
@pytest.mark.parametrize("boundary", ["writer", "verifier", "fdopen_failure", "stream_fallback"])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("error_type", [TimeoutError, PermissionError])
def test_r20_public_repro_fd_reuse(route, leaf, boundary, active, error_type, tmp_path, monkeypatch):
    import contextlib
    import io
    import os
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "cas")
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"foreign public descriptor")
    primary, secondary = TimeoutError("public body"), error_type("post close")
    real = {name: getattr(os, name) for name in ("open", "close", "fstat", "read", "fsync", "fdopen")}
    opened, selected, calls, handles = {}, [], [], []
    parent = tmp_path / ("out" if route == "caller" else "cas")
    def matches(path):
        return path.parent == parent if route == "caller" else parent in path.parents
    def opening(path, flags, *args, **kwargs):
        fd = real["open"](path, flags, *args, **kwargs)
        path = Path(path).absolute()
        opened[fd] = path
        wanted = (path.name.startswith("." + leaf + ".") and path.suffix == ".tmp" and flags & os.O_WRONLY
                  if boundary == "writer" else path.name == leaf and not flags & os.O_WRONLY)
        if not selected and matches(path) and wanted and boundary in {"writer", "verifier"}:
            selected.append(fd)
        return fd
    def closing(fd):
        if selected and fd == selected[0]:
            calls.append(fd)
            if len(calls) == 1:
                real["close"](fd)
                assert real["open"](foreign, os.O_RDONLY) == fd
                raise secondary
        return real["close"](fd)
    def body(name, fd, *args):
        if active and selected and fd == selected[0]:
            raise primary
        return real[name](fd, *args)
    class Reader(io.BufferedReader):
        def read(self, *args):
            return body("read", self.fileno(), 1 << 20)
        def close(self):
            raise secondary
    def fdopen(fd, *args, **kwargs):
        path = opened[fd]
        if not selected and boundary in {"fdopen_failure", "stream_fallback"} and matches(path) and path.name == leaf:
            selected.append(fd)
            if boundary == "fdopen_failure":
                raise primary
            stream = Reader(io.FileIO(fd, "rb", closefd=False))
            handles.append(stream)
            return stream
        return real["fdopen"](fd, *args, **kwargs)
    caught = result = None
    with monkeypatch.context() as patcher:
        for name, action in {"open": opening, "close": closing, "fdopen": fdopen,
                             "read": lambda fd, *args: body("read", fd, *args),
                             "fsync": lambda fd: body("fsync", fd)}.items():
            patcher.setattr(os, name, action)
        with contextlib.suppress(BaseException):
            try:
                result = repro_mcp("repro_op", {"occurrence": paged_binding(), "iters": 20020,
                    **({"output_dir": "out"} if route == "caller" else {})})
            finally:
                caught = sys.exc_info()[1]
    assert selected and calls == selected and all(handle.closed for handle in handles)
    if active or boundary == "fdopen_failure":
        assert caught is primary
    elif error_type is TimeoutError:
        assert caught is secondary
    else:
        assert caught is None and result["status"] == "delivery_failed"
    assert real["read"](selected[0], 100) == b"foreign public descriptor"
    real["close"](selected[0])

@pytest.mark.parametrize("field", ["world_size", "ranks_per_node", "nodes", "dies_per_package", "warmup", "iterations", "sizes", "ranks"])
@pytest.mark.parametrize("value", [True, 1.5, "2", -1, 2**63, 10**100, 10**5000], ids=["bool", "float", "str", "negative", "ssize", "finite_huge", "huge"])
def test_r20_planner_cardinalities(field, value):
    from mindie_profiler.events import ProfilerUsageError
    args = {field: [value] if field == "sizes" else value}
    if field == "ranks":
        args = {"topology": {"tiers": [{"tier": "intra_node_hccs", "ranks": [0, value]}]}}
    with pytest.raises(ProfilerUsageError):
        build_fabric_probe_plan(**args)

@pytest.mark.parametrize("field", ["warmup", "iters"])
@pytest.mark.parametrize("value", [True, 1.5, "2", -1, 2**63, 10**100, 10**5000], ids=["bool", "float", "str", "negative", "ssize", "finite_huge", "huge"])
def test_r20_repro_count_api_and_mcp(field, value):
    from mindie_profiler.events import ProfilerUsageError
    with pytest.raises(ProfilerUsageError):
        generate_single_op_reproducer(paged_binding(), stage=False, **{field: value})
    result = repro_mcp("repro_op", {"occurrence": paged_binding(), field: value})
    assert result["status"] == "blocked" and estimate_tokens(result) <= 250

@pytest.mark.parametrize("field", ["world-size", "ranks-per-node", "nodes", "dies-per-package", "warmup", "iterations"])
@pytest.mark.parametrize("value", [2**63, 10**100])
def test_r20_cardinality_cli_usage_exit(field, value, tmp_path):
    result = subprocess.run([sys.executable, "-B", "-m", "mindie_profiler", "fabric-probe",
        "--" + field, str(value), "--duration-s", "1"], cwd=tmp_path,
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr and "OverflowError" not in result.stderr

@pytest.mark.parametrize("field", ["world_size", "group_size", "ranks", "bytes", "warmup", "iterations"])
@pytest.mark.parametrize("value", [True, 2**63, 10**100])
def test_r20_supplied_plan_counts(field, value):
    from mindie_profiler.events import ProfilerUsageError
    plan = build_fabric_probe_plan(world_size=2, sizes=[4096], ops=["all_reduce"], warmup=0, iterations=1)
    target = plan if field == "world_size" else plan["tiers"][0] if field in {"group_size", "ranks"} else plan["admitted"][0]
    target[field] = [0, value] if field == "ranks" else value
    with pytest.raises(ProfilerUsageError):
        emit_fabric_probe_script(plan)

@pytest.mark.parametrize("field,limit", [("world_size", 4096), ("ranks_per_node", 4096),
    ("nodes", 4096), ("dies_per_package", 4096), ("warmup", 1_000_000),
    ("iterations", 1_000_000), ("sizes", 268_435_456)])
def test_r20_documented_planner_limits(field, limit):
    from mindie_profiler.events import ProfilerUsageError
    args = {"sizes": [4096], "ops": ["all_reduce"], field: [limit] if field == "sizes" else limit}
    plan = build_fabric_probe_plan(**args)
    assert plan["sizes_bytes" if field == "sizes" else "max_iterations" if field == "iterations" else field] == ([limit] if field == "sizes" else limit)
    args[field] = [limit + 1] if field == "sizes" else limit + 1
    with pytest.raises(ProfilerUsageError):
        build_fabric_probe_plan(**args)

@pytest.mark.parametrize("field", ["batch_size", "num_seqs", "expert_num", "shared_prefix_pages", "shape", "strides"])
@pytest.mark.parametrize("value", [True, 2**63, 10**100])
def test_r20_metadata_allocation_cardinalities(field, value):
    from mindie_profiler.events import ProfilerUsageError
    from mindie_profiler.repro_metadata import synthesize_metadata
    from mindie_profiler.repro_op import _compact_occurrence
    occurrence = _compact_occurrence(paged_binding())
    if field == "shape":
        occurrence["allocations"][0]["shape"] = [value]
    elif field == "strides":
        occurrence["allocations"][0]["storage"] = {"strides": [value]}
    else:
        occurrence["attrs"][field] = value
    with pytest.raises(ProfilerUsageError):
        synthesize_metadata(occurrence)

@pytest.mark.parametrize("field,value", [("flush_mb", True), ("flush_mb", 4097),
    ("l2_bytes", True), ("l2_bytes", 2**63)])
def test_r20_repro_byte_count_options(field, value):
    from mindie_profiler.events import ProfilerUsageError
    with pytest.raises(ProfilerUsageError):
        generate_single_op_reproducer(paged_binding(), stage=False, **{field: value})

@pytest.mark.parametrize("route", ["probe", "repro"])
def test_r20_emitted_cli_count_bounds(route, tmp_path):
    if route == "probe":
        script = emit_fabric_probe_script(build_fabric_probe_plan(world_size=2))["script"]
        args = ["--max-bytes", str(2**63)]
    else:
        script = generate_single_op_reproducer(paged_binding(), stage=False)["script"]
        args = ["--dry-run", "--iters", str(2**63)]
    target = tmp_path / "standalone.py"
    target.write_text(script)
    result = subprocess.run([sys.executable, "-B", str(target), *args], cwd=tmp_path,
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and "Traceback" not in result.stderr

@pytest.mark.parametrize("field", ["flush_mb", "l2_bytes", "flush_invocations", "swept_bytes", "min_per_sweep_bytes", "per_sweep_bytes"])
@pytest.mark.parametrize("value", [True, 2**63])
def test_r20_standalone_cache_counts(field, value):
    ns = {"__name__": "r20_cache"}
    exec(generate_single_op_reproducer(paged_binding(), stage=False)["script"], ns)
    args = {"flush_mb": 256, "l2_bytes": 1024, field: value}
    with pytest.raises(ns["ProfilerUsageError"]):
        ns["_cache_label"](**args)

def test_r20_unsized_cardinality_rejected_before_iteration():
    from mindie_profiler.events import ProfilerUsageError
    from mindie_profiler.repro_metadata import synthesize_metadata
    from mindie_profiler.repro_values import slot_allocations, validate_captured_values
    huge = range(2**63)
    binding = paged_binding()
    next(spec for spec in binding["inputs"] if spec.get("shape"))["shape"] = huge
    calls = [lambda: build_fabric_probe_plan(sizes=huge),
             lambda: supervise_probe_processes(huge), lambda: slot_allocations(binding),
             lambda: validate_captured_values(huge, "int32", [1]),
             lambda: synthesize_metadata({"allocations": [{"shape": huge}]})]
    calls.append(lambda: synthesize_metadata({"allocations": [
        {"name": str(i), "shape": [2], "storage": {"strides": [268_435_455]}}
        for i in range(2)]}))  # Each span fits; their aggregate must still be rejected.
    for call in calls:
        with pytest.raises(ProfilerUsageError):
            call()
