"""Round-6 closure: the final five partial findings become executable contracts.

[Role]
  Re-run Astra's exact Round-6 counterexamples for F1, F6, F7, F10 and F11:
  terminal/overlapping gap identity, validation abort notification, file-only
  peer wake-up, the outer process supervisor, the late final MIN gate, Unicode
  short-root receipts, immutable content-addressed slots, byte-verified
  resolution, bounded resolution envelopes and required/nullable call plans.

[Invariants]
  * Every assertion executes product code; no source grep substitutes for a
    runtime trace, protocol, deadline, storage or callable check.
  * Probe tests use the real two-thread rendezvous broker without any private
    ``_mindie_*`` hook, plus real subprocesses for the supervisor contract.
  * Storage tests use fresh ASCII roots under ``/tmp`` and clean every artifact.
  * The F11 table is independent of the catalog at assertion time.

[Timeouts & Locks]
  Two bounded broker threads per probe run; real subprocesses are supervised,
  terminated and killed with bounded waits; no NPU, HCCL or device command.

[Side-Effect Policy]
  Writes only pytest temporary files, short reproducer directories under
  ``/tmp`` and the caller-owned abort files used by the supervisor tests.
"""

from __future__ import annotations

import json
from pathlib import Path
__import__("pytest").importorskip("mindie_profiler")
from mindie_profiler import repro_atomic_io as checked_io
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

from mindie_profiler.fabric_probe import (
    emit_fabric_probe_script, supervise_probe_processes,
)
from mindie_profiler.fabric_probe_fit import fit_calibrated_fabric_profile
from mindie_profiler import repro_storage
from mindie_profiler.repro_op import call_mcp_tool
from mindie_profiler.repro_storage import resolve_repro_artifact, short_root
from mindie_profiler.token_budget import estimate_tokens
from mindie_sim.diff_trace import attribute_diff
from mindie_sim.diff_trace_release import release_delays, tail_gap

from nr_s6_round4_probe import BrokerDist, run_ranks
from nr_s6_support import paged_binding
from temp_cleanup import tempdir
from test_nr_s6_round4_closure import _probe_plan
from test_nr_s6_round5_closure import (
    _READY_TYPES, _invoke_all_ready, _ready_invoke_setup,
)


def _diff(predicted, captured):
    return attribute_diff(predicted, captured, launch_overhead_us=0)


def _codes(results):
    return sorted(code for code, _rows in results.values())


def _records(results):
    return [item for _rank, (_code, rows) in results.items() for item in rows]


def _fresh_short_root():
    return tempdir("r6cas-", directory="/tmp")


def _fake_mprof_root() -> Path:
    """Short, host-independent fallback root for receipt-budget tests."""
    parent = Path("/tmp") if Path("/tmp").is_dir() else Path(tempfile.gettempdir())
    return tempdir("mprof-f10-", directory=parent)


# ----------------------------------------------------------------------- F1
def test_f1_terminal_gap_identity_is_zero_in_every_bucket():
    trace = {
        "total_ms": 4,
        "gaps": [{"stream": 0, "start_ms": 2, "duration_ms": 2}],
        "ops": [
            {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": 1},
            {"name": "B", "stream": 0, "start_ms": 1, "duration_ms": 1},
        ],
    }
    result = _diff(trace, trace)
    assert result["stages"] == {"R0": 4.0, "R1": 4.0, "R2": 4.0,
                                "R3": 4.0, "R4_model": 4.0, "O": 4.0}
    assert all(value == 0.0 for value in result["buckets"].values())
    assert result["attribution"]["sched_delta_ms"] == 0.0
    assert result["attribution"]["residual_ms"] == 0.0
    assert result["reconciliation"]["reconciled"] is True
    assert result["status"] == "reconciled"


def test_f1_overlapping_stream_and_streamless_gaps_are_unioned():
    trace = {
        "total_ms": 3,
        "gaps": [{"start_ms": 1, "duration_ms": 1},
                 {"stream": 0, "start_ms": 1, "duration_ms": 1}],
        "ops": [
            {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": 1},
            {"name": "B", "stream": 0, "start_ms": 2, "duration_ms": 1},
        ],
    }
    result = _diff(trace, trace)
    assert result["stages"] == {"R0": 3.0, "R1": 3.0, "R2": 3.0,
                                "R3": 3.0, "R4_model": 3.0, "O": 3.0}
    assert all(value == 0.0 for value in result["buckets"].values())
    assert result["reconciliation"]["reconciled"] is True
    ops = [{"stream": 0, "start": 0, "duration": 1, "rank": 0},
           {"stream": 0, "start": 2, "duration": 1, "rank": 0}]
    gaps = [{"start": 1, "duration": 1},
            {"stream": 0, "start": 1, "duration": 1}]
    assert release_delays(ops, gaps, 0.0, [[], [0]]) == [0.0, 1.0]
    assert tail_gap(gaps, (0, 3), 3) == 0.0
    terminal = [{"start": 2, "duration": 2},
                {"stream": 0, "start": 2, "duration": 2}]
    assert tail_gap(terminal, (0, 4), 2) == 2.0


def test_f1_intermediate_multistream_and_inserted_occurrence_identity():
    intermediate = {
        "total_ms": 3,
        "gaps": [{"stream": 0, "start_ms": 1, "duration_ms": 1}],
        "ops": [
            {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": 1},
            {"name": "B", "stream": 0, "start_ms": 2, "duration_ms": 1},
        ],
    }
    multistream = {
        "total_ms": 4,
        "gaps": [{"stream": 0, "start_ms": 1, "duration_ms": 1},
                 {"stream": 1, "start_ms": 3, "duration_ms": 1}],
        "ops": [
            {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": 1},
            {"name": "B", "stream": 1, "start_ms": 1, "duration_ms": 2},
        ],
    }
    inserted = {
        "total_ms": 4,
        "gaps": [{"stream": 0, "start_ms": 2, "duration_ms": 1}],
        "ops": [
            {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": 1},
            {"name": "X", "stream": 0, "start_ms": 1, "duration_ms": 1,
             "deps": [0]},
            {"name": "B", "stream": 0, "start_ms": 3, "duration_ms": 1,
             "deps": [0]},
        ],
    }
    implicit = {
        "total_ms": 4,
        "ops": [
            {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": 1},
            {"name": "B", "stream": 0, "start_ms": 1, "duration_ms": 1},
        ],
    }
    for trace in (intermediate, multistream, inserted, implicit):
        result = _diff(trace, trace)
        assert all(value == 0.0 for value in result["buckets"].values())
        assert result["reconciliation"]["reconciled"] is True


def test_f1_captured_terminal_gap_replaces_not_adds_to_the_baseline():
    predicted = {
        "total_ms": 4,
        "gaps": [{"stream": 0, "start_ms": 2, "duration_ms": 2}],
        "ops": [
            {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": 1},
            {"name": "B", "stream": 0, "start_ms": 1, "duration_ms": 1},
        ],
    }
    captured = {
        "total_ms": 6,
        "gaps": [{"stream": 0, "start_ms": 2, "duration_ms": 4}],
        "ops": predicted["ops"],
    }
    result = _diff(predicted, captured)
    assert result["stages"]["R0"] == 4.0
    assert result["stages"]["R1"] == 6.0
    assert result["stages"]["R4_model"] == 6.0
    assert result["attribution"]["sched_delta_ms"] == pytest.approx(2.0)
    assert result["attribution"]["residual_ms"] == pytest.approx(0.0)
    assert result["reconciliation"]["reconciled"] is True


# ----------------------------------------------------------------------- F6
def test_f6_validation_data_failure_signals_abort_without_timeout():
    plan = _probe_plan(iterations=1, warmup=0)
    world, results = run_ranks(plan, data_actions={1: {1: "raise"}})
    assert world.errors == {}
    assert world.broker.mismatches == []
    assert world.broker.aborted is True
    assert _codes(results) == [2, 2]
    statuses = {item.get("status") for item in _records(results)
                if item.get("kind") == "failed"}
    assert "collective_failed" in statuses
    assert "peer_aborted" in statuses
    abort = world.abort_record
    assert abort is not None
    assert abort["rank"] == 1
    assert abort["step"] == "validation_data_collective"


def test_f6_file_only_peer_wakeup_without_private_hooks():
    assert "_mindie_signal_abort" not in BrokerDist.__dict__
    assert "_mindie_abort_signaled" not in BrokerDist.__dict__
    plan = _probe_plan(iterations=1, warmup=0)
    assert "_mindie_" not in emit_fabric_probe_script(plan)["script"]
    started = time.monotonic()
    world, results = run_ranks(plan, data_actions={1: {2: "raise"}})
    elapsed = time.monotonic() - started
    assert world.errors == {}
    assert world.broker.mismatches == []
    assert world.broker.aborted is True
    assert _codes(results) == [2, 2]
    assert any(item.get("status") == "peer_aborted"
               for item in _records(results))
    assert elapsed < 1.0


def _spawn(code, *args):
    return subprocess.Popen([sys.executable, "-c", code, *args])


def test_f6_supervisor_terminates_a_blocked_peer_promptly():
    tmp = tempdir("r6sup-")
    abort = Path(tmp, "abort.json")
    writer = (
        "import json, sys;"
        "open(sys.argv[1], 'w').write(json.dumps({'kind': 'status',"
        "'status': 'deadline_exceeded', 'reason': 'deadline_exceeded',"
        "'rank': 1})); sys.exit(2)")
    procs = [_spawn("import time; time.sleep(30)"),
             _spawn(writer, str(abort))]
    try:
        started = time.monotonic()
        result = supervise_probe_processes(procs, abort, wall_budget_s=30,
                                           poll_interval_s=0.01)
        elapsed = time.monotonic() - started
        assert result["status"] == "peer_aborted"
        assert result["exit_code"] == 2
        assert result["exit_codes"][1] == 2
        assert 0 in result["terminated_ranks"]
        assert procs[0].poll() is not None
        assert elapsed < 0.5
        assert result["abort_record"]["reason"] == "deadline_exceeded"
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
        shutil.rmtree(tmp, ignore_errors=True)


def test_f6_supervisor_enforces_the_wall_budget():
    procs = [_spawn("import time; time.sleep(30)"),
             _spawn("import time; time.sleep(30)")]
    try:
        result = supervise_probe_processes(procs, None, wall_budget_s=0.12,
                                           poll_interval_s=0.01)
        assert result["status"] == "deadline_exceeded"
        assert result["exit_code"] == 2
        assert sorted(result["terminated_ranks"]) == [0, 1]
        assert all(proc.poll() is not None for proc in procs)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)


# ----------------------------------------------------------------------- F7
def test_f7_late_final_min_gate_cannot_certify_a_sample():
    plan = _probe_plan(iterations=1, warmup=0)
    plan["budget_s"] = 1.0
    plan["deadline"]["measurement_budget_s"] = 0.9
    plan["deadline"]["max_wall_s"] = 1.0
    plan["deadline"]["cleanup_allowance_s"] = 0.1
    assert plan["status"] == "admitted"
    latency = {0: {4: ("advance", 0.0001)}, 1: {4: ("advance", 0.0001)}}
    baseline, base_results = run_ranks(plan, sync_actions=latency)
    assert _codes(base_results) == [0, 0]
    assert len(baseline.broker.decision_logs[1]) == 8
    world, results = run_ranks(
        plan, sync_actions=latency, decision_delay=(1, 8, 2.0))
    assert world.errors == {}
    assert world.broker.mismatches == []
    code, rows = results[1]
    assert code == 2
    assert any(item.get("kind") == "status"
               and item.get("status") == "deadline_exceeded" for item in rows)
    samples = [item for item in rows
               if item.get("schema") == "mindie-fabric-probe-sample/2"]
    assert samples
    assert all(item["validated"] is False for item in samples)
    assert all(item["disqualified_reason"] == "deadline_exceeded"
               for item in samples)
    abort = world.abort_record
    assert abort is not None
    assert abort["reason"] == "deadline_exceeded"
    profile = fit_calibrated_fabric_profile(plan, _records(results),
                                            abort_record=abort)
    assert profile["status"] == "partial"
    assert profile["usable_for_link_override"] is False
    assert profile["run_evidence_blocked"] is True
    assert any(item["reason"] == "incomplete_rank_set"
               for item in profile["rejected_samples"])


def test_f7_abort_record_alone_rejects_a_calibrated_override():
    plan = _probe_plan(iterations=1, warmup=0)
    plan["budget_s"] = 1.0
    plan["deadline"]["measurement_budget_s"] = 0.9
    plan["deadline"]["max_wall_s"] = 1.0
    world, results = run_ranks(
        plan, sync_actions={0: {4: ("advance", 0.0001)},
                            1: {4: ("advance", 0.0001)}})
    assert world.errors == {}
    assert _codes(results) == [0, 0]
    rows = [item for item in _records(results)
            if item.get("schema") == "mindie-fabric-probe-sample/2"]
    assert rows and all(item["latency_us"] > 0 for item in rows)
    abort = world.abort_record or {
        "kind": "status", "status": "deadline_exceeded",
        "reason": "deadline_exceeded", "rank": 1}
    profile = fit_calibrated_fabric_profile(plan, rows, abort_record=abort)
    assert profile["usable_for_link_override"] is False
    assert profile["run_evidence_blocked"] is True

# ---------------------------------------------------------------------- F10
def test_f10_unicode_short_root_falls_back_to_a_bounded_ascii_receipt(
        monkeypatch):
    fake_default = _fake_mprof_root()
    sentinel = fake_default / "host-sentinel.txt"
    sentinel.write_text("host-owned data")
    monkeypatch.setattr(repro_storage, "SHORT_ROOT_DEFAULT", str(fake_default))
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "/tmp/" + "界" * 40)
    assert short_root() == fake_default
    result = call_mcp_tool("repro_op", {"occurrence": paged_binding()})
    try:
        assert result["status"] == "ready"
        assert estimate_tokens(result) <= 250
        assert result["artifact_id"] == result["repro_sha256"]
        assert result["artifact"]["retrievable"] is True
        script = Path(result["artifact_path"])
        manifest = Path(result["manifest_path"])
        assert checked_io.path_is_file(script) and checked_io.path_is_file(manifest)
        assert script.is_relative_to(fake_default)
        assert sentinel.read_text() == "host-owned data"
        resolved = resolve_repro_artifact(result["artifact_id"])
        assert resolved["resolved"] is True
        assert resolved["script_sha256"] == result["artifact_id"]
        via_tool = call_mcp_tool(
            "repro_op", {"resolve_artifact_id": result["artifact_id"]})
        assert via_tool["resolved"] is True
        assert estimate_tokens(via_tool) <= 250
    finally:
        # The fallback default is monkeypatched to a pytest-owned directory;
        # never delete the host's real reproducer root to clean up here.
        assert sentinel.read_text() == "host-owned data"

def test_f10_registered_slot_is_immutable_and_resolution_verifies_bytes(
        monkeypatch):
    root = _fresh_short_root()
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
    try:
        first = call_mcp_tool("repro_op", {"occurrence": paged_binding(),
                                           "iters": 5})
        assert first["status"] == "ready"
        script = Path(first["artifact_path"])
        before = script.read_bytes()
        second = call_mcp_tool("repro_op", {
            "occurrence": paged_binding(), "iters": 6,
            "output_dir": str(script.parent)})
        assert second["status"] == "delivery_blocked"
        assert second["error"] == (
            "output_dir_conflicts_with_registered_artifact")
        assert script.read_bytes() == before
        assert resolve_repro_artifact(first["artifact_id"])["resolved"] is True
        script.write_text(script.read_text() + "\n# tampered\n")
        tampered = resolve_repro_artifact(first["artifact_id"])
        assert tampered["resolved"] is False
        assert tampered["retrievable"] is False
        assert tampered["error"] == "artifact_integrity_mismatch"
        bogus = call_mcp_tool(
            "repro_op", {"resolve_artifact_id": "sha256:" + "z" * 2000})
        assert estimate_tokens(bogus) <= 250
        assert bogus["status"] == "invalid_request"
        assert bogus["resolved"] is False
        assert "zzzz" not in json.dumps(bogus)
    finally:
        shutil.rmtree(root, ignore_errors=True)

def test_f10_ascii_short_root_is_byte_bounded(monkeypatch, tmp_path):
    fallback = tmp_path / "fallback-mprof"
    monkeypatch.setattr(repro_storage, "SHORT_ROOT_DEFAULT", str(fallback))
    accepted = "/tmp/" + "a" * 43
    assert len(accepted) == 48
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", accepted)
    assert short_root() == Path(accepted)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", accepted + "a")
    assert short_root() == fallback

# ---------------------------------------------------------------------- F11
def test_f11_generated_fa_plan_missing_required_keyword_raises():
    setup = _ready_invoke_setup("aclnnFlashAttentionVarLenScore")
    namespace, plan, inputs, outputs, lists, attrs, metadata = setup
    namespace["_invoke"](plan, inputs, outputs, lists, attrs, metadata)
    mutated = json.loads(json.dumps(plan))
    for name in ("head_num", "input_layout"):
        assert name in mutated["keyword"]
        mutated["keyword"].pop(name)
    with pytest.raises(TypeError):
        namespace["_invoke"](mutated, inputs, outputs, lists, attrs, metadata)

def test_f11_all_thirteen_unmodified_plans_still_execute():
    invoked = _invoke_all_ready()
    assert set(invoked) == set(_READY_TYPES)
    assert len(invoked) == 13

# R19: real descriptors, finite close faults, and descriptor-number reuse.
@pytest.mark.parametrize("error_type", [OSError, TimeoutError, PermissionError])
@pytest.mark.parametrize("mode", ["pre", "post", "persistent", "reused"])
@pytest.mark.parametrize("read_primary", [False, True])
def test_r19_reader_close_ownership(error_type, mode, read_primary, monkeypatch, tmp_path):
    import contextlib
    import errno
    import io
    import os
    target, foreign = tmp_path / "reader", tmp_path / "foreign"
    target.write_bytes(b"reader bytes")
    foreign.write_bytes(b"foreign bytes")
    primary, secondary = TimeoutError("read primary"), error_type("close fault")
    original_open, original_close = os.open, os.close
    handles, reused, raw_closes = [], [], []
    class Reader(io.BufferedReader):
        close_calls = 0
        def read(self, *args):
            if read_primary:
                raise primary
            return super().read(*args)
        def close(self):
            self.close_calls += 1
            if mode == "persistent" or (mode == "pre" and self.close_calls == 1):
                raise secondary
            super().close()
            if self.close_calls == 1:
                if mode == "reused":
                    fd = original_open(foreign, os.O_RDONLY)
                    if fd != handles[0][1]:
                        os.dup2(fd, handles[0][1])
                        original_close(fd)
                    reused.append(handles[0][1])
                raise secondary
    def fdopen(fd, *args, **kwargs):
        handle = Reader(io.FileIO(fd, "rb", closefd=False))
        handles.append((handle, fd))
        return handle
    def close(fd):
        raw_closes.append(fd)
        return original_close(fd)
    with monkeypatch.context() as patcher:
        patcher.setattr(os, "fdopen", fdopen)
        patcher.setattr(os, "close", close)
        if read_primary or error_type is TimeoutError:
            with pytest.raises(TimeoutError) as caught:
                checked_io.read_regular_bytes(target)
            assert caught.value is (primary if read_primary else secondary)
        else:
            assert checked_io.read_regular_bytes(target) is None
    assert len(handles) == 1
    handle, fd = handles[0]
    assert handle.close_calls == (4 if mode == "persistent" else 2 if mode == "pre" else 1)
    if mode == "reused":
        assert reused == [fd] and fd not in raw_closes
        info = os.fstat(fd)
        assert (info.st_dev, info.st_ino) == checked_io.inode_of(foreign)
        assert os.read(fd, 100) == b"foreign bytes"
        original_close(fd)
    else:
        with pytest.raises(OSError) as closed:
            os.fstat(fd)
        assert closed.value.errno == errno.EBADF
        assert (fd in raw_closes) is (mode == "persistent")
    with contextlib.suppress(OSError):
        io.BufferedReader.close(handle)

@pytest.mark.parametrize("writer", ["abort", "atomic"])
@pytest.mark.parametrize("error_type", [TimeoutError, PermissionError])
@pytest.mark.parametrize("faults", [1, 2, 3])
def test_r19_writer_recovers_fstat_before_close(writer, error_type, faults, monkeypatch, tmp_path):
    import builtins
    import errno
    import os
    target, fds, handles, hits = tmp_path / "record", {}, [], []
    fault = error_type("initial fstat")
    opened, fstat, fileopen = os.open, os.fstat, builtins.open
    def open_fd(path, flags, *args, **kwargs):
        fd = opened(path, flags, *args, **kwargs)
        if str(path).endswith(".tmp") and flags & os.O_WRONLY:
            fds[fd] = Path(path)
        return fd
    def stat_fd(fd):
        if fd in fds and len(hits) < faults:
            hits.append(fd)
            raise fault
        return fstat(fd)
    def open_file(*args, **kwargs):
        handle = fileopen(*args, **kwargs)
        handles.append(handle)
        return handle
    namespace = {"__name__": "r19_probe", "ABORT_FILE": None}
    exec(compile(emit_fabric_probe_script(_probe_plan())["script"], "<r19>", "exec"), namespace)
    namespace["ABORT_FILE"] = str(target)
    def invoke():
        if writer == "abort":
            return namespace["_write_abort_record"]({"kind": "abort"})
        return checked_io.safe_write_bytes(target, b"payload")
    with monkeypatch.context() as patcher:
        patcher.setattr(os, "open", open_fd)
        patcher.setattr(os, "fstat", stat_fd)
        patcher.setattr(builtins, "open", open_file)
        if error_type is TimeoutError:
            with pytest.raises(TimeoutError) as caught:
                invoke()
            assert caught.value is fault
        else:
            assert invoke() is (None if writer == "abort" else False)
    assert len(hits) == faults and fds and not list(tmp_path.iterdir())
    assert all(handle.closed for handle in handles)
    for fd in fds:
        with pytest.raises(OSError) as closed:
            fstat(fd)
        assert closed.value.errno == errno.EBADF

@pytest.mark.parametrize("replacement", ["regular", "symlink", "fifo"])
@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("contended", [False, True])
def test_r19_abort_publication_preserves_foreign(replacement, prior, contended, monkeypatch, tmp_path):
    import builtins
    import io
    import os
    target, foreign, payload = (tmp_path / name for name in ("abort", "foreign", "payload"))
    payload.write_bytes(b"foreign")
    if replacement == "regular":
        foreign.write_bytes(b"foreign")
    elif replacement == "symlink":
        foreign.symlink_to(payload)
    else:
        os.mkfifo(foreign, 0o600)
    identity = foreign.lstat()
    if prior:
        target.write_bytes(b"old abort")
    before = checked_io.inode_of(target)
    namespace = {"__name__": "r19_probe"}
    exec(compile(emit_fabric_probe_script(_probe_plan())["script"], "<r19>", "exec"), namespace)
    namespace["ABORT_FILE"] = str(target)
    fileopen, replace, link = builtins.open, os.replace, os.link
    handles, staging = [], []
    class Writer(io.TextIOWrapper):
        def close(self):
            path = Path(self.name)
            super().close()
            replace(foreign, path)
            staging.append(path)
    def open_file(path, *args, **kwargs):
        handle = Writer(fileopen(path, "xb", buffering=0, opener=kwargs["opener"]), encoding="utf-8")
        handles.append(handle)
        return handle
    def restore(src, dst, *args, **kwargs):
        if contended and str(src).endswith(".publish"):
            Path(dst).write_bytes(b"new occupant")
        return link(src, dst, *args, **kwargs)
    with monkeypatch.context() as patcher:
        patcher.setattr(builtins, "open", open_file)
        patcher.setattr(os, "link", restore)
        namespace["_write_abort_record"]({"kind": "abort", "rank": 0})
    assert len(handles) == len(staging) == 1 and handles[0].closed
    assert checked_io.inode_of(target) == before
    assert target.read_bytes() == b"old abort" if prior else not checked_io.path_lexists(target)
    preserved = [p for p in tmp_path.iterdir() if (p.lstat().st_dev, p.lstat().st_ino)
                 == (identity.st_dev, identity.st_ino)]
    assert len(preserved) == 1 and payload.read_bytes() == b"foreign"
    assert preserved[0].suffix == (".publish" if contended else ".tmp")
    if contended:
        assert staging[0].read_bytes() == b"new occupant"

@pytest.mark.parametrize("outer_active", [False, True])
@pytest.mark.parametrize("post_kernel", [False, True])
@pytest.mark.parametrize("error_type", [TimeoutError, PermissionError])
def test_r19_abort_publish_fault_retires_owned_name(outer_active, post_kernel, error_type, monkeypatch, tmp_path):
    import os
    target = tmp_path / "abort"
    namespace = {"__name__": "r19_probe"}
    exec(compile(emit_fabric_probe_script(_probe_plan())["script"], "<r19>", "exec"), namespace)
    namespace["ABORT_FILE"] = str(target)
    replace, hits = os.replace, []
    primary, secondary = ValueError("outer"), error_type("publication")
    def faulty(src, dst):
        hits.append(src)
        assert str(src).endswith(".publish")
        if post_kernel:
            replace(src, dst)
        raise secondary
    with monkeypatch.context() as patcher:
        patcher.setattr(os, "replace", faulty)
        if outer_active:
            with pytest.raises(ValueError) as caught:
                try:
                    raise primary
                finally:
                    namespace["_write_abort_record"]({"kind": "abort"}, primary)
            assert caught.value is primary
        elif error_type is TimeoutError:
            with pytest.raises(TimeoutError) as caught:
                namespace["_write_abort_record"]({"kind": "abort"})
            assert caught.value is secondary
        else:
            namespace["_write_abort_record"]({"kind": "abort"})
    assert len(hits) == 1 and list(tmp_path.iterdir()) == ([target] if post_kernel else [])

@pytest.mark.parametrize("exclusive", [False, True])
@pytest.mark.parametrize("replacement", ["regular", "symlink", "fifo"])
@pytest.mark.parametrize("boundary", ["close", "verified"])
def test_r19_atomic_publication_receipt(exclusive, replacement, boundary, monkeypatch, tmp_path):
    import os
    target, foreign, payload = (tmp_path / name for name in ("target", "foreign", "payload"))
    payload.write_bytes(b"foreign")
    if replacement == "regular":
        foreign.write_bytes(b"foreign")
    elif replacement == "symlink":
        foreign.symlink_to(payload)
    else:
        os.mkfifo(foreign, 0o600)
    inode = foreign.lstat().st_ino
    if not exclusive:
        target.write_bytes(b"prior")
    previous = checked_io.inode_of(target)
    original, close, opened = checked_io._verified_inode, os.close, os.open
    hits, writers, receipts = [], {}, []
    def swap(path):
        os.replace(foreign, path)
        hits.append(path)
    def verify(path, *args):
        result = original(path, *args)
        if boundary == "verified" and not hits and str(path).endswith(".tmp"):
            assert result
            swap(path)
        return result
    def open_fd(path, flags, *args, **kwargs):
        fd = opened(path, flags, *args, **kwargs)
        if flags & os.O_WRONLY:
            writers[fd] = path
        return fd
    def close_fd(fd):
        close(fd)
        if boundary == "close" and fd in writers and not hits:
            swap(writers[fd])
    with monkeypatch.context() as patcher:
        patcher.setattr(checked_io, "_verified_inode", verify)
        patcher.setattr(os, "open", open_fd)
        patcher.setattr(os, "close", close_fd)
        assert not checked_io.safe_write_bytes(target, b"new", exclusive=exclusive, installed_out=receipts)
    assert len(hits) == 1 and not receipts and checked_io.inode_of(target) == previous
    assert target.read_bytes() == b"prior" if not exclusive else not checked_io.path_lexists(target)
    preserved = [p for p in tmp_path.iterdir() if p.lstat().st_ino == inode]
    assert len(preserved) == 1 and preserved[0] == Path(hits[0])
    assert payload.read_bytes() == b"foreign"

@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("field", ["slot", "name", "code", "nested"])
@pytest.mark.parametrize("status", ["bound", "blocked"])
def test_r22_diagnostic_and_raw_descriptor_boundaries(sign, field, status):
    from mindie_profiler.events import ProfilerUsageError
    from mindie_profiler.op_api_registry import bind_captured_occurrence
    from mindie_profiler.repro_op import generate_single_op_reproducer
    from nr_s6_support import paged_binding, paged_row
    assert sys.get_int_max_str_digits() == 4300
    huge = sign * 10**5000
    binding = paged_binding()
    binding.update(status=status, issues=[{"code": "descriptor_invalid", field: {"x": [huge]} if field == "nested" else huge}])
    with pytest.raises(ProfilerUsageError):
        generate_single_op_reproducer(binding)
    envelope = call_mcp_tool("repro_op", {"occurrence": binding})
    assert envelope["status"] == "blocked" and estimate_tokens(envelope) <= 250
    json.dumps(envelope, allow_nan=False)
    for side in ("input", "output"):
        for shape in (huge, [True, huge], [[huge]], {"nested": [huge]}):
            row = paged_row()
            row[side + "_shapes"][0] = {"shape": shape, "dtype": "BF16", "layout": "ND"}
            with pytest.raises(ProfilerUsageError):
                bind_captured_occurrence(row)

@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("command,flag", [("repro-op", "--row"), ("repro-op", "--capture"), ("fabric-probe", "--topology"), ("fabric-probe", "--samples")])
def test_r22_cli_json_decode_boundary(sign, command, flag, tmp_path):
    from nr_s6_support import paged_binding
    value = paged_binding()
    value["attrs"]["k"] = "HUGE"
    path = tmp_path / "input.json"
    for text in (json.dumps(value).replace('"HUGE"', ("-" if sign < 0 else "") + "1" + "0" * 5000), "{malformed"):
        path.write_text(text)
        result = subprocess.run([sys.executable, "-B", "-m", "mindie_profiler", command, flag, str(path)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 2 and "Traceback" not in result.stderr
        assert len(result.stderr.encode()) < 1200
