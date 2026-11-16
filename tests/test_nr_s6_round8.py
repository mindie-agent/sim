"""Round-8 controls and Round-16 pair ownership / emitted timeout closure.

[Role]
  Re-run Astra's three remaining Round-8 counterexamples at their public
  boundaries: measurement-annotated and wrapper/standalone failure controls
  through parser, fitter and the real CLI; caller symlink destinations and old
  registered-slot subtrees through the reproducer storage surface; and a real
  filesystem failure during diff-artifact persistence through the MCP tool.

[Invariants]
  * Every assertion executes product code; no source grep substitutes for a
    parser, fitter, filesystem or token-budget observation.
  * Probe rows are the same schema the emitted script writes; the plan is
    planner-built with an explicit two-rank HCCS tier and is not patched.
  * Storage tests use fresh short ASCII roots under ``/tmp`` and restore the
    environment; they verify exact bytes and resolution after every attempt.
  * The persistence failure is a real ``OSError`` from an existing regular file
    used as ``MINDIE_REPRO_SHORT_ROOT``, not a simulated exception.

[Timeouts & Locks]
  In-process faults and bounded thread interleavings; no NPU or network work.

[Side-Effect Policy]
  Writes only fresh ``/tmp`` short roots, pytest temporary mirrors and a
  single regular blocker file used to force the persistence error.
"""

from __future__ import annotations

import ast
import contextlib
import errno
import json
import os
import sys
import threading
import types
from unittest.mock import patch
__import__("pytest").importorskip("mindie_profiler")
from mindie_profiler import repro_storage as storage
from mindie_profiler.fabric_probe import emit_fabric_probe_script
from pathlib import Path
from mindie_profiler import repro_atomic_io as checked_io
import shutil

import pytest

from mindie_profiler.fabric_probe import (
    build_fabric_probe_plan, main as fabric_main,
)
from mindie_profiler.fabric_probe_fit import (
    _sample_qualified, fit_calibrated_fabric_profile, parse_fabric_probe_samples,
)
from mindie_profiler.repro_op import (
    call_mcp_tool as repro_mcp, generate_single_op_reproducer,
)
from mindie_profiler.repro_storage import resolve_repro_artifact
from mindie_profiler.token_budget import estimate_tokens
from mindie_sim.diff_trace import (
    call_mcp_tool as diff_mcp, resolve_diff_artifact,
)

from nr_s6_support import paged_binding
from temp_cleanup import tempdir


TOPOLOGY = {"tiers": [{"tier": "intra_node_hccs", "ranks": [0, 1],
                       "protocol": "HCCS"}]}


def _plan():
    return build_fabric_probe_plan(
        world_size=2, ranks_per_node=2, sizes=[4096, 8192],
        ops=["all_reduce"], iterations=1, warmup=0, duration_s=1.0,
        topology=TOPOLOGY)


def _healthy_rows():
    rows = []
    for size, latency in ((4096, 100.0), (8192, 200.0)):
        for rank in (0, 1):
            rows.append({
                "schema": "mindie-fabric-probe-sample/2",
                "tier": "intra_node_hccs", "op": "all_reduce",
                "bytes": size, "latency_us": latency, "rank": rank,
                "world_size": 2, "group_size": 2, "group_ranks": [0, 1],
                "protocol": "HCCS", "link": "hccs", "regime": "isolated",
                "direction": "collective", "validated": True,
                "clock_mhz": 1000,
            })
    return rows


def _control():
    return {"kind": "cleanup_failed", "rank": 1,
            "cleanup_failed": True, "status": "cleanup_failed"}


def _assert_blocked(profile):
    assert profile["status"] == "partial"
    assert profile["run_evidence_blocked"] is True
    assert profile["usable_for_link_override"] is False
    assert profile["control_record_count"] >= 1


def _fresh_root(prefix="nr_s6_r8_"):
    return tempdir(prefix, directory="/tmp")


def _cli_profile(tmp_path, samples_text):
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(samples_text, encoding="utf-8")
    topology_path = tmp_path / "topology.json"
    topology_path.write_text(json.dumps(TOPOLOGY), encoding="utf-8")
    output_path = tmp_path / "profile.json"
    code = fabric_main([
        "--world-size", "2", "--ranks-per-node", "2", "--duration-s", "1",
        "--iterations", "1", "--warmup", "0", "--topology",
        str(topology_path), "--samples", str(samples_path), "--output",
        str(output_path)])
    assert code in (0, 2)
    return json.loads(output_path.read_text(encoding="utf-8"))


# ------------------------------------------------------------- R8-F7 controls
@pytest.mark.parametrize("validated", [True, False])
def test_f7_annotated_control_dominates_zero_one_two_parser_passes(validated):
    plan, healthy = _plan(), _healthy_rows()
    annotated = {**healthy[-1], **_control(), "validated": validated}
    records = healthy + [annotated]

    assert _sample_qualified(
        {**annotated, "_parsed_fabric_sample": True},
        plan["tiers"][0]) == (False, "terminal_failure")

    direct = fit_calibrated_fabric_profile(plan, records)
    _assert_blocked(direct)
    assert direct["sample_count"] == 4

    once = parse_fabric_probe_samples(records)
    assert any(item.get("_control_record") is True for item in once)
    first = fit_calibrated_fabric_profile(plan, once)
    _assert_blocked(first)

    twice = parse_fabric_probe_samples(once)
    assert any(item.get("_control_record") is True for item in twice)
    second = fit_calibrated_fabric_profile(plan, twice)
    _assert_blocked(second)
    assert second["control_record_count"] == first["control_record_count"]


def test_f7_wrapper_control_survives_extraction_and_real_cli(tmp_path):
    plan, healthy = _plan(), _healthy_rows()
    payload = json.dumps({**_control(), "samples": healthy})

    parsed = parse_fabric_probe_samples(payload)
    assert len([item for item in parsed
                if item.get("_control_record") is True]) == 1
    assert any(item.get("_wrapper_control") is True for item in parsed)
    _assert_blocked(fit_calibrated_fabric_profile(plan, payload))
    _assert_blocked(fit_calibrated_fabric_profile(plan, parsed))
    _assert_blocked(fit_calibrated_fabric_profile(
        plan, parse_fabric_probe_samples(parsed)))

    profile = _cli_profile(tmp_path, payload)
    _assert_blocked(profile)
    assert profile["sample_count"] == 4


def test_f7_standalone_and_control_only_jsonl_are_partial_profiles(tmp_path):
    plan = _plan()
    standalone = json.dumps(_control())
    parsed = parse_fabric_probe_samples(standalone)
    assert len(parsed) == 1
    assert parsed[0]["_control_record"] is True

    direct = fit_calibrated_fabric_profile(plan, standalone)
    _assert_blocked(direct)
    assert direct["sample_count"] == 0
    _assert_blocked(fit_calibrated_fabric_profile(plan, _control()))
    _assert_blocked(fit_calibrated_fabric_profile(
        plan, parse_fabric_probe_samples(parsed)))

    control_only = "\n".join(json.dumps(_control()) for _ in range(2))
    jsonl = parse_fabric_probe_samples(control_only)
    assert len(jsonl) == 2
    _assert_blocked(fit_calibrated_fabric_profile(plan, jsonl))

    profile = _cli_profile(tmp_path, standalone)
    _assert_blocked(profile)
    assert profile["sample_count"] == 0


# ---------------------------------------------------------- R8-F10 leaf writes
def test_f10_symlinked_script_and_manifest_leaves_cannot_touch_cas(
        monkeypatch, tmp_path):
    root = _fresh_root()
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
    try:
        first = repro_mcp("repro_op", {"occurrence": paged_binding(),
                                       "iters": 5})
        assert first["status"] == "ready"
        script = Path(first["artifact_path"])
        manifest = Path(first["manifest_path"])
        script_bytes = script.read_bytes()
        manifest_bytes = manifest.read_bytes()

        mirror = tmp_path / "script-mirror"
        mirror.mkdir()
        (mirror / "repro.py").symlink_to(script)
        second = generate_single_op_reproducer(
            paged_binding(), iters=7, output_dir=mirror)
        assert second["status"] == "delivery_blocked"
        assert second["storage"]["status"] == "delivery_blocked"
        assert second["error"] == (
            "output_dir_conflicts_with_registered_artifact")
        assert not checked_io.path_exists(mirror / "manifest.json")

        manifest_mirror = tmp_path / "manifest-mirror"
        manifest_mirror.mkdir()
        (manifest_mirror / "manifest.json").symlink_to(manifest)
        third = generate_single_op_reproducer(
            paged_binding(), iters=9, output_dir=manifest_mirror)
        assert third["status"] == "delivery_blocked"
        assert not checked_io.path_exists(manifest_mirror / "repro.py")

        assert script.read_bytes() == script_bytes
        assert manifest.read_bytes() == manifest_bytes
        assert resolve_repro_artifact(first["artifact_id"])["resolved"] is True
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_f10_old_registered_slot_subtree_is_closed_after_root_change(
        monkeypatch):
    old_root, new_root = _fresh_root(), _fresh_root()
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(old_root))
    try:
        first = repro_mcp("repro_op", {"occurrence": paged_binding(),
                                       "iters": 5})
        assert first["status"] == "ready"
        script = Path(first["artifact_path"])
        manifest = Path(first["manifest_path"])
        slot = script.parent
        script_bytes, manifest_bytes = script.read_bytes(), manifest.read_bytes()

        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(new_root))
        refreshed = repro_mcp("repro_op", {"occurrence": paged_binding(),
                                           "iters": 5})
        assert refreshed["status"] == "ready"
        assert refreshed["artifact_id"] == first["artifact_id"]
        assert Path(refreshed["artifact_path"]).parent != slot
        nested_path = slot / "nested" / "other.py"
        via_path = generate_single_op_reproducer(
            paged_binding(), iters=7, output_path=nested_path)
        assert via_path["status"] == "delivery_blocked"
        assert via_path["error"] == (
            "output_dir_conflicts_with_registered_artifact")
        assert not checked_io.path_exists(slot / "nested")

        via_dir = generate_single_op_reproducer(
            paged_binding(), iters=7, output_dir=slot / "nested")
        assert via_dir["status"] == "delivery_blocked"
        assert not checked_io.path_exists(slot / "nested")

        assert script.read_bytes() == script_bytes
        assert manifest.read_bytes() == manifest_bytes
        assert resolve_repro_artifact(first["artifact_id"])["resolved"] is True
    finally:
        shutil.rmtree(old_root, ignore_errors=True)
        shutil.rmtree(new_root, ignore_errors=True)


# ------------------------------------------------------------ R8-MCP delivery
def _identity():
    return {"total_ms": 2, "ops": [
        {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": 2}]}


def test_mcp_persistence_failure_is_explicit_bounded_and_unretrievable(
        monkeypatch):
    root = _fresh_root()
    blocker = _fresh_root("nr_s6_r8_b_") / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    identity = _identity()
    try:
        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
        normal = diff_mcp({"predicted": identity, "captured": identity})
        assert normal["status"] == "reconciled"
        assert normal["reconciled"] is True
        assert normal["retrievable"] is True
        assert normal["persisted"] is True
        assert checked_io.path_is_file(Path(normal["artifact_path"]))
        assert resolve_diff_artifact(normal["artifact_id"])["resolved"] is True
        artifact = Path(normal["artifact_path"])
        original = artifact.read_bytes()
        artifact.write_bytes(original + b"\n")
        tampered = resolve_diff_artifact(normal["artifact_id"])
        assert tampered["resolved"] is False
        assert tampered["retrievable"] is False
        assert tampered["error"] == "artifact_integrity_mismatch"
        artifact.write_bytes(original)
        assert resolve_diff_artifact(normal["artifact_id"])["resolved"] is True

        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(blocker))
        failed = diff_mcp({"predicted": identity, "captured": identity})
        assert failed["status"] == "delivery_failed"
        assert failed["reconciled"] is True
        assert failed["retrievable"] is False
        assert failed["persisted"] is False
        assert failed["error"] == "artifact_persistence_failed"
        assert "artifact_persistence_failed" in failed["issues"]
        assert failed["artifact_path"] is None
        assert estimate_tokens(failed) <= 250
        assert failed["receipt"]["within_budget"] is True
        missing = resolve_diff_artifact(failed["artifact_id"])
        assert missing["resolved"] is False
        assert missing["retrievable"] is False
        assert missing["error"] == "artifact_not_found"

        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
        assert resolve_diff_artifact(normal["artifact_id"])["resolved"] is True
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(blocker.parent, ignore_errors=True)


# Round 16: public ownership boundaries, not mocked writers or protection policy.
def _r16_request(monkeypatch, tmp_path, **extra):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "cas")
    return {"occurrence": paged_binding(), "iters": 16001, **extra}

def _r16_fds():
    original_open, original_fstat, fds = os.open, os.fstat, {}
    def opened(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        st = original_fstat(fd)
        fds[fd] = (Path(path), flags, (st.st_dev, st.st_ino))
        return fd
    return opened, fds

def _r16_no_leaks(fds, root):
    for fd in fds:
        with pytest.raises(OSError) as closed:
            os.fstat(fd)
        assert closed.value.errno == errno.EBADF
    assert not list(root.rglob("*.tmp"))

def test_r16_unknown_inode_never_authorizes_unlink(tmp_path):
    target = tmp_path / "foreign"
    target.write_bytes(b"foreign")
    before = checked_io.inode_of(target)
    checked_io._unlink_owned_path(target)
    assert checked_io.inode_of(target) == before and target.read_bytes() == b"foreign"

@pytest.mark.parametrize("leaf", ["repro.py", "manifest.json"])
def test_r16_public_foreign_destination_read_replacement(leaf, monkeypatch, tmp_path):
    request = _r16_request(monkeypatch, tmp_path)
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"owned by another caller")
    inode, primary, hits = checked_io.inode_of(foreign), TimeoutError(leaf), []
    opened, fds = _r16_fds()
    original_read = os.read
    def read(fd, *args):
        path = fds[fd][0]
        if path.name == leaf and not hits:
            os.replace(foreign, path)
            hits.append(path)
            raise primary
        return original_read(fd, *args)
    with patch.object(os, "open", opened), patch.object(os, "read", read):
        with pytest.raises(TimeoutError) as caught:
            repro_mcp("repro_op", request)
    assert caught.value is primary and len(hits) == 1
    assert checked_io.inode_of(hits[0]) == inode
    assert hits[0].read_bytes() == b"owned by another caller"
    assert repro_mcp("repro_op", request)["status"] != "ready"
    assert checked_io.inode_of(hits[0]) == inode
    _r16_no_leaks(fds, tmp_path)

@pytest.mark.parametrize("mode", ["thread", "reentrant"])
@pytest.mark.parametrize("fault", [False, True])
@pytest.mark.parametrize("schedule", ["open", "link"])
def test_r16_b_ready_survives_a_convergence_or_timeout(mode, fault, schedule, monkeypatch, tmp_path):
    request = _r16_request(monkeypatch, tmp_path)
    paused, released = threading.Event(), threading.Event()
    primary, outcome, before, hits = TimeoutError("caller A"), {}, {}, []
    tracked_open, fds = _r16_fds()
    original_read, original_link = os.read, os.link
    caller_a = threading.current_thread()
    def caller_b():
        result = repro_mcp("repro_op", request)
        assert result["status"] == "ready"
        outcome["b"] = result
        for key in ("artifact_path", "manifest_path"):
            path = Path(result[key])
            before[path] = (checked_io.inode_of(path), path.read_bytes())
        assert resolve_repro_artifact(result["artifact_id"])["retrievable"] is True
    def interleave():
        paused.set()
        if mode == "reentrant":
            caller_b()
        else:
            assert released.wait(10)
    def opened(path, flags, *args, **kwargs):
        if (schedule == "open" and threading.current_thread() is caller_a and flags & os.O_WRONLY
                and Path(path).name.startswith(".repro.py.") and not paused.is_set()):
            interleave()
        return tracked_open(path, flags, *args, **kwargs)
    def linked(src, dst, *args, **kwargs):
        if (schedule == "link" and threading.current_thread() is caller_a
                and Path(dst).name == "repro.py" and not paused.is_set()):
            interleave()
            if fault:
                hits.append(src)
                raise primary
        return original_link(src, dst, *args, **kwargs)
    def read(fd, *args):
        if (fault and threading.current_thread() is caller_a and "b" in outcome
                and fds[fd][0].name.startswith(".repro.py.") and not hits):
            hits.append(fd)
            raise primary
        return original_read(fd, *args)
    def invoke_a():
        try:
            outcome["a"] = repro_mcp("repro_op", request)
        except TimeoutError as exc:
            outcome["exception"] = exc
    with patch.object(os, "open", opened), patch.object(os, "read", read), \
            patch.object(os, "link", linked):
        if mode == "thread":
            caller_a = threading.Thread(target=invoke_a)
            caller_a.start()
            try:
                assert paused.wait(10)
                caller_b()
            finally:
                released.set()
                caller_a.join(10)
            assert not caller_a.is_alive()
        else:
            invoke_a()
    if fault:
        assert hits and outcome["exception"] is primary
    else:
        assert outcome["a"]["status"] == "ready"
        assert outcome["a"]["artifact_id"] == outcome["b"]["artifact_id"]
    assert {p: (checked_io.inode_of(p), p.read_bytes()) for p in before} == before
    resolved = resolve_repro_artifact(outcome["b"]["artifact_id"])
    assert resolved["resolved"] is True and resolved["retrievable"] is True
    _r16_no_leaks(fds, tmp_path)

@pytest.mark.parametrize("leaf", ["repro.py", "manifest.json", "index.json"])
def test_r16_nonexclusive_install_refreshes_live_protection(leaf, monkeypatch, tmp_path):
    request = _r16_request(monkeypatch, tmp_path, output_dir="mirror")
    target = Path("mirror") / leaf
    opened, hits, other = os.open, [], {}
    def open_hook(path, flags, *args, **kwargs):
        if (flags & os.O_WRONLY and Path(path).parent == target.absolute().parent
                and Path(path).name.startswith("." + leaf + ".") and not hits):
            hits.append(path)
            other.update(repro_mcp("repro_op", {"occurrence": paged_binding(), "iters": 16002}))
            assert other["status"] == "ready"
            os.link(other["artifact_path"], target)
        return opened(path, flags, *args, **kwargs)
    with patch.object(os, "open", open_hook):
        if leaf == "index.json":
            assert storage._atomic_write_bytes(target, b"index", snapshot=(set(), set())) is False
        else:
            assert repro_mcp("repro_op", request)["status"] == "delivery_failed"
    assert hits and target.read_bytes() == Path(other["artifact_path"]).read_bytes()
    assert checked_io.inode_of(target) == checked_io.inode_of(other["artifact_path"])
    assert resolve_repro_artifact(other["artifact_id"])["retrievable"] is True
    assert not list(tmp_path.rglob("*.tmp"))

@pytest.mark.parametrize("boundary,keep", [
    ("pre", None), ("read", None), ("pair_false", None), ("pair_timeout", None),
    ("pre", "prior_manifest"), ("pre", "foreign_script"),
    ("read", "foreign_manifest"), ("pair_false", "foreign_script"),
    ("pair_timeout", "foreign_manifest"),
])
@pytest.mark.parametrize("cleanup_fault", [False, True])
def test_r16_caller_pair_rollback(boundary, keep, cleanup_fault, monkeypatch, tmp_path):
    request = _r16_request(monkeypatch, tmp_path, output_dir="out")
    mirror = tmp_path / "out"
    mirror.mkdir()
    primary, secondary, hits, kept = TimeoutError(boundary), TimeoutError("cleanup"), [], {}
    if keep == "prior_manifest":
        path = mirror / "manifest.json"
        path.write_bytes(b'{"unowned": true}')
        kept[path] = (checked_io.inode_of(path), path.read_bytes())
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b'{"foreign": true}')
    opened, fds = _r16_fds()
    replace, read, verify, unlink = os.replace, os.read, storage._verify_pair_bytes, os.unlink
    cleanup_hits = []
    def fail():
        hits.append(boundary)
        if keep and keep.startswith("foreign_"):
            path = mirror / ("repro.py" if keep.endswith("script") else "manifest.json")
            replace(foreign, path)
            kept[path] = (checked_io.inode_of(path), path.read_bytes())
        if boundary != "pair_false":
            raise primary
        return False
    def replace_hook(src, dst, *args, **kwargs):
        if boundary == "pre" and Path(dst) == mirror / "manifest.json" and not hits:
            fail()
        return replace(src, dst, *args, **kwargs)
    def read_hook(fd, *args):
        if boundary == "read" and fds[fd][0] == mirror / "manifest.json" and not hits:
            fail()
        return read(fd, *args)
    def pair_hook(script, *args, **kwargs):
        if boundary.startswith("pair_") and Path(script).absolute().parent == mirror and not hits:
            return fail()
        return verify(script, *args, **kwargs)
    def unlink_hook(path, *args, **kwargs):
        if cleanup_fault and hits and Path(path).absolute().parent == mirror and not cleanup_hits:
            cleanup_hits.append(path)
            raise secondary
        return unlink(path, *args, **kwargs)
    with contextlib.ExitStack() as stack:
        for owner, name, fn in ((os, "open", opened), (os, "replace", replace_hook),
                (os, "read", read_hook), (storage, "_verify_pair_bytes", pair_hook),
                (os, "unlink", unlink_hook)):
            stack.enter_context(patch.object(owner, name, fn))
        if boundary == "pair_false" and not cleanup_fault:
            assert repro_mcp("repro_op", request)["status"] == "delivery_failed"
        else:
            with pytest.raises(TimeoutError) as caught:
                repro_mcp("repro_op", request)
            assert caught.value is (secondary if boundary == "pair_false" else primary)
    # A false verifier has no body exception: the first cleanup timeout must escape.
    assert hits
    assert {p: (checked_io.inode_of(p), p.read_bytes()) for p in kept} == kept
    assert set(mirror.iterdir()) == set(kept)
    _r16_no_leaks(fds, tmp_path)
    assert repro_mcp("repro_op", request)["status"] == "ready"

def _r16_probe_namespace():
    script = emit_fabric_probe_script(_plan())["script"]
    namespace = {"__name__": "r16_probe"}
    exec(compile(script, "<r16-probe>", "exec"), namespace)
    return script, namespace

def test_r16_emitted_ast_has_bare_timeout_guards_and_no_path_predicates():
    scripts = [_r16_probe_namespace()[0],
               generate_single_op_reproducer(paged_binding(), stage=False)["script"]]
    counts = []
    for script in scripts:
        count = 0
        for node in ast.walk(ast.parse(script)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"exists", "lexists", "isfile", "isdir", "islink",
                                              "is_file", "is_dir", "is_symlink"}
            if not isinstance(node, ast.Try):
                continue
            for i, handler in enumerate(node.handlers):
                assert handler.type is not None
                names = {n.id for n in ast.walk(handler.type) if isinstance(n, ast.Name)}
                if names & {"OSError", "IOError", "EnvironmentError", "Exception", "BaseException"}:
                    count += 1
                    assert any(isinstance(h.type, ast.Name) and h.type.id == "TimeoutError"
                               and len(h.body) == 1 and isinstance(h.body[0], ast.Raise)
                               and h.body[0].exc is None and h.body[0].cause is None
                               for h in node.handlers[:i])
        counts.append(count)
    assert counts == [16, 6]

@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("cleanup_fault", [False, True])
def test_r16_emitted_abort_timeout_cleans_and_preserves_primary(active, cleanup_fault, tmp_path):
    _, ns = _r16_probe_namespace()
    target = tmp_path / "abort.json"
    ns["ABORT_FILE"] = str(target)
    primary, secondary = ValueError("primary"), TimeoutError("abort replace")
    hits, unlink = [], os.unlink
    def replace(*args):
        raise secondary
    def remove(path):
        if cleanup_fault and not hits:
            hits.append(path)
            raise TimeoutError("unlink")
        return unlink(path)
    with patch.object(os, "replace", replace), patch.object(os, "unlink", remove):
        with pytest.raises((ValueError, TimeoutError)) as caught:
            if active:
                try:
                    raise primary
                finally:
                    ns["_write_abort_record"]({"kind": "abort"}, primary)
            else:
                ns["_write_abort_record"]({"kind": "abort"})
    assert caught.value is (primary if active else secondary)
    assert not list(tmp_path.iterdir())

@pytest.mark.parametrize("active", [False, True])
def test_r16_emitted_main_cleanup_keeps_primary(active, monkeypatch, tmp_path):
    _, ns = _r16_probe_namespace()
    primary, secondary, stopped = TimeoutError("measurement"), TimeoutError("destroy"), []
    torch, dist = types.ModuleType("torch"), types.ModuleType("torch.distributed")
    torch.npu = types.SimpleNamespace(device_count=lambda: 2, set_device=lambda rank: None)
    torch.distributed = dist
    dist.init_process_group = lambda **kwargs: None
    def destroy():
        raise secondary
    dist.destroy_process_group = destroy
    for name, module in (("torch", torch), ("torch_npu", types.ModuleType("torch_npu")),
                         ("torch.distributed", dist)):
        monkeypatch.setitem(sys.modules, name, module)
    ns["_tier_groups"] = lambda *args: {t["tier"]: (object(), True, t["ranks"])
                                        for t in ns["PLAN"]["tiers"]}
    ns["_world_decision"] = lambda *args: 1
    ns["_stop_abort_watcher"] = lambda: stopped.append(True)
    def decision(*args, **kwargs):
        raise primary
    ns["_continue_decision"] = decision
    if not active:
        ns["PLAN"]["admitted"] = []
    with pytest.raises(TimeoutError) as caught:
        ns["main"](["--abort-file", str(tmp_path / "abort.json")])
    assert caught.value is (primary if active else secondary)
    assert stopped == [True] and not list(tmp_path.glob("*.tmp"))

@pytest.mark.parametrize("fault", [None, "read", "unlink"])
def test_r16_atomic_reports_only_verified_commits(fault, tmp_path):
    target = tmp_path / "leaf"
    opened, fds = _r16_fds()
    original_read, original_unlink = os.read, os.unlink
    installed, hits, primary = [], [], TimeoutError("exclusive " + str(fault))
    def read(fd, *args):
        if fault == "read" and fds[fd][0] == target and not hits:
            hits.append(fd)
            raise primary
        return original_read(fd, *args)
    def unlink(path, *args, **kwargs):
        if fault == "unlink" and Path(path).name.endswith(".tmp") and not hits:
            hits.append(path)
            raise primary
        return original_unlink(path, *args, **kwargs)
    with patch.object(os, "open", opened), patch.object(os, "read", read), \
            patch.object(os, "unlink", unlink):
        if fault is None:
            assert checked_io.safe_write_bytes(target, b"pair", exclusive=True, installed_out=installed)
        else:
            with pytest.raises(TimeoutError) as caught:
                checked_io.safe_write_bytes(target, b"pair", exclusive=True, installed_out=installed)
            assert caught.value is primary and hits
    if fault is None:
        assert installed == [checked_io.inode_of(target)] and target.read_bytes() == b"pair"
    else:
        assert not installed and not list(tmp_path.iterdir())
    _r16_no_leaks(fds, tmp_path)

def test_r16_emitted_destroy_timeout_survives_watchdog_cleanup_fault():
    _, ns = _r16_probe_namespace()
    primary, secondary = TimeoutError("destroy primary"), TimeoutError("timer cleanup")
    def destroy():
        raise primary
    def cancel():
        raise secondary
    timer = types.SimpleNamespace(start=lambda: None, cancel=cancel)
    ns["threading"] = types.SimpleNamespace(Timer=lambda *args: timer,
        current_thread=lambda: "worker", main_thread=lambda: "main")
    with pytest.raises(TimeoutError) as caught:
        ns["_bounded_destroy"](types.SimpleNamespace(destroy_process_group=destroy), .1)
    assert caught.value is primary

@pytest.mark.parametrize("bad", [True, False, "1", 1.5, 10**5000, -10**5000, {"x": [10**5000]}], ids=["true", "false", "str", "fraction", "huge", "negative_huge", "nested"])
def test_r22_direct_precondition_schemas_and_values(bad):
    from mindie_profiler.repro_op import _compact_occurrence
    from mindie_profiler.repro_preconditions import precondition_problems, _HANDLERS
    from test_nr_s6_round2_closure import _ready_bindings
    assert sys.get_int_max_str_digits() == 4300
    recipes = _ready_bindings()
    for name, field in (("aclnnSwiGlu", "dim"), ("aclnnRmsNorm", "epsilon"), ("aclnnDynamicQuant", "dst_type")):
        occurrence = _compact_occurrence(recipes[name])
        occurrence["attrs"][field] = bad
        problems = precondition_problems(occurrence)
        assert problems
        json.dumps(problems, allow_nan=False)
        fn = _HANDLERS[occurrence["preconditions"]["kind"]]
        assert fn(occurrence, occurrence["attrs"], occurrence["allocations"], {}, occurrence["call_plan"], {})
    for value in (bad, 1.0):
        occurrence = _compact_occurrence(recipes["ChunkGatedDeltaRule"])
        spec = next(x for x in occurrence["allocations"] if x["name"] == "actual_seq_lengths")
        spec["values"] = [value] * spec["shape"][0]
        problems = precondition_problems(occurrence)
        assert problems
        json.dumps(problems, allow_nan=False)
    for name, field, value in (("aclnnSwiGlu", "dim", -1), ("aclnnRmsNorm", "epsilon", 0.001)):
        occurrence = _compact_occurrence(recipes[name])
        occurrence["attrs"][field] = value
        assert precondition_problems(occurrence) == []
