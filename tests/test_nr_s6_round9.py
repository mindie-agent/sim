"""Round-9 closure: inode aliases, colliding leaves and protocol cores.

[Role]
  Re-run Astra's Round-9 counterexamples at their public boundaries: hard links
  to registered CAS scripts through caller mirrors, dangling internal staging
  leaves, a fresh-process script/manifest collision, and the extended diff-MCP
  protocol matrix.

[Invariants]
  * Every storage assertion verifies exact old bytes and successful resolution.
  * Internal staging tests create no file inside the protected slot A.
  * The collision CLI is a real fresh process; exit status and JSON are checked.
  * Every extended envelope is measured on the returned dictionary and its
    schema/status/error/issue codes and delivery flags are exact.

[Timeouts & Locks]
  Ordinary bounded local file operations plus a few short-lived Python
  subprocesses; no NPU, HCCL, network or device access.

[Side-Effect Policy]
  Writes only fresh /tmp short roots and pytest temporary mirrors/outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
__import__("pytest").importorskip("mindie_profiler")
from mindie_profiler import repro_atomic_io as checked_io
import shutil
import subprocess
import sys
import threading
from unittest.mock import patch

import pytest

from mindie_profiler.fabric_probe_fit import parse_fabric_probe_samples, fit_calibrated_fabric_profile
from mindie_profiler.op_api_registry import (
    call_mcp_tool as registry_mcp, load_catalog,
)
from mindie_profiler.repro_op import (
    call_mcp_tool as repro_mcp, generate_single_op_reproducer,
)
from mindie_profiler.repro_storage import resolve_repro_artifact
from mindie_profiler.token_budget import TOKENIZER_ID, estimate_tokens
from mindie_sim.diff_trace import (
    attribute_diff, call_mcp_tool as diff_mcp, resolve_diff_artifact,
)

from nr_s6_support import paged_binding
from temp_cleanup import tempdir
from test_nr_s6_round2_closure import _ready_bindings
from test_nr_s6_round8 import _r16_fds, _r16_no_leaks

PROJECT_ROOT = Path(checked_io.__file__).resolve().parents[1]
TOPOLOGY = {"tiers": [{"tier": "intra_node_hccs", "ranks": [0, 1],
                       "protocol": "HCCS"}]}
IDENTITY = {"total_ms": 2, "ops": [
    {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": 2}]}
BASE_PRED = {"total_ms": 3, "ops": [{"name": "A", "duration_ms": 3}]}
CAPTURES = {
    "identity": IDENTITY,
    "missing_total": {"ops": IDENTITY["ops"]},
    "unmatched": {"total_ms": 2, "ops": [{"name": "B", "duration_ms": 2}]},
    "multi_issue": {
        "ops": [{"name": "B", "duration_ms": 0},
                {"name": "C", "start_ms": 0, "duration_ms": 2}],
        "alignment": {"coverage_complete": False},
    },
    "fractional": {
        "ops": [{"name": "B", "duration_ms": 0},
                {"name": "C", "start_ms": 0.123456789,
                 "duration_ms": 2.345678912}],
        "alignment": {"coverage_complete": False},
    },
    "cycles": {
        "ops": [{"name": "B", "duration_ms": 0, "deps": [1]},
                {"name": "C", "duration_ms": 2, "deps": [0]}],
        "alignment": {"coverage_complete": False},
    },
}
PADDING = {"aclnnGroupedMatmulV4": {
    "bucket_tokens": 64, "bound_args": {"group_list": [16, 32, 48, 64]}}}
PADDINGS = {
    "none": None,
    "one": PADDING,
    "two": {**PADDING, "aclnnFlashAttentionVarLenScore": {
        "bucket_tokens": 64, "logical_tokens": 32}},
    "invalid_two": {**PADDING, "UnknownOp": {
        "bucket_tokens": 64, "logical_tokens": 32}},
}

def _fresh_root(prefix):
    return tempdir(prefix, directory="/tmp")

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

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

def _annotated_control(validated):
    return {**_healthy_rows()[-1], "kind": "cleanup_failed",
            "cleanup_failed": True, "status": "cleanup_failed",
            "validated": validated}

def _run_child(arguments, env):
    return subprocess.run(
        [sys.executable, "-B", "-m", "mindie_profiler", *arguments],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=60)

# ------------------------------------------------ R9-F10 caller inode aliases
@pytest.mark.parametrize("mode,leaf", [("dir", "repro.py"),
                                       ("path", "custom.py")])
def test_r9_f10_hard_link_to_cas_script_is_blocked(monkeypatch, mode, leaf):
    old_root, mirror = _fresh_root("nr_s6_r9_hla_"), _fresh_root("nr_s6_r9_hlm_")
    try:
        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(old_root))
        first = repro_mcp("repro_op", {"occurrence": paged_binding(),
                                       "iters": 5})
        assert first["status"] == "ready"
        script = Path(first["artifact_path"])
        manifest = Path(first["manifest_path"])
        script_bytes, manifest_bytes = script.read_bytes(), manifest.read_bytes()
        link = mirror / leaf
        os.link(script, link)
        assert link.stat().st_ino == script.stat().st_ino

        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT",
                           str(_fresh_root("nr_s6_r9_hlb_")))
        kwargs = ({"output_dir": mirror} if mode == "dir"
                  else {"output_path": mirror / "custom.py"})
        second = generate_single_op_reproducer(
            paged_binding(), iters=7, **kwargs)
        assert second["status"] == "delivery_failed"
        assert second["storage"]["status"] == "delivery_failed"
        assert second["error"] == "protected_artifact_target"
        assert second["storage"]["protected_artifact_target"] is True
        assert second["storage"]["retrievable"] is False
        assert script.read_bytes() == script_bytes
        assert manifest.read_bytes() == manifest_bytes
        assert checked_io.path_exists(link) and link.stat().st_ino == script.stat().st_ino
        resolved = resolve_repro_artifact(first["artifact_id"])
        assert resolved["resolved"] is True
        assert resolved["retrievable"] is True
    finally:
        shutil.rmtree(old_root, ignore_errors=True)
        shutil.rmtree(mirror, ignore_errors=True)

# --------------------------------------------- R9-F10 internal staging leaves
@pytest.mark.parametrize("switched", [False, True])
@pytest.mark.parametrize("leaf", ["repro.py", "manifest.json"])
def test_r9_f10_dangling_internal_staging_leaf_writes_nothing(
        monkeypatch, switched, leaf):
    old_root = _fresh_root("nr_s6_r9_dga_")
    try:
        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(old_root))
        first = repro_mcp("repro_op", {"occurrence": paged_binding(),
                                       "iters": 5})
        assert first["status"] == "ready"
        old_script = Path(first["artifact_path"])
        old_manifest = Path(first["manifest_path"])
        script_bytes, manifest_bytes = old_script.read_bytes(), old_manifest.read_bytes()
        preview = generate_single_op_reproducer(paged_binding(), iters=7)
        slot_name = preview["sha256"].split(":", 1)[1][:12]
        new_root = _fresh_root("nr_s6_r9_dgb_")
        active_root = new_root if switched else old_root
        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(active_root))
        slot = active_root / slot_name
        slot.mkdir()
        victim = old_script.parent / f"new-from-{leaf}"
        (slot / leaf).symlink_to(victim)
        assert checked_io.path_is_symlink(slot / leaf)
        assert not checked_io.path_exists(slot / leaf)

        result = generate_single_op_reproducer(
            paged_binding(), iters=7, stage=True)
        assert result["status"] == "delivery_failed"
        assert result["storage"]["status"] == "delivery_failed"
        assert result["error"] == "artifact_persistence_failed"
        assert result["storage"]["protected_artifact_target"] is True
        assert result["storage"]["retrievable"] is False
        assert not checked_io.path_exists(victim)
        assert old_script.read_bytes() == script_bytes
        assert old_manifest.read_bytes() == manifest_bytes
        resolved = resolve_repro_artifact(first["artifact_id"])
        assert resolved["resolved"] is True
    finally:
        shutil.rmtree(old_root, ignore_errors=True)

def test_r9_f10_symlinked_internal_slot_directory_is_blocked(monkeypatch):
    old_root = _fresh_root("nr_s6_r9_sda_")
    try:
        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(old_root))
        first = repro_mcp("repro_op", {"occurrence": paged_binding(),
                                       "iters": 5})
        old_script = Path(first["artifact_path"])
        before = old_script.read_bytes()
        preview = generate_single_op_reproducer(paged_binding(), iters=7)
        slot_name = preview["sha256"].split(":", 1)[1][:12]
        victim_dir = old_script.parent / "victim-slot"
        (old_root / slot_name).symlink_to(victim_dir, target_is_directory=True)
        result = generate_single_op_reproducer(
            paged_binding(), iters=7, stage=True)
        assert result["status"] == "delivery_failed"
        assert result["storage"]["status"] == "delivery_failed"
        assert result["error"] == "artifact_persistence_failed"
        assert result["storage"]["protected_artifact_target"] is True
        assert not checked_io.path_exists(victim_dir)
        assert old_script.read_bytes() == before
        assert resolve_repro_artifact(first["artifact_id"])["resolved"] is True
    finally:
        shutil.rmtree(old_root, ignore_errors=True)

# ------------------------------------------- R9-DELIVERY colliding CLI leaves
def test_r9_delivery_cli_rejects_script_manifest_collision(tmp_path):
    root = tmp_path / "cas"
    row = tmp_path / "binding.json"
    row.write_text(json.dumps(paged_binding()), encoding="utf-8")
    output = tmp_path / "artifacts" / "manifest.json"
    env = dict(os.environ)
    env["MINDIE_REPRO_SHORT_ROOT"] = str(root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")])
    completed = _run_child(["repro-op", "--row", str(row), "--iters", "7",
                            "--output", str(output)], env)
    assert completed.returncode != 0
    result = json.loads(completed.stdout)
    assert result["status"] == "delivery_blocked"
    assert result["storage"]["status"] == "delivery_blocked"
    assert result["error"] == "colliding_output_paths"
    assert result["storage"]["retrievable"] is False
    assert result["storage"]["artifact_path"] is None
    assert not checked_io.path_exists(output)
    assert not checked_io.path_exists(output.parent)
    assert not checked_io.path_exists(root)

# -------------------------------------------- R9-MCP extended 48-case matrix
def test_r9_mcp_extended_matrix_keeps_exact_protocol_core(monkeypatch):
    root = _fresh_root("nr_s6_r9_mcpa_")
    blocker_root = _fresh_root("nr_s6_r9_mcpb_")
    blocker = blocker_root / "b"
    blocker.write_text("not a directory", encoding="utf-8")
    cases = 0
    try:
        for capture_name, captured in CAPTURES.items():
            for padding_name, padding in PADDINGS.items():
                arguments = {
                    "predicted": IDENTITY if capture_name == "identity"
                    else BASE_PRED,
                    "captured": captured,
                }
                if padding is not None:
                    arguments["padding"] = padding
                    arguments["cann_version"] = "9.0"
                full = attribute_diff(
                    arguments["predicted"], arguments["captured"],
                    padding=padding,
                    cann_version=arguments.get("cann_version"))
                full_codes = [str(issue.get("code") or "unspecified")
                              for issue in full.get("issues") or ()
                              if isinstance(issue, dict)]
                expected_failure = (["artifact_persistence_failed"]
                                    + [code for code in full_codes
                                       if code != "artifact_persistence_failed"][:5])
                for mode, short_root_value in (("normal", root),
                                               ("failed", blocker)):
                    monkeypatch.setenv(
                        "MINDIE_REPRO_SHORT_ROOT", str(short_root_value))
                    envelope = diff_mcp(arguments)
                    label = f"{capture_name}/{padding_name}/{mode}"
                    assert estimate_tokens(envelope) <= 250, label
                    assert envelope["receipt"]["within_budget"] is True, label
                    assert envelope["schema"] == "mindie-diff-trace-mcp/1", label
                    assert "..." not in envelope["schema"], label
                    failed = mode == "failed"
                    expected_status = ("delivery_failed" if failed
                                       else full["status"])
                    assert envelope["status"] == expected_status, label
                    assert "..." not in envelope["status"], label
                    assert envelope["reconciled"] == bool(
                        (full.get("reconciliation") or {}).get("reconciled")), label
                    assert envelope["persisted"] is (not failed), label
                    assert envelope["retrievable"] is (not failed), label
                    if failed:
                        assert envelope["error"] == "artifact_persistence_failed", label
                        assert envelope["artifact_path"] is None, label
                        expected_issues = expected_failure
                    else:
                        assert "error" not in envelope, label
                        expected_issues = full_codes[:6]
                    retained = envelope["issues"]
                    omitted = envelope.get("issues_omitted", 0)
                    assert retained == expected_issues[:len(retained)], label
                    assert len(retained) + omitted == len(expected_issues), label
                    assert all("..." not in code for code in retained), label
                    cases += 1
        assert cases == 48
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(blocker_root, ignore_errors=True)

# ------------------------------------------- R9-MCP all 273 declared routes
def test_r9_mcp_declared_routes_keep_exact_protocol_core(monkeypatch):
    root = _fresh_root("nr_s6_r9_routes_")
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
    envelopes = []
    try:
        document = load_catalog()
        for name, entry in document["operators"].items():
            for version in entry.get("cann_versions") or ():
                envelopes.append(registry_mcp(
                    "inspect_op_abi",
                    {"op_name": name, "cann_version": str(version)}))
            envelopes.append(registry_mcp("inspect_op_abi",
                                          {"op_name": name}))
        assert len(envelopes) == 132
        receipts = []
        for binding in _ready_bindings().values():
            receipt = repro_mcp("repro_op", {"occurrence": binding})
            assert receipt["status"] == "ready"
            receipts.append(receipt)
            envelopes.append(receipt)
        assert len(receipts) == 13
        for receipt in receipts:
            envelopes.append(repro_mcp(
                "repro_op", {"resolve_artifact_id": receipt["artifact_id"]}))
        handles = ["", " ", "sha256:", "sha256:abc", "sha256:" + "z" * 64,
                   "sha256:" + "A" * 64, "sha256:" + "0" * 63,
                   "sha256:" + "0" * 65, "sha256:" + "0" * 64 + "\n",
                   "sha256:" + "0" * 128, "sha256:" + "界" * 40,
                   "sha256:" + "a" * 2000, None, 0, 1.5, True, [], {},
                   ["sha256:" + "0" * 64]]
        while len(handles) < 106:
            handles.append("sha256:" + format(len(handles), "064x"))
        for handle in handles[:106]:
            envelopes.append(repro_mcp(
                "repro_op", {"resolve_artifact_id": handle}))
        assert len(envelopes) == 264

        multi = {"total_ms": 6, "ops": [
            {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": 1},
            {"name": "B", "stream": 0, "start_ms": 1, "duration_ms": 1}],
            "gaps": [{"start_ms": 2, "duration_ms": 4}]}
        diff_envelopes = [
            diff_mcp({"predicted": IDENTITY, "captured": IDENTITY}),
            diff_mcp({"predicted": IDENTITY, "captured": multi}),
            diff_mcp({"predicted": IDENTITY, "captured": IDENTITY,
                      "padding": PADDING, "cann_version": "9.0"}),
            diff_mcp({}),
        ]
        diff_envelopes.append(resolve_diff_artifact(
            diff_envelopes[0]["artifact_id"]))
        diff_envelopes.append(resolve_diff_artifact("sha256:" + "0" * 64))
        diff_envelopes.append(resolve_diff_artifact(""))
        diff_envelopes.append(resolve_diff_artifact(123))
        artifact = Path(diff_envelopes[0]["artifact_path"])
        original = artifact.read_bytes()
        artifact.write_bytes(original + b"\n")
        try:
            diff_envelopes.append(resolve_diff_artifact(
                diff_envelopes[0]["artifact_id"]))
            assert diff_envelopes[-1]["error"] == "artifact_integrity_mismatch"
        finally:
            artifact.write_bytes(original)
        envelopes.extend(diff_envelopes)
        assert len(envelopes) == 273
        for envelope in envelopes:
            assert estimate_tokens(envelope) <= 250
            assert envelope["receipt"]["within_budget"] is True
            assert envelope["receipt"]["tokenizer"] == TOKENIZER_ID
            schema = envelope.get("schema")
            assert isinstance(schema, str) and "..." not in schema
            for key in ("status", "error"):
                value = envelope.get(key)
                if isinstance(value, str):
                    assert "..." not in value
            for issue in envelope.get("issues") or ():
                code = issue.get("code") if isinstance(issue, dict) else issue
                if isinstance(code, str):
                    assert "..." not in code
            for flag in ("reconciled", "persisted", "retrievable"):
                if flag in envelope:
                    assert isinstance(envelope[flag], bool)
    finally:
        shutil.rmtree(root, ignore_errors=True)

# ------------------------------------- Round-8 annotated controls, real CLI
@pytest.mark.parametrize("validated", [True, False])
@pytest.mark.parametrize("parser_passes", [0, 1, 2, 3])
def test_r9_f7_annotated_control_fresh_cli_is_partial(
        tmp_path, validated, parser_passes):
    records = _healthy_rows() + [_annotated_control(validated)]
    for _ in range(parser_passes):
        records = parse_fabric_probe_samples(records)
    samples = tmp_path / "samples.jsonl"
    samples.write_text("\n".join(json.dumps(item) for item in records)
                       + "\n", encoding="utf-8")
    topology = tmp_path / "topology.json"
    topology.write_text(json.dumps(TOPOLOGY), encoding="utf-8")
    output = tmp_path / "profile.json"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")])
    completed = _run_child([
        "fabric-probe", "--world-size", "2", "--ranks-per-node", "2",
        "--duration-s", "1", "--iterations", "1", "--warmup", "0",
        "--topology", str(topology), "--samples", str(samples), "--output", str(output)], env)
    assert completed.returncode == 0, completed.stderr
    profile = json.loads(output.read_text(encoding="utf-8"))
    assert profile["status"] == "partial"
    assert profile["run_evidence_blocked"] is True
    assert profile["usable_for_link_override"] is False
    assert profile["control_record_count"] >= 1

@pytest.mark.parametrize("route", ["cas", "caller"])
@pytest.mark.parametrize("leaf", ["repro.py", "manifest.json"])
@pytest.mark.parametrize("mode", ["thread", "reentrant"])
@pytest.mark.parametrize("boundary", ["rename", "unlink"])
def test_r17_final_name_survives_rollback_swap(route, leaf, mode, boundary, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "cas")
    request = {"occurrence": paged_binding(), "iters": 17002}
    if route == "caller":
        request["output_dir"] = "mirror"
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"owned by B")
    inode, primary, hits, swaps = checked_io.inode_of(foreign), TimeoutError(leaf), [], []
    gate, release = threading.Event(), threading.Event()
    opened, fds = _r16_fds()
    read, rename, unlink, replace = os.read, os.rename, os.unlink, os.replace
    def selected(path):
        p = Path(path)
        return p.name == leaf and (p.parent == tmp_path / "mirror" if route == "caller"
                                   else tmp_path / "cas" in p.parents)
    def read_hook(fd, *args):
        if selected(fds[fd][0]) and not hits:
            hits.append(fds[fd][0])
            raise primary
        return read(fd, *args)
    def swap():
        replace(foreign, hits[0])  # Raw external mutation; no product locks/helpers.
        swaps.append(checked_io.inode_of(hits[0]))
        release.set()
    def schedule():
        gate.set()
        if mode == "reentrant":
            swap()
        else:
            assert release.wait(10)
    def rename_hook(src, dst, *args, **kwargs):
        if boundary == "rename" and hits and selected(src) and not swaps:
            schedule()  # After ownership lstat, immediately before namespace detach.
        return rename(src, dst, *args, **kwargs)
    def unlink_hook(path, *args, **kwargs):
        assert not selected(path), "rollback reached a shared final pathname"
        if (boundary == "unlink" and hits and not swaps and Path(path).parent == hits[0].parent
                and Path(path).name.startswith("." + leaf + ".") and Path(path).suffix == ".rollback"):
            schedule()  # Replace the final name while cleanup unlinks its private name.
        return unlink(path, *args, **kwargs)
    def other_thread():
        assert gate.wait(10)
        swap()
    thread = threading.Thread(target=other_thread) if mode == "thread" else None
    with patch.object(os, "open", opened), patch.object(os, "read", read_hook), \
            patch.object(os, "rename", rename_hook), patch.object(os, "unlink", unlink_hook):
        if thread:
            thread.start()
        try:
            with pytest.raises(TimeoutError) as caught:
                repro_mcp("repro_op", request)
        finally:
            if thread:
                thread.join(10)
    assert caught.value is primary and swaps == [inode]
    assert not thread or not thread.is_alive()
    assert checked_io.inode_of(hits[0]) == inode and hits[0].read_bytes() == b"owned by B"
    _r16_no_leaks(fds, tmp_path)
    assert not list(tmp_path.rglob("*.rollback"))

@pytest.mark.parametrize("phase", ["before_kernel", "after_kernel", "entropy"])
@pytest.mark.parametrize("faults", [1, 2, 3])
def test_r17_parking_retries_preserve_primary(phase, faults, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "cas")
    primary, secondary, hits, failures = TimeoutError("read"), TimeoutError("park"), [], []
    opened, fds = _r16_fds()
    read, rename, lstat, urandom = os.read, os.rename, os.lstat, os.urandom
    def read_hook(fd, *args):
        if fds[fd][0].name == "manifest.json" and not hits:
            hits.append(fd)
            raise primary
        return read(fd, *args)
    def rename_hook(src, dst):
        if phase != "entropy" and hits and not failures:
            failures.append(src)
            if phase == "after_kernel":
                rename(src, dst)
            raise secondary
        return rename(src, dst)
    def lstat_hook(path, *args, **kwargs):
        if phase != "entropy" and failures and len(failures) < faults and str(path).endswith(".rollback"):
            failures.append(path)
            raise secondary
        return lstat(path, *args, **kwargs)
    def random_hook(size):
        if phase == "entropy" and hits and len(failures) < faults:
            failures.append(size)
            raise secondary
        return urandom(size)
    request = {"occurrence": paged_binding(), "iters": 17003}
    with patch.object(os, "open", opened), patch.object(os, "read", read_hook), \
            patch.object(os, "rename", rename_hook), patch.object(os, "lstat", lstat_hook), \
            patch.object(os, "urandom", random_hook):
        with pytest.raises(TimeoutError) as caught:
            repro_mcp("repro_op", request)
    assert caught.value is primary and hits and len(failures) == faults
    assert not any(checked_io.path_is_file(p) for p in Path("cas").rglob("*"))
    _r16_no_leaks(fds, tmp_path)
    assert repro_mcp("repro_op", request)["status"] == "ready"

@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("case", ["valid", "large", "positive", "negative", "edge", "above", "below"])
def test_r19_captured_integer_float_boundaries(dtype, case):
    from mindie_profiler.repro_values import _FLOAT_MAX, validate_captured_values
    limit = _FLOAT_MAX[dtype]
    value = {"valid": 1, "large": 10**100, "positive": 10**400, "negative": -10**400,
             "edge": int(limit), "above": int(limit) + 1, "below": -int(limit) - 1}[case]
    values, problems = validate_captured_values([1], dtype, [value])
    if abs(value) <= limit:
        assert values == [value] and problems == []
    else:
        assert values is None and problems == [{"code": "captured_value_out_of_range", "index": 0,
            "dtype": dtype, "value": repr(value)[:120], "max_finite": limit}]

@pytest.mark.parametrize("value", [1, 10**100, 10**400, -10**400])
def test_r19_public_captured_integer_errors_are_bounded(value, monkeypatch, tmp_path):
    try:
        import tiktoken
    except ImportError:
        tiktoken = None
    from mindie_profiler.events import ProfilerUsageError
    from mindie_profiler.op_api_registry import bind_captured_occurrence
    from test_nr_s6_round2_closure import rms_norm_row, nd
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "cas")
    row = rms_norm_row()
    row["input_shapes"][0] = nd([4, 8], values=[value] * 32)
    bound = bind_captured_occurrence(json.loads(json.dumps(row, allow_nan=False)))
    if value == 1:
        assert generate_single_op_reproducer(bound, stage=False)["status"] == "ready"
    else:
        with pytest.raises(ProfilerUsageError, match="captured_value_out_of_range"):
            generate_single_op_reproducer(bound, stage=False)
    envelope = repro_mcp("repro_op", {"occurrence": bound})
    assert envelope["status"] == ("ready" if value == 1 else "blocked")
    serialized = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert estimate_tokens(envelope) <= 250
    if tiktoken is not None:
        for name in ("cl100k_base", "o200k_base"):
            try:
                enc = tiktoken.get_encoding(name)
            except TimeoutError:
                raise
            except Exception:
                break
            assert len(enc.encode(serialized)) <= estimate_tokens(envelope) <= 250

@pytest.mark.parametrize("sign", [-1, 1])
def test_r19_json_integer_overflow_entry_points(sign, monkeypatch, tmp_path):
    from mindie_profiler import fabric_probe as probe, op_abi_padding as padding
    from mindie_profiler import repro_preconditions as pre
    from mindie_profiler.events import ProfilerUsageError
    from mindie_profiler.op_abi_attributes import attr_issue
    from mindie_profiler.op_api_registry import bind_captured_occurrence
    from test_nr_s6_round2_closure import rms_norm_row
    huge = sign * 10**400
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "cas")
    assert attr_issue("epsilon", "float", huge)[0]
    row = rms_norm_row()
    row["attrs"]["epsilon"] = huge
    bound = bind_captured_occurrence(row)
    assert bound["status"] == "incomplete" and bound["issues"]
    assert repro_mcp("repro_op", {"occurrence": bound})["status"] == "blocked"
    assert padding.numeric_list([huge]) is None
    charge = padding.operator_charge({}, bucket=abs(huge), logical=1)
    assert charge["bucket_padding_tax_pct"] is None
    assert padding.operator_charge({}, bucket=abs(huge), logical=abs(huge))["bucket_padding_tax_pct"] == 0.0
    assert not padding.validate_sequence_lengths([huge], encoding="lengths")["valid"]
    assert not padding.validate_group_list([huge], 0, 10)["valid"]
    with pytest.raises(ValueError):
        padding.synthesize_group_list(10, huge)
    for fn, field in ((pre._rms_norm, "epsilon"), (pre._gdn_inplace, "scale"), (pre._dst_type, "dst_type_max")):
        assert fn({}, {field: huge}, [], {}, {}, {})
    with pytest.raises(ProfilerUsageError):
        generate_single_op_reproducer(paged_binding(), max_cv_pct=huge, stage=False)
    trace = {"total_ms": huge, "ops": [{"name": "A", "duration_ms": huge, "start_ms": huge}],
             "gaps": [{"start_ms": huge, "duration_ms": huge}]}
    assert isinstance(attribute_diff(trace, trace), dict)
    assert estimate_tokens(diff_mcp({"predicted": trace, "captured": trace})) <= 250
    with pytest.raises(ProfilerUsageError, match="bytes must be an integer"):
        parse_fabric_probe_samples([{"bytes": huge, "latency_us": huge}])
    with pytest.raises(ProfilerUsageError, match="bytes must be an integer"):
        fit_calibrated_fabric_profile({}, [{"bytes": huge, "latency_us": huge, "_parsed_fabric_sample": True}])
    with pytest.raises(ProfilerUsageError):
        probe.build_fabric_probe_plan(duration_s=huge)
    if sign > 0:
        with pytest.raises(ProfilerUsageError):
            probe.build_fabric_probe_plan(sizes=[huge])
    for field in ("wall_budget_s", "poll_interval_s"):
        with pytest.raises(ProfilerUsageError):
            probe.supervise_probe_processes([object()], **{field: huge})
    plan = probe.build_fabric_probe_plan(world_size=2, sizes=[4096], iterations=1, warmup=0)
    plan["deadline"]["measurement_budget_s"] = huge
    with pytest.raises(ProfilerUsageError):
        probe.emit_fabric_probe_script(plan)

def test_r19_captured_integer_repr_limit():
    from mindie_profiler.repro_values import validate_captured_values
    for value in (10**5000, -10**5000):
        values, problems = validate_captured_values([1], "float64", [value])
        assert values is None and len(problems) == 1
        assert problems[0]["code"] == "captured_value_out_of_range"
        assert len(problems[0]["value"]) <= 120

@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("field", ["bytes", "rank", "group_size", "group_ranks", "world_size", "local_rank", "iterations", "warmup"])
@pytest.mark.parametrize("bad", [True, False, "1", 1.5, 1e200, -1, 2**63, 10**5000, -10**5000], ids=["true", "false", "string", "fraction", "float_huge", "negative", "excessive", "huge", "negative_huge"])
def test_r21_fit_strict_counts(normalized, field, bad):
    from mindie_profiler.fabric_probe_fit import parse_fabric_probe_samples, fit_calibrated_fabric_profile
    from mindie_profiler.fabric_probe import build_fabric_probe_plan
    from mindie_profiler.events import ProfilerUsageError
    plan = build_fabric_probe_plan(world_size=2, ranks_per_node=2,
        topology={"tiers": [{"tier": "intra_node_hccs", "ranks": [0, 1], "protocol": "HCCS"}]})
    samples = [dict(tier="intra_node_hccs", op="all_reduce", bytes=size, latency_us=latency,
        rank=0, group_size=2, group_ranks=[0, 1], protocol="HCCS", validated=True)
        for size, latency in ((4096, 1), (8192, 2))]
    assert fit_calibrated_fabric_profile(plan, samples)["status"] == "calibrated"
    if normalized:
        samples = parse_fabric_probe_samples(samples)
    samples[0][field] = [0, bad] if field == "group_ranks" else bad
    with pytest.raises(ProfilerUsageError):
        fit_calibrated_fabric_profile(plan, samples)
    samples[0].update(kind="abort", error="terminal wins")
    assert fit_calibrated_fabric_profile(plan, samples)["run_evidence_blocked"]

@pytest.mark.parametrize("bad", [True, 4096.5, 1e200, "1e200", 10**200])
def test_r21_fit_cli_clean_usage(bad, tmp_path):
    import json
    import os
    import subprocess
    import sys
    sample = tmp_path / "samples.json"
    sample.write_text(json.dumps([dict(bytes=bad, latency_us=1, tier="intra_node_hccs",
        op="all_reduce", rank=0, group_size=2, group_ranks=[0, 1], validated=True)]))
    result = subprocess.run([sys.executable, "-B", "-m", "mindie_profiler", "fabric-probe",
        "--world-size", "2", "--samples", str(sample)], text=True, capture_output=True,
        timeout=15, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 2 and "Traceback" not in result.stderr

@pytest.mark.parametrize("parsed", [False, True])
@pytest.mark.parametrize("bad", [True, False, "1", 0, -1, float("nan"), float("inf"), 1e155, 1e200, 10**200, 10**5000, -10**5000], ids=["true", "false", "str", "zero", "negative", "nan", "inf", "square", "float_huge", "int_huge", "huge", "negative_huge"])
def test_r22_fit_numeric_boundaries(parsed, bad, tmp_path):
    import subprocess
    from mindie_profiler.events import ProfilerUsageError
    from mindie_profiler.fabric_probe import build_fabric_probe_plan
    from mindie_profiler.fabric_probe_fit import _num, parse_fabric_probe_samples, fit_calibrated_fabric_profile
    assert sys.get_int_max_str_digits() == 4300
    assert _num(True) is None and _num(False) is None
    topology = {"tiers": [{"tier": "intra_node_hccs", "ranks": [0, 1], "protocol": "HCCS"}]}
    plan = build_fabric_probe_plan(world_size=2, ranks_per_node=2, topology=topology)
    samples = [dict(tier="intra_node_hccs", op="all_reduce", bytes=b, latency_us=l, rank=0, group_size=2, group_ranks=[0, 1], protocol="HCCS", validated=True) for b, l in ((4096, 1), (8192, 2))]
    assert fit_calibrated_fabric_profile(plan, samples)["status"] == "calibrated"
    if parsed:
        samples = parse_fabric_probe_samples(samples)
    samples[0]["latency_us"] = bad
    with pytest.raises(ProfilerUsageError):
        parse_fabric_probe_samples(samples)
    with pytest.raises(ProfilerUsageError):
        fit_calibrated_fabric_profile(plan, samples)
    samples[0]["latency_us"] = "HUGE" if isinstance(bad, int) and abs(bad) > 10**400 else bad
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(samples).replace('"HUGE"', ("-" if isinstance(bad, int) and bad < 0 else "") + "1" + "0" * 5000))
    result = subprocess.run([sys.executable, "-B", "-m", "mindie_profiler", "fabric-probe", "--samples", str(path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "Traceback" not in result.stderr and len(result.stderr.encode()) < 1200
    samples[0].update(kind="abort", latency_us=bad, rank=10**5000, error={"nested": [-10**5000]})
    result = fit_calibrated_fabric_profile(plan, samples)
    assert result["run_evidence_blocked"] and not result["usable_for_link_override"]
    json.dumps(result, allow_nan=False)
