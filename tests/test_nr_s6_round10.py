"""Round-10 closure: shared atomic writes, BPE bounds and literal identities.

[Role]
  Re-run the exact Round-10 counterexamples at their public boundaries: the nine
  index/diff `.tmp` hard-link, symlink and dangling-symlink cases; all 321
  inspection/repro/resolution/diff envelopes with an optional real tiktoken
  cross-check; long caller keys/issue names; literal ellipses; and
  case-insensitive manifest destination aliases.

[Invariants]
  * A pre-existing `.tmp` alias is never opened or truncated: old script and
    manifest bytes are unchanged and the old handle still resolves.
  * No dangling `.tmp` symlink creates a file inside a protected slot.
  * Every returned envelope is bounded by the estimator and reports
    ``within_budget=True``; when tiktoken is importable the estimator is at or
    above both real encodings and the real count is at most 250.
  * Literal ``...`` bytes in caller input never raise and never become a
    protocol value.
  * A case-insensitive reserved destination is rejected before any CAS or
    caller directory/file is created.

[Timeouts & Locks]
  Synchronous local file writes plus bounded subprocesses; no NPU, network or
  device access.

[Side-Effect Policy]
  Writes only fresh /tmp roots, pytest temporary mirrors and one CLI output.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import stat
import threading
from unittest.mock import patch
__import__("pytest").importorskip("mindie_profiler")
from mindie_profiler import repro_storage as storage
from test_nr_s6_round8 import _r16_fds, _r16_no_leaks, _r16_probe_namespace
import json
import os
from pathlib import Path
from mindie_profiler import repro_atomic_io as checked_io
import subprocess
import sys

import pytest

from mindie_profiler.op_api_registry import (call_mcp_tool as registry_mcp, load_catalog,
)
from mindie_profiler.repro_op import (call_mcp_tool as repro_mcp, generate_single_op_reproducer,
)
from mindie_profiler.repro_storage import (resolve_repro_artifact, short_root,
)
from mindie_profiler.token_budget import bounded_envelope, estimate_tokens
from mindie_sim.diff_trace import (attribute_diff, call_mcp_tool as diff_mcp, resolve_diff_artifact,
)

from nr_s6_support import paged_binding
from temp_cleanup import tempdir
from test_nr_s6_round2_closure import _ready_bindings

PROJECT_ROOT = Path(checked_io.__file__).resolve().parents[1]
TRACE = {"total_ms": 2, "ops": [
    {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": 2}]}
MULTI = {"total_ms": 6, "ops": [
    {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": 1},
    {"name": "B", "stream": 0, "start_ms": 1, "duration_ms": 1}],
    "gaps": [{"start_ms": 2, "duration_ms": 4}]}
BASE_PRED = {"total_ms": 3, "ops": [{"name": "A", "duration_ms": 3}]}
PADDING = {"aclnnGroupedMatmulV4": {
    "bucket_tokens": 64, "bound_args": {"group_list": [16, 32, 48, 64]}}}
CAPTURES = {
    "identity": TRACE,
    "missing_total": {"ops": TRACE["ops"]},
    "unmatched": {"total_ms": 2, "ops": [{"name": "B", "duration_ms": 2}]},
    "multi_issue": {
        "ops": [{"name": "B", "duration_ms": 0}, {"name": "C", "start_ms": 0, "duration_ms": 2}],
        "alignment": {"coverage_complete": False},
    },
    "fractional": {
        "ops": [{"name": "B", "duration_ms": 0}, {"name": "C", "start_ms": 0.123456789,
                 "duration_ms": 2.345678912}],
        "alignment": {"coverage_complete": False},
    },
    "cycles": {
        "ops": [{"name": "B", "duration_ms": 0, "deps": [1]}, {"name": "C", "duration_ms": 2, "deps": [0]}],
        "alignment": {"coverage_complete": False},
    },
}
PADDINGS = {
    "none": None,
    "one": PADDING,
    "two": {**PADDING, "aclnnFlashAttentionVarLenScore": {
        "bucket_tokens": 64, "logical_tokens": 32}},
    "invalid_two": {**PADDING, "UnknownOp": {
        "bucket_tokens": 64, "logical_tokens": 32}},
}
SCOPED = ("mindie_profiler/op_api_registry.py", "mindie_profiler/op_api_paging.py",
    "mindie_profiler/op_abi_descriptors.py",
    "mindie_profiler/op_abi_attributes.py",
    "mindie_profiler/op_abi_binding.py",
    "mindie_profiler/op_abi_padding.py",
    "mindie_profiler/repro_metadata.py",
    "mindie_profiler/repro_values.py",
    "mindie_profiler/repro_op.py",
    "mindie_profiler/repro_mcp.py",
    "mindie_profiler/repro_storage.py",
    "mindie_profiler/repro_atomic_io.py",
    "mindie_profiler/repro_preconditions.py",
    "mindie_profiler/repro_script_template.py",
    "mindie_profiler/token_budget.py",
    "mindie_profiler/fabric_probe.py",
    "mindie_profiler/fabric_probe_fit.py",
    "mindie_profiler/fabric_probe_script_template.py",
    "mindie_sim/diff_trace.py",
    "mindie_sim/diff_trace_release.py",
    "mindie_sim/diff_trace_replay.py",
    "tests/nr_s6_support.py",
    "tests/nr_s6_round3_probe.py",
    "tests/nr_s6_round4_probe.py",
    "tests/test_op_abi_repro_and_diff_trace.py",
    "tests/test_nr_s6_adversarial_contracts.py",
    "tests/test_nr_s6_round2_closure.py",
    "tests/test_nr_s6_round2_runtime.py",
    "tests/test_nr_s6_round3_closure.py",
    "tests/test_nr_s6_round4_closure.py",
    "tests/test_nr_s6_round5_closure.py",
    "tests/test_nr_s6_round6_closure.py",
    "tests/test_nr_s6_round7_closure.py",
    "tests/test_nr_s6_round8.py",
    "tests/test_nr_s6_round9.py",
    "tests/test_nr_s6_round10.py",
    "tests/test_nr_s6_round11.py",
    "tests/test_nr_s6_round12.py",
    "tests/test_nr_s6_round13.py",
)

def _fresh_root(prefix="nr_s6_r10_"):
    return tempdir(prefix, directory="/tmp")

def _compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _child(arguments, env):
    return subprocess.run([sys.executable, "-B", "-m", "mindie_profiler", *arguments],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=60)

def _index_attack(monkeypatch, relation, switched):
    old_root, new_root = _fresh_root("nr_s6_r10_idx_"), _fresh_root("nr_s6_r10_idxb_")
    try:
        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(old_root))
        first = repro_mcp("repro_op", {"occurrence": paged_binding(), "iters": 5})
        assert first["status"] == "ready"
        script = Path(first["artifact_path"])
        manifest = Path(first["manifest_path"])
        script_bytes, manifest_bytes = script.read_bytes(), manifest.read_bytes()
        root = new_root if switched else old_root
        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
        leaf = root / "index.json.tmp"
        if relation == "dangling":
            victim = script.parent / "new-from-index.json"
            leaf.symlink_to(victim)
        else:
            victim = script
            if relation == "hardlink":
                os.link(victim, leaf)
            else:
                leaf.symlink_to(victim)
        second = repro_mcp("repro_op", {"occurrence": paged_binding(),
                                        "iters": 7})
        assert second["status"] == "ready"
        assert second["artifact"]["retrievable"] is True
        assert script.read_bytes() == script_bytes
        assert manifest.read_bytes() == manifest_bytes
        assert resolve_repro_artifact(first["artifact_id"])["resolved"] is True
        if relation == "dangling":
            assert not checked_io.path_exists(victim)
        else:
            assert checked_io.path_exists(victim)
    finally:
        for root in (old_root, new_root):
            import shutil
            shutil.rmtree(root, ignore_errors=True)

@pytest.mark.parametrize("switched", [False, True])
@pytest.mark.parametrize("relation", ["hardlink", "symlink", "dangling"])
def test_r10_f10_index_tmp_aliases_cannot_touch_registered_cas(
        monkeypatch, relation, switched):
    _index_attack(monkeypatch, relation, switched)

@pytest.mark.parametrize("relation", ["hardlink", "symlink", "dangling"])
def test_r10_f10_diff_tmp_aliases_cannot_touch_registered_cas(
        monkeypatch, relation):
    root = _fresh_root("nr_s6_r10_dtmp_")
    try:
        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
        first = repro_mcp("repro_op", {"occurrence": paged_binding(),
                                       "iters": 5})
        assert first["status"] == "ready"
        script = Path(first["artifact_path"])
        script_bytes = script.read_bytes()
        payload = attribute_diff(TRACE, TRACE)
        data = (_compact(payload) + "\n").encode()
        digest = hashlib.sha256(data).hexdigest()[:12]
        slot = root / "diff" / digest
        slot.mkdir(parents=True)
        leaf = slot / "diff.json.tmp"
        if relation == "dangling":
            victim = script.parent / "new-from-diff.json"
            leaf.symlink_to(victim)
        else:
            victim = script
            if relation == "hardlink":
                os.link(victim, leaf)
            else:
                leaf.symlink_to(victim)
        result = diff_mcp({"predicted": TRACE, "captured": TRACE})
        assert result["status"] == "reconciled"
        assert result["persisted"] is True
        assert result["retrievable"] is True
        assert script.read_bytes() == script_bytes
        assert resolve_repro_artifact(first["artifact_id"])["resolved"] is True
        if relation == "dangling":
            assert not checked_io.path_exists(victim)
        else:
            assert checked_io.path_exists(victim)
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)

# --------------------------------------------------------------- R10-BPE
def _declared_envelopes(monkeypatch):
    root = _fresh_root("nr_s6_r10_bpe_")
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
    envelopes = []
    document = load_catalog()
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
        envelopes.append(repro_mcp("repro_op",
                                   {"resolve_artifact_id": handle}))
    diff_envelopes = [
        diff_mcp({"predicted": TRACE, "captured": TRACE}),
        diff_mcp({"predicted": TRACE, "captured": MULTI}), diff_mcp({"predicted": TRACE, "captured": TRACE,
                  "padding": PADDING, "cann_version": "9.0"}), diff_mcp({}), ]
    diff_envelopes.append(resolve_diff_artifact( diff_envelopes[0]["artifact_id"]))
    diff_envelopes.append(resolve_diff_artifact("sha256:" + "0" * 64))
    diff_envelopes.append(resolve_diff_artifact(""))
    diff_envelopes.append(resolve_diff_artifact(123))
    artifact = Path(diff_envelopes[0]["artifact_path"])
    original = artifact.read_bytes()
    artifact.write_bytes(original + b"\n")
    try:
        diff_envelopes.append(resolve_diff_artifact( diff_envelopes[0]["artifact_id"]))
        assert diff_envelopes[-1]["error"] == "artifact_integrity_mismatch"
    finally:
        artifact.write_bytes(original)
    envelopes.extend(diff_envelopes)
    assert len(envelopes) == 273
    return root, envelopes

def _extended_envelopes(monkeypatch):
    root = _fresh_root("nr_s6_r10_ext_")
    blocker_root = _fresh_root("nr_s6_r10_extb_")
    blocker = blocker_root / "b"
    blocker.write_text("not a directory", encoding="utf-8")
    rows = []
    for capture_name, captured in CAPTURES.items():
        for padding_name, padding in PADDINGS.items():
            arguments = { "predicted": TRACE if capture_name == "identity" else BASE_PRED, "captured": captured, }
            if padding is not None:
                arguments["padding"] = padding
                arguments["cann_version"] = "9.0"
            full = attribute_diff(arguments["predicted"], arguments["captured"],
                                  padding=padding, cann_version=arguments.get("cann_version"))
            for mode, short_root_value in (("normal", root), ("failed", blocker)):
                monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(short_root_value))
                envelope = diff_mcp(arguments)
                rows.append((f"{capture_name}/{padding_name}/{mode}", envelope, full, mode))
    assert len(rows) == 48
    return root, blocker_root, rows

def test_r10_bpe_all_321_envelopes_are_bounded_and_real(monkeypatch):
    import shutil
    root, envelopes = _declared_envelopes(monkeypatch)
    ext_root, blocker_root, extended = _extended_envelopes(monkeypatch)
    try:
        for label, envelope, full, mode in extended:
            assert estimate_tokens(envelope) <= 250, label
            assert envelope["receipt"]["within_budget"] is True, label
            if mode == "normal":
                assert envelope["persisted"] is True, label
                assert envelope["retrievable"] is True, label
            else:
                assert envelope["status"] == "delivery_failed", label
                assert envelope["error"] == "artifact_persistence_failed", label
        for envelope in envelopes:
            assert estimate_tokens(envelope) <= 250
            assert envelope["receipt"]["within_budget"] is True
        try:
            import tiktoken
        except ImportError:
            tiktoken = None
        if tiktoken is not None:
            encoders = [tiktoken.get_encoding(name) for name in ("cl100k_base", "o200k_base")]
            for envelope in envelopes:
                text = _compact(envelope)
                real = max(len(encoder.encode(text)) for encoder in encoders)
                assert estimate_tokens(envelope) >= real
                assert real <= 250
            for label, envelope, _full, _mode in extended:
                text = _compact(envelope)
                real = max(len(encoder.encode(text)) for encoder in encoders)
                assert estimate_tokens(envelope) >= real, label
                assert real <= 250, label
    finally:
        for path in (root, ext_root, blocker_root):
            shutil.rmtree(path, ignore_errors=True)

def test_r10_bpe_pathological_hex_unicode_punctuation_is_upper_bounded():
    try:
        import tiktoken
    except ImportError:
        return
    encoders = [tiktoken.get_encoding(name) for name in ("cl100k_base", "o200k_base")]
    samples = [
        "a1B2c3D4" * 128, "0123456789abcdef" * 128, "a" * 2048, "A" * 2048, "Ab" * 1024, "界" * 512, "😀" * 512,
        "{}[]:,\"/-_.'" * 128, ("custom_parameter_" + "Q7z9_" * 256)[:1024], ]
    for sample in samples:
        real = max(len(encoder.encode(sample)) for encoder in encoders)
        assert estimate_tokens(sample) >= real
    for sample in samples:
        envelope = bounded_envelope({ "schema": "s", "status": "blocked", "error": "invalid_request",
            "issues": [{"code": "invalid_attribute_type", "name": sample}]}, limit=250)
        assert estimate_tokens(envelope) <= 250
        assert envelope["receipt"]["within_budget"] is True

# ------------------------------------------------------- R10-MCP-LONG/LITERAL
@pytest.mark.parametrize("size", [256, 1024, 4096])
def test_r10_mcp_long_issue_name_is_shed_whole(monkeypatch, size):
    root = _fresh_root("nr_s6_r10_long_")
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
    import shutil
    try:
        key = ("custom_parameter_" + "Q7z9_" * size)[:size]
        envelope = repro_mcp("repro_op", {"occurrence": {
            "schema": "mindie-op-occurrence-binding/2", "status": "blocked",
            "issues": [{"code": "invalid_attribute_type", "name": key}]}})
        assert envelope["status"] == "blocked"
        assert estimate_tokens(envelope) <= 250
        assert envelope["receipt"]["within_budget"] is True
        assert envelope["issues"][0]["code"] == "invalid_attribute_type"
        assert "Q7z9_" not in _compact(envelope)
    finally:
        shutil.rmtree(root, ignore_errors=True)

def test_r10_mcp_long_custom_key_never_exceeds_limit():
    key = ("custom_parameter_" + "Q7z9_" * 1024)[:1024]
    envelope = bounded_envelope({ "schema": "mindie-diff-trace-mcp/1", "status": "delivery_failed",
        "error": "artifact_persistence_failed", "reconciled": True,
        "persisted": False, "retrievable": False, key: 1}, limit=250)
    assert estimate_tokens(envelope) <= 250
    assert envelope["receipt"]["within_budget"] is True
    assert key not in _compact(envelope)
    long_error = bounded_envelope({
        "schema": "s", "status": "blocked", "error": "E" * 4096, "issues": [{"code": "C" * 4096}]}, limit=250)
    assert estimate_tokens(long_error) <= 250
    assert long_error["error"] == "invalid_request"
    assert long_error["issues"] == [{"code": "invalid_occurrence_issue"}]

@pytest.mark.parametrize("handle", ["...", "sha256:bad..."])
def test_r10_mcp_literal_ellipsis_handles_are_structured(handle):
    envelope = diff_mcp({"resolve_artifact_id": handle})
    assert envelope["status"] == "invalid_request"
    assert envelope["error"] == "invalid_artifact_id"
    assert envelope.get("artifact_id") is None
    assert "..." not in _compact(envelope).replace(handle, "")
    assert estimate_tokens(envelope) <= 250

def test_r10_mcp_literal_ellipsis_in_a_valid_root(monkeypatch):
    root = tempdir("r10...-", directory="/tmp")
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
    import shutil
    try:
        envelope = diff_mcp({"predicted": TRACE, "captured": TRACE})
        assert envelope["status"] == "reconciled"
        assert envelope["persisted"] is True
        assert "..." in envelope["artifact_path"]
        assert estimate_tokens(envelope) <= 250
    finally:
        shutil.rmtree(root, ignore_errors=True)

# ---------------------------------------------------------- R10-DELIVERY
@pytest.mark.parametrize("name", ["manifest.json", "MANIFEST.JSON", "Manifest.json", "manifest.JSON"])
def test_r10_delivery_case_alias_is_blocked_before_any_write(monkeypatch, name):
    base = _fresh_root("nr_s6_r10_case_")
    root = base / "cas"
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
    import shutil
    try:
        output = base / "out" / name
        result = generate_single_op_reproducer( paged_binding(), iters=7, output_path=output)
        assert result["status"] == "delivery_blocked"
        assert result["error"] == "colliding_output_paths"
        assert result["storage"]["status"] == "delivery_blocked"
        assert result["storage"]["artifact_path"] is None
        assert not checked_io.path_exists(root)
        assert not checked_io.path_exists(output.parent)
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_r10_delivery_cli_case_alias_exits_4_without_writes(monkeypatch):
    base = _fresh_root("nr_s6_r10_casecli_")
    root = base / "cas"
    row = base / "binding.json"
    row.write_text(json.dumps(paged_binding()), encoding="utf-8")
    output = base / "out" / "MANIFEST.JSON"
    env = dict(os.environ)
    env["MINDIE_REPRO_SHORT_ROOT"] = str(root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")])
    import shutil
    try:
        completed = _child(["repro-op", "--row", str(row), "--iters", "7", "--output", str(output)], env)
        assert completed.returncode == 4
        result = json.loads(completed.stdout)
        assert result["status"] == "delivery_blocked"
        assert result["error"] == "colliding_output_paths"
        assert not checked_io.path_exists(root)
        assert not checked_io.path_exists(output.parent)
    finally:
        shutil.rmtree(base, ignore_errors=True)

# ---------------------------------------------------------- module hygiene
def test_r10_every_scoped_python_file_is_below_730_lines():
    markers = ("[Role]", "[Invariants]", "[Timeouts & Locks]", "[Side-Effect Policy]")
    for relative in SCOPED:
        import mindie_sim
        sim_root = Path(mindie_sim.__file__).resolve().parents[1]
        path = (PROJECT_ROOT / relative) if (PROJECT_ROOT / relative).exists() else (sim_root / relative)
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        lines = text.splitlines()
        assert len(lines) < 725, (relative, len(lines))
        assert b"\r" not in raw, relative
        first = lines[:30]
        for marker in markers:
            assert any(marker in line for line in first), (relative, marker)
    storage = (PROJECT_ROOT / "mindie_profiler/repro_storage.py").read_text()
    atomic = (PROJECT_ROOT / "mindie_profiler/repro_atomic_io.py").read_text()
    assert len(storage.splitlines()) < 660
    assert len(atomic.splitlines()) < 420

def _r18_inside(function):
    frame = sys._getframe(1)
    while frame:
        if frame.f_code is function.__code__:
            return True
        frame = frame.f_back
    return False

@pytest.mark.parametrize("route,leaf,secondary_type", [ (route, leaf, secondary) for route in ("cas", "caller")
    for leaf in ("repro.py", "manifest.json") for secondary in (None, TimeoutError, PermissionError)])
def test_r18_pair_reader_primary(route, leaf, secondary_type, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "cas")
    primary, hits, handles = TimeoutError("pair read"), [], []
    opened, fds = _r16_fds()
    fdopen = os.fdopen
    class Reader(io.BufferedReader):
        def read(self, *args):
            hits.append("read")
            raise primary
        def close(self):
            super().close()
            hits.append("close")
            if secondary_type:
                raise secondary_type("close")
    def wrap(fd, *args, **kwargs):
        path = fds[fd][0].absolute()
        selected = path.parent == tmp_path / "out" if route == "caller" else tmp_path / "cas" in path.parents
        if selected and path.name == leaf and _r18_inside(storage._verify_pair_bytes) and not hits:
            stream = Reader(io.FileIO(fd, "rb", closefd=True))
            handles.append(stream)
            return stream
        return fdopen(fd, *args, **kwargs)
    request = {"occurrence": paged_binding(), "iters": 18001}
    if route == "caller":
        request["output_dir"] = "out"
    with patch.object(os, "open", opened), patch.object(os, "fdopen", wrap):
        with pytest.raises(TimeoutError) as caught:
            repro_mcp("repro_op", request)
    assert caught.value is primary and hits == ["read", "close"] and all(h.closed for h in handles)
    _r16_no_leaks(fds, tmp_path)

@pytest.mark.parametrize("fault_type", [TimeoutError, PermissionError])
def test_r18_fdopen_failure_closes_raw_fd(fault_type, tmp_path):
    path = tmp_path / "reader"
    path.write_bytes(b"old")
    opened, fds = _r16_fds()
    primary = fault_type("fdopen")
    with patch.object(os, "open", opened), patch.object(os, "fdopen", side_effect=primary):
        if fault_type is TimeoutError:
            with pytest.raises(TimeoutError) as caught:
                checked_io.read_regular_bytes(path)
            assert caught.value is primary
        else:
            assert checked_io.read_regular_bytes(path) is None
    _r16_no_leaks(fds, tmp_path)

@pytest.mark.parametrize("exclusive", [False, True])
@pytest.mark.parametrize("kind", ["same_bytes", "fifo", "missing", "symlink", "success"])
def test_r18_opened_destination_identity(exclusive, kind, tmp_path):
    target, foreign = tmp_path / "leaf", tmp_path / "foreign"
    if kind == "fifo":
        os.mkfifo(foreign, 0o600)
    else:
        foreign.write_bytes(b"payload")
    tracked, fds = _r16_fds()
    hits, reads, receipts, read = [], [], [], os.read
    def opened(path, flags, *args, **kwargs):
        if Path(path) == target and not flags & os.O_WRONLY and not hits:
            assert flags & os.O_NONBLOCK
            hits.append(path)
            if kind in ("same_bytes", "fifo"):
                os.replace(foreign, target)
            elif kind in ("missing", "symlink"):
                os.unlink(target)
                if kind == "symlink":
                    target.symlink_to(foreign)
        return tracked(path, flags, *args, **kwargs)
    def reading(fd, *args):
        if fds[fd][0] == target:
            reads.append(fd)
        return read(fd, *args)
    with patch.object(os, "open", opened), patch.object(os, "read", reading):
        assert checked_io.safe_write_bytes(target, b"payload", exclusive=exclusive,
                                            installed_out=receipts) is (kind == "success")
    assert hits and bool(reads) is (kind == "success") and bool(receipts) is (kind == "success")
    if kind == "success":
        assert receipts == [checked_io.inode_of(target)] and target.read_bytes() == b"payload"
    if kind == "fifo":
        assert stat.S_ISFIFO(os.lstat(target).st_mode)
    _r16_no_leaks(fds, tmp_path)

@pytest.mark.parametrize("boundary,foreign", [
    (boundary, foreign) for boundary in ("before", "after", "outer") for foreign in (False, True)
] + [("inner_false", False), ("outer_false", False), ("success", False)])
def test_r18_caller_prior_pair(boundary, foreign, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "cas")
    out = tmp_path / "out"
    out.mkdir()
    script, manifest, other = out / "repro.py", out / "manifest.json", tmp_path / "foreign"
    script.write_bytes(b"old script\n")
    manifest.write_bytes(b'{"owner":"old"}\n')
    other.write_bytes(b"foreign")
    before = {p: (checked_io.inode_of(p), p.read_bytes()) for p in (script, manifest)}
    foreign_inode, primary, hits = checked_io.inode_of(other), TimeoutError(boundary), []
    opened, fds = _r16_fds()
    replace, verify = os.replace, storage._verify_pair_bytes
    def fail():
        hits.append(boundary)
        if foreign:
            replace(other, script)
        if boundary.endswith("false"):
            return False
        raise primary
    def replacing(src, dst, *args, **kwargs):
        if Path(dst) == manifest and boundary in ("before", "after") and not hits:
            if boundary == "after":
                replace(src, dst, *args, **kwargs)
            fail()
        return replace(src, dst, *args, **kwargs)
    def verifying(a, b, *args):
        inside = _r18_inside(storage._write_caller_pair)
        if (Path(a).absolute() == script and not hits and
                ((boundary.startswith("outer") and not inside) or (boundary == "inner_false" and inside))):
            return fail()
        return verify(a, b, *args)
    with patch.object(os, "open", opened), patch.object(os, "replace", replacing), \
            patch.object(storage, "_verify_pair_bytes", verifying):
        request = {"occurrence": paged_binding(), "iters": 18002, "output_dir": "out"}
        if boundary in ("before", "after", "outer"):
            with pytest.raises(TimeoutError) as caught:
                repro_mcp("repro_op", request)
            assert caught.value is primary
        else:
            assert repro_mcp("repro_op", request)["status"] == ("ready" if boundary == "success" else "delivery_failed")
    after = {p: (checked_io.inode_of(p), p.read_bytes()) for p in before}
    recovery = list(out.glob("*.rollback"))
    if foreign:
        assert after[script] == (foreign_inode, b"foreign") and after[manifest] == before[manifest]
        assert len(recovery) == 1 and (checked_io.inode_of(recovery[0]), recovery[0].read_bytes()) == before[script]
    else:
        assert not recovery and (after == before) is (boundary != "success")
    assert not list(out.glob("*.backup")) and bool(hits) is (boundary != "success")
    _r16_no_leaks(fds, tmp_path)

@pytest.mark.parametrize("mode", ["foreign", "two_writers"])
@pytest.mark.parametrize("secondary_type", [None, TimeoutError, PermissionError])
def test_r18_abort_staging_ownership(mode, secondary_type, tmp_path):
    _, ns = _r16_probe_namespace()
    target, foreign = tmp_path / "abort.json", tmp_path / "foreign"
    ns["ABORT_FILE"] = str(target)
    foreign.write_bytes(b"foreign")
    foreign_inode, primary = checked_io.inode_of(foreign), TimeoutError("write")
    a_closed, b_ready, a_done = threading.Event(), threading.Event(), threading.Event()
    handles, paths, caught = [], {}, []
    real_open, replace = builtins.open, os.replace
    class Writer(io.TextIOWrapper):
        def write(self, data):
            raise primary
        def close(self):
            super().close()
            if mode == "foreign":
                replace(foreign, self.name)
            else:
                a_closed.set()
                assert b_ready.wait(10)
            if secondary_type:
                raise secondary_type("close")
    def opened(path, *args, **kwargs):
        name = threading.current_thread().name
        paths[name] = next(p for p in tmp_path.glob("*.tmp") if p.stat().st_ino == os.fstat(path).st_ino)
        if mode == "foreign" or name == "r18-A":
            h = Writer(real_open(path, "xb", buffering=0, opener=kwargs["opener"]), encoding="utf-8")
        else:
            h = real_open(path, *args, **kwargs)
        handles.append(h)
        return h
    def replacing(src, dst):
        if threading.current_thread().name == "r18-B":
            before = checked_io.inode_of(src)
            b_ready.set()
            assert a_done.wait(10) and checked_io.inode_of(src) == before
        return replace(src, dst)
    def a_call():
        try:
            ns["_write_abort_record"]({"rank": 0})
        except TimeoutError as exc:
            caught.append(exc)
        finally:
            a_done.set()
    with patch.object(builtins, "open", opened), patch.object(os, "replace", replacing):
        if mode == "foreign":
            a_call()
        else:
            a = threading.Thread(target=a_call, name="r18-A")
            b = threading.Thread(target=lambda: ns["_write_abort_record"]({"rank": 1}), name="r18-B")
            a.start()
            assert a_closed.wait(10)
            b.start()
            a.join(10)
            b.join(10)
            assert not a.is_alive() and not b.is_alive() and paths["r18-A"] != paths["r18-B"]
    assert caught == [primary] and handles and all(h.closed for h in handles)
    if mode == "foreign":
        path = next(iter(paths.values()))
        assert checked_io.inode_of(path) == foreign_inode and path.read_bytes() == b"foreign"
    else:
        assert json.loads(target.read_text()) == {"rank": 1} and not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.rollback"))

def test_r18_abort_success_preserves_legacy_temporary(tmp_path):
    _, ns = _r16_probe_namespace()
    target, legacy = tmp_path / "abort.json", tmp_path / "abort.json.tmp"
    ns["ABORT_FILE"] = str(target)
    legacy.write_bytes(b"legacy foreign")
    for rank in (0, 1):
        ns["_write_abort_record"]({"rank": rank})
        assert json.loads(target.read_text()) == {"rank": rank}
    assert legacy.read_bytes() == b"legacy foreign" and sorted(p.name for p in tmp_path.iterdir()) == [target.name, legacy.name]
