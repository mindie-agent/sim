"""Round-11 closure: provable token bounds, path transitions, ABI continuation.

[Role]
  Exercise the seven Round-11 findings against the public helpers: the real
  tiktoken upper-bound matrix, the guaranteed-fitting terminal envelope, the
  33/48-byte root path transition, the compact ABI catalog span round trip, the
  protected-slot zero-creation rule, cached-diff policy checks and multi-handle
  index persistence.

[Invariants]
  * Every dynamic/caller/path/digest byte is charged, so the estimate is at or
    above both real tokenizers on every generated case.
  * A bounded envelope always reports ``within_budget=True`` and never exceeds
    its limit, including the hostile emoji terminal example.
  * An unrepresentable path is a structured ``delivery_blocked`` transition,
    not an ``AssertionError``.
  * Protected slot ancestry is rejected before any directory creation.
  * The persisted index contains every successfully registered handle.

[Timeouts & Locks]
  In-process filesystem writes under per-test temporary roots; no device or
  subprocess work.

[Side-Effect Policy]
  Writes only under temporary directories created by this module and removes
  them in ``finally`` blocks.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
__import__("pytest").importorskip("mindie_profiler")
from mindie_profiler import repro_atomic_io as checked_io
import shutil
import builtins
import io
import sys
import threading
from unittest.mock import patch

import pytest

from mindie_profiler.op_api_registry import (
    call_mcp_tool as registry_mcp, inspect_op_abi, load_catalog,
    resolve_op_abi_catalog_span,
)
from mindie_profiler.repro_op import (
    call_mcp_tool as repro_mcp, generate_single_op_reproducer,
)
from mindie_profiler.repro_storage import resolve_repro_artifact, short_root
from mindie_profiler.token_budget import bounded_envelope, estimate_tokens
from mindie_sim.diff_trace import (
    attribute_diff, call_mcp_tool as diff_mcp, resolve_diff_artifact,
)
from nr_s6_support import paged_binding
from mindie_profiler import repro_storage as storage
from temp_cleanup import tempdir
from test_nr_s6_round8 import _r16_fds, _r16_no_leaks, _r16_probe_namespace


def _compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _encoders():
    tiktoken = pytest.importorskip("tiktoken")
    return [tiktoken.get_encoding(name)
            for name in ("cl100k_base", "o200k_base")]


def _real(value, encoders):
    text = _compact(value)
    return max(len(encoder.encode(text)) for encoder in encoders)


def _root(prefix="nr_s6_r11_"):
    base = tempdir(prefix, directory="/tmp")
    return base, base / "cas"


def _tree(root: Path) -> dict[str, str]:
    if not checked_io.path_exists(root):
        return {}
    return {str(path.relative_to(root)):
            ("dir" if checked_io.path_is_dir(path) else
             hashlib.sha256(path.read_bytes()).hexdigest())
            for path in root.rglob("*")}


# ---------------------------------------------------------------- R11-BPE/CORE
def test_r11_bpe_360_case_matrix_is_a_real_upper_bound():
    encoders = _encoders()
    under = over = 0
    for field in ("name", "code", "slot"):
        for size in (16, 32, 48, 60, 64):
            for count in range(1, 9):
                for pattern in ("qz", "zq", "JQ"):
                    text = (pattern * size)[:size]
                    issues = [{"code": text if field == "code"
                               else "invalid_attribute_type"}
                              for _ in range(count)]
                    for index, item in enumerate(issues):
                        if field == "name":
                            item["name"] = text
                        if field == "slot":
                            item["name"] = text
                            if index % 2 == 0:
                                item["slot"] = text
                    envelope = repro_mcp("repro_op", {"occurrence": {
                        "schema": "mindie-op-occurrence-binding/2",
                        "status": "blocked", "issues": issues}})
                    estimate = estimate_tokens(envelope)
                    real = _real(envelope, encoders)
                    assert envelope["receipt"]["within_budget"] is True
                    assert estimate <= 250
                    assert real <= 250
                    if real > estimate:
                        under += 1
                    if real > 250:
                        over += 1
    assert (under, over) == (0, 0)


def test_r11_core_terminal_envelope_is_always_measured_and_fitting():
    encoders = _encoders()
    payloads = [
        {key: "😀" * 64 for key in ("schema", "status", "error")},
        {"schema": "mindie-diff-trace-mcp/1", "status": "delivery_failed",
         "error": "😀" * 64, "reconciled": True, "persisted": False,
         "retrievable": False},
        {"schema": "s", "status": "blocked", "detail": "qz" * 512},
    ]
    for payload in payloads:
        envelope = bounded_envelope(payload, limit=250)
        real = _real(envelope, encoders)
        assert estimate_tokens(envelope) <= 250
        assert real <= 250
        if envelope.get("error") == "envelope_budget_exceeded":
            assert real <= envelope["receipt"]["token_estimate"] <= 250
        assert envelope["receipt"]["within_budget"] is True
        if "error" in envelope:
            assert envelope["error"] in {
                "envelope_budget_exceeded",
                "artifact_paths_not_representable", "invalid_request"}
    hostile = bounded_envelope(
        {key: "界" * 64 for key in ("schema", "status", "error")}, limit=250)
    assert hostile["schema"] == "mindie-mcp-envelope/1"
    assert hostile["status"] == "blocked"
    assert hostile["receipt"]["within_budget"] is True


# ---------------------------------------------------------------- R11-RESOLVE
def test_r11_resolve_long_path_is_a_structured_transition(monkeypatch):
    path = "/tmp/" + "x" * 4096 + "/repro.py"
    envelope = bounded_envelope({
        "schema": "s", "status": "ready", "artifact_path": path,
        "retrievable": True,
        "artifact": {"retrievable": True, "sha256": "sha256:" + "b" * 64},
    }, limit=250)
    assert envelope["status"] == "delivery_blocked"
    assert envelope["error"] == "artifact_paths_not_representable"
    assert envelope["retrievable"] is False
    assert envelope["artifact"]["retrievable"] is False
    assert envelope.get("artifact_path_omitted") is True
    assert envelope.get("artifact_path") is None
    assert estimate_tokens(envelope) <= 250


@pytest.mark.parametrize("root_length", [33, 48])
def test_r11_resolve_33_and_48_byte_roots_do_not_raise(
        monkeypatch, root_length):
    base = tempdir("nr_s6_r11root_", directory="/tmp")
    root = base / ("r" * max(1, root_length - len(str(base)) - 1))
    assert len(str(root)) == root_length
    try:
        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
        result = generate_single_op_reproducer(
            paged_binding(), iters=103, stage=True)
        assert result["status"] == "ready"
        direct = resolve_repro_artifact(result["sha256"])
        assert direct["resolved"] is True
        assert direct["retrievable"] is True
        envelope = repro_mcp("repro_op",
                             {"resolve_artifact_id": result["sha256"]})
        assert estimate_tokens(envelope) <= 250
        assert envelope["receipt"]["within_budget"] is True
        if root_length == 48:
            assert envelope["status"] == "delivery_blocked"
            assert envelope["error"] == "artifact_paths_not_representable"
            assert envelope["resolved"] is False
            assert envelope["retrievable"] is False
            assert envelope.get("artifact_path") is None
        else:
            assert envelope["resolved"] is True
            assert envelope["retrievable"] is True
            assert checked_io.path_is_file(Path(envelope["artifact_path"]))
            assert checked_io.path_is_file(Path(envelope["manifest_path"]))
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ---------------------------------------------------------------- R11-ABI-REF
def test_r11_abi_catalog_span_round_trip_and_full_direct_card():
    encoders = _encoders()
    document = load_catalog()
    spans = {}
    for name in document["operators"]:
        envelope = registry_mcp("inspect_op_abi", {"op_name": name})
        artifact = envelope["artifact"]
        assert artifact["kind"] == "catalog_span"
        assert len(artifact["catalog_sha256_16"]) == 16
        assert len(artifact["span"]) == 2
        assert estimate_tokens(envelope) <= 250
        assert _real(envelope, encoders) <= 250
        assert envelope["receipt"]["within_budget"] is True
        spans[name] = tuple(artifact["span"])
        full = resolve_op_abi_catalog_span(
            artifact["span"],
            catalog_sha256_16=artifact["catalog_sha256_16"])
        assert full["api"] == name
        assert full["constraints"]["slot_specs"]
        assert full["padding_semantics"]["mode"]
        assert tuple(full["span"]) == tuple(artifact["span"])
        assert full["entry"]["abi_id"] == full["abi_id"]
    assert len(spans) == 32
    direct = inspect_op_abi("aclnnFlashAttentionVarLenScore",
                            cann_version="8.0")
    card = direct["card"]
    for key in ("api", "abi_id", "inputs", "outputs", "attrs", "recipe",
                "qualification", "padding", "signature", "complete"):
        assert key in card
    assert card["api"] == "aclnnFlashAttentionVarLenScore"
    assert card["abi_id"].startswith("sha256:")
    assert card["recipe"]["call"]
    assert direct["artifact"]["kind"] == "catalog_span"
    assert direct["receipt"]["within_budget"] is True


# --------------------------------------------------------------- R11-F10-DIRS
def test_r11_f10_dirs_protected_slot_creates_zero_directories(monkeypatch):
    base, root = _root("nr_s6_r11_dirs_")
    try:
        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
        first = repro_mcp("repro_op", {"occurrence": paged_binding(),
                                       "iters": 5})
        assert first["status"] == "ready"
        slot = Path(first["artifact_path"]).parent
        link = base / "ln"
        link.symlink_to(slot, target_is_directory=True)
        for destination in (slot, slot / "i", link / "i"):
            assert len(str(destination)) <= 48
            before = _tree(slot)
            monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(destination))
            result = generate_single_op_reproducer(
                paged_binding(), iters=11, stage=True)
            after = _tree(slot)
            assert result["status"] == "delivery_failed"
            assert result["storage"]["status"] == "delivery_failed"
            assert result["error"] == "artifact_persistence_failed"
            assert set(after) == set(before)
            assert after == before
    finally:
        shutil.rmtree(base, ignore_errors=True)


# --------------------------------------------------------------- R11-F10-DIFF
def test_r11_f10_diff_cached_bytes_never_bypass_policy(monkeypatch):
    base, root = _root("nr_s6_r11_diff_")
    try:
        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
        first = repro_mcp("repro_op", {"occurrence": paged_binding(),
                                       "iters": 5})
        assert first["status"] == "ready"
        slot = Path(first["artifact_path"]).parent
        trace = {"total_ms": 2, "ops": [{"name": "A", "duration_ms": 2}]}
        normal = diff_mcp({"predicted": trace, "captured": trace})
        assert normal["status"] == "reconciled"
        assert normal["persisted"] is True
        artifact = Path(normal["artifact_path"])
        data = artifact.read_bytes()
        alias = base / "alias"
        (alias / "diff").mkdir(parents=True)
        (alias / "diff" / artifact.parent.name).symlink_to(
            artifact.parent, target_is_directory=True)
        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(alias))
        symlinked = diff_mcp({"predicted": trace, "captured": trace})
        assert symlinked["status"] == "delivery_failed"
        assert symlinked["error"] == "artifact_persistence_failed"
        assert symlinked["persisted"] is False
        resolved = resolve_diff_artifact(normal["artifact_id"])
        assert resolved["resolved"] is False
        assert resolved["retrievable"] is False
        protected = base / "protected"
        (protected / "diff").mkdir(parents=True)
        (protected / "diff" / artifact.parent.name).symlink_to(
            slot, target_is_directory=True)
        (slot / "diff.json").write_bytes(data)
        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(protected))
        cached = diff_mcp({"predicted": trace, "captured": trace})
        assert cached["status"] == "delivery_failed"
        assert cached["persisted"] is False
        assert resolve_diff_artifact(normal["artifact_id"])["resolved"] is False
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ------------------------------------------------------------------ R11-INDEX
def test_r11_index_persists_every_registered_handle(monkeypatch):
    base, root = _root("nr_s6_r11_idx_")
    try:
        monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
        handles = []
        for iters in (5, 7, 9):
            result = repro_mcp("repro_op", {"occurrence": paged_binding(),
                                            "iters": iters})
            assert result["status"] == "ready"
            handles.append(result["artifact_id"])
        assert len(set(handles)) == 3
        index = json.loads((root / "index.json").read_text())
        assert set(handles) <= set(index)
        for handle in handles:
            assert index[handle]["artifact_id"] == handle
        assert resolve_repro_artifact(handles[-1])["resolved"] is True
        assert short_root() == root
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.mark.parametrize("mode", ["thread", "reentrant"])
@pytest.mark.parametrize("boundary", ["manifest_read", "pair_open"])
def test_r17_committed_pair_survives_installer_timeout(mode, boundary, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "cas")
    request = {"occurrence": paged_binding(), "iters": 17001}
    primary, hits, outcome, before = TimeoutError(boundary), [], {}, {}
    gate, release = threading.Event(), threading.Event()
    tracked, fds = _r16_fds()
    original_read, caller_a = os.read, threading.current_thread()
    def caller_b():
        result = repro_mcp("repro_op", request)
        assert result["status"] == "ready"
        outcome["b"] = result
        assert resolve_repro_artifact(result["artifact_id"])["retrievable"]
        for key in ("artifact_path", "manifest_path"):
            path = Path(result[key])
            before[path] = checked_io.inode_of(path), path.read_bytes()
    def pause(path):
        hits.append(path)
        gate.set()
        if mode == "reentrant":
            caller_b()
        else:
            assert release.wait(10)
        raise primary
    def opened(path, flags, *args, **kwargs):
        if (boundary == "pair_open" and threading.current_thread() is caller_a
                and not hits and Path(path).name == "repro.py" and not flags & os.O_WRONLY):
            frame = sys._getframe(1)
            while frame:
                if frame.f_code is storage._verify_pair_bytes.__code__:
                    pause(path)
                frame = frame.f_back
        return tracked(path, flags, *args, **kwargs)
    def read(fd, *args):
        if (boundary == "manifest_read" and threading.current_thread() is caller_a
                and not hits and fds[fd][0].name == "manifest.json"):
            pause(fds[fd][0])
        return original_read(fd, *args)
    def invoke():
        with pytest.raises(TimeoutError) as caught:
            repro_mcp("repro_op", request)
        outcome["a"] = caught.value
    with patch.object(os, "open", opened), patch.object(os, "read", read):
        if mode == "thread":
            caller_a = threading.Thread(target=invoke)
            caller_a.start()
            try:
                assert gate.wait(10)
                caller_b()
            finally:
                release.set()
                caller_a.join(10)
            assert not caller_a.is_alive()
        else:
            invoke()
    assert hits and outcome["a"] is primary
    assert {p: (checked_io.inode_of(p), p.read_bytes()) for p in before} == before
    resolved = resolve_repro_artifact(outcome["b"]["artifact_id"])
    assert resolved["resolved"] and resolved["retrievable"]
    _r16_no_leaks(fds, tmp_path)
    assert not list(tmp_path.rglob("*.rollback"))


@pytest.mark.parametrize("external,write_fault,secondary_type", [
    (external, True, secondary) for external in (False, True)
    for secondary in (None, TimeoutError, PermissionError)
] + [(False, False, TimeoutError)])
def test_r17_emitted_abort_close_preserves_primary(
        external, write_fault, secondary_type, tmp_path):
    _, ns = _r16_probe_namespace()
    target = tmp_path / "abort.json"
    ns["ABORT_FILE"] = str(target)
    primary, outer = TimeoutError("write"), ValueError("outer")
    secondary = secondary_type("close") if secondary_type else None
    handles, hits, real_open = [], [], builtins.open
    class FaultyText(io.TextIOWrapper):
        def write(self, data):
            hits.append("write")
            if write_fault:
                raise primary
            return super().write(data)
        def close(self):
            super().close()
            hits.append("close")
            if secondary is not None:
                raise secondary
    def opened(path, *args, **kwargs):
        handle = FaultyText(real_open(path, "wb", buffering=0), encoding="utf-8")
        handles.append(handle)
        return handle
    with patch.object(builtins, "open", opened):
        with pytest.raises((ValueError, TimeoutError)) as caught:
            if external:
                try:
                    raise outer
                finally:
                    ns["_write_abort_record"]({"kind": "abort"}, outer)
            else:
                ns["_write_abort_record"]({"kind": "abort"})
    assert caught.value is (outer if external else primary if write_fault else secondary)
    assert hits == ["write", "close"] and all(h.closed for h in handles)
    assert not list(tmp_path.iterdir())


def test_r17_incomplete_pair_cannot_commit(tmp_path):
    script, manifest = tmp_path / "repro.py", tmp_path / "manifest.json"
    script.write_bytes(b"provisional")
    inode = checked_io.inode_of(script)
    storage._remember_inodes(script, manifest)
    assert inode not in storage._PROTECTED_INODES
    checked_io._unlink_owned_path(script, inode)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("recipe,value", [("MoeInitRoutingV2", 0), ("MoeInitRoutingV2", 1),
    ("aclnnDynamicQuant", "dynamic"), ("aclnnDynamicQuant", "static")])
def test_r23_operator_scalar_schema(recipe, value, tmp_path, monkeypatch):
    import subprocess
    from test_nr_s6_round2_closure import _ready_bindings
    from mindie_profiler.repro_op import _compact_occurrence
    from mindie_profiler.repro_preconditions import precondition_problems, _HANDLERS
    binding = _ready_bindings()[recipe]
    binding["attrs"]["quant_mode"] = value
    occurrence = _compact_occurrence(binding)
    assert not precondition_problems(occurrence)
    handler = _HANDLERS[occurrence["preconditions"]["kind"]]
    assert not handler(occurrence, occurrence["attrs"], occurrence["allocations"],
        {a["name"]: a for a in occurrence["allocations"]}, occurrence["call_plan"], occurrence["preconditions"])
    assert generate_single_op_reproducer(binding)["status"] == "ready"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "r23scalar")
    assert repro_mcp("repro_op", {"occurrence": binding})["status"] == "ready"
    row = tmp_path / "row.json"
    row.write_text(json.dumps(binding))
    result = subprocess.run([sys.executable, "-B", "-m", "mindie_profiler", "repro-op",
        "--row", str(row), "--dry-run"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and not result.stderr
    assert not json.loads(result.stdout)["failed_checks"]


@pytest.mark.parametrize("recipe,role", [("aclnnGroupedMatmulV4", "group_list"),
    ("aclnnPagedAttention", "block_table"), ("MoeInitRoutingV2", "index")])
@pytest.mark.parametrize("case", ["valid", "empty", "long", "semantic", "geometry", "duplicate_name"])
@pytest.mark.parametrize("first", [False, True])
def test_r23_every_metadata_allocation(recipe, role, case, first):
    import copy
    import math
    from test_nr_s6_round2_closure import _ready_bindings
    from mindie_profiler.repro_op import _compact_occurrence
    from mindie_profiler.repro_metadata import synthesize_metadata
    from mindie_profiler.repro_preconditions import precondition_problems, _HANDLERS
    occurrence = _compact_occurrence(_ready_bindings()[recipe])
    target = next(a for a in occurrence["allocations"] if a["role"] == role and a["direction"] == "input")
    valid = next(i["values"] for i in occurrence["metadata"]["install"] if i["tensor"] == target["name"])
    extra = copy.deepcopy(target)
    extra["name"] = target["name"] if case == "duplicate_name" else "extra_" + target["name"]
    extra["values"] = ([] if case == "empty" else valid + [0] if case == "long" else
        [268435456] * math.prod(extra["shape"]) if case == "semantic" else list(valid))
    if case == "geometry":
        extra["shape"][0] += 1
    occurrence["allocations"].insert(0 if first else len(occurrence["allocations"]), extra)
    result = synthesize_metadata(occurrence)
    problems = precondition_problems(occurrence)
    handler = _HANDLERS[occurrence["preconditions"]["kind"]]
    handler_problems = handler(occurrence, occurrence["attrs"], occurrence["allocations"],
        {a["name"]: a for a in occurrence["allocations"]}, occurrence["call_plan"], occurrence["preconditions"])
    if case == "valid":
        assert result["checks"]["metadata_installable"] and not problems and not handler_problems
        for name in (target["name"], extra["name"]):
            item = next(i for i in result["install"] if i["tensor"] == name)
            assert item["values"] == valid
        assert next(i for i in result["install"] if i["tensor"] == extra["name"])["captured_valid"]
    else:
        assert result["checks"]["metadata_installable"] is False
        assert result["violations"] and problems and handler_problems


@pytest.mark.parametrize("recipe,name", [("MoeInitRoutingV2", "expanded_row_idx_out"),
    ("MoeGatingTopKSoftmax", "expert_idx")])
@pytest.mark.parametrize("case", ["valid", "short", "long", "semantic", "inflated_experts"])
def test_r23_output_metadata_capacity(recipe, name, case):
    import math
    from test_nr_s6_round2_closure import _ready_bindings
    from mindie_profiler.repro_op import _compact_occurrence
    from mindie_profiler.repro_metadata import synthesize_metadata
    from mindie_profiler.repro_preconditions import precondition_problems
    occurrence = _compact_occurrence(_ready_bindings()[recipe])
    spec = next(a for a in occurrence["allocations"] if a["name"] == name)
    count = math.prod(spec["shape"])
    spec["values"] = [] if case == "short" else [0] * (count + 1) if case == "long" else [
        268435456 if case == "semantic" else 0] * count
    if case == "inflated_experts":
        occurrence["attrs"]["expert_num"] = 1024
        spec["values"] = [count] * count
    result = synthesize_metadata(occurrence)
    if case == "valid":
        assert result["checks"]["metadata_installable"] and not precondition_problems(occurrence)
        installed = next(i for i in result["install"] if i["tensor"] == name)
        assert installed["captured_valid"] and installed["values"] == spec["values"]
    else:
        assert not result["checks"]["metadata_installable"]
        assert result["violations"] and precondition_problems(occurrence)
        assert next(i for i in result["install"] if i["tensor"] == name)["captured_valid"] is False


@pytest.mark.parametrize("position", ["before", "after"])
@pytest.mark.parametrize("invalid_target", ["forwarded", "extra"])
@pytest.mark.parametrize("value", [256, 257, 2048])
def test_r24_all_sequence_capacities(position, invalid_target, value, tmp_path, monkeypatch):
    import copy
    import subprocess
    from nr_s6_support import paged_binding
    from mindie_profiler.events import ProfilerUsageError
    from mindie_profiler.repro_op import _compact_occurrence
    from mindie_profiler.repro_metadata import synthesize_metadata
    from mindie_profiler.repro_preconditions import precondition_problems, _HANDLERS
    assert sys.get_int_max_str_digits() == 4300
    binding = paged_binding()
    original = next(a for a in binding["inputs"] if a["name"] == "context_lens")
    extra = copy.deepcopy(original)
    extra.update(name="r24_control_context_lens", values=[128] * 16)
    original["values"] = [128] * 16
    (original if invalid_target == "forwarded" else extra)["values"] = [value] + [0] * 15
    binding["inputs"].insert(0 if position == "before" else len(binding["inputs"]), extra)
    occurrence = _compact_occurrence(binding)
    metadata = synthesize_metadata(occurrence)
    valid = value == 256
    assert metadata["checks"]["metadata_installable"] is valid
    assert bool(precondition_problems(occurrence)) is not valid
    handler = _HANDLERS[occurrence["preconditions"]["kind"]]
    assert bool(handler(occurrence, occurrence["attrs"], occurrence["allocations"],
        {a["name"]: a for a in occurrence["allocations"]}, occurrence["call_plan"], {})) is not valid
    failed_name = "context_lens" if invalid_target == "forwarded" else extra["name"]
    assert next(i for i in metadata["install"] if i["tensor"] == failed_name)["captured_valid"] is valid
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "r24seq")
    assert repro_mcp("repro_op", {"occurrence": binding})["status"] == ("ready" if valid else "blocked")
    if valid:
        assert generate_single_op_reproducer(binding)["status"] == "ready"
    else:
        with pytest.raises(ProfilerUsageError):
            generate_single_op_reproducer(binding)
    path = tmp_path / "row.json"
    path.write_text(json.dumps(binding))
    run = subprocess.run([sys.executable, "-B", "-m", "mindie_profiler", "repro-op", "--row", str(path),
        "--dry-run"], capture_output=True, text=True, timeout=30)
    assert run.returncode == (0 if valid else 2) and "Traceback" not in run.stderr
    assert sys.get_int_max_str_digits() == 4300


@pytest.mark.parametrize("case", ["rows", "rank", "cache_size", "cache_capacity", "missing_table", "narrow_table"])
@pytest.mark.parametrize("reverse", [False, True])
def test_r24_sequence_geometry_association(case, reverse):
    import copy
    from nr_s6_support import paged_binding
    from mindie_profiler.repro_op import _compact_occurrence
    from mindie_profiler.repro_metadata import synthesize_metadata
    from mindie_profiler.repro_preconditions import precondition_problems
    occurrence = _compact_occurrence(paged_binding())
    allocations = occurrence["allocations"]
    sequence = copy.deepcopy(next(a for a in allocations if a["role"] == "seq_lens"))
    sequence.update(name="extra_context_lens", values=[128] * 16)
    allocations.append(sequence)
    if case == "rows":
        sequence.update(shape=[17], values=[0] * 17)
    elif case == "rank":
        sequence["shape"] = [16, 1]
    elif case in {"cache_size", "cache_capacity"}:
        cache = copy.deepcopy(next(a for a in allocations if a["role"] == "kv_cache"))
        cache["name"] = "extra_cache"
        cache["shape"][1 if case == "cache_size" else 0] = 8 if case == "cache_size" else 1
        allocations.append(cache)
    elif case == "missing_table":
        allocations[:] = [a for a in allocations if a["role"] != "block_table"]
    else:
        table = copy.deepcopy(next(a for a in allocations if a["role"] == "block_table"))
        table.update(name="extra_table", shape=[16, 4])
        allocations.append(table)
    if reverse:
        allocations.reverse()
    metadata = synthesize_metadata(occurrence)
    assert not metadata["checks"]["metadata_installable"] and metadata["violations"]
    assert precondition_problems(occurrence)
