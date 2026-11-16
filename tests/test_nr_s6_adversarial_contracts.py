"""Adversarial contracts for NR-S6: every R1..R15 counterexample and header audit.

[Role]
  These tests cover R7..R15: metadata installation, incomplete-binding gates,
  distributed probe pairing/geometry/deadlines, fits, steady-state honesty,
  occurrence selectors, bounded receipts and documentation claims.  R1..R6
  counterexamples live in ``test_op_abi_repro_and_diff_trace.py``.

[Invariants]
  * No fixture is generated from the catalog under test; C signatures are parsed
    from the pinned headers.
  * Each assertion is falsifiable by the specific reviewed defect.
  * No test writes outside ``tmp_path`` or touches an NPU.

[Timeouts & Locks]
  Pure in-memory tests plus one bounded generated dry-run subprocess.

[Side-Effect Policy]
  Temporary files only under pytest's ``tmp_path``.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
__import__("pytest").importorskip("mindie_profiler")
from mindie_profiler import repro_atomic_io as checked_io
import sys
import types

import pytest

from temp_cleanup import tempdir

@pytest.fixture(autouse=True)
def _isolate_short_root(monkeypatch):
    import shutil as _shutil
    short = tempdir('mp_', directory='/tmp')
    monkeypatch.setenv('MINDIE_REPRO_SHORT_ROOT', str(short))
    try:
        yield
    finally:
        _shutil.rmtree(short, ignore_errors=True)

from mindie_profiler.events import ProfilerUsageError
from mindie_profiler.fabric_probe import (
    build_fabric_probe_plan, emit_fabric_probe_script,
)
from mindie_profiler.fabric_probe_fit import fit_calibrated_fabric_profile
from mindie_profiler.op_api_registry import (
    bind_captured_occurrence, call_mcp_tool, inspect_op_abi,
)
from mindie_profiler.repro_op import (
    _compact_occurrence, _row_from_args, generate_single_op_reproducer,
    main as repro_main, synthesize_metadata,
)
from mindie_profiler.token_budget import TOKENIZER_ID, estimate_tokens
from mindie_sim.diff_trace import attribute_diff

ROOT = Path("/Users/maoxx241/code/refs/ops-transformer")
VLLM = Path("/Users/maoxx241/code/vllm-ascend")


# ---------------------------------------------------------------- fixtures
from nr_s6_support import exec_script, fa_row, paged_row, _fake_torch, probe_namespace


# ------------------------------------------------------------- R7 / R8
def test_r7_metadata_install_shape_and_numel_are_hard_errors():
    binding = bind_captured_occurrence(paged_row())
    occurrence = _compact_occurrence(binding)
    script = generate_single_op_reproducer(binding)["script"]
    namespace = exec_script(script)
    metadata = namespace["synthesize_metadata"](occurrence)
    install = metadata["install"][0]
    inputs = {item["tensor"]: _fake_torch().tensor(shape=tuple(item["shape"]))
              for item in metadata["install"]}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    try:
        namespace["_apply_metadata"](inputs, metadata)
        broken = json.loads(json.dumps(metadata))
        broken["install"][0]["shape"] = [3, 3, 3, 3, 3]
        with pytest.raises(Exception):
            namespace["_apply_metadata"](inputs, broken)
        broken_numel = json.loads(json.dumps(metadata))
        broken_numel["install"][0]["values"] = broken_numel["install"][0]["values"][:-1]
        with pytest.raises(Exception):
            namespace["_apply_metadata"](inputs, broken_numel)
    finally:
        monkeypatch.undo()
    assert install["shape"] == [16]


def test_r7_dry_run_fails_nonzero_when_a_check_fails():
    script = generate_single_op_reproducer(bind_captured_occurrence(fa_row()))["script"]
    namespace = exec_script(script)
    namespace["OCCURRENCE"]["padding"]["qualification"]["tail_rule"] = "eq_bucket"
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = namespace["main"](["--dry-run"])
    assert code == 2
    assert "dry_run_failed" in buffer.getvalue()


def test_r8_incomplete_bindings_cannot_emit_ready():
    binding = bind_captured_occurrence({
        "op_type": "aclnnMatmul", "cann_version": "9.2",
        "input_shapes": ["[-1,2]", "[2,2]"],
        "input_dtypes": ["BF16", "BF16"], "input_formats": ["ND", "ND"],
        "output_shapes": ["[2,2]"], "output_dtypes": ["BF16"],
        "output_formats": ["ND"]})
    with pytest.raises(ProfilerUsageError):
        generate_single_op_reproducer(binding)


def test_r8_paged_attention_forwards_the_output_buffer():
    binding = bind_captured_occurrence(paged_row())
    occurrence = _compact_occurrence(binding)
    plan = occurrence["call_plan"]
    assert plan["keyword"]["out"] == {"kind": "output", "name": "out"}
    assert "out" in [spec["name"] for spec in occurrence["allocations"]
                     if spec["direction"] == "output"]


def test_r8_stateful_reset_is_outside_the_timed_region():
    script = generate_single_op_reproducer(bind_captured_occurrence(paged_row()))["script"]
    namespace = exec_script(script)
    assert "reset" in namespace["_measure"].__code__.co_varnames
    start = script.index("def _measure(")
    source = script[start:script.index("def _percentile(", start)]
    assert source.index("reset()") < source.index("start.record()")


def inspect_source(function) -> str:
    import inspect
    return inspect.getsource(function)


def test_r8_captured_metadata_is_preferred_and_invalid_is_rejected():
    def occurrence(values):
        return {
            "attrs": {"kv_len": 2048},
            "padding": {"bucket_tokens": 16, "effective_tokens": 16,
                        "qualification": {"layout": "PA_BSND", "pa": True,
                                          "kv_mode": {"PA": "lengths",
                                                      "non_PA": "cumsum"}}},
            "allocations": [
                {"name": "query", "direction": "input", "role": "activation",
                 "shape": [16, 8, 128], "dtype": "bf16"},
                {"name": "context_lens", "direction": "input", "role": "seq_lens",
                 "shape": [16], "dtype": "int32", "values": values}],
        }

    kept = synthesize_metadata(occurrence([100] * 16))
    install = next(item for item in kept["install"]
                   if item["tensor"] == "context_lens")
    assert install["values"] == [100] * 16
    assert install["encoding"].endswith("|captured")
    assert kept["fidelity"] == "metadata_captured_activations_synthetic"
    rejected = synthesize_metadata(occurrence([999] * 16))
    install = next(item for item in rejected["install"]
                   if item["tensor"] == "context_lens")
    assert install["values"] == [999] * 16
    assert any(item["code"] == "captured_metadata_invalid"
               for item in rejected["violations"])
    assert rejected["fidelity"] == "metadata_captured_activations_synthetic"


def test_r8_output_directory_writes_manifest_and_repro(tmp_path):
    result = generate_single_op_reproducer(
        bind_captured_occurrence(paged_row()), output_dir=tmp_path)
    assert result["status"] == "ready"
    script = (tmp_path / "repro.py").read_text()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["repro_sha256"] == result["sha256"]
    assert manifest["fidelity"] == "descriptor_synthetic"
    assert manifest["catalog_abi_id"] == result["manifest"]["catalog_abi_id"]
    assert hashlib.sha256(script.encode()).hexdigest() == result["sha256"].split(":")[1]


# ------------------------------------------------------- R9 / R10 / R11
def probe_namespace() -> dict:
    plan = build_fabric_probe_plan(world_size=16, duration_s=60)
    script = emit_fabric_probe_script(plan)["script"]
    namespace = {"__name__": "probe_test"}
    exec(compile(script, "<probe>", "exec"), namespace)
    return namespace


def test_r9_send_recv_only_matched_pairs_participate():
    namespace = probe_namespace()

    class FakeDist:
        def __init__(self):
            self.calls = []

        def send(self, tensor, dst, tag):
            self.calls.append(("send", dst))

        def recv(self, tensor, src, tag):
            self.calls.append(("recv", src))

    fake = FakeDist()
    participants = []
    for rank in range(16):
        acted = namespace["_call_once"](None, fake, "send_recv", object(), None,
                                        None, [0, 1], rank, 1000)
        if acted:
            participants.append(rank)
    assert participants == [0, 1]
    assert fake.calls == [("send", 1), ("recv", 0)]


def test_r9_reduce_scatter_uses_two_tensor_form_and_preallocated_groups():
    namespace = probe_namespace()

    class FakeDist:
        def __init__(self):
            self.args = None

        def reduce_scatter_tensor(self, output, input, group=None):
            self.args = (output, input, group)

        def all_gather_into_tensor(self, output, input, group=None):
            self.args = (output, input, group)

    fake = FakeDist()
    namespace["_call_once"](None, fake, "reduce_scatter", "out", "flat", "grp",
                            None, 0, 0)
    assert fake.args == ("out", "flat", "grp")
    namespace["_call_once"](None, fake, "all_gather", "inp", "out", "grp",
                            None, 0, 0)
    assert fake.args == ("out", "inp", "grp")
    source = Path(namespace["__file__"] if "__file__" in namespace else
                  str(Path(checked_io.__file__).resolve().parent / "fabric_probe_script_template.py")).read_text()
    assert "reduce_scatter_tensor(buf, second" in source
    assert "[output.clone()" not in source


def test_r9_device_is_set_before_group_initialization_and_cleanup_is_barrier_free():
    plan = build_fabric_probe_plan(world_size=16, duration_s=60)
    source = emit_fabric_probe_script(plan)["script"]
    assert source.index("set_device(local_rank)") < source.index("init_process_group")
    cleanup = source[source.index("finally:"):]
    assert "barrier" not in cleanup
    assert source.count("dist.new_group") == 1


def test_r10_tiny_budget_admits_nothing_and_reports_unmeasured():
    plan = build_fabric_probe_plan(world_size=16, duration_s=0.01)
    assert plan["status"] == "rejected"
    assert plan["admitted"] == []
    assert plan["unmeasured"]
    assert plan["deadline"]["mode"] == "time.monotonic"
    assert plan["deadline"]["cleanup_allowance_s"] > 0
    assert plan["estimated_s"] >= plan["cleanup_allowance_s"]


def test_r10_template_enforces_a_monotonic_deadline_before_validation():
    plan = build_fabric_probe_plan(world_size=16, duration_s=60)
    source = emit_fabric_probe_script(plan)["script"]
    assert "deadline = started + float(PLAN[\"deadline\"][\"measurement_budget_s\"])" in source
    assert "time.monotonic() >= deadline" in source
    assert "def _continue_decision(" in source
    assert source.index("def _continue_decision(") < source.index(
        "validated = _continue_decision(")
    assert "init_timeout_s" in source
    assert "min(remaining, max_wall)" in source


def test_r11_real_tier_groups_and_no_pooling():
    plan = build_fabric_probe_plan(world_size=16, ranks_per_node=8, duration_s=60)
    tiers = {tier["tier"]: tier for tier in plan["tiers"]}
    assert tiers["intra_package_die"]["ranks"] == [0, 1]
    assert tiers["intra_node_hccs"]["ranks"] == list(range(8))
    assert tiers["inter_node_roce"]["ranks"] == [0, 8]
    samples = [
        {"tier": "intra_node_hccs", "op": op, "bytes": size,
         "latency_us": alpha + beta * size / 1000.0, "group_size": 8,
         "protocol": "HCCS"}
        for op, alpha, beta in (("all_reduce", 8.0, 0.1), ("all_gather", 12.0, 0.2))
        for size in (4096, 65536, 1048576)]
    profile = fit_calibrated_fabric_profile(plan, samples)
    operations = profile["tiers"]["intra_node_hccs"]["operations"]
    assert operations["all_reduce"]["beta_ns_per_byte"] == pytest.approx(0.1, rel=1e-3)
    assert operations["all_gather"]["beta_ns_per_byte"] == pytest.approx(0.2, rel=1e-3)
    assert operations["all_reduce"]["congestion_factor"] is None
    assert "intra_package_die" in profile["missing_tiers"]


def test_r11_single_size_cannot_produce_a_slope():
    plan = build_fabric_probe_plan(world_size=16, duration_s=60)
    profile = fit_calibrated_fabric_profile(plan, [
        {"tier": "intra_node_hccs", "op": "all_reduce", "bytes": 4096,
         "latency_us": 9.0, "group_size": 8, "protocol": "HCCS"}])
    fit = profile["tiers"]["intra_node_hccs"]["operations"]["all_reduce"]
    assert fit["beta_ns_per_byte"] is None
    assert fit["fit_quality"] == "single_size"
    assert profile["usable_for_link_override"] is False


def test_r11_nonnegative_fit_never_invents_bandwidth():
    plan = build_fabric_probe_plan(world_size=16, duration_s=60)
    samples = [{"tier": "intra_node_hccs", "op": "all_reduce", "bytes": size,
                "latency_us": 100.0 - size / 1e6, "group_size": 8,
                "protocol": "HCCS"} for size in (4096, 65536, 1048576)]
    fit = fit_calibrated_fabric_profile(plan, samples)["tiers"]["intra_node_hccs"][
        "operations"]["all_reduce"]
    assert fit["alpha_us"] >= 0 and fit["beta_ns_per_byte"] >= 0
    assert fit["monotonicity_violation"] is True


def test_r11_piecewise_fit_reports_a_breakpoint_and_uncertainty():
    plan = build_fabric_probe_plan(world_size=16, duration_s=60)
    sizes = [4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]
    samples = []
    for size in sizes:
        beta = 0.05 if size <= 262144 else 0.40
        samples.append({"tier": "intra_node_hccs", "op": "all_reduce",
                        "bytes": size, "latency_us": 8.0 + beta * size / 1000.0,
                        "group_size": 8, "protocol": "HCCS"})
    fit = fit_calibrated_fabric_profile(plan, samples)["tiers"]["intra_node_hccs"][
        "operations"]["all_reduce"]
    assert fit["fit_quality"] in {"piecewise", "affine"}
    assert fit["alpha_us"] >= 0 and fit["beta_ns_per_byte"] >= 0
    if fit["fit_quality"] == "piecewise":
        assert fit["breakpoint_bytes"] in sizes
        assert len(fit["segments"]) == 2
    assert fit["se_alpha_us"] is not None and fit["se_beta_ns_per_byte"] is not None


def test_r11_congestion_requires_both_regimes():
    plan = build_fabric_probe_plan(world_size=16, duration_s=60)
    samples = []
    for size in (4096, 65536, 1048576):
        samples.append({"tier": "intra_node_hccs", "op": "all_reduce",
                        "bytes": size, "latency_us": 8.0 + 0.1 * size / 1000.0,
                        "group_size": 8, "protocol": "HCCS", "regime": "isolated"})
        samples.append({"tier": "intra_node_hccs", "op": "all_reduce",
                        "bytes": size, "latency_us": 16.0 + 0.2 * size / 1000.0,
                        "group_size": 8, "protocol": "HCCS", "regime": "loaded"})
    fit = fit_calibrated_fabric_profile(plan, samples)["tiers"]["intra_node_hccs"][
        "operations"]["all_reduce"]
    assert fit["congestion_factor"] == pytest.approx(2.0, rel=0.1)


def test_r11_unidentified_tiers_are_not_fabricated():
    plan = build_fabric_probe_plan(world_size=16, duration_s=60)
    profile = fit_calibrated_fabric_profile(plan, [
        {"tier": "not_a_tier", "op": "all_reduce", "bytes": 4096,
         "latency_us": 3.0, "group_size": 2, "protocol": "HCCS"}])
    assert "not_a_tier" in profile["unidentified_tiers"]
    assert "not_a_tier" not in profile["tiers"]


# ------------------------------------------------------------------ R12
def test_r12_gate_uses_sample_stdev_and_honors_configuration():
    namespace = exec_script(generate_single_op_reproducer(
        bind_captured_occurrence(fa_row()))["script"])
    samples = [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 9.7]
    gate = namespace["steady_state"](samples, 5.0, window=4, consecutive=2)
    import statistics
    expected = (statistics.stdev(samples[3:7]) / statistics.fmean(samples[3:7]) * 100)
    assert gate["windows"][0]["end"] == 7
    assert gate["windows"][0]["cv_pct"] == pytest.approx(expected)
    assert gate["max_cv_pct"] == 5.0
    strict = namespace["steady_state"](samples, 0.0001, window=4, consecutive=2)
    assert strict["qualified"] is False


def test_r12_zero_mean_cv_is_unknown_not_zero():
    namespace = exec_script(generate_single_op_reproducer(
        bind_captured_occurrence(fa_row()))["script"])
    gate = namespace["steady_state"]([0.0] * 8, 5.0, window=4, consecutive=2)
    assert gate["windows"][0]["cv_pct"] is None
    assert gate["qualified"] is False


def test_r12_generated_threshold_is_the_configured_value():
    script = generate_single_op_reproducer(
        bind_captured_occurrence(fa_row()), max_cv_pct=1.0)["script"]
    assert "--max-cv-pct\", type=float, default=1.0" in script
    assert "status = (\"ok\" if qualified and alloc_stable is True" in script
    assert '"status": status' in script


def test_r12_clock_instability_forces_unstable():
    script = generate_single_op_reproducer(bind_captured_occurrence(fa_row()))["script"]
    assert 'gate["reason"] = "clock_unstable"' in script
    assert "used = samples[-window:] if qualified else []" in script


def test_r12_cache_labels_are_honest():
    namespace = exec_script(generate_single_op_reproducer(
        bind_captured_occurrence(fa_row()))["script"])
    warm = namespace["_cache_label"](0, None)
    assert warm["mode"] == "warm"
    cold = namespace["_cache_label"](256, None)
    assert cold["mode"] == "cold_attempted" and cold["cold_verified"] is False
    inferred = namespace["_cache_label"](1, 4096)
    assert inferred["cold_verified"] is False
    verified = namespace["_cache_label"](1, 4096, 8, 8192)
    assert verified["cold_verified"] is True
    assert verified["flush_invocations"] == 8


# ------------------------------------------------------------------ R13
def test_r13_occurrence_index_is_applied_before_name_filtering(tmp_path):
    capture = {"rows": [
        {"op_type": "aclnnRmsNorm", "input_shapes": ["[4,8]"]},
        {"op_type": "aclnnMatmul", "input_shapes": ["[2,2]"]}]}
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(capture))
    args = types.SimpleNamespace(row=None, capture=str(path), op_name="aclnnMatmul",
                                 occurrence_index=1, index=None, version="9.2")
    binding, index, _ = _row_from_args(args)
    assert index == 1
    assert binding["operator"] == "aclnnMatmul"
    args.occurrence_index = 0
    with pytest.raises(ProfilerUsageError):
        _row_from_args(args)


def test_r13_repeated_names_require_an_immutable_index(tmp_path):
    capture = {"rows": [{"op_type": "aclnnMatmul", "input_shapes": ["[2,2]"]},
                        {"op_type": "aclnnMatmul", "input_shapes": ["[4,4]"]}]}
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(capture))
    args = types.SimpleNamespace(row=None, capture=str(path), op_name="aclnnMatmul",
                                 occurrence_index=None, index=None, version="9.2")
    with pytest.raises(ProfilerUsageError):
        _row_from_args(args)


def test_r13_commands_quote_paths_and_target_an_output_dir(tmp_path):
    predicted = {"total_ms": 10, "ops": [{"name": "Matmul", "stream": 0,
                                          "start_ms": 0, "duration_ms": 10}]}
    captured = {"total_ms": 12, "ops": [{"name": "Matmul", "stream": 0,
                                         "start_ms": 0, "duration_ms": 12}]}
    result = attribute_diff(predicted, captured, launch_overhead_us=0,
                            capture_path="/tmp/a path/capture.json")
    command = result["kernel_roofline_and_clock"]["top_offenders"][0]["repro_command"]
    assert "'/tmp/a path/capture.json'" in command
    assert "--output-dir" in command

def test_r13_output_dir_cli_writes_repro_and_manifest(tmp_path):
    binding = bind_captured_occurrence(paged_row())
    row_path = tmp_path / "binding.json"
    row_path.write_text(json.dumps(binding))
    code = repro_main(["--row", str(row_path), "--output-dir", str(tmp_path / "art")])
    assert code == 0
    assert checked_io.path_exists(tmp_path / "art" / "repro.py")
    manifest = json.loads((tmp_path / "art" / "manifest.json").read_text())
    assert manifest["repro_sha256"].startswith("sha256:")

# ------------------------------------------------------------------ R14
def test_r14_indexed_lookup_and_bounded_receipts():
    card = call_mcp_tool("inspect_op_abi", {"op_name": "aclnnPagedAttention",
                                            "cann_version": "8.0"})
    assert card["receipt"]["tokenizer"] == TOKENIZER_ID
    assert card["receipt"]["token_estimate"] <= 250
    assert card["receipt"]["within_budget"] is True
    assert card["artifact"]["kind"] == "catalog_span"
    assert card["receipt"]["token_estimate"] == estimate_tokens(
        {key: value for key, value in card.items() if key != "receipt"})
    full = inspect_op_abi("aclnnFlashAttentionVarLenScore", cann_version="8.0")
    assert full["identity"]["source_sha256"]
    assert full["card"]["schema_pinned"] is True
    assert full["card"]["capture_verified"] is False
    assert full["card"]["version_qualified"] is False
    assert full["card"]["qualification_state"] == "schema_pinned_unverified"
    assert full["card"]["complete"] is False
    assert full["card"]["api"] == "aclnnFlashAttentionVarLenScore"
    assert full["card"]["abi_id"].startswith("sha256:")
    assert full["card"]["inputs"] and full["card"]["outputs"]
    assert full["card"]["attrs"] and full["card"]["recipe"]["call"]
    assert estimate_tokens(full["card"]) <= full["receipt"]["token_limit"]
    mcp_card = call_mcp_tool("inspect_op_abi",
                             {"op_name": "aclnnFlashAttentionVarLenScore",
                              "cann_version": "8.0"})
    assert mcp_card["receipt"]["within_budget"] is True
    assert mcp_card["receipt"]["token_estimate"] <= 250
    unversioned = inspect_op_abi("aclnnFlashAttentionVarLenScore")
    assert unversioned["card"]["version_qualified"] is False
    assert unversioned["card"]["complete"] is False

def test_r14_lookup_reads_only_its_own_byte_span(monkeypatch):
    import mindie_profiler.op_api_registry as registry
    calls = []
    original = registry._entry

    def recording(index, name):
        calls.append(name)
        return original(index, name)

    monkeypatch.setattr(registry, "_entry", recording)
    registry.inspect_op_abi("aclnnPagedAttention", cann_version="8.0")
    assert calls == ["aclnnPagedAttention"]

def test_r14_repro_mcp_receipt_has_no_inline_script():
    binding = bind_captured_occurrence(paged_row())
    result = call_mcp_tool("repro_op", {"occurrence": binding})
    assert "script" not in result
    assert result["artifact"]["inline_script_omitted"] is True
    assert result["receipt"]["token_estimate"] <= 250

@pytest.mark.parametrize("route,faults", [("abort_native", 0), ("abort_wrapper", 0), ("abort_wrapper", 4), ("reader_wrapper", 0), ("reader_wrapper", 4), ("reader_native", 0)])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("interleave", [False, True])
def test_r21_stream_transfer_preserves_foreign_fd(route, faults, active, interleave, tmp_path, monkeypatch):
    import builtins
    import gc
    import os
    import threading
    import weakref
    target, foreign = tmp_path / "target", tmp_path / "foreign"
    target.write_bytes(b"old")
    foreign.write_bytes(b"foreign")
    primary, secondary = TimeoutError("primary"), PermissionError("close")
    opened_fds, streams, close_calls, caught, live_fds = [], [], [], [], []
    ready, resume = threading.Event(), threading.Event()
    real_open, fileopen, fdopen, dup2 = os.open, builtins.open, os.fdopen, os.dup2
    borrow = checked_io._borrow_stream_fd
    def closing(base):
        close_calls.append(1)
        if len(close_calls) <= faults:
            raise secondary
        return base()
    class Reader(io.BufferedReader):
        def read(self, *args):
            if active:
                raise primary
            return super().read(*args)
        def close(self):
            return closing(super().close)
    class Writer(io.TextIOWrapper):
        def close(self):
            return closing(super().close)
    def opened(fd, *args, **kwargs):
        handle = (Writer(io.FileIO(fd, "w", closefd=True), encoding="utf-8")
                  if route == "abort_wrapper" else fileopen(fd, *args, **kwargs))
        streams.append(handle)
        opened_fds.append(fd)
        return handle
    def fdopened(fd, *args, **kwargs):
        handle = (Reader(io.FileIO(fd, "rb", closefd=True)) if route == "reader_wrapper"
                  else fdopen(fd, *args, **kwargs))
        streams.append(handle)
        opened_fds.append(fd)
        return handle
    def pause():
        if interleave and not ready.is_set():
            ready.set()
            assert resume.wait(5)
    def transferring(stream, fd, inode):
        result = borrow(stream, fd, inode)
        live_fds.append(result)
        pause()
        return result
    def destructive_transfer(src, dst, *args, **kwargs):
        pause()  # The original implementation exposes a free dst here.
        return dup2(src, dst, *args, **kwargs)
    ns = {"__name__": "r21_transfer"}
    exec(emit_fabric_probe_script(build_fabric_probe_plan(world_size=2))["script"], ns)
    ns.update(ABORT_FILE=str(target), open=opened, _borrow_stream_fd=transferring)
    def worker():
        with contextlib.suppress(BaseException):
            try:
                if route.startswith("reader"):
                    checked_io.read_regular_bytes(target)
                elif active:
                    try:
                        raise primary
                    finally:
                        ns["_write_abort_record"]({"kind": "abort"}, primary)
                else:
                    ns["_write_abort_record"]({"kind": "abort"})
            finally:
                caught.append(sys.exc_info()[1])
    with monkeypatch.context() as patcher:
        patcher.setattr(os, "fdopen", fdopened)
        patcher.setattr(os, "dup2", destructive_transfer)
        patcher.setattr(checked_io, "_borrow_stream_fd", transferring)
        thread = threading.Thread(target=worker)
        thread.start()
        if interleave:
            try:
                assert ready.wait(5)
                foreign_fd = real_open(foreign, os.O_RDWR)
                receipt = os.fstat(foreign_fd)
            finally:
                resume.set()
        thread.join(5)
        assert not thread.is_alive()
    if not interleave:
        foreign_fd = real_open(foreign, os.O_RDWR)
        receipt = os.fstat(foreign_fd)
        assert foreign_fd == opened_fds[0]
    assert streams and all(s.closed and checked_io._stream_raw(s).closed for s in streams)
    assert caught == ([primary] if active and route != "reader_native" else [None])
    for exc in (primary, secondary, *caught):
        if exc:
            exc.__traceback__ = exc.__context__ = exc.__cause__ = None
    caught.clear()
    refs = [weakref.ref(s) for s in streams]
    streams.clear()
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert os.fstat(foreign_fd).st_ino == receipt.st_ino
    assert os.read(foreign_fd, 100) == b"foreign"
    os.close(foreign_fd)
    if route.startswith("abort") and faults:
        assert target.read_bytes() == b"old"

@pytest.mark.parametrize("route", ["reader", "abort_native", "abort_owning"])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("error_type", [TimeoutError, PermissionError])
@pytest.mark.parametrize("failures", [1, 2])
def test_r21_dup_failure_retires_owner(route, active, error_type, failures, tmp_path, monkeypatch):
    import builtins
    import os
    target = tmp_path / "target"
    target.write_bytes(b"old")
    primary, secondary = ValueError("primary"), error_type("dup")
    handles, fds, calls = [], [], []
    fileopen, real_dup = builtins.open, os.dup
    def opened(fd, *args, **kwargs):
        if route == "abort_owning":
            kwargs["closefd"] = True
        handle = fileopen(fd, *args, **kwargs)
        handles.append(handle)
        fds.append(fd)
        return handle
    def fdopened(fd, *args, **kwargs):
        handle = io.BufferedReader(io.FileIO(fd, "rb", closefd=True))
        handles.append(handle)
        fds.append(fd)
        return handle
    def duplicate(fd):
        calls.append(fd)
        if len(calls) <= failures:
            raise secondary
        return real_dup(fd)
    ns = {"__name__": "r21_dup"}
    exec(emit_fabric_probe_script(build_fabric_probe_plan(world_size=2))["script"], ns)
    ns.update(ABORT_FILE=str(target), open=opened)
    caught = None
    with monkeypatch.context() as patcher:
        patcher.setattr(os, "dup", duplicate)
        patcher.setattr(os, "fdopen", fdopened)
        with contextlib.suppress(BaseException):
            try:
                if route == "reader":
                    if active:
                        try:
                            raise primary
                        except ValueError:
                            checked_io.read_regular_bytes(target)
                    else:
                        checked_io.read_regular_bytes(target)
                elif active:
                    try:
                        raise primary
                    finally:
                        ns["_write_abort_record"]({"kind": "abort"}, primary)
                else:
                    ns["_write_abort_record"]({"kind": "abort"})
            finally:
                caught = sys.exc_info()[1]
    assert handles and all(h.closed and checked_io._stream_raw(h).closed for h in handles)
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    expected = primary if active and route.startswith("abort") else secondary if route != "abort_native" and error_type is TimeoutError else None
    assert caught is expected
    assert len(calls) == (0 if route == "abort_native" else 1)

@pytest.mark.parametrize("bad", [True, False, "1", 1.0, 1.5, -1, 2**63, 10**5000, -10**5000, ["invalid", 10**5000]], ids=["true", "false", "str", "float", "fraction", "negative", "range", "huge", "negative_huge", "mixed"])
@pytest.mark.parametrize("role", ["seq_lens", "block_table", "group_list", "index"])
def test_r22_direct_metadata_integer_values(bad, role):
    import json
    import sys
    from mindie_profiler.repro_metadata import synthesize_metadata
    from mindie_profiler.repro_op import _compact_occurrence
    from nr_s6_support import paged_binding, gmm_binding
    from test_nr_s6_round2_closure import _ready_bindings
    assert sys.get_int_max_str_digits() == 4300
    binding = (gmm_binding() if role == "group_list" else
               _ready_bindings()["MoeInitRoutingV2"] if role == "index" else paged_binding())
    occurrence = _compact_occurrence(binding)
    spec = next(x for x in occurrence["allocations"] if x["role"] == role)
    count = 1
    for dim in spec["shape"]:
        count *= dim
    spec["values"] = [bad] * count
    result = synthesize_metadata(occurrence)
    json.dumps(result, allow_nan=False)
    assert result["checks"]["metadata_installable"] is False
    installed = next(x for x in result["install"] if x["tensor"] == spec["name"])
    assert installed.get("captured_valid") is False and not installed["captured"]
    assert any("captured_metadata" in x["code"] for x in result["violations"])
