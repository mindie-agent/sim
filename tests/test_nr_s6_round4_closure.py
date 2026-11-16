"""Round-4 closure: F1..F11 executable counterexamples and rendezvous evidence.

[Role]
  Re-run the Round-4 review counterexamples for host attribution, fixed-size
  callable arrays, symmetric sequence metadata, physical KV/sparse bounds,
  dtype representability, collective ordering, post-step deadlines, rank
  evidence, duplicate UIDs, bounded receipts and independent callable types.

[Invariants]
  * The callable type table below is transcribed from the external
    vllm-ascend/op-plugin contract, never read from the catalog under test.
  * The two probe ranks run concurrently through one rendezvous broker; peer
    decisions are computed from both ranks at the same collective position.
  * No NPU, HCCL, device subprocess or network access is used.

[Timeouts & Locks]
  In-memory work plus two bounded threads and a condition-variable rendezvous
  timeout; no generated process is left running.

[Side-Effect Policy]
  Writes only pytest temporary files and short content-addressed reproducer
  directories created by the product.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
__import__("pytest").importorskip("mindie_profiler")
from mindie_profiler import repro_atomic_io as checked_io
import shutil
import types

import pytest
from mindie_profiler.events import ProfilerUsageError

from mindie_profiler.fabric_probe import build_fabric_probe_plan
from mindie_profiler.fabric_probe_fit import fit_calibrated_fabric_profile
from mindie_profiler.op_api_registry import (
    bind_captured_occurrence, catalog_entry,
)
from mindie_profiler.repro_op import (
    _compact_occurrence, _row_from_args, _row_uid,
    call_mcp_tool, generate_single_op_reproducer,
)
from mindie_profiler.repro_values import validate_captured_values
from mindie_profiler.token_budget import bounded_envelope, estimate_tokens
from mindie_sim.diff_trace import attribute_diff, effective_operator_rows

from nr_s6_round4_probe import run_ranks
from nr_s6_support import exec_script, paged_binding
from test_nr_s6_round3_closure import (
    _bsh_fa_row, _fit_samples, _gmm_row, _moe_row, _paged_row, _rms_row,
)

# ----------------------------------------------------------------------- F1
def _chain(pred_a, pred_b, cap_a, cap_b, *, cap_start=None, gap=None, total=0):
    captured = {"total_ms": total, "ops": [
        {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": cap_a},
        {"name": "B", "stream": 0, "start_ms": (cap_a if cap_start is None
                                                else cap_start),
         "duration_ms": cap_b, "deps": [0]}]}
    if gap is not None:
        captured["gaps"] = [{"stream": 0, "start_ms": gap[0],
                             "duration_ms": gap[1]}]
    return ({"total_ms": pred_a + pred_b, "ops": [
        {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": pred_a},
        {"name": "B", "stream": 0, "start_ms": pred_a, "duration_ms": pred_b,
         "deps": [0]}]}, captured)

def test_f1_pure_host_gap_is_scheduled_not_operator_time():
    predicted, captured = _chain(1, 1, 1, 1, cap_start=2, gap=(1, 1), total=3)
    result = attribute_diff(predicted, captured, launch_overhead_us=0)
    assert result["stages"] == {"R0": 2.0, "R1": 3.0, "R2": 3.0,
                                "R3": 3.0, "R4_model": 3.0, "O": 3.0}
    assert result["attribution"]["sched_delta_ms"] == pytest.approx(1.0)
    assert result["attribution"]["op_delta_ms"] == pytest.approx(0.0)
    assert result["attribution"]["residual_ms"] == pytest.approx(0.0)
    assert result["host_bound_ratio"] == pytest.approx(1 / 3)
    assert result["reconciliation"]["reconciled"] is True

def test_f1_pure_operator_speedup_is_not_host_bound():
    predicted, captured = _chain(2, 2, 1, 2, cap_start=1, total=3)
    result = attribute_diff(predicted, captured, launch_overhead_us=0)
    assert result["attribution"]["sched_delta_ms"] == pytest.approx(0.0)
    assert result["attribution"]["op_delta_ms"] == pytest.approx(-1.0)
    assert result["attribution"]["residual_ms"] == pytest.approx(0.0)
    assert result["stages"]["R4_model"] == pytest.approx(3.0)

def test_f1_combined_host_and_operator_change_do_not_double_count():
    predicted, captured = _chain(2, 2, 1, 1, cap_start=1.5,
                                 gap=(1, 0.5), total=2.5)
    result = attribute_diff(predicted, captured, launch_overhead_us=0)
    assert result["attribution"]["sched_delta_ms"] == pytest.approx(0.5)
    assert result["attribution"]["op_delta_ms"] == pytest.approx(-2.0)
    assert result["attribution"]["residual_ms"] == pytest.approx(0.0)
    assert result["conservation"]["exact"] is True
    assert result["reconciliation"]["reconciled"] is True

# ----------------------------------------------------------------------- F2
def test_f2_half_open_active_expert_range_allows_expert_num_endpoint():
    for value in ([0, 4], [0, 1]):
        row = _moe_row()
        row["attrs"]["active_expert_range"] = value
        assert _compact_occurrence(
            bind_captured_occurrence(row))["status"] == "ready", value
    for value, code in (([0, 1, 2], "invalid_array_length"),
                        ([3, 1], "invalid_array_order"),
                        ([-1, 4], "invalid_array_range"),
                        ([True, 4], "invalid_attribute_type")):
        row = _moe_row()
        row["attrs"]["active_expert_range"] = value
        binding = bind_captured_occurrence(row)
        assert binding["status"] != "bound", value
        assert any(issue["code"] == code for issue in binding["issues"]), value
        with pytest.raises(Exception):
            _compact_occurrence(binding)
    for value in ([0, 5], [4, 4]):
        row = _moe_row()
        row["attrs"]["active_expert_range"] = value
        binding = bind_captured_occurrence(row)
        assert binding["status"] == "bound", value
        with pytest.raises(Exception) as excinfo:
            _compact_occurrence(binding)
        assert "invalid_precondition" in str(excinfo.value)
    catalog = catalog_entry("MoeInitRoutingV2")["repro_recipe"]
    spec = catalog["signature"]["params"]["active_expert_range"]
    assert spec["fixed_length"] == 2 and spec["allowed_lengths"] == [0, 2]

# ----------------------------------------------------------------------- F3
def test_f3_kv_uses_per_sequence_and_batch_arity():
    row = _bsh_fa_row([3, 2])
    row["attrs"]["actual_seq_kv_len"] = [6, 0]
    binding = bind_captured_occurrence(row)
    assert binding["status"] != "bound"
    assert any(item["code"] == "sequence_kv_exceeds_per_sequence_extent"
               for item in binding["padding"]["violations"])
    arity = bind_captured_occurrence(_bsh_fa_row([1, 1, 1]))
    codes = {item["code"] for item in arity["padding"]["violations"]}
    assert {"sequence_q_batch_arity", "sequence_kv_batch_arity"} <= codes
    assert _compact_occurrence(bind_captured_occurrence(
        _bsh_fa_row([3, 2])))["status"] == "ready"

def test_f3_explicit_logical_tokens_reject_non_integer_charges():
    for value in ("nonsense", float("nan"), 2.5, True, -1):
        charge = effective_operator_rows(
            "aclnnGroupedMatmulV4", 128, value,
            {"group_list": [16, 32, 48, 73]}, cann_version="9.0")
        assert charge["charged_tokens"] == 0, value
        assert charge["valid"] is False
        assert any(item["code"] == "effective_tokens_invalid"
                   for item in charge["violations"])
    valid = effective_operator_rows(
        "aclnnGroupedMatmulV4", 128, 73,
        {"group_list": [16, 32, 48, 73]}, cann_version="9.0")
    assert valid["charged_tokens"] == 73 and valid["valid"] is True

# ----------------------------------------------------------------------- F4
def test_f4_value_cache_and_sparse_expert_indices_are_bounded():
    paged = _paged_row()
    paged["input_shapes"][2]["shape"] = [1, 16, 8, 128]
    occurrence = _compact_occurrence(bind_captured_occurrence(paged))
    assert occurrence["status"] == "blocked"
    checks = occurrence["metadata"]["checks"]
    assert checks["allocated_pages"] == 1
    assert checks["kv_cache_page_capacities"] == [256, 1]
    assert checks["block_table_physical_fit"] is False
    assert occurrence["checks"]["block_table_in_bounds"] is False

    gmm = _gmm_row()
    gmm["attrs"]["group_list_type"] = 2
    gmm["attrs"]["group_list"] = [0, 32, 1, 32]
    gmm["input_shapes"][2]["values"] = [9999, 32, 10000, 32]
    binding = bind_captured_occurrence(gmm)
    occurrence = _compact_occurrence(binding)
    assert occurrence["status"] == "blocked"
    assert occurrence["checks"]["group_list_valid"] is False
    assert occurrence["checks"]["group_list_num_groups"] == 4
    install = next(item for item in occurrence["metadata"]["install"]
                   if item["tensor"] == "group_list")
    assert install["captured"] is True and install["captured_valid"] is False

# ----------------------------------------------------------------------- F5
def test_f5_destination_dtype_representability_is_exact():
    for dtype, value, code in (
            ("int32", [2 ** 40], "captured_value_out_of_range"),
            ("int8", [128], "captured_value_out_of_range"),
            ("bf16", [1e100], "captured_value_out_of_range"),
            ("fp16", [65505.0], "captured_value_out_of_range"),
            ("fp8_e4m3fn", [449.0], "captured_value_out_of_range"),
            ("bool", [2], "captured_value_type_mismatch"),
            ("int32", [True], "captured_value_type_mismatch")):
        values, problems = validate_captured_values([1], dtype, value)
        assert values is None
        assert code in {item["code"] for item in problems}
    assert validate_captured_values([1], "bf16", [1e30])[0] == [1e30]
    assert validate_captured_values([1], "fp8_e5m2", [57344.0])[0] == [57344.0]
    row = _rms_row([1e100] * 32)
    with pytest.raises(Exception) as excinfo:
        generate_single_op_reproducer(bind_captured_occurrence(row))
    assert "captured_value_out_of_range" in str(excinfo.value)

# ------------------------------------------------------------------ F6 / F7
def _probe_plan(*, iterations=2, warmup=1):
    return build_fabric_probe_plan(
        world_size=2, ranks_per_node=2, sizes=[4096], ops=["all_reduce"],
        iterations=iterations, warmup=warmup, duration_s=60.0,
        topology={"tiers": [{"tier": "intra_node_hccs", "ranks": [0, 1],
                             "protocol": "HCCS"}]})

def _assert_order(world, results):
    assert world.errors == {}
    assert world.broker.mismatches == []
    assert world.broker.sequence[0] == world.broker.sequence[1]
    assert world.broker.data_counts[0] == world.broker.data_counts[1]
    assert {code for code, _ in results.values()} == {2}

def test_f6_pre_and_post_sync_failures_keep_one_collective_protocol():
    cases = (
        ({1: {3: "raise"}}, None),
        ({1: {4: "raise"}}, None),
        (None, {1: True}),
    )
    for sync, alloc in cases:
        world, results = run_ranks(
            _probe_plan(), sync_actions=sync, fail_alloc=alloc)
        _assert_order(world, results)
        assert not any(item.get("validated") is True
                       for _, records in results.values() for item in records)

def test_f7_post_sync_deadline_rejects_the_final_sample():
    plan = _probe_plan(iterations=1, warmup=0)
    plan["deadline"]["measurement_budget_s"] = 0.9
    plan["deadline"]["cleanup_allowance_s"] = 0.1
    plan["deadline"]["max_wall_s"] = 1.0
    world, results = run_ranks(
        plan, sync_actions={0: {4: ("advance", 2.0)},
                            1: {4: ("advance", 2.0)}})
    _assert_order(world, results)
    records = [item for _, rows in results.values() for item in rows]
    assert not [item for item in records
                if item.get("schema") == "mindie-fabric-probe-sample/2"]
    assert any(item.get("status") == "deadline_exceeded" for item in records)

# ----------------------------------------------------------------------- F8
def test_f8_malformed_rank_evidence_never_calibrates():
    plan = build_fabric_probe_plan(
        world_size=8, ranks_per_node=8, duration_s=60,
        topology={"tiers": [{"tier": "intra_node_hccs", "ranks": list(range(8)),
                             "protocol": "HCCS"}]})
    for bad in ("garbage", 0.9, False, None):
        samples = _fit_samples(validated=True, rank=0,
                               group_ranks=list(range(8)))
        samples[0]["rank"] = bad
        if bad is not None:
            with pytest.raises(ProfilerUsageError):
                fit_calibrated_fabric_profile(plan, samples)
        else:
            assert not fit_calibrated_fabric_profile(plan, samples)["usable_for_link_override"]
    samples = _fit_samples(validated=True, rank=0, group_ranks=list(range(8)))
    samples[0]["group_ranks"] = [0, 0, 2, 3, 4, 5, 6, 7]
    profile = fit_calibrated_fabric_profile(plan, samples)
    assert profile["usable_for_link_override"] is False
    valid = fit_calibrated_fabric_profile(
        plan, _fit_samples(validated=True, rank=0,
                           group_ranks=list(range(8))))
    assert valid["status"] == "calibrated"

# ----------------------------------------------------------------------- F9
def test_f9_duplicate_op_uid_requires_the_explicit_index(tmp_path):
    rows = []
    for duration in (12, 14):
        row = _paged_row()
        row.update({"op_uid": "same-capture-uid", "stream": 0,
                    "start_ms": 0, "duration_ms": duration})
        rows.append(row)
    capture = {"run_id": "run-1", "cann_version": "8.0", "total_ms": 14,
               "rows": rows}
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(capture))
    args = types.SimpleNamespace(
        row=None, capture=str(path), op_name=None, occurrence_index=1,
        index=None, version=None, op_uid="same-capture-uid")
    binding, index, _ = _row_from_args(args)
    assert index == 1 and binding["operator"] == "aclnnPagedAttention"
    ambiguous = types.SimpleNamespace(
        row=None, capture=str(path), op_name=None, occurrence_index=None,
        index=None, version=None, op_uid="same-capture-uid")
    with pytest.raises(Exception) as excinfo:
        _row_from_args(ambiguous)
    assert "ambiguous_op_uid" in str(excinfo.value)
    canonical = _row_uid("run-1", rows[1], 1)
    mismatch = types.SimpleNamespace(
        row=None, capture=str(path), op_name=None, occurrence_index=0,
        index=None, version=None, op_uid=canonical)
    with pytest.raises(Exception) as excinfo:
        _row_from_args(mismatch)
    assert "selector_mismatch" in str(excinfo.value)

# ---------------------------------------------------------------------- F10
def test_f10_wide_dict_and_identity_path_stay_bounded():
    wide = {"schema": "s", "status": "blocked",
            "issues": [{"code": "x", "detail": {str(i): "x" for i in range(1000)}}]}
    envelope = bounded_envelope(wide, limit=3000, list_limit=8)
    assert estimate_tokens(envelope) <= 3000
    assert envelope["receipt"]["within_budget"] is True
    assert envelope["issues"][0]["detail"]["_omitted_keys"] > 0
    path = "/tmp/" + "a" * 2000 + "/repro.py"
    omitted = bounded_envelope(
        {"schema": "s", "status": "ready", "artifact_path": path,
         "artifact": {"retrievable": True, "sha256": "sha256:" + "b" * 64}},
        limit=250)
    assert estimate_tokens(omitted) <= 250
    assert omitted["receipt"]["within_budget"] is True
    assert omitted.get("artifact_path_omitted") is True
    assert omitted["artifact"]["retrievable"] is False
    assert "..." not in str(omitted.get("artifact_path", ""))
    result = call_mcp_tool("repro_op", {
        "occurrence": bind_captured_occurrence(_paged_row()),
        "output_dir": "/tmp/" + "a" * 180})
    try:
        assert result["artifact"]["retrievable"] is True
        assert "..." not in result["artifact_path"]
        assert checked_io.path_exists(Path(result["artifact_path"]))
    finally:
        shutil.rmtree(Path(result["artifact_path"]).parent, ignore_errors=True)

# ---------------------------------------------------------------------- F11
# Transcribed from vllm-ascend/op-plugin callable signatures, not the catalog.
_EXPECTED_TYPES = {
    "aclnnDynamicQuant": {
        "positional": [("x", "tensor")],
        "keyword": {"smooth_scales": "tensor", "group_index": "tensor",
                    "dst_type": "int", "quant_mode": "str",
                    "dst_type_max": "float"}},
    "aclnnPagedAttention": {
        "positional": [],
        "keyword": {name: "tensor" for name in
                    ("query", "key_cache", "value_cache", "block_table",
                     "context_lens", "out")}
        | {"num_kv_heads": "int", "num_heads": "int", "scale_value": "float"}},
    "MoeInitRoutingV2": {
        "positional": [("x", "tensor"), ("expert_idx", "tensor")],
        "keyword": {"active_num": "int", "expert_capacity": "int",
                    "expert_num": "int", "drop_pad_mode": "int",
                    "expert_tokens_num_type": "int",
                    "expert_tokens_num_flag": "bool",
                    "active_expert_range": "int_array", "quant_mode": "int",
                    "row_idx_type": "int"}},
    "aclnnFlashAttentionVarLenScore": {
        "positional": [("query", "tensor"), ("key", "tensor"),
                       ("value", "tensor")],
        "keyword": {"head_num": "int", "input_layout": "str",
                    "scale": "float", "keep_prob": "float",
                    "pre_tockens": "int", "next_tockens": "int",
                    "inner_precise": "int", "prefix": "int_array",
                    "actual_seq_qlen": "int_array",
                    "actual_seq_kvlen": "int_array", "sparse_mode": "int"}},
    "aclnnGroupedMatmulV4": {
        "positional": [],
        "keyword": {"x": "tensor_list", "weight": "tensor_list",
                    "bias": "tensor_list", "scale": "tensor_list",
                    "group_list": "tensor", "split_item": "int",
                    "group_type": "int", "group_list_type": "int",
                    "act_type": "int"}},
}

def test_f11_independent_callable_types_and_fixed_array_arity():
    for op_name, expected in _EXPECTED_TYPES.items():
        recipe = catalog_entry(op_name)["repro_recipe"]
        params = recipe["signature"]["params"]
        assert [(name, params[name]["kind"])
                for name in recipe.get("positional") or ()] == expected["positional"]
        for name, kind in expected["keyword"].items():
            assert params[name]["kind"] == kind, (op_name, name)
    binding = bind_captured_occurrence(_dynamic_quant())
    occurrence = _compact_occurrence(binding)
    assert occurrence["call_plan"]["positional_names"] == ["x"]
    stub = _typed_stub({"x": "tensor", "smooth_scales": "tensor",
                        "group_index": "tensor", "dst_type": "int",
                        "quant_mode": "str", "dst_type_max": "float"})
    namespace = exec_script(generate_single_op_reproducer(binding)["script"])
    namespace["_resolve"] = lambda _path: stub
    inputs = {spec["name"]: _FakeTensor(spec["shape"])
              for spec in occurrence["allocations"]
              if spec["direction"] == "input"}
    namespace["_invoke"](occurrence["call_plan"], inputs, {}, {},
                         occurrence["attrs"], {})
    bad = copy.deepcopy(occurrence["call_plan"])
    bad["keyword"]["quant_mode"]["value"] = 7
    with pytest.raises((TypeError, ValueError)):
        namespace["_invoke"](bad, inputs, {}, {}, occurrence["attrs"], {})

def test_f11_generated_telemetry_can_be_disabled_without_subprocess(monkeypatch):
    import subprocess
    calls = []
    monkeypatch.setattr(subprocess, "run",
                        lambda *args, **kwargs: calls.append(args))
    monkeypatch.setenv("MINDIE_REPRO_DISABLE_TELEMETRY", "1")
    namespace = exec_script(generate_single_op_reproducer(paged_binding())["script"])
    assert namespace["_npu_smi"]() == {"aic_mhz": None, "power_w": None}
    assert calls == []

class _FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

def _typed_stub(kinds, fixed_lengths=None):
    fixed_lengths = fixed_lengths or {}

    def stub(*args, **kwargs):
        for value in args:
            if not isinstance(value, _FakeTensor):
                raise TypeError(f"expected tensor, got {type(value)!r}")
        for name, value in kwargs.items():
            kind = kinds.get(name)
            if kind is None:
                raise TypeError(f"unknown keyword {name!r}")
            if value is None:
                continue
            if kind == "tensor":
                if not isinstance(value, _FakeTensor):
                    raise TypeError(f"{name}: expected tensor")
            elif kind == "tensor_list":
                if not isinstance(value, (list, tuple)) or not all(
                        isinstance(item, _FakeTensor) for item in value):
                    raise TypeError(f"{name}: expected tensor[]")
            elif kind == "bool":
                if not isinstance(value, bool):
                    raise TypeError(f"{name}: expected bool")
            elif kind == "int":
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f"{name}: expected non-bool int")
            elif kind == "str":
                if not isinstance(value, str):
                    raise TypeError(f"{name}: expected str")
            elif kind == "float":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"{name}: expected non-bool number")
            elif kind in {"int_array", "array"}:
                if not isinstance(value, (list, tuple)) or not all(
                        isinstance(item, int) and not isinstance(item, bool)
                        for item in value):
                    raise TypeError(f"{name}: expected strict int[]")
                length = fixed_lengths.get(name)
                if length is not None and len(value) != length:
                    raise ValueError(f"{name}: expected length {length}")
        return ("y", "scale")
    return stub

def _dynamic_quant():
    return {
        "op_type": "aclnnDynamicQuant", "cann_version": "9.0",
        "input_shapes": ["[4,8]", "[4,8]"], "input_names": ["x", "smooth_scales"],
        "input_dtypes": ["BF16", "BF16"], "input_formats": ["ND", "ND"],
        "output_shapes": ["[4,8]", "[4,8]"], "output_names": ["y", "scale"],
        "output_dtypes": ["INT8", "FP32"], "output_formats": ["ND", "ND"],
        "attrs": {"dst_type": 1, "quant_mode": "per_token"},
    }

@pytest.mark.parametrize("route", ["reader", "abort"])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("error_type", [OSError, TimeoutError, PermissionError])
@pytest.mark.parametrize("faults", [3, 4, "reused"])
def test_r20_stream_owner_survives_reuse_and_gc(route, active, error_type, faults, tmp_path, monkeypatch):
    import builtins
    import contextlib
    import gc
    import io
    import os
    import sys
    import weakref
    from mindie_profiler.fabric_probe import emit_fabric_probe_script
    target, foreign = tmp_path / "target", tmp_path / "foreign"
    target.write_bytes(b"old target")
    foreign.write_bytes(b"foreign bytes" * 16)
    primary, secondary = TimeoutError("primary"), error_type("close")
    handles, calls, reused = [], [], []
    real_open, real_close, fileopen = os.open, os.close, builtins.open
    def closing(stream, base):
        calls.append(1)
        if faults == "reused":
            fd = stream.fileno()
            reserved = real_open(foreign, os.O_RDWR)
            real_close(fd)
            newfd = real_open(foreign, os.O_RDWR)
            assert newfd == fd
            reused.append(fd)
            handles[0] = (stream, fd)
            real_close(reserved)
            raise secondary  # The stream remains open, but fd now belongs to someone else.
        if len(calls) <= faults:
            raise secondary
        return base()
    class Reader(io.BufferedReader):
        def read(self, *args):
            if active:
                raise primary
            return super().read(*args)
        def close(self):
            return closing(self, super().close)
    class Writer(io.TextIOWrapper):
        def close(self):
            return closing(self, super().close)
    def fdopen(fd, *args, **kwargs):
        handle = Reader(io.FileIO(fd, "rb", closefd=True))
        handles.append((handle, fd))
        return handle
    def opened(path, *args, **kwargs):
        handle = Writer(fileopen(path, "xb", buffering=0, opener=kwargs["opener"]), encoding="utf-8")
        handles.append((handle, handle.fileno()))
        return handle
    ns = {"__name__": "r20_owner"}
    exec(emit_fabric_probe_script(build_fabric_probe_plan(world_size=2))["script"], ns)
    ns.update(ABORT_FILE=str(target), open=opened)
    caught = None
    with monkeypatch.context() as patcher:
        patcher.setattr(os, "fdopen", fdopen)
        with contextlib.suppress(BaseException):
            try:
                if route == "reader":
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
    assert len(handles) == 1 and handles[0][0].closed
    assert len(calls) == (1 if faults == "reused" else 4)
    if active:
        assert caught is primary
    elif error_type is TimeoutError:
        assert caught is secondary
    else:
        assert caught is None
    assert target.read_bytes() == b"old target"
    fd = handles[0][1]
    for error in (primary, secondary, caught):
        if error is not None:
            error.__traceback__ = error.__context__ = error.__cause__ = None
    caught = None
    gc.collect()
    if not reused:
        with pytest.raises(OSError):
            os.fstat(fd)
        assert real_open(foreign, os.O_RDWR) == fd
    assert checked_io.inode_of(foreign) == (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
    reference = weakref.ref(handles[0][0])
    handles.clear()
    gc.collect()
    assert reference() is None
    assert os.read(fd, 1000) == b"foreign bytes" * 16
    assert foreign.read_bytes() == b"foreign bytes" * 16
    real_close(fd)
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob("*.publish"))

@pytest.mark.parametrize("route", ["writer", "verifier", "fdopen_failure", "stream_fallback", "unknown_receipt"])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("error_type", [TimeoutError, PermissionError])
@pytest.mark.parametrize("reuse", [False, True])
def test_r20_raw_close_receipt(route, active, error_type, reuse, tmp_path, monkeypatch):
    import contextlib
    import hashlib
    import io
    import os
    import sys
    target, foreign = tmp_path / "target", tmp_path / "foreign"
    target.write_bytes(b"owned")
    foreign.write_bytes(b"foreign")
    primary, secondary = TimeoutError("body"), error_type("post close")
    real = {name: getattr(os, name) for name in ("open", "close", "fstat", "fsync", "read", "fdopen")}
    tracked, calls, handles = [], [], []
    def opened(path, *args, **kwargs):
        fd = real["open"](path, *args, **kwargs)
        if not tracked:
            tracked.append(fd)
        return fd
    def close(fd):
        if fd == tracked[0]:
            calls.append(fd)
            if len(calls) == 1:
                real["close"](fd)
                if reuse:
                    assert real["open"](foreign, os.O_RDONLY) == fd
                raise secondary
        return real["close"](fd)
    def body(name, fd, *args):
        if active:
            raise primary
        return real[name](fd, *args)
    class Reader(io.BufferedReader):
        def read(self, *args):
            return body("read", self.fileno(), 100)
        def close(self):
            raise secondary
    def fdopen(fd, *args, **kwargs):
        if route == "fdopen_failure":
            raise primary
        stream = Reader(io.FileIO(fd, "rb", closefd=False))
        handles.append(stream)
        return stream
    caught = None
    with monkeypatch.context() as patcher:
        for name, action in {"open": opened, "close": close, "fdopen": fdopen,
                             "fsync": lambda fd: body("fsync", fd),
                             "read": lambda fd, *args: body("read", fd, *args)}.items():
            patcher.setattr(os, name, action)
        with contextlib.suppress(BaseException):
            try:
                if route == "writer":
                    checked_io.safe_write_bytes(target, b"owned")
                elif route == "verifier":
                    checked_io._verified_inode(target, checked_io.inode_of(target), hashlib.sha256(b"owned").hexdigest())
                elif route == "unknown_receipt":
                    fd = opened(target, os.O_RDONLY)
                    patcher.setattr(os, "fstat", lambda fd: (_ for _ in ()).throw(PermissionError("receipt")))
                    checked_io._close_owned_fd(fd, active_exc=primary if active else None)
                else:
                    checked_io.read_regular_bytes(target)
            finally:
                caught = sys.exc_info()[1]
    assert len(calls) == 1
    if route == "unknown_receipt" and active:
        assert caught is None
    elif active or route == "fdopen_failure":
        assert caught is primary
    elif error_type is TimeoutError or route == "unknown_receipt":
        assert caught is secondary
    else:
        assert caught is None
    assert all(handle.closed for handle in handles)
    if reuse:
        assert real["read"](tracked[0], 100) == b"foreign"
        real["close"](tracked[0])
    else:
        with pytest.raises(OSError):
            real["fstat"](tracked[0])

@pytest.mark.parametrize("boundary", ["write", "flush", "fsync", "close", "short_write"])
@pytest.mark.parametrize("active", [False, True])
def test_r20_abort_requires_completed_write(boundary, active, tmp_path, monkeypatch):
    import builtins
    import contextlib
    import io
    import os
    import sys
    from mindie_profiler.fabric_probe import emit_fabric_probe_script
    target = tmp_path / "abort"
    target.write_bytes(b"existing abort")
    primary, secondary = ValueError("outer"), TimeoutError("abort commit")
    handles, real_open = [], builtins.open
    class Writer(io.TextIOWrapper):
        def write(self, data):
            if boundary == "short_write":
                return 0
            result = super().write(data)
            if boundary == "write":
                raise secondary
            return result
        def flush(self):
            if boundary == "flush":
                raise secondary
            return super().flush()
        def close(self):
            if boundary == "close":
                raise secondary
            return super().close()
    def opened(path, *args, **kwargs):
        handle = Writer(real_open(path, "xb", buffering=0, opener=kwargs["opener"]), encoding="utf-8")
        handles.append(handle)
        return handle
    ns = {"__name__": "r20_abort_commit"}
    exec(emit_fabric_probe_script(build_fabric_probe_plan(world_size=2))["script"], ns)
    ns.update(open=opened, ABORT_FILE=str(target))
    if boundary == "fsync":
        monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(secondary))
    caught = None
    with contextlib.suppress(BaseException):
        try:
            if active:
                try:
                    raise primary
                finally:
                    ns["_write_abort_record"]({"kind": "abort"}, primary)
            else:
                ns["_write_abort_record"]({"kind": "abort"})
        finally:
            caught = sys.exc_info()[1]
    assert caught is (primary if active else None if boundary == "short_write" else secondary)
    assert handles and all(handle.closed for handle in handles)
    assert target.read_bytes() == b"existing abort" and list(tmp_path.iterdir()) == [target]
